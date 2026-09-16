from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from pm4py.objects.bpmn.exporter.variants.etree import get_xml_string

import powl.conversion.variants.to_bpmn as pminer_bpmn_transformer
from powl.visualization.bpmn.layout import layout_bpmn_xml

# Standard BPMN Namespaces
NAMESPACES = {
    "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
    "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
    "dc": "http://www.omg.org/spec/DD/20100524/DC",
    "di": "http://www.omg.org/spec/DD/20100524/DI",
}


def _pools_to_tasks(
    activity_to_pool_lane: Dict[str, Tuple[str, str]]
) -> Dict[str, Set[str]]:
    """Convert activity-to-pool-lane mapping to pool-to-tasks mapping."""
    pool_tasks = defaultdict(set)
    for activity, (pool, _) in activity_to_pool_lane.items():
        pool_name = pool if pool is not None else "DefaultPool"
        pool_tasks[pool_name].add(activity)
    return dict(pool_tasks)


def _get_gateway_type(G: nx.DiGraph, node) -> str:
    predecessors = list(G.predecessors(node))
    successors = list(G.successors(node))
    if len(predecessors) > len(successors):
        return "converging"
    elif len(predecessors) < len(successors):
        return "diverging"
    else:
        return "mixed"


def _color_subgraph(
    graph: nx.DiGraph,
    start_node,
    end_node,
    coloring: Dict[str, str],
    main_color: str,
) -> Dict[str, str]:
    list_of_paths = list(nx.all_simple_paths(graph, source=start_node, target=end_node))
    nodes_in_subgraph = set(node for path in list_of_paths for node in path)
    subgraph = graph.subgraph(nodes_in_subgraph).copy()
    used_colors = [
        coloring[str(node)]
        for node in subgraph.nodes
        if str(node) in coloring and graph.nodes[node].get("content") is not None
    ]
    if len(set(used_colors)) == 1:
        color = used_colors[0]
        for node in subgraph.nodes:
            coloring[str(node)] = color
    return coloring


def _get_predecessors_color(G: nx.DiGraph, node, coloring: Dict[str, str]) -> Optional[str]:
    for predecessor in G.predecessors(node):
        if str(predecessor) in coloring:
            return coloring[str(predecessor)]
    return None


def _get_successors_color(G: nx.DiGraph, node, coloring: Dict[str, str]) -> Optional[str]:
    for successor in G.successors(node):
        if str(successor) in coloring:
            return coloring[str(successor)]
    return None


def _color_graph(graph: nx.DiGraph, pools: Dict[str, Set[str]]) -> Dict[str, str]:
    """Color the graph nodes based on pool assignments."""
    pool_with_most_tasks = (
        max(pools, key=lambda k: len(pools[k])) if pools else "DefaultPool"
    )
    coloring: Dict[str, str] = {}

    for pool, tasks in pools.items():
        for task in tasks:
            node_for_task = next(
                (
                    n
                    for n, attr in graph.nodes(data=True)
                    if attr.get("content") == task
                ),
                None,
            )
            if node_for_task is not None:
                coloring[str(node_for_task)] = pool

    for node in graph.nodes:
        if graph.nodes[node].get("type") == "diverging":
            counterparts = graph.nodes[node].get("paired_with") or []
            for counterpart in counterparts:
                if counterpart in graph.nodes:
                    coloring = _color_subgraph(
                        graph, node, counterpart, coloring, pool_with_most_tasks
                    )

    for node in graph.nodes:
        node_str = str(node)
        if node_str not in coloring:
            if "gateway" in node_str.lower():
                gw_type = _get_gateway_type(graph, node)
                if gw_type == "converging":
                    c = _get_predecessors_color(graph, node, coloring)
                    if c:
                        coloring[node_str] = c
                elif gw_type == "diverging":
                    c = _get_successors_color(graph, node, coloring)
                    if c:
                        coloring[node_str] = c
                if node_str not in coloring:
                    coloring[node_str] = pool_with_most_tasks

    for node in graph.nodes:
        node_str = str(node)
        if node_str not in coloring:
            node_type = str(graph.nodes[node].get("type", ""))
            if node_type == "startEvent" or "start" in node_str.lower():
                successors = list(graph.successors(node))
                visited = set()
                while successors:
                    cur = successors.pop(0)
                    if cur in visited:
                        continue
                    visited.add(cur)
                    if str(cur) in coloring:
                        coloring[node_str] = coloring[str(cur)]
                        break
                    successors.extend(graph.successors(cur))
            elif node_type == "endEvent" or "end" in node_str.lower():
                predecessors = list(graph.predecessors(node))
                visited = set()
                while predecessors:
                    cur = predecessors.pop(0)
                    if cur in visited:
                        continue
                    visited.add(cur)
                    if str(cur) in coloring:
                        coloring[node_str] = coloring[str(cur)]
                        break
                    predecessors.extend(graph.predecessors(cur))

    for node in graph.nodes:
        node_str = str(node)
        if node_str not in coloring:
            coloring[node_str] = pool_with_most_tasks

    return coloring


def _add_intermediate_events_to_graph(
    G: nx.DiGraph, coloring: Dict[str, str]
) -> Tuple[nx.DiGraph, Dict[str, str]]:
    """Insert intermediate throw and catch message events for edges crossing pools."""
    G_new = G.copy()
    coloring_new = coloring.copy()
    edges_in_graph = list(G.edges())
    edges_to_add = []
    edges_to_remove = []

    for u, v in edges_in_graph:
        u_str, v_str = str(u), str(v)
        if coloring.get(u_str) != coloring.get(v_str):
            throw_node_id = f"IntermediateThrowEvent_{abs(hash(u_str))}_{abs(hash(v_str))}"
            catch_node_id = f"IntermediateCatchEvent_{abs(hash(u_str))}_{abs(hash(v_str))}"

            G_new.add_node(throw_node_id, type="throw")
            coloring_new[throw_node_id] = coloring[u_str]

            G_new.add_node(catch_node_id, type="catch")
            coloring_new[catch_node_id] = coloring[v_str]

            edges_to_remove.append((u, v))
            edges_to_add.append((u, throw_node_id))
            edges_to_add.append((throw_node_id, catch_node_id))
            edges_to_add.append((catch_node_id, v))

    G_new.remove_edges_from(edges_to_remove)
    G_new.add_edges_from(edges_to_add)
    return G_new, coloring_new


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _build_unlayouted_collaboration_xml(
    flat_xml_str: str,
    activity_to_pool_lane: Dict[str, Tuple[str, str]],
    element_id_to_pool: Dict[str, str],
) -> str:
    """Restructure flat BPMN 2.0 XML into collaboration with pools, lanes, and message flows."""
    for prefix, uri in NAMESPACES.items():
        ET.register_namespace(prefix, uri)

    root = ET.fromstring(flat_xml_str)
    bpmn_ns = NAMESPACES["bpmn"]

    source_process = None
    for child in root:
        if _strip_ns(child.tag) == "process":
            source_process = child
            break

    if source_process is None:
        return flat_xml_str

    # Remove any existing diagram
    for child in list(root):
        if _strip_ns(child.tag) == "BPMNDiagram":
            root.remove(child)

    # Collect distinct pools and lanes
    pools_set: Set[str] = set()
    pool_to_lanes: Dict[str, List[str]] = defaultdict(list)
    activity_to_lane: Dict[str, str] = {}

    for act, (p, l) in activity_to_pool_lane.items():
        pool_name = p if p is not None else "DefaultPool"
        lane_name = l if l is not None else "DefaultLane"
        pools_set.add(pool_name)
        if lane_name not in pool_to_lanes[pool_name]:
            pool_to_lanes[pool_name].append(lane_name)
        activity_to_lane[act] = lane_name

    # Ensure all pools in element_id_to_pool are registered
    for p in element_id_to_pool.values():
        pools_set.add(p)
        if not pool_to_lanes[p]:
            pool_to_lanes[p].append("DefaultLane")

    # Map node id to pool and lane
    node_to_pool: Dict[str, str] = {}
    node_to_lane: Dict[str, str] = {}
    flow_nodes: Dict[str, ET.Element] = {}
    sequence_flows: List[ET.Element] = []

    for child in source_process:
        tag_local = _strip_ns(child.tag)
        child_id = child.attrib.get("id", "")
        child_name = child.attrib.get("name", "")

        if tag_local == "sequenceFlow":
            sequence_flows.append(child)
        else:
            flow_nodes[child_id] = child
            pool = element_id_to_pool.get(child_id)
            if not pool and child_name in activity_to_pool_lane:
                pool = activity_to_pool_lane[child_name][0]
            if not pool:
                pool = list(pools_set)[0] if pools_set else "DefaultPool"
            node_to_pool[child_id] = pool

            lane = activity_to_lane.get(child_name)
            if not lane and pool in pool_to_lanes and pool_to_lanes[pool]:
                lane = pool_to_lanes[pool][0]
            if not lane:
                lane = "DefaultLane"
            node_to_lane[child_id] = lane

    # Classify sequence flows vs message flows
    intra_process_flows: Dict[str, List[ET.Element]] = defaultdict(list)
    message_flows: List[Tuple[str, str, str]] = []

    for sf in sequence_flows:
        flow_id = sf.attrib.get("id", "")
        src = sf.attrib.get("sourceRef", "")
        tgt = sf.attrib.get("targetRef", "")
        src_pool = node_to_pool.get(src)
        tgt_pool = node_to_pool.get(tgt)

        if src_pool == tgt_pool and src_pool is not None:
            intra_process_flows[src_pool].append(sf)
        else:
            message_flows.append((flow_id, src, tgt))
            # If source is intermediate throw event, mark as message event
            src_elem = flow_nodes.get(src)
            if src_elem is not None and "ThrowEvent" in _strip_ns(src_elem.tag):
                has_msg_def = any(
                    _strip_ns(c.tag) == "messageEventDefinition" for c in src_elem
                )
                if not has_msg_def:
                    ET.SubElement(
                        src_elem,
                        f"{{{bpmn_ns}}}messageEventDefinition",
                        attrib={"id": f"MessageEventDefinition_{abs(hash(src))}"},
                    )
            # If target is intermediate catch event, mark as message event
            tgt_elem = flow_nodes.get(tgt)
            if tgt_elem is not None and "CatchEvent" in _strip_ns(tgt_elem.tag):
                has_msg_def = any(
                    _strip_ns(c.tag) == "messageEventDefinition" for c in tgt_elem
                )
                if not has_msg_def:
                    ET.SubElement(
                        tgt_elem,
                        f"{{{bpmn_ns}}}messageEventDefinition",
                        attrib={"id": f"MessageEventDefinition_{abs(hash(tgt))}"},
                    )

    # Create collaboration element
    collab = ET.Element(f"{{{bpmn_ns}}}collaboration", attrib={"id": "Collaboration_1"})

    new_processes: List[ET.Element] = []
    for pool_name in sorted(pools_set):
        proc_id = f"Process_{abs(hash(pool_name))}"
        ET.SubElement(
            collab,
            f"{{{bpmn_ns}}}participant",
            attrib={
                "id": f"Participant_{proc_id}",
                "name": pool_name,
                "processRef": proc_id,
            },
        )

        proc = ET.Element(
            f"{{{bpmn_ns}}}process",
            attrib={"id": proc_id, "name": pool_name, "isExecutable": "false"},
        )

        lanes_for_pool = pool_to_lanes[pool_name] or ["DefaultLane"]
        lane_set = ET.SubElement(
            proc, f"{{{bpmn_ns}}}laneSet", attrib={"id": f"LaneSet_{proc_id}"}
        )
        lane_elems = {}
        for l_name in lanes_for_pool:
            lane_id = f"Lane_{abs(hash(l_name))}_{abs(hash(pool_name))}"
            lane_elem = ET.SubElement(
                lane_set,
                f"{{{bpmn_ns}}}lane",
                attrib={"id": lane_id, "name": l_name},
            )
            lane_elems[l_name] = lane_elem

        # Add flow nodes for this process
        for nid, elem in flow_nodes.items():
            if node_to_pool.get(nid) == pool_name:
                proc.append(elem)
                assigned_lane = node_to_lane.get(nid, lanes_for_pool[0])
                if assigned_lane not in lane_elems:
                    assigned_lane = lanes_for_pool[0]
                ref_elem = ET.SubElement(
                    lane_elems[assigned_lane], f"{{{bpmn_ns}}}flowNodeRef"
                )
                ref_elem.text = nid

        # Add sequence flows for this process
        for sf in intra_process_flows.get(pool_name, []):
            proc.append(sf)

        new_processes.append(proc)

    # Add message flows to collaboration
    for mf_id, src_id, tgt_id in message_flows:
        ET.SubElement(
            collab,
            f"{{{bpmn_ns}}}messageFlow",
            attrib={"id": mf_id, "sourceRef": src_id, "targetRef": tgt_id},
        )

    # Reconstruct root
    root.remove(source_process)
    root.append(collab)
    for p in new_processes:
        root.append(p)

    return ET.tostring(root, encoding="utf-8").decode("utf-8")


def apply(activity_to_pool_lane: Dict[str, Tuple[str, str]], powl) -> str:
    """
    Convert a POWL model to BPMN with resources (pools & lanes), and return
    a cleanly layouted BPMN 2.0 XML string.

    :param activity_to_pool_lane: Mapping of activity label to (pool_name, lane_name)
    :param powl: TaggedPOWL model
    :return: Layouted BPMN 2.0 XML string
    """
    original_pools = set(
        p for _, (p, _) in activity_to_pool_lane.items() if p is not None
    )
    activity_to_pool_lane_copy = activity_to_pool_lane.copy()
    if len(original_pools) == 0:
        activity_to_pool_lane = {
            activity: ("ProcessPool", lane if lane is not None else "DefaultLane")
            for activity, (_, lane) in activity_to_pool_lane.items()
        }
    for activity, (pool, lane) in activity_to_pool_lane_copy.items():
        if pool is None and lane is None:
            activity_to_pool_lane[activity] = ("ProcessPool", "DefaultLane")
        elif pool is None:
            activity_to_pool_lane[activity] = ("ProcessPool", lane)
        elif lane is None:
            activity_to_pool_lane[activity] = (pool, "DefaultLane")

    pools = _pools_to_tasks(activity_to_pool_lane)
    _, G, _ = pminer_bpmn_transformer.apply(powl)

    coloring: Dict[str, str] = {}
    if len(pools) > 1:
        coloring = _color_graph(G, pools)
        G, coloring = _add_intermediate_events_to_graph(G, coloring)
    else:
        pool_name = list(pools.keys())[0] if pools else "DefaultPool"
        coloring = {str(node): pool_name for node in G.nodes()}

    bpmn, elements_mapping = pminer_bpmn_transformer.__transform_to_bpmn(G)
    raw_bpmn_xml = get_xml_string(bpmn).decode("utf-8")

    # Map transformed BPMN element IDs to pool names
    element_id_to_pool: Dict[str, str] = {}
    for orig_node, pool_name in coloring.items():
        elem_id = elements_mapping.get(orig_node, orig_node)
        element_id_to_pool[elem_id] = pool_name

    # Build collaboration XML with pools and lanes
    unlayouted_xml = _build_unlayouted_collaboration_xml(
        raw_bpmn_xml, activity_to_pool_lane, element_id_to_pool
    )

    # Run the new Graphviz + post-processing layouter pipeline
    return layout_bpmn_xml(unlayouted_xml)
