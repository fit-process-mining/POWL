from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

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


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


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
    coloring: Dict[Any, str],
    main_color: str,
) -> Dict[Any, str]:
    list_of_paths = list(nx.all_simple_paths(graph, source=start_node, target=end_node))
    nodes_in_subgraph = set(node for path in list_of_paths for node in path)
    subgraph = graph.subgraph(nodes_in_subgraph).copy()
    used_colors = [
        coloring[node]
        for node in subgraph.nodes
        if node in coloring and graph.nodes[node].get("content") is not None
    ]
    if len(set(used_colors)) == 1:
        color = used_colors[0]
        for node in subgraph.nodes:
            coloring[node] = color
    return coloring


def _get_predecessors_color(G: nx.DiGraph, node, coloring: Dict[Any, str]) -> Optional[str]:
    predecessors = list(G.predecessors(node))
    # Prioritize concrete tasks with explicit resource assignments
    for pred in predecessors:
        if pred in coloring and G.nodes[pred].get("content") is not None:
            return coloring[pred]
    for pred in predecessors:
        if pred in coloring:
            return coloring[pred]
    return None


def _get_successors_color(G: nx.DiGraph, node, coloring: Dict[Any, str]) -> Optional[str]:
    successors = list(G.successors(node))
    # Prioritize concrete tasks with explicit resource assignments
    for succ in successors:
        if succ in coloring and G.nodes[succ].get("content") is not None:
            return coloring[succ]
    for succ in successors:
        if succ in coloring:
            return coloring[succ]
    return None


def _color_graph(graph: nx.DiGraph, pools: Dict[str, Set[str]]) -> Dict[Any, str]:
    """Color the graph nodes based on pool assignments."""
    pool_with_most_tasks = (
        max(pools, key=lambda k: len(pools[k])) if pools else "DefaultPool"
    )
    coloring: Dict[Any, str] = {}

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
                coloring[node_for_task] = pool

    for node in graph.nodes:
        counterparts = graph.nodes[node].get("paired_with") or []
        if graph.nodes[node].get("type") == "diverging":
            for counterpart in counterparts:
                if counterpart in graph.nodes:
                    coloring = _color_subgraph(
                        graph, node, counterpart, coloring, pool_with_most_tasks
                    )

    for node in graph.nodes:
        if node not in coloring:
            if "Gateway" in str(node):
                gw_type = _get_gateway_type(graph, node)
                if gw_type == "converging":
                    c = _get_predecessors_color(graph, node, coloring)
                    if c:
                        coloring[node] = c
                elif gw_type == "diverging":
                    c = _get_successors_color(graph, node, coloring)
                    if c:
                        coloring[node] = c
                if node not in coloring:
                    coloring[node] = pool_with_most_tasks

    for node in graph.nodes:
        if node not in coloring:
            node_type = str(graph.nodes[node].get("type", ""))
            if node_type == "startEvent" or "start" in str(node).lower():
                successors = list(graph.successors(node))
                visited = set()
                while successors:
                    cur = successors.pop(0)
                    if cur in visited:
                        continue
                    visited.add(cur)
                    if cur in coloring:
                        coloring[node] = coloring[cur]
                        break
                    successors.extend(graph.successors(cur))
            elif node_type == "endEvent" or "end" in str(node).lower():
                predecessors = list(graph.predecessors(node))
                visited = set()
                while predecessors:
                    cur = predecessors.pop(0)
                    if cur in visited:
                        continue
                    visited.add(cur)
                    if cur in coloring:
                        coloring[node] = coloring[cur]
                        break
                    predecessors.extend(graph.predecessors(cur))

    for node in graph.nodes:
        if node not in coloring:
            coloring[node] = pool_with_most_tasks

    return coloring


def _add_intermediate_events_to_graph(
    G: nx.DiGraph, coloring: Dict[Any, str]
) -> Tuple[nx.DiGraph, Dict[Any, str]]:
    """Insert intermediate throw and catch message events for edges crossing pools."""
    G_new = G.copy()
    coloring_new = coloring.copy()
    edges_in_graph = list(G.edges())
    edges_to_add = []
    edges_to_remove = []

    for u, v in edges_in_graph:
        if coloring.get(u) != coloring.get(v):
            throw_node_id = f"IntermediateThrowEvent_{abs(hash(str(u)))}_{abs(hash(str(v)))}"
            catch_node_id = f"IntermediateCatchEvent_{abs(hash(str(u)))}_{abs(hash(str(v)))}"

            G_new.add_node(throw_node_id, type="throw")
            coloring_new[throw_node_id] = coloring[u]

            G_new.add_node(catch_node_id, type="catch")
            coloring_new[catch_node_id] = coloring[v]

            edges_to_remove.append((u, v))
            edges_to_add.append((u, throw_node_id))
            edges_to_add.append((throw_node_id, catch_node_id))
            edges_to_add.append((catch_node_id, v))

    G_new.remove_edges_from(edges_to_remove)
    G_new.add_edges_from(edges_to_add)
    return G_new, coloring_new


def _propagate_lanes(
    G: nx.DiGraph,
    coloring: Dict[Any, str],
    activity_to_pool_lane: Dict[str, Tuple[str, str]],
) -> Dict[Any, str]:
    """Assign every flow node to a lane within its pool by following task lane assignments."""
    node_to_lane: Dict[Any, str] = {}
    for n, d in G.nodes(data=True):
        content = d.get("content")
        if content in activity_to_pool_lane:
            node_to_lane[n] = activity_to_pool_lane[content][1]

    # Propagate along intra-pool edges
    for _ in range(5):
        for n in list(G.nodes):
            if n in node_to_lane:
                continue
            pool = coloring.get(n)
            preds = [p for p in G.predecessors(n) if coloring.get(p) == pool and p in node_to_lane]
            succs = [s for s in G.successors(n) if coloring.get(s) == pool and s in node_to_lane]
            if "Gateway" in str(n):
                gw_type = _get_gateway_type(G, n)
                if gw_type == "converging" and preds:
                    node_to_lane[n] = node_to_lane[preds[0]]
                elif gw_type == "diverging" and succs:
                    node_to_lane[n] = node_to_lane[succs[0]]
                elif preds:
                    node_to_lane[n] = node_to_lane[preds[0]]
                elif succs:
                    node_to_lane[n] = node_to_lane[succs[0]]
            else:
                if preds:
                    node_to_lane[n] = node_to_lane[preds[0]]
                elif succs:
                    node_to_lane[n] = node_to_lane[succs[0]]

    pool_to_first_lane = {}
    for a, (p, l) in activity_to_pool_lane.items():
        if p not in pool_to_first_lane:
            pool_to_first_lane[p] = l
    for n in G.nodes:
        if n not in node_to_lane:
            p = coloring.get(n)
            node_to_lane[n] = pool_to_first_lane.get(p, "DefaultLane")

    return node_to_lane


def _build_unlayouted_collaboration_xml(
    flat_xml_str: str,
    activity_to_pool_lane: Dict[str, Tuple[str, str]],
    element_id_to_pool: Dict[str, str],
    element_id_to_lane: Dict[str, str],
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

    pools_set: Set[str] = set(element_id_to_pool.values())

    # Build pool to ordered lanes mapping
    pool_to_lanes: Dict[str, List[str]] = defaultdict(list)
    for a, (p, l) in activity_to_pool_lane.items():
        if p and l and l not in pool_to_lanes[p]:
            pool_to_lanes[p].append(l)

    flow_nodes: Dict[str, ET.Element] = {}
    sequence_flows: List[ET.Element] = []
    for child in list(source_process):
        tag_local = _strip_ns(child.tag)
        child_id = child.attrib.get("id", "")
        if tag_local == "sequenceFlow":
            sequence_flows.append(child)
        else:
            flow_nodes[child_id] = child

    intra_process_flows: Dict[str, List[ET.Element]] = defaultdict(list)
    message_flows: List[Tuple[str, str, str]] = []

    for sf in sequence_flows:
        fid = sf.attrib.get("id", "")
        src = sf.attrib.get("sourceRef", "")
        tgt = sf.attrib.get("targetRef", "")
        src_p = element_id_to_pool.get(src)
        tgt_p = element_id_to_pool.get(tgt)

        if src_p == tgt_p and src_p is not None:
            intra_process_flows[src_p].append(sf)
        else:
            message_flows.append((fid, src, tgt))
            src_elem = flow_nodes.get(src)
            if src_elem is not None and "ThrowEvent" in _strip_ns(src_elem.tag):
                if not any(_strip_ns(c.tag) == "messageEventDefinition" for c in src_elem):
                    ET.SubElement(
                        src_elem,
                        f"{{{bpmn_ns}}}messageEventDefinition",
                        attrib={"id": f"MessageEventDefinition_{abs(hash(src))}"},
                    )
            tgt_elem = flow_nodes.get(tgt)
            if tgt_elem is not None and "CatchEvent" in _strip_ns(tgt_elem.tag):
                if not any(_strip_ns(c.tag) == "messageEventDefinition" for c in tgt_elem):
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

        # Add flow nodes for this process and assign flowNodeRefs to their lane
        for nid, elem in flow_nodes.items():
            if element_id_to_pool.get(nid) == pool_name:
                proc.append(elem)
                assigned_lane = element_id_to_lane.get(nid, lanes_for_pool[0])
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

    coloring: Dict[Any, str] = {}
    if len(pools) > 1:
        coloring = _color_graph(G, pools)
        G, coloring = _add_intermediate_events_to_graph(G, coloring)
    else:
        pool_name = list(pools.keys())[0] if pools else "DefaultPool"
        coloring = {node: pool_name for node in G.nodes()}

    node_to_lane = _propagate_lanes(G, coloring, activity_to_pool_lane)

    bpmn, elements_mapping = pminer_bpmn_transformer.__transform_to_bpmn(G)
    raw_bpmn_xml = get_xml_string(bpmn).decode("utf-8")

    # Map transformed BPMN element IDs to pool names and lanes
    element_id_to_pool: Dict[str, str] = {}
    element_id_to_lane: Dict[str, str] = {}
    for orig_node, pool_name in coloring.items():
        elem_id = elements_mapping.get(str(orig_node), str(orig_node))
        element_id_to_pool[elem_id] = pool_name
        element_id_to_lane[elem_id] = node_to_lane.get(orig_node, "DefaultLane")

    # Build collaboration XML with pools and lanes
    unlayouted_xml = _build_unlayouted_collaboration_xml(
        raw_bpmn_xml, activity_to_pool_lane, element_id_to_pool, element_id_to_lane
    )

    # Run the new Graphviz + post-processing layouter pipeline
    return layout_bpmn_xml(unlayouted_xml)

