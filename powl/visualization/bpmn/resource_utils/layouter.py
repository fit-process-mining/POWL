"""BPMN Layouter using Graphviz.
Computes 2D position coordinates for BPMN flow nodes using Graphviz directed graph layout (rankdir=LR).
"""

from __future__ import annotations
import json
from typing import Dict, Any, Optional

import graphviz

from .input_parser import BPMNModel, Process, FlowNode, Bounds


def layout_process(process: Process) -> Dict[str, Bounds]:
    """Compute (x, y) coordinates for all flow nodes in a process using Graphviz."""
    if not process.flow_nodes:
        return {}

    dot = graphviz.Digraph(format="json")
    dot.attr(rankdir="LR", nodesep="0.6", ranksep="0.8", margin="0.2")

    # Add all flow nodes directly without cluster bounding subgraphs so Graphviz
    # aligns directly sequential nodes on their exact center Y axes.
    for nid, node in process.flow_nodes.items():
        dot.node(
            nid,
            label="",
            width=str(node.bounds.width / 72.0),
            height=str(node.bounds.height / 72.0),
            fixedsize="true",
            shape="box",
        )

    # Add sequence flow edges
    for edge in process.sequence_flows:
        if edge.source_ref in process.flow_nodes and edge.target_ref in process.flow_nodes:
            dot.edge(edge.source_ref, edge.target_ref)

    # Execute graphviz pipe
    json_bytes = dot.pipe(format="json")
    data = json.loads(json_bytes.decode("utf-8"))

    # Graphviz coordinates have bottom-left origin; convert to top-left origin
    bb_str = data.get("bb", "0,0,100,100")
    llx, lly, urx, ury = map(float, bb_str.split(","))

    node_bounds: Dict[str, Bounds] = {}
    for obj in data.get("objects", []):
        name = obj.get("name")
        pos_str = obj.get("pos")
        if name and pos_str and name in process.flow_nodes:
            cx_str, cy_str = pos_str.split(",")
            cx, cy = float(cx_str), float(cy_str)
            w = process.flow_nodes[name].bounds.width
            h = process.flow_nodes[name].bounds.height
            # Invert Y coordinate
            top_y = (ury - cy) - (h / 2.0)
            left_x = cx - (w / 2.0)
            node_bounds[name] = Bounds(x=round(left_x, 1), y=round(top_y, 1), width=w, height=h)

    # Apply computed bounds to process flow nodes
    for nid, b in node_bounds.items():
        process.flow_nodes[nid].bounds = b

    return node_bounds


def layout_model(model: BPMNModel) -> BPMNModel:
    """Run Graphviz layout on the model.
    If collaboration message flows or multiple processes are present, performs
    a unified directed graph layout respecting sequence flows and message flows
    between flow nodes to horizontally space out sub-graphs across pools and lanes.
    """
    if not model.processes:
        return model

    all_flow_nodes: Dict[str, FlowNode] = {}
    for proc in model.processes:
        for nid, node in proc.flow_nodes.items():
            all_flow_nodes[nid] = node

    if not all_flow_nodes:
        return model

    # Check if unified multi-process or message-flow layout is needed
    has_cross_message_flows = False
    if model.collaboration and model.collaboration.message_flows:
        for mf in model.collaboration.message_flows:
            if mf.source_ref in all_flow_nodes and mf.target_ref in all_flow_nodes:
                has_cross_message_flows = True
                break

    if len(model.processes) == 1 and not has_cross_message_flows:
        layout_process(model.processes[0])
        return model

    # Unified layout across all processes
    adj: Dict[str, list[str]] = {nid: [] for nid in all_flow_nodes}
    all_edges: list[tuple[str, str, str]] = []

    for proc in model.processes:
        for edge in proc.sequence_flows:
            if edge.source_ref in all_flow_nodes and edge.target_ref in all_flow_nodes:
                adj[edge.source_ref].append(edge.target_ref)
                all_edges.append((edge.source_ref, edge.target_ref, "sequenceFlow"))

    if model.collaboration:
        for mf in model.collaboration.message_flows:
            if mf.source_ref in all_flow_nodes and mf.target_ref in all_flow_nodes:
                adj[mf.source_ref].append(mf.target_ref)
                all_edges.append((mf.source_ref, mf.target_ref, "messageFlow"))

    # Cycle / feedback loop detection using DFS
    state: Dict[str, int] = {nid: 0 for nid in all_flow_nodes}  # 0: unvisited, 1: visiting, 2: visited
    back_edges: set[tuple[str, str]] = set()

    def dfs(u: str):
        state[u] = 1
        for v in adj[u]:
            if state[v] == 1:
                back_edges.add((u, v))
            elif state[v] == 0:
                dfs(v)
        state[u] = 2

    in_degree: Dict[str, int] = {nid: 0 for nid in all_flow_nodes}
    for u, v, _ in all_edges:
        in_degree[v] += 1

    # Visit in-degree 0 nodes first (especially start events)
    start_nodes = [
        nid for nid in all_flow_nodes
        if in_degree[nid] == 0 or all_flow_nodes[nid].element_type.lower().endswith("startevent")
    ]
    for s in start_nodes:
        if state[s] == 0:
            dfs(s)
    for nid in all_flow_nodes:
        if state[nid] == 0:
            dfs(nid)

    dot = graphviz.Digraph(format="json")
    dot.attr(rankdir="LR", nodesep="0.6", ranksep="0.8", margin="0.2")

    for nid, node in all_flow_nodes.items():
        dot.node(
            nid,
            label="",
            width=str(node.bounds.width / 72.0),
            height=str(node.bounds.height / 72.0),
            fixedsize="true",
            shape="box",
        )

    for u, v, kind in all_edges:
        if (u, v) in back_edges:
            dot.edge(u, v, constraint="false")
        else:
            dot.edge(u, v)

    # Execute graphviz pipe
    json_bytes = dot.pipe(format="json")
    data = json.loads(json_bytes.decode("utf-8"))

    bb_str = data.get("bb", "0,0,100,100")
    llx, lly, urx, ury = map(float, bb_str.split(","))

    for obj in data.get("objects", []):
        name = obj.get("name")
        pos_str = obj.get("pos")
        if name and pos_str and name in all_flow_nodes:
            cx_str, cy_str = pos_str.split(",")
            cx, cy = float(cx_str), float(cy_str)
            w = all_flow_nodes[name].bounds.width
            h = all_flow_nodes[name].bounds.height
            top_y = (ury - cy) - (h / 2.0)
            left_x = cx - (w / 2.0)
            all_flow_nodes[name].bounds = Bounds(
                x=round(left_x, 1),
                y=round(top_y, 1),
                width=w,
                height=h,
            )

    return model