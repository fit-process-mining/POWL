"""The reference method's weighted DFG noise filter, including empty traces.

The start-to-end edge for an empty trace competes with activity start edges in
the same relative frequency filter.  Connectivity repair protects activities;
it does not restore a filtered empty-trace edge.  Boundary objects are private
sentinels, so user labels such as ``__START__`` remain ordinary activities.
"""

from __future__ import annotations

from collections import Counter, deque
from heapq import heapify, heappop, heappush
from math import inf

import networkx as nx
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.objects.dfg.obj import DFG

_START = object()
_END = object()
_EMPTY = object()


def _node_key(node):
    # Preserve the source's lexical boundary order without reserving labels.
    if node is _START:
        return "__START__", 0
    if node is _END:
        return "__END__", 0
    return node, 1


def _augmented_graph(variants):
    """Build the source's graph ordering directly from weighted variants."""
    relations = Counter()
    for trace, frequency in variants.items():
        if trace:
            relations[_START, trace[0]] += frequency
            relations[trace[-1], _END] += frequency
            for a, b in zip(trace, trace[1:]):
                relations[a, b] += frequency
        else:
            relations[_START, _END] += frequency

    ordinary = nx.DiGraph()
    for (a, b), frequency in relations.items():
        if a is _START and b is _END:
            ordinary.add_edge(_EMPTY, _EMPTY, weight=frequency)
        elif a is _START:
            ordinary.add_node(b, start=frequency)
        elif b is _END:
            ordinary.add_node(a, end=frequency)
        else:
            ordinary.add_edge(a, b, weight=frequency)

    augmented = nx.DiGraph()
    augmented.add_nodes_from((_START, _END))
    for node, data in ordinary.nodes(data=True):
        if node is _EMPTY:
            continue
        augmented.add_node(node)
        if data.get("start", 0):
            augmented.add_edge(_START, node, weight=data["start"])
        if data.get("end", 0):
            augmented.add_edge(node, _END, weight=data["end"])
    for a, b, data in ordinary.edges(data=True):
        if a is _EMPTY:
            augmented.add_edge(_START, _END, weight=data["weight"])
        else:
            augmented.add_edge(a, b, weight=data["weight"])
    return augmented


def _reachable(graph, initial, *, reverse=False):
    reached = set(initial)
    pending = deque(initial)
    neighbors = graph.predecessors if reverse else graph.successors
    while pending:
        for neighbor in neighbors(pending.popleft()):
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return reached


def _widest_links(graph, initial, *, reverse=False):
    """Parents, or next hops, along maximum-bottleneck original paths."""
    width = {node: inf for node in initial}
    links = {node: None for node in initial}
    heap = [(-inf, _node_key(node), node) for node in initial]
    heapify(heap)
    neighbors = graph.predecessors if reverse else graph.successors
    while heap:
        negative_width, _, node = heappop(heap)
        node_width = -negative_width
        if node_width != width.get(node):
            continue
        for neighbor in neighbors(node):
            a, b = (neighbor, node) if reverse else (node, neighbor)
            capacity = min(node_width, graph[a][b]["weight"])
            if capacity > width.get(neighbor, -1):
                width[neighbor] = capacity
                links[neighbor] = node
                heappush(heap, (-capacity, _node_key(neighbor), neighbor))
    return links


def filter_data(obj: IMDataStructureUVCL, threshold: float):
    """Return ``(filtered_obj, has_empty_trace_arc)`` without changing traces.

    The filtered DFG has no sentinel activities.  Its accompanying boolean
    tells the miner whether the source method would attempt its empty-trace
    fall-through.  Projecting a cut still uses the original weighted traces.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("noise threshold must satisfy 0 <= x <= 1.")
    if threshold == 0 or not obj.data_structure:
        return obj, bool(obj.data_structure.get((), 0))
    original = _augmented_graph(obj.data_structure)
    activities = set(original) - {_START, _END}
    reachable = _reachable(original, {_START})
    coreachable = _reachable(original, {_END}, reverse=True)
    if activities - (reachable & coreachable):
        raise ValueError("Every activity must lie on an original start-to-end path.")

    outgoing_max = {
        node: max(data["weight"] for _, _, data in original.out_edges(node, data=True))
        for node in original
        if original.out_degree(node)
    }
    filtered = nx.DiGraph()
    filtered.add_nodes_from(original)
    for a, b, data in original.edges(data=True):
        if data["weight"] >= threshold * outgoing_max[a]:
            filtered.add_edge(a, b, weight=data["weight"])

    # These guards mirror the reference; at least a maximum-weight start/end
    # arc normally survives any threshold in [0, 1].
    if filtered.out_degree(_START) == 0 and original.out_degree(_START):
        a, b, data = max(
            original.out_edges(_START, data=True),
            key=lambda edge: (edge[2]["weight"], _node_key(edge[1])),
        )
        filtered.add_edge(a, b, weight=data["weight"])
    if filtered.in_degree(_END) == 0 and original.in_degree(_END):
        a, b, data = max(
            original.in_edges(_END, data=True),
            key=lambda edge: (edge[2]["weight"], _node_key(edge[0])),
        )
        filtered.add_edge(a, b, weight=data["weight"])

    while True:
        reachable = _reachable(filtered, {_START})
        coreachable = _reachable(filtered, {_END}, reverse=True)
        broken = activities - (reachable & coreachable)
        if not broken:
            break
        parents = _widest_links(original, reachable)
        next_hops = _widest_links(original, coreachable, reverse=True)
        progressed = False
        for activity in sorted(broken):
            if activity not in parents or activity not in next_hops:
                raise ValueError(
                    "Noise-filter repair could not restore activity connectivity."
                )
            path = []
            cursor = activity
            while cursor is not None and cursor not in reachable:
                path.append(cursor)
                cursor = parents[cursor]
            for node in reversed(path):
                parent = parents[node]
                if parent is not None:
                    filtered.add_edge(
                        parent, node, weight=original[parent][node]["weight"]
                    )
                    progressed = True
            cursor = activity
            while cursor is not None and cursor not in coreachable:
                following = next_hops[cursor]
                if following is None:
                    break
                filtered.add_edge(
                    cursor, following, weight=original[cursor][following]["weight"]
                )
                progressed = True
                cursor = following
        if not progressed:
            raise ValueError("Noise-filter connectivity repair made no progress.")

    dfg = DFG()
    for a, b, data in filtered.edges(data=True):
        if a is _START and b is _END:
            continue
        if a is _START:
            dfg.start_activities[b] = data["weight"]
        elif b is _END:
            dfg.end_activities[a] = data["weight"]
        else:
            dfg.graph[a, b] = data["weight"]
    return IMDataStructureUVCL(obj.data_structure, dfg), filtered.has_edge(_START, _END)
