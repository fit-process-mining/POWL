from collections import defaultdict, deque
from copy import copy
from typing import TypeVar

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.objects.dfg.obj import DFG

from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT,
)

T = TypeVar("T", bound=IMDataStructureUVCL)


# Hidden artificial nodes (not stored in DFG.graph; only used for reasoning)
_START = "__START__"
_END = "__END__"


def _build_augmented_adjacency(graph, start_activities, end_activities):
    """
    Builds adjacency including hidden START/END:
      START -> a  for a in start_activities
      a -> END    for a in end_activities
      a -> b      for (a,b) in graph
    Returns (out_adj, in_adj, edge_weight_fn)
    """
    out_adj = defaultdict(list)
    in_adj = defaultdict(list)

    # DFG internal edges
    for (a, b), f in graph.items():
        out_adj[a].append(b)
        in_adj[b].append(a)

    # START arcs
    for a in start_activities.keys():
        out_adj[_START].append(a)
        in_adj[a].append(_START)

    # END arcs
    for a in end_activities.keys():
        out_adj[a].append(_END)
        in_adj[_END].append(a)

    def edge_weight(u, v):
        if u == _START:
            return start_activities[v]
        if v == _END:
            return end_activities[u]
        return graph[(u, v)]

    return out_adj, in_adj, edge_weight


def _bfs_reachable(starts, out_adj):
    q = deque(starts)
    seen = set(starts)
    while q:
        x = q.popleft()
        for y in out_adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                q.append(y)
    return seen


def _bfs_parents_multi_source(sources, out_adj):
    """
    Multi-source BFS that records one parent for each visited node.
    parent[src] is None for all sources.
    """
    q = deque(sources)
    parent = {s: None for s in sources}
    while q:
        u = q.popleft()
        for v in out_adj.get(u, ()):
            if v not in parent:
                parent[v] = u
                q.append(v)
    return parent


def _reverse_bfs_next_multi_sink(sinks, in_adj):
    """
    Multi-sink reverse BFS that records one 'next hop' towards some sink.
    next_hop[x] = y means x -> y moves you closer to a sink.
    For sinks themselves, next_hop[sink] = None.
    """
    q = deque(sinks)
    next_hop = {t: None for t in sinks}
    while q:
        v = q.popleft()
        for u in in_adj.get(v, ()):
            if u not in next_hop:
                next_hop[u] = v
                q.append(u)
    return next_hop


def _add_augmented_edge_to_dfg(dfg, u, v, w):
    """
    Writes an augmented edge back into the DFG object:
      START->a => dfg.start_activities[a] = w
      a->END   => dfg.end_activities[a] = w
      a->b     => dfg.graph[(a,b)] = w
    """
    if u == _START:
        # START -> v
        dfg.start_activities[v] = max(dfg.start_activities.get(v, 0), w)
    elif v == _END:
        # u -> END
        dfg.end_activities[u] = max(dfg.end_activities.get(u, 0), w)
    else:
        dfg.graph[(u, v)] = max(dfg.graph.get((u, v), 0), w)


import math
import heapq

def _widest_parents_multi_source(sources, out_adj, edge_weight):
    """
    Multi-source widest-path (maximum bottleneck) that records one parent.
    width[x] = best bottleneck capacity from any source to x.
    parent[src] = None for all sources.
    """
    width = {s: math.inf for s in sources}
    parent = {s: None for s in sources}

    # max-heap via negative keys
    heap = [(-width[s], s) for s in sources]
    heapq.heapify(heap)

    while heap:
        neg_w, u = heapq.heappop(heap)
        w_u = -neg_w
        if w_u != width.get(u, None):
            continue  # stale heap entry

        for v in out_adj.get(u, ()):
            cap = min(w_u, edge_weight(u, v))
            if cap > width.get(v, -1):
                width[v] = cap
                parent[v] = u
                heapq.heappush(heap, (-cap, v))

    return parent


def _reverse_widest_next_multi_sink(sinks, in_adj, edge_weight):
    """
    Multi-sink reverse widest-path that records one 'next hop' towards some sink.
    width[x] = best bottleneck capacity from x to any sink (in forward direction).
    next_hop[x] = y means x -> y is the chosen next step.

    For sinks themselves: next_hop[sink] = None.
    """
    width = {t: math.inf for t in sinks}
    next_hop = {t: None for t in sinks}

    heap = [(-width[t], t) for t in sinks]
    heapq.heapify(heap)

    while heap:
        neg_w, v = heapq.heappop(heap)
        w_v = -neg_w
        if w_v != width.get(v, None):
            continue  # stale

        # go backwards: u -> v (forward), so u is predecessor of v
        for u in in_adj.get(v, ()):
            cap = min(w_v, edge_weight(u, v))
            if cap > width.get(u, -1):
                width[u] = cap
                next_hop[u] = v
                heapq.heappush(heap, (-cap, u))

    return next_hop


def filter_dfg_noise_keep_activities_and_repair(obj, noise_threshold):
    # Original structures
    start0 = copy(obj.dfg.start_activities)
    end0 = copy(obj.dfg.end_activities)
    graph0 = copy(obj.dfg.graph)

    if not end0 or not start0:
        raise ValueError("Assumption violated: no original start or end activities!")

    # Universe of activities from original DFG
    activities = set(start0.keys()) | set(end0.keys())
    for (a, b) in graph0.keys():
        activities.add(a)
        activities.add(b)

    # --- Check the assumption on the ORIGINAL augmented graph:
    out0, in0, w0 = _build_augmented_adjacency(graph0, start0, end0)
    R0 = _bfs_reachable({_START}, out0)
    # Compute nodes that can reach END in original:
    next_to_end0 = _reverse_bfs_next_multi_sink({_END}, in0)
    on_path0 = {a for a in activities if a in R0 and a in next_to_end0}
    if on_path0 != activities:
        missing = activities - on_path0
        raise ValueError(
            f"Assumption violated: {len(missing)} activities are not on any START->END path in the original DFG."
        )

    # --- Noise filter
    outgoing_max_occ = {}
    for (a, b), f in graph0.items():
        outgoing_max_occ[a] = max(outgoing_max_occ.get(a, 0), f)
    for a, f in end0.items():
        outgoing_max_occ[a] = max(outgoing_max_occ.get(a, 0), f)

    # Filter internal edges
    graph1 = {
        (a, b): f
        for (a, b), f in graph0.items()
        if f >= noise_threshold * outgoing_max_occ.get(a, f)
    }

    dfg = DFG()
    dfg.graph.update(graph1)

    # Filter start arcs
    if start0:
        start_max = max(start0.values())
        for a, f in start0.items():
            if f >= noise_threshold * start_max:
                dfg.start_activities[a] = f

    # Filter end arcs
    for a, f in end0.items():
        if f >= noise_threshold * outgoing_max_occ.get(a, f):
            dfg.end_activities[a] = f

    # If we filtered all ends/starts away, that’s OK — repair is allowed to re-add them.
    # But we need at least one end to define END reachability during repair.
    if not dfg.end_activities:
        # pick most frequent original end (deterministic)
        a, f = max(end0.items(), key=lambda kv: kv[1])
        dfg.end_activities[a] = f

    if not dfg.start_activities:
        # pick most frequent original start if any; otherwise repair will re-add via paths anyway
        if start0:
            a, f = max(start0.items(), key=lambda kv: kv[1])
            dfg.start_activities[a] = f
        else:
            # If original start_activities is empty (rare), that contradicts standard DFG semantics
            raise ValueError("Original start_activities is empty; cannot define START.")

    # --- Guaranteed repair using ORIGINAL augmented paths
    while True:
        out_cur, in_cur, w_cur = _build_augmented_adjacency(dfg.graph, dfg.start_activities, dfg.end_activities)

        # current backbone: reachable from START and can reach END
        R = _bfs_reachable({_START}, out_cur)
        next_to_end = _reverse_bfs_next_multi_sink({_END}, in_cur)
        backbone = {a for a in activities if a in R and a in next_to_end}

        broken = activities - backbone
        if not broken:
            break

        # Build “how to get there” from current reachable set, but using ORIGINAL graph (so we can re-add missing arcs)
        # Sources for forward parents: everything currently reachable (including START)
        # Sinks for reverse next: everything currently co-reachable to END (i.e., present in next_to_end)
        # Note: co-reachable set is exactly keys of next_to_end (nodes that can reach END in current)
        co_reachable = set(next_to_end.keys())

        parent_from_R_in_orig = _widest_parents_multi_source(R, out0, w0)
        next_to_C_in_orig = _reverse_widest_next_multi_sink(co_reachable, in0, w0)

        progressed = False

        for a in broken:
            # 1) Add a path from some node in R to a (in original augmented graph)
            if a not in parent_from_R_in_orig:
                # With your assumption + because START is in R, this should not happen
                raise ValueError(f"Repair failed: cannot reach {a} from current reachable set in original graph.")

            # Reconstruct forward path: ... -> a
            path_nodes = []
            x = a
            while x is not None and x not in R:
                path_nodes.append(x)
                x = parent_from_R_in_orig[x]
            # Now x is in R; parent pointers give edges parent->child
            for node in reversed(path_nodes):
                u = parent_from_R_in_orig[node]
                v = node
                if u is None:
                    # node itself is a source; shouldn't occur for nodes not in R
                    continue
                _add_augmented_edge_to_dfg(dfg, u, v, w0(u, v))
                progressed = True

            # 2) Add a path from a to some node that can reach END (co_reachable), in original augmented graph
            if a not in next_to_C_in_orig:
                raise ValueError(f"Repair failed: cannot connect {a} to END from current co-reachable set in original graph.")

            x = a
            while x is not None and x not in co_reachable:
                y = next_to_C_in_orig[x]
                if y is None:
                    break
                _add_augmented_edge_to_dfg(dfg, x, y, w0(x, y))
                progressed = True
                x = y

        if not progressed:
            raise ValueError("Repair made no progress; this should be impossible under the connectivity assumption.")

    if isinstance(obj, IMDataStructurePOT):
        return IMDataStructurePOT(obj.data_structure, dfg, obj.efg)
    return IMDataStructureUVCL(obj.data_structure, dfg)
