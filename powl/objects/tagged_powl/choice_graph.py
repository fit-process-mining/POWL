from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Set, Tuple

import networkx as nx

from .activity import Activity
from .graph_base import GraphBacked
from .base import TaggedPOWL
from .partial_order import PartialOrder
from .types import ModelType
from ..utils.graph_sequentialization import split_graph_into_stages


# Internal nodes (not TaggedPOWL; users shouldn't ever touch them)
@dataclass(frozen=True, slots=True)
class _ChoiceGraphStart:
    def __repr__(self) -> str:
        return "_ChoiceGraphStart()"


@dataclass(frozen=True, slots=True)
class _ChoiceGraphEnd:
    def __repr__(self) -> str:
        return "_ChoiceGraphEnd()"

class ChoiceGraph(GraphBacked):
    """
    ChoiceGraph:
      - Directed graph among TaggedPOWL nodes
      - Has internal START and END nodes
      - Start marking is encoded as START -> node edges
      - End marking is encoded as node -> END edges

    API:
      - mark_start / unmark_start
      - mark_end / unmark_end
      - start_nodes / end_nodes
      - add/remove nodes/edges
    """

    __slots__ = ("_start", "_end")

    def __init__(
        self,
        nodes: Optional[Iterable[TaggedPOWL]] = None,
        edges: Optional[Iterable[Tuple[TaggedPOWL, TaggedPOWL]]] = None,
        *,
        start_nodes: Optional[Iterable[TaggedPOWL]] = None,
        end_nodes: Optional[Iterable[TaggedPOWL]] = None,
        min_freq: int = 1,
        max_freq: Optional[int] = 1,
    ) -> None:
        super().__init__(ModelType.ChoiceGraph, min_freq=min_freq, max_freq=max_freq)

        self._start = _ChoiceGraphStart()
        self._end = _ChoiceGraphEnd()
        self._g.add_node(self._start)
        self._g.add_node(self._end)

        if nodes is not None:
            self.add_nodes(nodes)
        if edges is not None:
            self.add_edges(edges)
        if start_nodes is not None:
            for n in start_nodes:
                self.mark_start(n)
        if end_nodes is not None:
            for n in end_nodes:
                self.mark_end(n)

    def validate_connectivity(self) -> None:

        # Every node must be on a path START -> ... -> END
        # Equivalent: reachable from START AND can reach END.
        reachable_from_start = nx.descendants(self._g, self._start)
        can_reach_end = nx.ancestors(self._g, self._end)

        for n in self.get_nodes():
            if (n not in reachable_from_start) or (n not in can_reach_end):
                raise ValueError(
                    "ChoiceGraph validity failed: every user node must lie on a path "
                    "from START to END. "
                    f"This node violate the requirement: {n}"
                )

    # --- override node selection to exclude internal ---
    def get_nodes(self) -> Set[TaggedPOWL]:
        return {n for n in self._g.nodes if isinstance(n, TaggedPOWL)}

    def get_edges(self) -> Set[Tuple[TaggedPOWL, TaggedPOWL]]:
        out: Set[Tuple[TaggedPOWL, TaggedPOWL]] = set()
        for u, v in self._g.edges:
            if isinstance(u, TaggedPOWL) and isinstance(v, TaggedPOWL):
                out.add((u, v))
        return out

    def predecessors(self, node: TaggedPOWL) -> Set[TaggedPOWL]:
        return {p for p in self._g.predecessors(node) if isinstance(p, TaggedPOWL)}

    def successors(self, node: TaggedPOWL) -> Set[TaggedPOWL]:
        return {s for s in self._g.successors(node) if isinstance(s, TaggedPOWL)}

    # --- start/end management ---
    def start_nodes(self) -> Set[TaggedPOWL]:
        return {v for v in self._g.successors(self._start) if isinstance(v, TaggedPOWL)}

    def end_nodes(self) -> Set[TaggedPOWL]:
        return {u for u in self._g.predecessors(self._end) if isinstance(u, TaggedPOWL)}

    def is_start(self, node: TaggedPOWL) -> bool:
        return self._g.has_edge(self._start, node)

    def is_end(self, node: TaggedPOWL) -> bool:
        return self._g.has_edge(node, self._end)

    def mark_start(self, node: TaggedPOWL) -> None:
        self.add_node(node)
        self._g.add_edge(self._start, node)

    def unmark_start(self, node: TaggedPOWL) -> None:
        if self._g.has_edge(self._start, node):
            self._g.remove_edge(self._start, node)

    def mark_end(self, node: TaggedPOWL) -> None:
        self.add_node(node)
        self._g.add_edge(node, self._end)

    def unmark_end(self, node: TaggedPOWL) -> None:
        if self._g.has_edge(node, self._end):
            self._g.remove_edge(node, self._end)

    def set_start_nodes(self, nodes: Iterable[TaggedPOWL]) -> None:
        # clear existing
        for n in list(self.start_nodes()):
            self.unmark_start(n)
        for n in nodes:
            self.mark_start(n)

    def set_end_nodes(self, nodes: Iterable[TaggedPOWL]) -> None:
        for n in list(self.end_nodes()):
            self.unmark_end(n)
        for n in nodes:
            self.mark_end(n)

    def pretty(self) -> str:
        return (
            "ChoiceGraph(\n"
            f"  nodes={len(self.get_nodes())}, edges={len(self.get_edges())},\n"
            f"  start_nodes={len(self.start_nodes())}, end_nodes={len(self.end_nodes())},\n"
            f"  min={self.min_freq}, max={self.max_freq}\n"
            ")"
        )

    def clone(self, *, deep: bool = True) -> "ChoiceGraph":
        new = ChoiceGraph(
            nodes=self.get_nodes(),
            edges=self.get_edges(),
            start_nodes=self.start_nodes(),
            end_nodes=self.end_nodes(),
            min_freq=self.min_freq,
            max_freq=self.max_freq,
        )
        return new

    def normalize(self) -> TaggedPOWL:
        node_map = {n: n.normalize() for n in self.children}
        return self.map_nodes(node_map)._reduce_silent_activities()

    def to_dict(self) -> dict[str, Any]:

        nodes = list(self.get_nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for (u, v) in self.get_edges()]
        start = [idx[n] for n in self.start_nodes() if n in idx]
        end = [idx[n] for n in self.end_nodes() if n in idx]

        return {
            "type": self.model_type.value,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "nodes": [n.to_dict() for n in nodes],
            "edges": edges,
            "start_nodes": start,
            "end_nodes": end,
        }

    def _reduce_silent_activities(self) -> TaggedPOWL:
        """
        Reduces silent activities by merging redundant edges, handling global self-loops,
        and abstracting isolated subgraphs that are dominated by a silent loop transition.
        """
        # 1. Trivial leaf self-loops
        for (u, v) in self.get_edges():
            if u == v:
                self.remove_edge(u, v)
                u.max_freq = None

        self._reduce_simple_silent_transitions()
        self._mark_skippable_nodes()

        node_map = {n: n.normalize() for n in self.get_nodes()}
        self._map_graph(node_map)
        self._abstract_self_loop()
        return self._apply_advanced_reductions()

    def _apply_advanced_reductions(self) -> "TaggedPOWL":
        if len(self.get_nodes()) == 1:
            return self._flatten_single_node()
        self._abstract_sccs()
        return self._abstract_sequences()


    def _reduce_simple_silent_transitions(self):
        # Simple Reduction (1-in or 1-out)

        def is_silent(n: Any) -> bool:
            return isinstance(n, Activity) and n.is_silent()

        changed = True
        while changed:

            changed = False
            silent_nodes = [n for n in self.get_nodes() if is_silent(n)]

            for tau in silent_nodes:

                if tau not in self._g.nodes:
                    pass

                preds = set(self._g.predecessors(tau))
                succs = set(self._g.successors(tau))

                if len(preds) <= 1 or len(succs) <= 1:
                    self._bypass_silent_node(tau, preds, succs)
                    changed = True


    def _mark_skippable_nodes(self):
        changed = True
        while changed:
            edges_to_remove = set()
            changed = False
            current_edges = list(self._g.edges)

            for node in self.get_nodes():

                preds = set(self._g.predecessors(node))
                succs = set(self._g.successors(node))

                if all((p, s) in current_edges for p in preds for s in succs):
                    edges_to_remove.update({(p, s) for p in preds for s in succs})
                    node.min_freq = 0

            if edges_to_remove:
                self._g.remove_edges_from(edges_to_remove)
                changed = True


    def _abstract_self_loop(self) -> bool:


        silent_nodes = [n for n in self.get_nodes() if isinstance(n, Activity) and n.is_silent()]
        start_nodes = set(self._g.successors(self._start))
        end_nodes = set(self._g.predecessors(self._end))

        for tau in silent_nodes:

            preds = set(self._g.predecessors(tau))
            succs = set(self._g.successors(tau))

            # Skippable Self-Loop
            if start_nodes == {tau} == end_nodes:
                self._g.remove_node(tau)
                self.min_freq = 0
                self.max_freq = None
                for p in preds - {self._start}:
                    self.mark_end(p)
                for s in succs - {self._end}:
                    self.mark_start(s)
                return True

            # Non-Skippable Self-Loop
            elif preds == end_nodes and succs == start_nodes:
                self._g.remove_node(tau)
                self.max_freq = None
                return True


        edges = self._g.edges()

        back_edges = []
        for u in end_nodes:
            for v in start_nodes:
                if (u, v) in edges:
                    back_edges.append((u, v))
                else:
                    return False

        if back_edges:
            self.max_freq = None
            self._g.remove_edges_from(back_edges)
            return True
        else:
            raise ValueError("This code should be unreachable!")

    def _abstract_sequences(self) -> TaggedPOWL:
        """
        General sequential chunking for ChoiceGraph using dominators/post-dominators.

        Produces a sequential PartialOrder:
            chunk1 -> chunk2 -> chunk3 -> ...

        Each chunk may itself be complex (ChoiceGraph / PartialOrder / etc.).
        This is a generalization of head/tail peeling.

        Efficiency notes:
        - Uses dominator spines and a monotone assignment to avoid O(V * spine) scanning.
        - Avoids cloning nodes; builds induced subgraphs using original TaggedPOWL objects.
        - Merges away trivial boundary regions caused by artificial START/END nodes.
        """

        self._abstract_self_loop()

        G = self._g
        START = self._start
        END = self._end
        stages, is_skippable = split_graph_into_stages(G, START, END)
        if len(stages) < 2:
            raise Exception("Something went wrong!")
        if {START} != stages[0] or {END} != stages[-1]:
            raise Exception("Something went wrong!")
        stages = stages[1:-1]
        is_skippable = is_skippable[1:-1]

        if len(stages) < 2:
            return self

        def build_chunk(chunk_nodes: set[TaggedPOWL], skippable_chunk: bool) -> TaggedPOWL:

            if len(chunk_nodes) == 1:
                chunk_node = chunk_nodes.pop()
                if skippable_chunk:
                    chunk_node.min_freq = 0
                return chunk_node

            chunk_edges = [(u, v) for (u, v) in self.get_edges() if u in chunk_nodes and v in chunk_nodes]

            chunk_start: set[TaggedPOWL] = set()
            chunk_end: set[TaggedPOWL] = set()

            for n in chunk_nodes:
                preds = list(G.predecessors(n))
                succs = list(G.successors(n))

                if any(p not in chunk_nodes for p in preds):
                    chunk_start.add(n)
                if any(s not in chunk_nodes for s in succs):
                    chunk_end.add(n)

            sub = ChoiceGraph(
                nodes=chunk_nodes,
                edges=chunk_edges,
                start_nodes=chunk_start,
                end_nodes=chunk_end,
                min_freq= 0 if skippable_chunk else 1,
                max_freq=1,
            )

            return sub._abstract_sequences()

        chunks: list[TaggedPOWL] = [build_chunk(stages[i], is_skippable[i]) for i in range(len(stages))]

        chunks = [c for c in chunks if not (isinstance(c, Activity) and c.is_silent())]

        po_edges = [(chunks[i], chunks[i + 1]) for i in range(len(chunks) - 1)]

        return PartialOrder(
            nodes=chunks,
            edges=po_edges,
            min_freq=self.min_freq,
            max_freq=self.max_freq,
        ).flatten()

    def _flatten_single_node(self):
        assert len(self._g.edges) == 2  # assuming leaf-level self-loops were abstracted before
        remaining_node = self.get_nodes().pop()
        remaining_node.min_freq = min(remaining_node.min_freq, self.min_freq)
        if remaining_node.max_freq is None or self.max_freq is None:
            remaining_node.max_freq = None
        else:
            remaining_node.max_freq = max(self.max_freq, remaining_node.max_freq)
        return remaining_node

    def _bypass_silent_node(self, tau: TaggedPOWL, preds: Set[TaggedPOWL], succs: Set[TaggedPOWL]) -> None:
        """Removes tau and connects preds directly to succs."""
        for p in preds:
            for s in succs:
                if p is self._start and s is self._end:
                    self.min_freq = 0
                elif p is self._end or s is self._start:
                    raise ValueError("This code should be unreachable!")
                elif p == s:
                    p.max_freq = None  # Collapse local loop
                else:
                    self._g.add_edge(p, s)

        self._g.remove_node(tau)

    def _abstract_sccs(self):
        """
        Detects Strongly Connected Components (SCCs).
        If an SCC has size > 1, and has a unique/complete entry set and exit set,
        it abstracts the SCC into a nested ChoiceGraph.
        """

        sccs = list(nx.strongly_connected_components(self._g))

        valid_abstractions = []
        nodes_to_abstract = set()

        for scc in sccs:

            if len(scc) <= 1:
                continue

            if self._start in scc or self._end in scc:
                raise ValueError("This code should be unreachable!")

            A = set()  # Outside Sources
            B = set()  # Inside Entries
            edges_in_count = 0

            C = set()  # Inside Exits
            D = set()  # Outside Targets
            edges_out_count = 0

            for (u, v) in self._g.edges:
                if u not in scc and v in scc:
                    A.add(u)
                    B.add(v)
                    edges_in_count += 1
                elif u in scc and v not in scc:
                    C.add(u)
                    D.add(v)
                    edges_out_count += 1


            # Check Condition: ALL nodes in C must connect to ALL nodes in D
            if len(A) > 0 and len(B) > 0:
                if edges_in_count != len(A) * len(B):
                    continue
            else:
                raise ValueError("This code should be unreachable!")

            # Check Exit Condition: Total Edges == |C| * |D|
            if len(C) > 0 and len(D) > 0:
                if edges_out_count != len(C) * len(D):
                    continue
            else:
                raise ValueError("This code should be unreachable!")

            valid_abstractions.append({
                "nodes": scc,
                "start_nodes": B,
                "end_nodes": C
            })
            nodes_to_abstract.update(scc)

        if not valid_abstractions:
            return

        replacement_map = {}

        for item in valid_abstractions:
            scc_nodes = item["nodes"]

            sub_edges = []
            for u in scc_nodes:
                for v in self._g.successors(u):
                    if v in scc_nodes:
                        sub_edges.append((u, v))

            sub_graph = self.__class__(
                nodes=scc_nodes,
                edges=sub_edges,
                start_nodes=item["start_nodes"],
                end_nodes=item["end_nodes"],
                min_freq=1,
                max_freq=1,
            )
            changed = sub_graph._abstract_self_loop()
            if changed:
                sub_graph = sub_graph._apply_advanced_reductions()

            for n in scc_nodes:
                replacement_map[n] = sub_graph


        new_edges = set()
        nodes_to_keep = [n for n in self.get_nodes() if n not in nodes_to_abstract]

        for u, v in self._g.edges:
            u_new = replacement_map.get(u, u)
            v_new = replacement_map.get(v, v)

            if u_new is v_new:
                if u in nodes_to_abstract:
                    # Case A: Edge internal to an SCC being removed -> Discard
                    continue
                else:
                    raise ValueError("This code should be unreachable!")
            else:
                new_edges.add((u_new, v_new))

        self._g = nx.DiGraph()

        for n in nodes_to_keep:
            self.add_node(n)

        added_replacements = set()
        for sub in replacement_map.values():
            if sub not in added_replacements:
                self.add_node(sub)
                added_replacements.add(sub)

        self._g.add_edges_from(new_edges)


    def _map_graph(self, node_map: dict) -> None:
        """Refreshes the internal graph structure based on a node mapping."""
        new_nodes = list(node_map.values())
        new_edges = [(node_map[u], node_map[v]) for u, v in self.get_edges()]
        old_start = self.start_nodes()
        old_end = self.end_nodes()

        self._g = nx.DiGraph()
        self._g.add_node(self._start)
        self._g.add_node(self._end)
        self.add_nodes(new_nodes)
        self.add_edges(new_edges)

        for n in old_start:
            self.mark_start(node_map[n])
        for n in old_end:
            self.mark_end(node_map[n])

    def map_nodes(self, mapping: dict[TaggedPOWL, TaggedPOWL]) -> "ChoiceGraph":
        return ChoiceGraph(
            nodes=[mapping[n] for n in self.children],
            edges=[(mapping[u], mapping[v]) for (u, v) in self.get_edges()],
            start_nodes=[mapping[n] for n in self.start_nodes()],
            end_nodes=[mapping[n] for n in self.end_nodes()],
            min_freq=self.min_freq,
            max_freq=self.max_freq,
        )
