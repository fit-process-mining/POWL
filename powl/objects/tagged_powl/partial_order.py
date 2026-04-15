from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import networkx as nx

from .activity import Activity
from .base import TaggedPOWL
from .graph_base import GraphBacked
from .types import ModelType


class PartialOrder(GraphBacked):

    def __init__(
        self,
        nodes: Optional[Iterable[TaggedPOWL]] = None,
        edges: Optional[Iterable[Tuple[TaggedPOWL, TaggedPOWL]]] = None,
        *,
        min_freq: int = 1,
        max_freq: Optional[int] = 1,
    ) -> None:
        super().__init__(ModelType.PartialOrder, min_freq=min_freq, max_freq=max_freq)

        if nodes is not None:
            self.add_nodes(nodes)
        if edges is not None:
            for u, v in edges:
                self.add_edge(u, v)

    def validate(self) -> None:
        # For a partial order represented as a DAG, acyclicity is the key structural requirement.
        if not nx.is_directed_acyclic_graph(self._g):
            cycle = nx.find_cycle(self._g, orientation="original")
            raise ValueError(f"PartialOrder must be acyclic. Found cycle: {cycle}")

    def validate_and_remove_transitive_edges(self) -> None:
        try:
            self._g = nx.transitive_reduction(self._g)
        except Exception as e:
            cycle = nx.find_cycle(self._g, orientation="original")
            raise ValueError(f"PartialOrder must be acyclic. Found cycle: {cycle}")

    def get_transitive_closure(self, fail_if_cyclic=True) -> nx.DiGraph:
        if fail_if_cyclic:
            return nx.transitive_closure_dag(self._g)
        else:
            return nx.transitive_closure(self._g)

    def pretty(self) -> str:
        return (
            "PartialOrder(\n"
            f"  nodes={len(self.get_nodes())}, edges={len(self.get_edges())},\n"
            f"  min={self.min_freq}, max={self.max_freq}\n"
            ")"
        )

    def to_dict(self) -> dict[str, Any]:
        nodes = list(self.get_nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for (u, v) in self.get_edges()]
        return {
            "type": self.model_type.value,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "nodes": [n.to_dict() for n in nodes],
            "edges": edges,
        }

    def _reduce_silent_activities(self) -> "PartialOrder":

        old_nodes = list(self.get_nodes())
        node_map = {n: n.normalize() for n in old_nodes}

        new = PartialOrder(
            nodes=node_map.values(),
            edges=((node_map[u], node_map[v]) for (u, v) in self.get_edges()),
            min_freq=self.min_freq,
            max_freq=self.max_freq,
        )

        def is_silent_activity(n: object) -> bool:
            return isinstance(n, Activity) and n.is_silent()

        changed = True
        while changed:
            changed = False
            silent_nodes = [n for n in list(new.get_nodes()) if is_silent_activity(n)]
            if not silent_nodes:
                break

            for tau in silent_nodes:
                preds = list(new._g.predecessors(tau))
                succs = list(new._g.successors(tau))

                if len(preds) <= 1 or len(succs) <= 1:
                    for p in preds:
                        for s in succs:
                            new.add_edge(p, s)

                    new._g.remove_node(tau)
                    changed = True

        return new

    def normalize(self) -> TaggedPOWL:
        node_map = {n: n.normalize() for n in self.children}
        simplified = self.map_nodes(node_map)._reduce_silent_activities()
        if isinstance(simplified, PartialOrder):
            return simplified.flatten()
        return simplified

    def map_nodes(self, mapping: dict[TaggedPOWL, TaggedPOWL]) -> "PartialOrder":
        self._validate_node_mapping(mapping)

        image_nodes = set(mapping.values())
        if len(mapping) == len(image_nodes):
            return PartialOrder(
                nodes=[mapping[n] for n in self.children],
                edges=[(mapping[u], mapping[v]) for (u, v) in self.get_edges()],
                min_freq=self.min_freq,
                max_freq=self.max_freq,
            )

        tc = self.get_transitive_closure(fail_if_cyclic=True)

        preorder = nx.DiGraph()
        preorder.add_nodes_from(image_nodes)

        for u, v in tc.edges():
            U = mapping[u]
            V = mapping[v]
            if U != V:
                preorder.add_edge(U, V)

        preorder_tc = nx.transitive_closure(preorder)

        final_edges = set()
        for U in image_nodes:
            for V in image_nodes:
                if U == V:
                    continue
                uv = preorder_tc.has_edge(U, V)
                vu = preorder_tc.has_edge(V, U)
                if uv and not vu:
                    final_edges.add((U, V))

        result = PartialOrder(
            nodes=image_nodes,
            edges=final_edges,
            min_freq=self.min_freq,
            max_freq=self.max_freq,
        )
        return result

    def flatten(self) -> TaggedPOWL:

        nodes = self.get_nodes()

        if len(nodes) == 1:

            node = list(nodes)[0]
            node.min_freq = min(self.min_freq, node.min_freq)
            if node.max_freq is None or self.max_freq is None:
                node.max_freq = None
            else:
                node.max_freq = max(self.max_freq, node.max_freq)
            if isinstance(node, PartialOrder):
                node = node.flatten()

            return node

        result = PartialOrder(
            min_freq=self.min_freq,
            max_freq=self.max_freq
        )

        entry_points: dict[TaggedPOWL, list[TaggedPOWL]] = {}
        exit_points: dict[TaggedPOWL, list[TaggedPOWL]] = {}

        for node in nodes:
            if isinstance(node, PartialOrder) and node.min_freq == 1 == node.max_freq:

                flat_child = node.flatten()

                if isinstance(flat_child, PartialOrder) and flat_child.min_freq == 1 == flat_child.max_freq:

                    for child_node in flat_child.get_nodes():
                        result.add_node(child_node)
                    for u, v in flat_child.get_edges():
                        result.add_edge(u, v)

                    c_starts = [n for n in flat_child.get_nodes() if flat_child._g.in_degree(n) == 0]
                    c_ends = [n for n in flat_child.get_nodes() if flat_child._g.out_degree(n) == 0]

                    entry_points[node] = c_starts
                    exit_points[node] = c_ends
                else:
                    result.add_node(flat_child)
                    entry_points[node] = [flat_child]
                    exit_points[node] = [flat_child]
            else:
                result.add_node(node)
                entry_points[node] = [node]
                exit_points[node] = [node]

        for u, v in self.get_edges():
            u_exist_points = exit_points[u]
            v_entry_points = entry_points[v]
            if len(u_exist_points) > 1 and len(v_entry_points) > 1:
                silent_connector = Activity(label=None)
                result.add_node(silent_connector)
                for u_exit in u_exist_points:
                    result.add_edge(u_exit, silent_connector)
                for v_entry in v_entry_points:
                    result.add_edge(silent_connector, v_entry)
            else:
                for u_exit in u_exist_points:
                    for v_entry in v_entry_points:
                        result.add_edge(u_exit, v_entry)

        return result
