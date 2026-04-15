from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Set, Tuple

import networkx as nx

from .base import TaggedPOWL
from .types import ModelType


class GraphBacked(TaggedPOWL, ABC):
    """
    Base for graph-backed models.

    Stores an nx.DiGraph in self._g.
    Subclasses may add internal nodes (ChoiceGraph does).
    """

    __slots__ = ("_g",)

    def __init__(self, model_type: ModelType, *, min_freq: int = 1, max_freq: Optional[int] = 1) -> None:
        super().__init__(model_type, min_freq=min_freq, max_freq=max_freq)
        self._g: nx.DiGraph = nx.DiGraph()

    # --- graph access ---
    @property
    def graph(self) -> nx.DiGraph:
        """Direct access (mutable). Use with care."""
        return self._g

    def to_networkx(self, *, include_internal: bool = False) -> nx.DiGraph:
        """
        Default export: shallow copy of the internal DiGraph.
        Subclasses may override to exclude internal nodes when include_internal=False.
        """
        return self._g.copy()

    # --- node/edge helpers ---
    def get_nodes(self) -> Set[TaggedPOWL]:
        """Default: all nodes that are TaggedPOWL instances."""
        return {n for n in self._g.nodes}

    @property
    def children(self) -> list[TaggedPOWL]:
        return [n for n in self._g.nodes if isinstance(n, TaggedPOWL)]

    def get_edges(self) -> Set[Tuple[TaggedPOWL, TaggedPOWL]]:
        """Default: edges whose endpoints are TaggedPOWL instances."""
        out: Set[Tuple[TaggedPOWL, TaggedPOWL]] = set()
        for u, v in self._g.edges:
            out.add((u, v))
        return out

    def has_node(self, node: TaggedPOWL) -> bool:
        return self._g.has_node(node)

    def is_edge(self, node_1: TaggedPOWL, node_2: TaggedPOWL) -> bool:
        return self._g.has_edge(node_1, node_2)

    def add_node(self, node: TaggedPOWL, **attrs: Any) -> None:
        if not isinstance(node, TaggedPOWL):
            raise TypeError(f"Node must be a TaggedPOWL instance: {node}")
        self._g.add_node(node, **attrs)

    def add_nodes(self, nodes: Iterable[TaggedPOWL]) -> None:
        for n in nodes:
            self.add_node(n)

    def remove_node(self, node: TaggedPOWL) -> None:
        self._g.remove_node(node)

    def add_edge(self, u: TaggedPOWL, v: TaggedPOWL, **attrs: Any) -> None:
        if not isinstance(u, TaggedPOWL) or not isinstance(v, TaggedPOWL):
            raise TypeError("u and v must be TaggedPOWL instances")
        self._g.add_node(u)
        self._g.add_node(v)
        self._g.add_edge(u, v, **attrs)

    def add_edges(self, edges: Iterable[Tuple[TaggedPOWL, TaggedPOWL]]) -> None:
        for u, v in edges:
            self.add_edge(u, v)

    def remove_edge(self, u: TaggedPOWL, v: TaggedPOWL) -> None:
        self._g.remove_edge(u, v)

    def predecessors(self, node: TaggedPOWL) -> Set[TaggedPOWL]:
        return {p for p in self._g.predecessors(node)}

    def successors(self, node: TaggedPOWL) -> Set[TaggedPOWL]:
        return {s for s in self._g.successors(node)}

    # --- convenience ---
    def in_degree(self, node: TaggedPOWL) -> int:
        return int(self._g.in_degree(node))

    def out_degree(self, node: TaggedPOWL) -> int:
        return int(self._g.out_degree(node))

    def is_empty(self) -> bool:
        return self._g.number_of_nodes() == 0

    def clone(self, *, deep: bool = True) -> "GraphBacked":
        """
        Default clone: copies tags + graph.
        Subclasses should override to preserve their internal invariants
        (e.g., ChoiceGraph internal start/end).
        """
        cls = type(self)
        new = cls.__new__(cls)  # type: ignore[misc]
        # Manually init base parts (bypassing subclass __init__)
        TaggedPOWL.__init__(new, self.model_type, self.min_freq, self.max_freq)  # type: ignore[arg-type]
        GraphBacked.__init__(new, self.model_type, min_freq=self.min_freq, max_freq=self.max_freq)  # type: ignore[misc]
        new._g = self._g.copy()
        return new

    def pretty(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self.get_nodes())}, "
            f"edges={len(self.get_edges())}, "
            f"min={self.min_freq}, max={self.max_freq})"
        )

    def same_structure(self, other: object) -> bool:
        if not isinstance(other, GraphBacked):
            return False
        if type(self) is not type(other):
            return False
        if not self.same_signature(other):
            return False
        # Conservative: identity-based node set + edge set
        return self.user_nodes() == other.user_nodes() and self.user_edges() == other.user_edges()

    # --- DAG utilities (usable by subclasses that are DAGs) ---
    def is_acyclic(self) -> bool:
        return nx.is_directed_acyclic_graph(self._g)

    def assert_acyclic(self) -> None:
        if not self.is_acyclic():
            raise ValueError(f"{self.__class__.__name__} must be acyclic but contains a cycle.")

    def topological_sort(self) -> list[object]:
        # May include internal nodes depending on subclass; override if needed.
        return list(nx.topological_sort(self._g))

    def transitive_closure(self) -> nx.DiGraph:
        self.assert_acyclic()
        return nx.transitive_closure_dag(self._g)

    def transitive_reduction(self) -> nx.DiGraph:
        self.assert_acyclic()
        return nx.transitive_reduction(self._g)

    def reachable(self, u: object, v: object) -> bool:
        return nx.has_path(self._g, u, v)

    def _validate_node_mapping(self, mapping: dict[TaggedPOWL, TaggedPOWL]) -> None:
        expected = set(self.get_nodes())
        actual = set(mapping.keys())

        missing = expected - actual
        extra = actual - expected

        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing keys: {missing}")
            if extra:
                parts.append(f"extra keys: {extra}")
            raise ValueError("Invalid node mapping: " + "; ".join(parts))
