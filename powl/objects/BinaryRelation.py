import copy
from itertools import product
from typing import Hashable, List as TList, Set as TSet, TypeVar

T = TypeVar("T", bound=Hashable)


class BinaryRelation:
    def __init__(self, nodes: TList[T]):
        self._number_nodes = 0
        self._set_nodes(nodes)
        self._edges = [
            [False for _ in range(self._number_nodes)]
            for _ in range(self._number_nodes)
        ]
        self._start_nodes = None
        self._end_nodes = None

    def get_nodes(self) -> TList[T]:
        return self._nodes

    def _set_nodes(self, nodes: TList[T]) -> None:
        self._nodes = [n for n in nodes]
        self._map_node_to_id = {}
        self._map_id_to_node = {}
        n = 0
        for node in self._nodes:
            self._map_node_to_id[node] = n
            self._map_id_to_node[n] = node
            n = n + 1
        self._number_nodes = n

    def add_edge(self, source: T, target: T) -> None:
        try:
            i = self._map_node_to_id[source]
            j = self._map_node_to_id[target]
        except Exception:
            raise Exception("Unable to create edge! Invalid  source or target!")
        else:
            self._edges[i][j] = True

    def remove_edge(self, source: T, target: T) -> None:
        try:
            i = self._map_node_to_id[source]
            j = self._map_node_to_id[target]
        except Exception:
            raise Exception("Unable to remove edge! Invalid  source or target!")
        else:
            self._edges[i][j] = False

    def remove_edge_without_violating_transitivity(self, source: T, target: T) -> None:
        try:
            i = self._map_node_to_id[source]
            j = self._map_node_to_id[target]
        except Exception:
            raise Exception("Unable to remove edge! Invalid  source or target!")
        else:
            self._edges[i][j] = False
            n = len(self.nodes)
            changed = True
            while changed:
                changed = False
                for i, j, k in product(range(n), range(n), range(n)):
                    if (
                        i != j
                        and j != k
                        and self._edges[i][j]
                        and self._edges[j][k]
                        and not self._edges[i][k]
                    ):
                        self._edges[j][k] = False
                        changed = True

    def add_node(self, node: T) -> None:
        if node not in self._nodes:
            self._nodes.append(node)
            n = self._number_nodes
            self._map_node_to_id[node] = n
            self._map_id_to_node[n] = node
            new_edges = [[False for _ in range(n + 1)] for _ in range(n + 1)]
            for i in range(n):
                for j in range(n):
                    new_edges[i][j] = self._edges[i][j]

            self._edges = new_edges
            self._number_nodes = n + 1

    def is_edge(self, source, target) -> bool:
        try:
            i = self._map_node_to_id[source]
            j = self._map_node_to_id[target]
        except Exception:
            print(source)
            print(target)
            print(self._nodes)
            raise Exception("Unable to create edge! Invalid  source or target!")
        else:
            return self._edges[i][j]

    def is_edge_id(self, i: int, j: int) -> bool:
        return self._edges[i][j]

    def get_transitive_reduction(self) -> "BinaryRelation":
        if not self.is_irreflexive():
            raise ValueError(
                "Cannot generate transitive reduction! Reflexivity detected!"
            )

        tc = BinaryRelation(self.nodes)
        tc._edges = copy.deepcopy(self._edges)
        tc.add_transitive_edges()

        tr = BinaryRelation(self.nodes)
        tr._edges = copy.deepcopy(self._edges)

        n = len(self.nodes)
        for i, j, k in product(range(n), range(n), range(n)):
            if (
                i != j
                and j != k
                and tc.edges[i][j]
                and tc.edges[j][k]
                and tc.edges[i][k]
            ):
                tr._edges[i][k] = False
        return tr

    def add_transitive_edges(self) -> None:
        n = len(self.nodes)
        changed = True
        while changed:
            changed = False
            for i, j, k in product(range(n), range(n), range(n)):
                if (
                    i != j
                    and j != k
                    and self._edges[i][j]
                    and self._edges[j][k]
                    and not self.is_edge_id(i, k)
                ):
                    self._edges[i][k] = True
                    changed = True

    def is_strict_partial_order(self) -> bool:
        return self.is_irreflexive() and self.is_transitive()

    def get_start_nodes(self) -> TSet[T]:
        return {
            self._map_id_to_node[j]
            for j in range(len(self.nodes))
            if not any(self._edges[i][j] for i in range(len(self.nodes)))
        }

    def get_end_nodes(self) -> TSet[T]:
        return {
            self._map_id_to_node[i]
            for i in range(len(self.nodes))
            if not any(self._edges[i][j] for j in range(len(self.nodes)))
        }

    def is_irreflexive(self) -> bool:
        n = len(self.nodes)
        for i in range(n):
            if self.is_edge_id(i, i):
                return False
        return True

    def is_transitive(self) -> bool:
        n = len(self.nodes)
        for i, j, k in product(range(n), range(n), range(n)):
            if (
                self.is_edge_id(i, j)
                and self.is_edge_id(j, k)
                and not self.is_edge_id(i, k)
            ):
                return False
        return True

    def __repr__(self) -> str:
        res = "(nodes = {  "
        for node in self._nodes:
            res = res + node.__repr__() + ", "
        res = res[:-2]
        res = res + "  }, order = {  "
        for node in self._nodes:
            i = self._map_node_to_id[node]
            for node2 in self._nodes:
                j = self._map_node_to_id[node2]
                if self._edges[i][j]:
                    res = res + node.__repr__() + "-->" + node2.__repr__() + ", "
        res = res[:-2]
        return res + "  })"

    nodes = property(get_nodes, _set_nodes)

    @property
    def edges(self):
        return self._edges

    def get_preset(self, child):
        res = set()
        for node in self.nodes:
            if self.is_edge(node, child):
                res.add(node)
        return res

    def get_postset(self, child):
        res = set()
        for node in self.nodes:
            if self.is_edge(child, node):
                res.add(node)
        return res
