from typing import Set

from powl.objects.tagged_powl.activity import Activity
from powl.objects.tagged_powl.builders import loop, xor
from powl.objects.tagged_powl.partial_order import PartialOrder

ENABLE_DUPLICATION = True


class XOR:
    def __init__(self, children: frozenset):
        if not isinstance(children, frozenset):
            raise TypeError("Children must be provided as a frozenset.")
        if len(children) < 1:
            raise ValueError("XOR must have at least one child.")
        self.children = children

    def __repr__(self):
        return f"XOR({', '.join(repr(child) for child in sorted(self.children))})"

    def __eq__(self, other):
        if isinstance(other, XOR):
            return self.children == other.children
        return False

    def __hash__(self):
        return hash(("XOR", self.children))

    def __lt__(self, other):
        if isinstance(other, XOR):
            return sorted(self.children) < sorted(other.children)
        elif isinstance(other, LOOP) or isinstance(
            other, Graph
        ):  # ActivityInstance < XOR < Loop < Graph
            return True
        elif isinstance(other, ActivityInstance):
            return False
        else:
            return NotImplemented


class Skip(XOR):
    _allow_init = False

    def __init__(self, element):
        if not Skip._allow_init:
            raise RuntimeError("You must use create() to create this object!")
        super().__init__(frozenset([element, ActivityInstance(None, 1)]))
        self.element = element

    @classmethod
    def create(cls, element):
        if isinstance(element, Skip) or isinstance(element, SkipSelfLoop):
            instance = element
        elif isinstance(element, SelfLoop):
            return SkipSelfLoop.create(element.element)
        else:
            cls._allow_init = True
            instance = cls(element)
            cls._allow_init = False

        return instance

    @property
    def allow_init(self):
        return self._allow_init


class LOOP:
    def __init__(self, body, redo):
        self.body = body
        self.redo = redo

    def __repr__(self):
        return f"LOOP(body={repr(self.body)}, redo={repr(self.redo)})"

    def __eq__(self, other):
        if isinstance(other, LOOP):
            return self.body == other.body and self.redo == other.redo
        return False

    def __hash__(self):
        return hash(("LOOP", self.body, self.redo))

    def __lt__(self, other):
        if isinstance(other, LOOP):
            return (self.body, self.redo) < (other.body, other.redo)
        elif isinstance(other, Graph):  # ActivityInstance < XOR < Loop < Graph
            return True
        elif isinstance(other, XOR) or isinstance(other, ActivityInstance):
            return False
        else:
            return NotImplemented


class SelfLoop(LOOP):
    _allow_init = False

    def __init__(self, element):
        if not SelfLoop._allow_init:
            raise RuntimeError("You must use create() to create this object!")
        super().__init__(element, ActivityInstance(None, 1))
        self.element = element

    @classmethod
    def create(cls, element):
        if isinstance(element, SelfLoop):
            return element
        elif isinstance(element, Skip) or isinstance(element, SkipSelfLoop):
            return SkipSelfLoop.create(element.element)
        else:
            cls._allow_init = True
            instance = cls(element)
            cls._allow_init = False
            return instance


class SkipSelfLoop(LOOP):
    def __init__(self, element):
        if (
            isinstance(element, SkipSelfLoop)
            or isinstance(element, SelfLoop)
            or isinstance(element, Skip)
        ):
            element = element.element
        super().__init__(ActivityInstance(None, 1), element)
        self.element = element

    @classmethod
    def create(cls, element):
        return cls(element)


class ActivityInstance:
    def __init__(self, label: str | None, number: int):
        if not ENABLE_DUPLICATION:
            number = 1
        if number < 1:
            raise ValueError("Activity number must be at least 1.")
        self.label = label
        self.number = number

    def __repr__(self):
        if self.number == 1:
            return f"{self.label}"
        return f"({self.label}, {self.number})"

    def __eq__(self, other):
        if isinstance(other, ActivityInstance):
            return self.label == other.label and self.number == other.number
        return False

    def __hash__(self):
        return hash((self.label, self.number))

    def __lt__(self, other):
        if isinstance(other, ActivityInstance):
            if self.label and not other.label:
                return False
            elif not self.label and other.label:
                return True
            return (self.label, self.number) < (other.label, other.number)
        # ActivityInstance < XOR < Loop < Graph
        elif (
            isinstance(other, Graph)
            or isinstance(other, XOR)
            or isinstance(other, LOOP)
        ):
            return True
        else:
            return NotImplemented


class Graph:
    def __init__(self, nodes: frozenset, edges: frozenset, additional_information=None):
        if additional_information is None:
            additional_information = {}
        if not isinstance(nodes, frozenset):
            raise TypeError("Nodes must be a frozenset.")
        if not isinstance(edges, frozenset):
            raise TypeError("Edges must be a frozenset.")
        for edge in edges:
            if not (isinstance(edge, tuple) and len(edge) == 2):
                raise ValueError(
                    f"Each edge must be a (source, target) tuple, found: {edge}"
                )
            if edge[0] not in nodes or edge[1] not in nodes:
                raise ValueError(f"Edge {edge} refers to nodes not in the node set.")

        self.nodes = nodes
        self.edges = edges
        self.additional_information = (
            additional_information if additional_information else {}
        )

    def __repr__(self):
        nodes_repr = ", ".join(sorted(map(repr, self.nodes)))
        edges_repr = ", ".join(
            f"{repr(src)}->{repr(tgt)}" for src, tgt in sorted(self.edges)
        )
        return f"Graph(Nodes: {{{nodes_repr}}}, Edges: {{{edges_repr}}}, {self.additional_information})"

    def __eq__(self, other):
        if isinstance(other, Graph):
            return self.nodes == other.nodes and self.edges == other.edges
        return False

    def __hash__(self):
        return hash(("Graph", self.nodes, self.edges))

    def __lt__(self, other):
        if isinstance(other, Graph):
            return (sorted(self.nodes), sorted(self.edges)) < (
                sorted(other.nodes),
                sorted(other.edges),
            )
        # ActivityInstance < XOR < Loop < Graph
        elif (
            isinstance(other, XOR)
            or isinstance(other, ActivityInstance)
            or isinstance(other, LOOP)
        ):
            return False
        else:
            return NotImplemented


def get_leaves(node) -> Set[str]:
    if isinstance(node, ActivityInstance):
        return {node.label} if node.label else set()
    if isinstance(node, Graph):
        children = node.nodes
    elif isinstance(node, XOR):
        children = node.children
    elif isinstance(node, LOOP):
        children = [node.body, node.redo]
    else:
        raise TypeError
    res = set()
    for child in children:
        res.update(get_leaves(child))
    return res


def _simplified_model_to_powl(model, add_instance_number=False):
    if isinstance(model, ActivityInstance):
        if not model.label:
            return Activity(label=None)
        if add_instance_number:
            label = f"({model.label}, {model.number})"
        else:
            label = model.label
        return Activity(label=label)
    elif isinstance(model, XOR):
        return xor([_simplified_model_to_powl(child) for child in model.children])
    elif isinstance(model, LOOP):
        return loop(
            _simplified_model_to_powl(model.body),
            _simplified_model_to_powl(model.redo),
        )
    elif not isinstance(model, Graph):
        raise NotImplementedError

    po = PartialOrder()
    submodels = model.nodes
    edges = model.edges

    powl_map = {}
    for submodel in submodels:
        powl_child = _simplified_model_to_powl(submodel)
        powl_map[submodel] = powl_child
        po.add_node(powl_child)

    for m1, m2 in edges:
        po.add_edge(powl_map[m1], powl_map[m2])

    len_all = len(po.children)

    start_nodes = [n for n in po.children if po.in_degree(n) == 0]
    start_len = len(start_nodes)
    if start_len > 1 and start_len != len_all:
        start = Activity(label=None)
        po.add_node(start)
        for node in list(po.children):
            if node is not start:
                po.add_edge(start, node)

    end_nodes = [n for n in po.children if po.out_degree(n) == 0]
    end_len = len(end_nodes)
    if end_len > 1 and end_len != len_all:
        end = Activity(label=None)
        po.add_node(end)
        for node in list(po.children):
            if node is not end:
                po.add_edge(node, end)

    po.validate()

    return po


def generate_powl(model, add_instance_number=False):
    powl = _simplified_model_to_powl(model, add_instance_number=add_instance_number)
    return powl.normalize()
