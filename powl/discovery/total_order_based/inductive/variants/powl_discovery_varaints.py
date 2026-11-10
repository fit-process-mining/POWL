from enum import auto, Enum


class POWLDiscoveryVariant(Enum):
    TREE = auto()  # base IM with no partial orders
    BRUTE_FORCE = auto()
    MAXIMAL = auto()
    DYNAMIC_CLUSTERING = auto()
    DECISION_GRAPH_MAX = auto()
    DECISION_GRAPH_CLUSTERING = auto()
    DECISION_GRAPH_CYCLIC = auto()
    DECISION_GRAPH_CYCLIC_STRICT = auto()
    RULE_BASED = auto()
