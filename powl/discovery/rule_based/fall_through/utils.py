"""Graph and projection helpers shared by rule-directed fall-throughs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import networkx as nx
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.rule_based.rules import (
    AbstractRule,
    ChainPrecedenceRule,
    ChainResponseRule,
    CoExistenceRule,
    EndRule,
    InitializationRule,
    NotCoExistenceRule,
    NotSuccessionRule,
    PrecedenceRule,
    RespondedExistenceRule,
    ResponseRule,
)

# Private object sentinels cannot collide with user activity labels.
ARTIFICIAL_START = object()
ARTIFICIAL_END = object()


def alphabet_of(obj: IMDataStructureUVCL) -> set[str]:
    return {activity for trace in obj.data_structure for activity in trace}


def project_alphabet(
    obj: IMDataStructureUVCL, alphabet: set[str], *, distinct: bool = False
) -> IMDataStructureUVCL:
    projected = Counter()
    for trace, count in obj.data_structure.items():
        child = tuple(activity for activity in trace if activity in alphabet)
        if distinct:
            projected[child] = 1
        else:
            projected[child] += count
    return IMDataStructureUVCL(projected)


def supported_rules_alphabet(
    alphabet: set[str], rules: Iterable[AbstractRule]
) -> tuple[AbstractRule, ...]:
    """Select rules local to a complete branch, as in the reference miner."""
    return tuple(rule for rule in rules if set(rule.args) <= alphabet)


def dfg_graph(obj: IMDataStructureUVCL) -> nx.DiGraph:
    """Adapt POWL's weighted DFG for the rule-dependency graph heuristic."""
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(alphabet_of(obj)))
    for activity, count in obj.dfg.start_activities.items():
        graph.nodes[activity]["start"] = count
    for activity, count in obj.dfg.end_activities.items():
        graph.nodes[activity]["end"] = count
    for (source, target), count in obj.dfg.graph.items():
        graph.add_edge(source, target, weight=count)
    return graph


def build_cdg(rules: Sequence[AbstractRule], alphabet: set[str]) -> nx.DiGraph:
    """Construct the student's constraint-dependency graph (CDG)."""
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(alphabet))
    graph.add_nodes_from((ARTIFICIAL_START, ARTIFICIAL_END))
    for rule in rules:
        if isinstance(rule, InitializationRule):
            graph.add_edge(ARTIFICIAL_START, rule.target_activity)
        elif isinstance(rule, EndRule):
            graph.add_edge(rule.target_activity, ARTIFICIAL_END)
        elif isinstance(
            rule,
            (ResponseRule, PrecedenceRule, ChainPrecedenceRule, ChainResponseRule),
        ):
            graph.add_edge(rule.activity_a, rule.activity_b)
        elif isinstance(rule, NotSuccessionRule):
            graph.add_edge(rule.activity_b, rule.activity_a)
        elif isinstance(rule, (RespondedExistenceRule, CoExistenceRule)):
            graph.add_nodes_from((rule.activity_a, rule.activity_b))
    return graph


def build_signed_cdg(
    rules: Sequence[AbstractRule], alphabet: set[str]
) -> tuple[nx.DiGraph, nx.DiGraph]:
    positive = build_cdg(rules, alphabet)
    positive.remove_nodes_from((ARTIFICIAL_START, ARTIFICIAL_END))
    negative = nx.DiGraph()
    negative.add_nodes_from(sorted(alphabet))
    for rule in rules:
        if isinstance(rule, NotCoExistenceRule):
            negative.add_edge(rule.activity_a, rule.activity_b)
        elif isinstance(rule, (RespondedExistenceRule, CoExistenceRule)):
            positive.add_edge(rule.activity_a, rule.activity_b)
    return positive, negative
