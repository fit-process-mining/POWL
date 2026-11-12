"""Rule-directed XOR fall-through using the reference signed-CDG coloring."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.rule_based.cuts import RuleAwareExclusiveChoiceCutUVCL
from powl.discovery.rule_based.rules import (
    EndRule,
    ExistenceRule,
    InitializationRule,
    NotCoExistenceRule,
    NotSuccessionRule,
)
from powl.discovery.total_order_based.inductive.modeling import XorSpec, build_model

from .utils import alphabet_of, build_signed_cdg, project_alphabet


def normalize_not_succession_pairs(rules):
    """Replace the two directed prohibitions by one NotCoExistence rule."""
    rules = tuple(rules)
    pairs = {
        (rule.activity_a, rule.activity_b)
        for rule in rules
        if isinstance(rule, NotSuccessionRule)
    }
    symmetric = {tuple(sorted((a, b))) for a, b in pairs if (b, a) in pairs}
    normalized = [
        rule
        for rule in rules
        if not (
            isinstance(rule, NotSuccessionRule)
            and tuple(sorted((rule.activity_a, rule.activity_b))) in symmetric
        )
    ]
    existing = {
        tuple(sorted((rule.activity_a, rule.activity_b)))
        for rule in normalized
        if isinstance(rule, NotCoExistenceRule)
    }
    normalized.extend(NotCoExistenceRule(a, b) for a, b in sorted(symmetric - existing))
    return tuple(normalized)


def forced_absences(rules):
    required = {
        rule.target_activity
        for rule in rules
        if isinstance(rule, (ExistenceRule, InitializationRule, EndRule))
    }
    forbidden = set()
    for rule in rules:
        if not isinstance(rule, NotCoExistenceRule):
            continue
        if rule.activity_a in required and rule.activity_b not in required:
            forbidden.add(rule.activity_b)
        if rule.activity_b in required and rule.activity_a not in required:
            forbidden.add(rule.activity_a)
    return forbidden


def detect_groups(positive_graph, negative_graph):
    """Color positive components so negative neighbors enter different XORs."""
    components = list(nx.weakly_connected_components(positive_graph))
    component_of = {
        activity: i for i, group in enumerate(components) for activity in group
    }
    conflicts = nx.Graph()
    conflicts.add_nodes_from(range(len(components)))
    for source, target in negative_graph.edges:
        a, b = component_of.get(source), component_of.get(target)
        # An unresolved contradiction within a positive component cannot be
        # represented by this fall-through. Do not color a self-loop or lose it.
        if a is None or b is None or a == b:
            return None
        conflicts.add_edge(a, b)
    colors = nx.coloring.greedy_color(conflicts, strategy="largest_first")
    groups = defaultdict(set)
    for component, color in colors.items():
        groups[color].update(components[component])
    return list(groups.values())


def rule_xor(obj: IMDataStructureUVCL, rules, mine, parameters=None):
    rules = normalize_not_succession_pairs(rules)
    if not any(isinstance(rule, NotCoExistenceRule) for rule in rules):
        return None
    alphabet = alphabet_of(obj)
    forbidden = forced_absences(rules) & alphabet
    if forbidden:
        # The prototype projected onto only the required activity here, which
        # accidentally deleted unrelated activities. Remove only the side(s)
        # prohibited by existence and keep the remaining obligations.
        remaining = alphabet - forbidden
        child = project_alphabet(obj, remaining)
        child_rules = RuleAwareExclusiveChoiceCutUVCL.project_rules(rules, [remaining])[
            0
        ]
        return mine(child, tuple(child_rules))
    positive, negative = build_signed_cdg(rules, alphabet)
    groups = detect_groups(positive, negative)
    if not groups or len(groups) <= 1:
        return None
    projected_rules = RuleAwareExclusiveChoiceCutUVCL.project_rules(rules, groups)
    children = [project_alphabet(obj, group, distinct=True) for group in groups]
    models = [
        mine(child, tuple(local)) for child, local in zip(children, projected_rules)
    ]
    if any(model is None for model in models):
        return None
    return build_model(XorSpec(len(models)), models)
