"""The reference unary-constrained flower fallback, built directly as POWL."""

from __future__ import annotations

from collections import Counter

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.rule_based.dtypes import RuleDecomposition
from powl.discovery.rule_based.rules import AtMostOnceRule, ExistenceRule
from powl.discovery.total_order_based.inductive.modeling import PartialOrderSpec
from powl.objects.tagged_powl import Activity
from powl.objects.tagged_powl.builders import loop, xor

from .utils import alphabet_of, project_alphabet


def build_flower(activities):
    activities = sorted(activities)
    if not activities:
        return Activity(label=None)
    alternatives = [Activity(label=activity) for activity in activities]
    redo = alternatives[0] if len(alternatives) == 1 else xor(alternatives)
    return loop(Activity(label=None), redo)


def flower(obj: IMDataStructureUVCL, rules=(), parameters=None):
    """Separate unary obligations, leaving other obligations for the caller.

    A plain flower is not a proof of rule satisfaction. The discovery entry
    point verifies the completed model and reports any remaining violation.
    """
    activities = alphabet_of(obj)
    existence = {
        rule.target_activity for rule in rules if isinstance(rule, ExistenceRule)
    }
    at_most_once = {
        rule.target_activity
        for rule in rules
        if isinstance(rule, AtMostOnceRule) and rule.target_activity in activities
    }
    constrained = existence | at_most_once
    if not constrained:
        return build_flower(activities)
    children = []
    projected_rules = []
    for activity in sorted(constrained):
        synthetic = Counter({(activity,): 1})
        if activity not in existence:
            synthetic[()] = 1
        children.append(IMDataStructureUVCL(synthetic))
        projected_rules.append(
            tuple(
                rule
                for rule in rules
                if isinstance(rule, (ExistenceRule, AtMostOnceRule))
                and rule.target_activity == activity
            )
        )
    remaining = activities - constrained
    if remaining:
        children.append(project_alphabet(obj, remaining))
        projected_rules.append(
            tuple(
                rule
                for rule in rules
                if not (
                    isinstance(rule, (ExistenceRule, AtMostOnceRule))
                    and rule.target_activity in constrained
                )
            )
        )
    return RuleDecomposition(
        PartialOrderSpec(len(children)), tuple(children), tuple(projected_rules)
    )
