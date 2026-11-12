"""Rule-aware loops with the source method's single combined redo branch."""

from powl.discovery.rule_based.rules import (
    AtMostOnceRule,
    ChainPrecedenceRule,
    ChainResponseRule,
    CoExistenceRule,
    EndRule,
    ExistenceRule,
    InitializationRule,
    NotCoExistenceRule,
    NotSuccessionRule,
    PrecedenceRule,
    RespondedExistenceRule,
    ResponseRule,
)
from powl.discovery.total_order_based.inductive.cuts.loop import POWLLoopCutUVCL

from .base import RuleAwareCut, separated


class RuleAwareLoopCutUVCL(RuleAwareCut, POWLLoopCutUVCL):
    @classmethod
    def holds(cls, obj, parameters=None):
        groups = super().holds(obj, parameters)
        if groups is None:
            return None
        return [set(groups[0]), set().union(*groups[1:])]

    @staticmethod
    def check_rules(rules, groups):
        rejected = set()
        do, redo = groups[0], groups[1:]
        for rule in rules:
            if isinstance(rule, (ExistenceRule, InitializationRule, EndRule)):
                if any(rule.target_activity in group for group in redo):
                    rejected.add(rule)
            elif isinstance(rule, AtMostOnceRule):
                if any(rule.target_activity in group for group in groups):
                    rejected.add(rule)
            elif isinstance(rule, (CoExistenceRule, ResponseRule, ChainResponseRule)):
                if separated(rule, groups):
                    rejected.add(rule)
            elif isinstance(rule, (PrecedenceRule, ChainPrecedenceRule)):
                if rule.activity_b in do and any(
                    rule.activity_a in group for group in redo
                ):
                    rejected.add(rule)
            elif isinstance(rule, RespondedExistenceRule):
                if rule.activity_a in do and any(
                    rule.activity_b in group for group in redo
                ):
                    rejected.add(rule)
            elif isinstance(rule, (NotSuccessionRule, NotCoExistenceRule)):
                if any(rule.activity_a in group for group in groups) and any(
                    rule.activity_b in group for group in groups
                ):
                    rejected.add(rule)
        return rejected
