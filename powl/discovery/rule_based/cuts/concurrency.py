"""Concurrent cuts, including the source method's conservative chain checks."""

from powl.discovery.rule_based.rules import (
    ChainPrecedenceRule,
    ChainResponseRule,
    EndRule,
    InitializationRule,
    NotCoExistenceRule,
    NotSuccessionRule,
    PrecedenceRule,
    ResponseRule,
)
from powl.discovery.total_order_based.inductive.cuts.concurrency import (
    POWLConcurrencyCutUVCL,
)

from .base import RuleAwareCut, separated


class RuleAwareConcurrencyCutUVCL(RuleAwareCut, POWLConcurrencyCutUVCL):
    @staticmethod
    def check_rules(rules, groups):
        rejected = set()
        for rule in rules:
            if isinstance(rule, (InitializationRule, EndRule)):
                if any(rule.target_activity in group for group in groups):
                    rejected.add(rule)
            elif isinstance(rule, (ChainPrecedenceRule, ChainResponseRule)):
                # Even a chain wholly inside one branch can be interrupted by
                # an event from another concurrent branch.
                rejected.add(rule)
            elif isinstance(
                rule,
                (NotCoExistenceRule, NotSuccessionRule, PrecedenceRule, ResponseRule),
            ) and separated(rule, groups):
                rejected.add(rule)
        return rejected
