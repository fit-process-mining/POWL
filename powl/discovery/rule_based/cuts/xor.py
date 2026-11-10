"""Exclusive choice with the source method's rule checks and noisy projection."""

from powl.discovery.rule_based.rules import (
    ChainPrecedenceRule,
    ChainResponseRule,
    CoExistenceRule,
    EndRule,
    ExistenceRule,
    InitializationRule,
    PrecedenceRule,
    RespondedExistenceRule,
    ResponseRule,
)
from powl.discovery.total_order_based.inductive.cuts.xor import (
    POWLExclusiveChoiceCutUVCL,
)

from .base import RuleAwareCut, separated


class RuleAwareExclusiveChoiceCutUVCL(RuleAwareCut, POWLExclusiveChoiceCutUVCL):
    @staticmethod
    def check_rules(rules, groups):
        return {
            rule
            for rule in rules
            if isinstance(rule, (InitializationRule, ExistenceRule, EndRule))
            or (
                isinstance(
                    rule,
                    (
                        PrecedenceRule,
                        ChainPrecedenceRule,
                        CoExistenceRule,
                        ResponseRule,
                        ChainResponseRule,
                        RespondedExistenceRule,
                    ),
                )
                and separated(rule, groups)
            )
        }

    @classmethod
    def holds(cls, obj, parameters=None):
        groups = super().holds(obj, parameters)
        if groups is None:
            return None
        # This order is significant for the source method's projection tie break.
        return sorted(groups, key=lambda group: (-len(group), " ".join(sorted(group))))
