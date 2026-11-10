"""Sequence cut rejection and projection, sharing the existing POWL cuts."""

from collections import Counter

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

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
from powl.discovery.total_order_based.inductive.cuts.sequence import (
    POWLSequenceCutUVCL,
    POWLStrictSequenceCutUVCL,
)

from .base import RuleAwareCut


class _SequenceRules(RuleAwareCut):
    @staticmethod
    def check_rules(rules, groups):
        rejected = set()
        for rule in rules:
            if isinstance(rule, InitializationRule):
                if rule.target_activity not in groups[0]:
                    rejected.add(rule)
            elif isinstance(rule, EndRule):
                if rule.target_activity not in groups[-1]:
                    rejected.add(rule)
            elif isinstance(rule, NotCoExistenceRule):
                a = {i for i, group in enumerate(groups) if rule.activity_a in group}
                b = {i for i, group in enumerate(groups) if rule.activity_b in group}
                if a and b and a != b:
                    rejected.add(rule)
            elif isinstance(rule, (ChainResponseRule, ResponseRule)):
                a = {i for i, group in enumerate(groups) if rule.activity_a in group}
                b = {i for i, group in enumerate(groups) if rule.activity_b in group}
                if a and b and a != b:
                    if isinstance(rule, ChainResponseRule):
                        valid = all(i + 1 in b for i in a)
                    else:
                        valid = all(any(i < j for j in b) for i in a)
                    if not valid:
                        rejected.add(rule)
            elif isinstance(rule, (PrecedenceRule, ChainPrecedenceRule)):
                a = {i for i, group in enumerate(groups) if rule.activity_a in group}
                b = {i for i, group in enumerate(groups) if rule.activity_b in group}
                if a and b and a != b:
                    if isinstance(rule, ChainPrecedenceRule):
                        valid = all(j - 1 in a for j in b)
                    else:
                        valid = all(any(i < j for i in a) for j in b)
                    if not valid:
                        rejected.add(rule)
            elif isinstance(rule, NotSuccessionRule):
                a = [i for i, group in enumerate(groups) if rule.activity_a in group]
                b = [i for i, group in enumerate(groups) if rule.activity_b in group]
                if a and b and a[0] < b[-1]:
                    rejected.add(rule)
        return rejected

    @classmethod
    def relaxed_projection(cls, obj, groups, parameters=None):
        """Weighted form of the source rule-order fallback's relaxed projection.

        Ordinary cut projection is inherited unchanged.  Here a group whose
        cost-minimizing segment is empty can claim its next occurrence.  This
        distinction matters when a rule-derived order disagrees with the log.
        """
        logs = [Counter() for _ in groups]
        for trace, count in obj.data_structure.items():
            split = 0
            ignored = set()
            for i, group in enumerate(groups):
                new_split = cls._source_split_point(trace, group, split, ignored)
                projected = tuple(a for a in trace[split:new_split] if a in group)
                if not projected:
                    for j in range(split, len(trace)):
                        if trace[j] in group:
                            new_split = j + 1
                            projected = (trace[j],)
                            break
                logs[i][projected] += count
                split = new_split
                ignored.update(group)
        return [IMDataStructureUVCL(log) for log in logs]

    @staticmethod
    def _source_split_point(trace, group, start, ignored):
        best_cost = cost = 0
        position = start
        for index in range(start, len(trace)):
            if trace[index] in group:
                cost -= 1
            elif trace[index] not in ignored:
                cost += 1
            if cost < best_cost:
                best_cost = cost
                position = index + 1
        return position


class RuleAwareSequenceCutUVCL(_SequenceRules, POWLSequenceCutUVCL):
    """Sequence rule logic, also used for rule-derived fall-through orders."""


class RuleAwareStrictSequenceCutUVCL(_SequenceRules, POWLStrictSequenceCutUVCL):
    """The normal strict sequence candidate in rule-aware discovery."""

    @classmethod
    def holds(cls, obj, parameters=None):
        groups = super().holds(obj, parameters)
        # Some PM4Py versions return the single group left by strict merging.
        # The source method requires a genuine split, otherwise recursion would
        # repeatedly rediscover the same log.
        return groups if groups is not None and len(groups) > 1 else None
