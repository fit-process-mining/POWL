"""Rule obligations layered on POWL's existing inductive cuts.

The ordinary cut API remains available.  ``evaluate`` additionally returns the
rules that rejected a structurally valid cut, or the obligations for each child.
Keeping these obligations next to the projected logs avoids shared, mutable
recursion parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.rule_based.rules import (
    AbstractRule,
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
from powl.discovery.total_order_based.inductive.modeling import InductiveSpec

Groups = Sequence[set[str] | frozenset[str]]
ProjectedRules = tuple[tuple[AbstractRule, ...], ...]


@dataclass(frozen=True)
class RuleAwareCutResult:
    """A structural candidate and its rule decision.

    Rejected candidates have empty ``children`` and ``projected_rules``.  Their
    ``groups`` and ``rejected_rules`` are retained for the miner's repair policy.
    """

    spec: InductiveSpec
    groups: tuple[frozenset[str], ...]
    children: tuple[IMDataStructureUVCL, ...] = ()
    projected_rules: ProjectedRules = ()
    rejected_rules: frozenset[AbstractRule] = frozenset()


class RuleAwareCut:
    """Mixin for a POWL cut with declarative rejection and rule projection."""

    @classmethod
    def evaluate(
        cls,
        obj: IMDataStructureUVCL,
        rules: Iterable[AbstractRule] = (),
        parameters: Mapping[str, Any] | None = None,
    ) -> RuleAwareCutResult | None:
        groups = cls.holds(obj, parameters)
        if groups is None:
            return None
        rules = tuple(rules)
        rejected = frozenset(cls.check_rules(rules, groups))
        spec = replace(cls.operator(parameters), size=len(groups))
        frozen_groups = tuple(frozenset(group) for group in groups)
        if rejected:
            return RuleAwareCutResult(spec, frozen_groups, rejected_rules=rejected)
        return RuleAwareCutResult(
            spec,
            frozen_groups,
            tuple(cls.project(obj, groups, parameters)),
            cls.project_rules(rules, groups),
        )

    @classmethod
    def apply(
        cls,
        obj: IMDataStructureUVCL,
        parameters: Mapping[str, Any] | None = None,
    ):
        """Retain the base-cut API; use ``evaluate`` to inspect rejections."""
        rules = () if parameters is None else parameters.get("rules", ())
        result = cls.evaluate(obj, rules, parameters)
        if result is None or result.rejected_rules:
            return None
        return result.spec, list(result.children)

    @staticmethod
    def check_rules(rules: Iterable[AbstractRule], groups: Groups) -> set[AbstractRule]:
        raise NotImplementedError

    @staticmethod
    def project_rules(
        rules: Iterable[AbstractRule] | None, groups: Groups
    ) -> ProjectedRules:
        """Project the student's obligations, including cross-branch rules.

        A binary rule that the parent discharges can still impose unary rules on
        its children.  For example, a response crossing a sequence boundary
        requires existence of its target, and a chain response requires an end
        and an initialization boundary.  Missing positive obligations are kept
        for recursive repair instead of silently disappearing.
        """
        if not groups:
            return ()
        projected: list[list[AbstractRule]] = [[] for _ in groups]
        for rule in rules or ():
            if hasattr(rule, "target_activity"):
                index = next(
                    (
                        i
                        for i, group in enumerate(groups)
                        if rule.target_activity in group
                    ),
                    0,
                )
                projected[index].append(rule)
                continue

            # A rule-derived sequence can repeat an activity in several groups.
            # Preserve the source projection's last matching binary endpoint.
            a = next(
                (
                    i
                    for i in reversed(range(len(groups)))
                    if rule.activity_a in groups[i]
                ),
                None,
            )
            b = next(
                (
                    i
                    for i in reversed(range(len(groups)))
                    if rule.activity_b in groups[i]
                ),
                None,
            )
            if a is None or b is None:
                index = a if a is not None else b
                if index is not None and not isinstance(
                    rule, (NotCoExistenceRule, NotSuccessionRule)
                ):
                    projected[index].append(rule)
                continue
            if a == b:
                projected[a].append(rule)
            elif isinstance(rule, ChainResponseRule):
                projected[a].extend(
                    (AtMostOnceRule(rule.activity_a), EndRule(rule.activity_a))
                )
                projected[b].append(InitializationRule(rule.activity_b))
            elif isinstance(rule, ChainPrecedenceRule):
                projected[a].append(EndRule(rule.activity_a))
                projected[b].extend(
                    (
                        AtMostOnceRule(rule.activity_b),
                        InitializationRule(rule.activity_b),
                    )
                )
            elif isinstance(rule, (ResponseRule, RespondedExistenceRule)):
                projected[b].append(ExistenceRule(rule.activity_b))
            elif isinstance(rule, CoExistenceRule):
                projected[a].append(ExistenceRule(rule.activity_a))
                projected[b].append(ExistenceRule(rule.activity_b))
            elif isinstance(rule, PrecedenceRule):
                projected[a].append(ExistenceRule(rule.activity_a))
        return tuple(tuple(child_rules) for child_rules in projected)


def group_of(activity: str, groups: Groups):
    """Return the first containing group, as in the source rule checks."""
    return next((group for group in groups if activity in group), None)


def separated(rule: AbstractRule, groups: Groups) -> bool:
    a, b = group_of(rule.activity_a, groups), group_of(rule.activity_b, groups)
    return a is not None and b is not None and a != b
