"""Optional, non-mutating validation of user-supplied declarative rules.

This module evaluates an existing rule set. It does not discover, rank, filter,
or modify rules, and discovery never calls it implicitly to discard rules.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from powl.discovery.rule_based.rules import AbstractRule, normalize_rules


@dataclass(frozen=True, slots=True)
class RuleViolationExample:
    """One observed violating activity sequence and its case frequency."""

    activities: tuple[str, ...]
    frequency: int


@dataclass(frozen=True, slots=True)
class RuleValidationResult:
    """Trace-weighted metrics for a single rule on one event log.

    ``support`` is satisfied traces / total traces, including vacuous
    satisfaction. ``confidence`` is satisfied activated traces / activated
    traces. Activation means A for Response, ChainResponse,
    RespondedExistence and NotSuccession; B for Precedence and ChainPrecedence;
    either A or B for CoExistence and NotCoExistence; and every trace for unary
    rules. Thus a unary rule's confidence equals its support.

    Empty traces count as real traces. On an empty log, support is 0.0. Binary
    templates with no activations have confidence 1.0 by convention; this is
    vacuous satisfaction, not evidence, and ``activated_traces`` makes it
    visible. Unary confidence on an empty log is 0.0.
    """

    rule: AbstractRule
    total_traces: int
    satisfied_traces: int
    violated_traces: int
    activated_traces: int
    satisfied_activated_traces: int
    vacuously_satisfied_traces: int
    support: float
    confidence: float
    violation_examples: tuple[RuleViolationExample, ...] = ()

    @property
    def violation_rate(self) -> float:
        """Violating fraction, or 0.0 when the log has no traces."""
        return self.violated_traces / self.total_traces if self.total_traces else 0.0


def validate_rules(
    log: Any,
    rules: Iterable[AbstractRule],
    *,
    activity_key: str = "concept:name",
    case_id_key: str = "case:concept:name",
    timestamp_key: str | None = "time:timestamp",
    max_examples: int = 0,
) -> tuple[RuleValidationResult, ...]:
    """Evaluate the supplied rule set without changing it or the event log.

    Accept the same log formats as rule-aware discovery: a pandas DataFrame,
    PM4Py EventLog, iterable of activity traces, or mapping of trace tuples to
    integer frequencies. Frequencies weight every metric; an empty trace is
    evaluated rather than silently skipped. DataFrame column keys can be
    customized with the keyword arguments.

    Results are deterministically ordered by template name and arguments, with
    duplicate rules evaluated once. They contain counts as well as fractions so
    vacuous confidence can be distinguished from confidence backed by data.
    Set ``max_examples`` to include up to that many violating variants, ordered
    by descending case frequency and then activity labels. The original log is
    never modified; examples are immutable copies of the observed sequences.
    """
    from powl.discovery.rule_based.log_utils import to_variants

    if isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples < 0:
        raise ValueError("max_examples must be a nonnegative integer.")
    supplied = normalize_rules(rules)
    variants = to_variants(
        log,
        activity_key=activity_key,
        case_id_key=case_id_key,
        timestamp_key=timestamp_key,
    )
    total = sum(variants.values())
    results = []
    for rule in supplied:
        satisfied = activated = satisfied_activated = vacuous = 0
        violations = []
        for trace, frequency in variants.items():
            active = rule.is_activated(trace)
            valid = rule.accepts(trace)
            if active:
                activated += frequency
            if valid:
                satisfied += frequency
                if active:
                    satisfied_activated += frequency
                else:
                    vacuous += frequency
            elif max_examples:
                violations.append(RuleViolationExample(tuple(trace), frequency))
        support = satisfied / total if total else 0.0
        confidence = (
            satisfied_activated / activated
            if activated
            else 0.0 if len(rule.args) == 1 else 1.0
        )
        results.append(
            RuleValidationResult(
                rule=rule,
                total_traces=total,
                satisfied_traces=satisfied,
                violated_traces=total - satisfied,
                activated_traces=activated,
                satisfied_activated_traces=satisfied_activated,
                vacuously_satisfied_traces=vacuous,
                support=support,
                confidence=confidence,
                violation_examples=tuple(
                    sorted(violations, key=lambda example: (-example.frequency, example.activities))[
                        :max_examples
                    ]
                ),
            )
        )
    return tuple(results)


__all__ = ["RuleValidationResult", "RuleViolationExample", "validate_rules"]
