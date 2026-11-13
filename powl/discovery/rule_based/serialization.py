"""Data-only declarative rule contracts for applications and saved projects.

Rules cross application boundaries as template names and activity labels. No
Python source is needed to construct or restore them.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any

from .exceptions import InconsistentRulesError
from .preprocessing import product_automaton
from .rules import (
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
    normalize_rules,
)


_TEMPLATES: tuple[tuple[type[AbstractRule], int, str], ...] = (
    (ExistenceRule, 1, "A occurs at least once in every case."),
    (AtMostOnceRule, 1, "A occurs at most once in every case."),
    (InitializationRule, 1, "Every case starts with A."),
    (EndRule, 1, "Every case ends with A."),
    (ResponseRule, 2, "Every A is eventually followed by B."),
    (PrecedenceRule, 2, "Every B has an earlier A."),
    (ChainResponseRule, 2, "Every A is immediately followed by B."),
    (ChainPrecedenceRule, 2, "Every B is immediately preceded by A."),
    (RespondedExistenceRule, 2, "If A occurs, B occurs somewhere in the same case."),
    (CoExistenceRule, 2, "A and B both occur, or both are absent."),
    (NotCoExistenceRule, 2, "A and B never occur together in a case."),
    (NotSuccessionRule, 2, "B never occurs after A."),
)
_TYPES = {cls.__name__.removesuffix("Rule"): (cls, arity) for cls, arity, _ in _TEMPLATES}


def rule_catalog() -> list[dict[str, Any]]:
    """Return the supported templates in a stable, user-facing order.

    Returned containers are fresh copies and may safely be edited by callers.
    The activity placeholders A and B follow the order in ``activities``.
    """
    return [
        {"template": cls.__name__.removesuffix("Rule"), "arity": arity, "description": description}
        for cls, arity, description in _TEMPLATES
    ]


def rule_to_dict(rule: AbstractRule) -> dict[str, Any]:
    """Serialize a supported rule as ``{template, activities}``."""
    # Reject subclasses whose semantics could differ from the named template.
    if type(rule) not in {cls for cls, _, _ in _TEMPLATES}:
        raise TypeError("Only built-in declarative rule templates can be serialized.")
    return {"template": rule.name, "activities": list(rule.args)}


def _activity_labels(activities: Iterable[str]) -> set[str]:
    if isinstance(activities, (str, bytes)):
        raise TypeError("Activities must be an iterable of activity labels, not a string.")
    labels = set()
    for activity in activities:
        if not isinstance(activity, str) or not activity:
            raise ValueError("Activity labels must be nonempty strings.")
        labels.add(activity)
    return labels


def rule_from_dict(
    spec: Mapping[str, Any], *, activities: Iterable[str] | None = None
) -> AbstractRule:
    """Construct a rule from a strict, JSON-compatible specification.

    For example ``{"template": "Response", "activities": ["Approve", "Pay"]}``.
    Labels are matched exactly; they are never trimmed, renamed, or inferred.
    ``Init`` is accepted as an alias for the canonical ``Initialization`` name.
    Unknown fields are rejected so that misspelled options cannot disappear.
    """
    if not isinstance(spec, Mapping):
        raise TypeError("A rule specification must be an object.")
    if set(spec) != {"template", "activities"}:
        raise ValueError("A rule specification requires exactly 'template' and 'activities'.")
    name = spec["template"]
    if not isinstance(name, str):
        raise ValueError("Rule template must be a string.")
    name = "Initialization" if name == "Init" else name
    if name not in _TYPES:
        raise ValueError(f"Unsupported rule template: {name!r}.")
    rule_type, arity = _TYPES[name]
    args = spec["activities"]
    if not isinstance(args, (list, tuple)) or len(args) != arity:
        raise ValueError(f"{name} requires exactly {arity} activity label(s).")
    rule = rule_type(*args)
    if activities is not None:
        allowed = _activity_labels(activities)
        unknown = rule.activities - allowed
        if unknown:
            raise ValueError(f"Unknown activity label(s): {', '.join(repr(a) for a in sorted(unknown))}.")
    return rule


def validate_rule_consistency(
    rules: Iterable[AbstractRule],
    *,
    activities: Iterable[str] | None = None,
    max_states: int = 100_000,
) -> tuple[str, ...]:
    """Prove that the rules admit a finite trace and return a shortest witness.

    A witness can be empty: binary rules alone may all be satisfied vacuously.
    This checks logical consistency, not empirical support or whether every
    listed activity occurs in an accepted trace. Unknown labels are rejected
    when an explicit alphabet is supplied. State exhaustion raises
    ``RuleDiscoveryLimitError``; it is never reported as inconsistency.
    """
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1:
        raise ValueError("max_states must be a positive integer.")
    supplied = normalize_rules(rules)
    referenced = {activity for rule in supplied for activity in rule.args}
    alphabet = referenced if activities is None else _activity_labels(activities)
    unknown = referenced - alphabet
    if unknown:
        raise ValueError(f"Unknown activity label(s): {', '.join(repr(a) for a in sorted(unknown))}.")
    product = product_automaton(supplied, alphabet, max_states=max_states)
    if not product.final_states:
        raise InconsistentRulesError("The declarative rules admit no finite trace.")

    # Store predecessor links rather than copying potentially long traces for
    # every state. Sorted symbols make equally short witnesses deterministic.
    initial = product.initial_state
    predecessors = {initial: None}
    queue = deque([initial])
    while queue:
        state = queue.popleft()
        if state in product.final_states:
            witness = []
            while predecessors[state] is not None:
                state, symbol = predecessors[state]
                witness.append(symbol)
            return tuple(reversed(witness))
        for symbol in sorted(alphabet):
            target = product.transitions[state][symbol]
            if target not in predecessors:
                predecessors[target] = (state, symbol)
                queue.append(target)
    raise InconsistentRulesError("The declarative rules admit no finite trace.")


__all__ = ["rule_catalog", "rule_from_dict", "rule_to_dict", "validate_rule_consistency"]
