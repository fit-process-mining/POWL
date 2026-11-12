"""Immutable declarative rules used by rule-aware POWL discovery.

The twelve templates and their deletion/trimming repairs follow the attached
rule-based Inductive Miner. Predicates use the templates' DFA semantics: in
particular *every* activation of Response must be fulfilled, and CoExistence is
true when neither activity occurs. Rule instances contain no log-specific state;
use :func:`validate_rules` to calculate support and confidence separately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from automata.fa.dfa import DFA

Trace = Sequence[str]


class AbstractRule(ABC):
    """A rule over activity labels, independent of any particular event log."""

    __slots__ = ()
    _states: ClassVar[tuple[int, ...]]
    _final_states: ClassVar[frozenset[int]]

    @property
    @abstractmethod
    def args(self) -> tuple[str, ...]:
        """The activity labels in the rule's declared argument order."""

    @property
    def activities(self) -> frozenset[str]:
        return frozenset(self.args)

    @property
    def name(self) -> str:
        return type(self).__name__.removesuffix("Rule")

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def accepts(self, trace: Trace) -> bool:
        """Return whether the complete finite trace satisfies this rule."""

    def is_satisfied(self, trace: Trace) -> bool:
        """Alias for :meth:`accepts`."""
        return self.accepts(trace)

    def is_activated(self, trace: Trace) -> bool:
        """Whether a trace contributes to the confidence denominator.

        Unary rules are evaluated on every trace. Directed binary templates use
        their activation activity; symmetric templates use either activity.
        """
        return True

    def apply(self, log: Iterable[Trace]) -> list[list[str]]:
        """Filter complete traces; never mutate the log or this rule."""
        return [list(trace) for trace in log if self.accepts(trace)]

    @abstractmethod
    def repair_trace(self, trace: Trace) -> tuple[str, ...] | None:
        """Apply the rule's event-level repair, or return None to drop a trace.

        Repairs delete events, trim prefixes/suffixes, or drop traces. They do
        not insert events. Edit-distance repair is implemented by the miner.
        """

    def repair(self, log: Iterable[Trace]) -> list[list[str]]:
        """Return copied, repaired traces without changing the input log."""
        repaired = []
        for trace in log:
            result = self.repair_trace(trace)
            if result is not None:
                repaired.append(list(result))
        return repaired

    @abstractmethod
    def _step(self, state: int, activity: str) -> int:
        """Transition function for the template's finite-state automaton."""

    def to_automaton(self, alphabet: Iterable[str]) -> DFA:
        """Build the rule DFA over exactly ``alphabet``.

        Activity labels referenced by a rule need not belong to this alphabet.
        This is necessary after projecting a log, and gives the natural rule
        language over the supplied alphabet (possibly empty or universal).
        Multi-character labels are symbols; pass sequences of labels to the DFA.
        """
        from automata.fa.dfa import DFA

        symbols = set(alphabet)
        if any(not isinstance(activity, str) for activity in symbols):
            raise TypeError("Activity labels in the alphabet must be strings.")
        return DFA(
            states=set(self._states),
            input_symbols=symbols,
            transitions={
                state: {activity: self._step(state, activity) for activity in symbols}
                for state in self._states
            },
            initial_state=0,
            final_states=set(self._final_states),
        )


def _check_activity(activity: str) -> None:
    if not isinstance(activity, str):
        raise TypeError("Declarative rule activity labels must be strings.")
    if not activity:
        raise ValueError("Declarative rule activity labels must be nonempty strings.")


@dataclass(frozen=True, slots=True, repr=False)
class _UnaryRule(AbstractRule):
    target_activity: str

    def __post_init__(self) -> None:
        _check_activity(self.target_activity)

    @property
    def args(self) -> tuple[str, ...]:
        return (self.target_activity,)


@dataclass(frozen=True, slots=True, repr=False)
class _BinaryRule(AbstractRule):
    activity_a: str
    activity_b: str

    def __post_init__(self) -> None:
        _check_activity(self.activity_a)
        _check_activity(self.activity_b)
        if self.activity_a == self.activity_b:
            raise ValueError(
                "Binary declarative rules require two distinct activities."
            )

    @property
    def args(self) -> tuple[str, ...]:
        return (self.activity_a, self.activity_b)

    def is_activated(self, trace: Trace) -> bool:
        return self.activity_a in trace


@dataclass(frozen=True, slots=True, repr=False)
class ExistenceRule(_UnaryRule):
    """The target activity occurs at least once in every trace."""

    _states = (0, 1)
    _final_states = frozenset({1})

    def accepts(self, trace: Trace) -> bool:
        return self.target_activity in trace

    def repair_trace(self, trace: Trace) -> tuple[str, ...] | None:
        return tuple(trace) if self.accepts(trace) else None

    def _step(self, state: int, activity: str) -> int:
        return int(state == 1 or activity == self.target_activity)


@dataclass(frozen=True, slots=True, repr=False)
class AtMostOnceRule(_UnaryRule):
    """The target activity occurs zero or one times in every trace."""

    _states = (0, 1, 2)
    _final_states = frozenset({0, 1})

    def accepts(self, trace: Trace) -> bool:
        return trace.count(self.target_activity) <= 1

    def repair_trace(self, trace: Trace) -> tuple[str, ...]:
        seen = False
        repaired = []
        for activity in trace:
            if activity == self.target_activity:
                if seen:
                    continue
                seen = True
            repaired.append(activity)
        return tuple(repaired)

    def _step(self, state: int, activity: str) -> int:
        return min(2, state + int(activity == self.target_activity))


@dataclass(frozen=True, slots=True, repr=False)
class InitializationRule(_UnaryRule):
    """Every trace is nonempty and starts with the target activity."""

    _states = (0, 1, 2)
    _final_states = frozenset({1})

    def accepts(self, trace: Trace) -> bool:
        return bool(trace) and trace[0] == self.target_activity

    def repair_trace(self, trace: Trace) -> tuple[str, ...] | None:
        if self.target_activity not in trace:
            return None
        return tuple(trace[trace.index(self.target_activity) :])

    def _step(self, state: int, activity: str) -> int:
        return (1 if activity == self.target_activity else 2) if state == 0 else state


@dataclass(frozen=True, slots=True, repr=False)
class EndRule(_UnaryRule):
    """Every trace is nonempty and ends with the target activity."""

    _states = (0, 1)
    _final_states = frozenset({1})

    def accepts(self, trace: Trace) -> bool:
        return bool(trace) and trace[-1] == self.target_activity

    def repair_trace(self, trace: Trace) -> tuple[str, ...] | None:
        for index in range(len(trace) - 1, -1, -1):
            if trace[index] == self.target_activity:
                return tuple(trace[: index + 1])
        return None

    def _step(self, state: int, activity: str) -> int:
        return int(activity == self.target_activity)


@dataclass(frozen=True, slots=True, repr=False)
class ResponseRule(_BinaryRule):
    """Every occurrence of A is followed eventually by an occurrence of B."""

    _states = (0, 1)
    _final_states = frozenset({0})

    def accepts(self, trace: Trace) -> bool:
        # The last A must precede a B; then all earlier A's are fulfilled too.
        for activity in reversed(trace):
            if activity == self.activity_b:
                return True
            if activity == self.activity_a:
                return False
        return True

    def repair_trace(self, trace: Trace) -> tuple[str, ...]:
        seen_b = False
        repaired = []
        for activity in reversed(trace):
            if activity == self.activity_b:
                seen_b = True
            if activity != self.activity_a or seen_b:
                repaired.append(activity)
        return tuple(reversed(repaired))

    def _step(self, state: int, activity: str) -> int:
        if activity == self.activity_a:
            return 1
        return 0 if activity == self.activity_b else state


@dataclass(frozen=True, slots=True, repr=False)
class PrecedenceRule(_BinaryRule):
    """Every occurrence of B is preceded somewhere by an occurrence of A."""

    _states = (0, 1, 2)
    _final_states = frozenset({0, 1})

    def is_activated(self, trace: Trace) -> bool:
        return self.activity_b in trace

    def accepts(self, trace: Trace) -> bool:
        for activity in trace:
            if activity == self.activity_a:
                return True
            if activity == self.activity_b:
                return False
        return True

    def repair_trace(self, trace: Trace) -> tuple[str, ...]:
        seen_a = False
        repaired = []
        for activity in trace:
            if activity == self.activity_a:
                seen_a = True
            if activity != self.activity_b or seen_a:
                repaired.append(activity)
        return tuple(repaired)

    def _step(self, state: int, activity: str) -> int:
        if state != 0:
            return state
        if activity == self.activity_a:
            return 1
        return 2 if activity == self.activity_b else 0


@dataclass(frozen=True, slots=True, repr=False)
class ChainResponseRule(ResponseRule):
    """Every occurrence of A is immediately followed by B."""

    _states = (0, 1, 2)
    _final_states = frozenset({0})

    def accepts(self, trace: Trace) -> bool:
        return all(
            index + 1 < len(trace) and trace[index + 1] == self.activity_b
            for index, activity in enumerate(trace)
            if activity == self.activity_a
        )

    def repair_trace(self, trace: Trace) -> tuple[str, ...]:
        return tuple(
            activity
            for index, activity in enumerate(trace)
            if activity != self.activity_a
            or (index + 1 < len(trace) and trace[index + 1] == self.activity_b)
        )

    def _step(self, state: int, activity: str) -> int:
        if state == 2:
            return 2
        if state == 1:
            return 0 if activity == self.activity_b else 2
        return int(activity == self.activity_a)


@dataclass(frozen=True, slots=True, repr=False)
class ChainPrecedenceRule(PrecedenceRule):
    """Every occurrence of B is immediately preceded by A."""

    _states = (0, 1, 2)
    _final_states = frozenset({0, 1})

    def accepts(self, trace: Trace) -> bool:
        return all(
            index > 0 and trace[index - 1] == self.activity_a
            for index, activity in enumerate(trace)
            if activity == self.activity_b
        )

    def repair_trace(self, trace: Trace) -> tuple[str, ...]:
        return tuple(
            activity
            for index, activity in enumerate(trace)
            if activity != self.activity_b
            or (index > 0 and trace[index - 1] == self.activity_a)
        )

    def _step(self, state: int, activity: str) -> int:
        if state == 2 or (state == 0 and activity == self.activity_b):
            return 2
        return int(activity == self.activity_a)


@dataclass(frozen=True, slots=True, repr=False)
class RespondedExistenceRule(_BinaryRule):
    """If A occurs, B also occurs somewhere in the same trace."""

    _states = (0, 1, 2, 3)
    _final_states = frozenset({0, 2, 3})

    def accepts(self, trace: Trace) -> bool:
        return self.activity_a not in trace or self.activity_b in trace

    def repair_trace(self, trace: Trace) -> tuple[str, ...]:
        if self.activity_b in trace:
            return tuple(trace)
        return tuple(activity for activity in trace if activity != self.activity_a)

    def _step(self, state: int, activity: str) -> int:
        return state | (
            1
            if activity == self.activity_a
            else 2 if activity == self.activity_b else 0
        )


@dataclass(frozen=True, slots=True, repr=False)
class CoExistenceRule(_BinaryRule):
    """A and B both occur or both are absent."""

    _states = (0, 1, 2, 3)
    _final_states = frozenset({0, 3})

    def is_activated(self, trace: Trace) -> bool:
        return self.activity_a in trace or self.activity_b in trace

    def accepts(self, trace: Trace) -> bool:
        return (self.activity_a in trace) == (self.activity_b in trace)

    def repair_trace(self, trace: Trace) -> tuple[str, ...]:
        if self.accepts(trace):
            return tuple(trace)
        return tuple(activity for activity in trace if activity not in self.activities)

    def _step(self, state: int, activity: str) -> int:
        return state | (
            1
            if activity == self.activity_a
            else 2 if activity == self.activity_b else 0
        )


@dataclass(frozen=True, slots=True, repr=False)
class NotCoExistenceRule(_BinaryRule):
    """A and B never occur together in the same trace."""

    _states = (0, 1, 2, 3)
    _final_states = frozenset({0, 1, 2})

    def is_activated(self, trace: Trace) -> bool:
        return self.activity_a in trace or self.activity_b in trace

    def accepts(self, trace: Trace) -> bool:
        return not (self.activity_a in trace and self.activity_b in trace)

    def repair_trace(self, trace: Trace) -> tuple[str, ...]:
        if self.accepts(trace):
            return tuple(trace)
        # Keep the activity that appeared first, matching the original repair.
        removed = (
            self.activity_b
            if trace.index(self.activity_a) < trace.index(self.activity_b)
            else self.activity_a
        )
        return tuple(activity for activity in trace if activity != removed)

    def _step(self, state: int, activity: str) -> int:
        return state | (
            1
            if activity == self.activity_a
            else 2 if activity == self.activity_b else 0
        )


@dataclass(frozen=True, slots=True, repr=False)
class NotSuccessionRule(_BinaryRule):
    """No occurrence of B occurs after an occurrence of A."""

    _states = (0, 1, 2)
    _final_states = frozenset({0, 1})

    def accepts(self, trace: Trace) -> bool:
        seen_a = False
        for activity in trace:
            if activity == self.activity_a:
                seen_a = True
            if activity == self.activity_b and seen_a:
                return False
        return True

    def repair_trace(self, trace: Trace) -> tuple[str, ...]:
        seen_b = False
        repaired = []
        for activity in reversed(trace):
            if activity == self.activity_b:
                seen_b = True
            if activity != self.activity_a or not seen_b:
                repaired.append(activity)
        return tuple(reversed(repaired))

    def _step(self, state: int, activity: str) -> int:
        if state == 2 or (state == 1 and activity == self.activity_b):
            return 2
        return 1 if activity == self.activity_a else state


SUPPORTED_RULE_TYPES = (
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


def normalize_rules(rules: Iterable[AbstractRule] | None) -> tuple[AbstractRule, ...]:
    """Validate, deduplicate, and deterministically order a set of rules."""
    if rules is None:
        return ()
    supplied = tuple(rules)
    if any(not isinstance(rule, AbstractRule) for rule in supplied):
        raise TypeError("rules must contain declarative AbstractRule objects.")
    return tuple(
        sorted(set(supplied), key=lambda rule: (type(rule).__name__, rule.args))
    )


__all__ = [
    "AbstractRule",
    "AtMostOnceRule",
    "ChainPrecedenceRule",
    "ChainResponseRule",
    "CoExistenceRule",
    "EndRule",
    "ExistenceRule",
    "InitializationRule",
    "NotCoExistenceRule",
    "NotSuccessionRule",
    "PrecedenceRule",
    "RespondedExistenceRule",
    "ResponseRule",
]
