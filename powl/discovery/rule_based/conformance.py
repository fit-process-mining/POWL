"""Internal, exact language check before returning a rule-aware POWL model.

This is a discovery correctness guard, not log/model quality evaluation. It
uses POWL's existing Petri-net conversion and explores the reachable marking
space together with each supplied rule's finite automaton. Loops need no
arbitrary trace-length cutoff: revisited product states are memoized.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.obj import Marking

from powl.conversion.variants import to_petri_net
from powl.discovery.rule_based.exceptions import (
    RuleConformanceError,
    RuleDiscoveryLimitError,
)
from powl.discovery.rule_based.rules import AbstractRule, normalize_rules
from powl.objects.tagged_powl.base import TaggedPOWL

ProductState = tuple[Marking, int]
Predecessor = tuple[ProductState, str | None] | None


def _witness(
    target: ProductState,
    predecessors: dict[ProductState, Predecessor],
) -> tuple[str, ...]:
    labels = []
    current = target
    while predecessors[current] is not None:
        previous, label = predecessors[current]
        if label is not None:
            labels.append(label)
        current = previous
    return tuple(reversed(labels))


def assert_model_satisfies_rules(
    model: TaggedPOWL,
    rules: Iterable[AbstractRule],
    *,
    max_states: int = 100_000,
) -> None:
    """Prove that every complete model trace satisfies each supplied rule.

    Convert the model with POWL's existing converter, which expands frequency
    tags. Silent transitions leave the DFA state unchanged. Parallel branches
    are explored with every enabled interleaving. Only the exact final marking
    counts as a complete trace; an unfinished prefix is not a counterexample.

    ``max_states`` bounds distinct reachable (marking, DFA-state) pairs *per
    rule*. Reaching the bound before exhaustion raises
    :class:`RuleDiscoveryLimitError`; a partial search never counts as proof.
    Net successors are cached and shared across the per-rule searches.
    A counterexample raises :class:`RuleConformanceError`, whose ``trace`` is a
    complete, visible trace of the exported model and whose ``rule`` identifies
    the violated constraint. The supplied model and rules are not modified.
    """
    if (
        isinstance(max_states, bool)
        or not isinstance(max_states, int)
        or max_states <= 0
    ):
        raise ValueError("max_states must be a positive integer.")
    supplied = normalize_rules(rules)
    if not supplied:
        return

    net, initial_marking, final_marking = to_petri_net.apply(model)
    alphabet = {
        transition.label
        for transition in net.transitions
        if transition.label is not None
    }

    # Conversion gives visible transitions UUID names. Structural ordering keeps
    # counterexamples reproducible without relying on those random identifiers.
    transitions = sorted(
        net.transitions,
        key=lambda transition: (
            transition.label is not None,
            transition.label or "",
            tuple(sorted((arc.source.name, arc.weight) for arc in transition.in_arcs)),
            tuple(sorted((arc.target.name, arc.weight) for arc in transition.out_arcs)),
        ),
    )
    successors: dict[Marking, tuple[tuple[str | None, Marking], ...]] = {}

    def next_markings(marking: Marking) -> tuple[tuple[str | None, Marking], ...]:
        if marking not in successors:
            found = []
            for transition in transitions:
                # execute returns a fresh Marking, or None when not enabled.
                following = semantics.execute(transition, net, marking)
                if following is not None:
                    found.append((transition.label, following))
            successors[marking] = tuple(found)
        return successors[marking]

    for rule in supplied:
        automaton = rule.to_automaton(alphabet)
        initial = (initial_marking, automaton.initial_state)
        predecessors: dict[ProductState, Predecessor] = {initial: None}
        pending = deque([initial])
        while pending:
            current = pending.popleft()
            marking, state = current
            if marking == final_marking and state not in automaton.final_states:
                raise RuleConformanceError(rule, _witness(current, predecessors))
            for label, following in next_markings(marking):
                next_state = (
                    state if label is None else automaton.transitions[state][label]
                )
                target = (following, next_state)
                if target in predecessors:
                    continue
                if len(predecessors) >= max_states:
                    raise RuleDiscoveryLimitError(
                        f"Model conformance for {rule} exceeded max_states={max_states}; "
                        "the supplied rules have not been proved for this model."
                    )
                predecessors[target] = (current, label)
                pending.append(target)


__all__ = ["assert_model_satisfies_rules"]
