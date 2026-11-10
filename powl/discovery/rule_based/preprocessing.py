"""Rule consistency and logically justified preprocessing; no rule mining."""

from __future__ import annotations

from collections import Counter, defaultdict, deque

from automata.fa.dfa import DFA

from .exceptions import InconsistentRulesError, RuleDiscoveryLimitError
from .rules import AbstractRule, NotCoExistenceRule, NotSuccessionRule, normalize_rules


def product_automaton(rules, alphabet, *, max_states: int = 100_000) -> DFA:
    """Construct only reachable states of the conjunction of the rule DFAs."""
    rules = normalize_rules(rules)
    symbols = tuple(sorted(alphabet))
    automata = tuple(rule.to_automaton(symbols) for rule in rules)
    initial = tuple(automaton.initial_state for automaton in automata)
    states = {initial}
    queue = deque([initial])
    transitions = {}
    final_states = set()
    while queue:
        state = queue.popleft()
        if all(q in automaton.final_states for q, automaton in zip(state, automata)):
            final_states.add(state)
        outgoing = {}
        for activity in symbols:
            target = tuple(
                automaton.transitions[q][activity]
                for q, automaton in zip(state, automata)
            )
            outgoing[activity] = target
            if target not in states:
                if len(states) >= max_states:
                    raise RuleDiscoveryLimitError(
                        f"Rule conjunction exceeded max_states={max_states}."
                    )
                states.add(target)
                queue.append(target)
        transitions[state] = outgoing
    return DFA(
        states=states,
        input_symbols=set(symbols),
        transitions=transitions,
        initial_state=initial,
        final_states=final_states,
    )


def remove_tautologies(rules, alphabet) -> tuple[AbstractRule, ...]:
    """Discard only rules satisfied by *every* trace over this alphabet.

    In particular, missing Existence/Initialization/End targets are retained.
    Disappearance from a repaired log must not erase an outstanding obligation.
    """
    remaining = []
    for rule in normalize_rules(rules):
        # Keep strong references: automata-lib's cached methods use weak
        # references to their owning automaton.
        automaton = rule.to_automaton(alphabet)
        complement = automaton.complement()
        if not complement.isempty():
            remaining.append(rule)
    return tuple(remaining)


def _merge_negative_pairs(rules):
    negatives = {
        (rule.activity_a, rule.activity_b)
        for rule in rules
        if isinstance(rule, NotSuccessionRule)
    }
    merged = []
    for rule in rules:
        if (
            isinstance(rule, NotSuccessionRule)
            and (rule.activity_b, rule.activity_a) in negatives
        ):
            a, b = sorted(rule.args)
            merged.append(NotCoExistenceRule(a, b))
        else:
            merged.append(rule)
    return normalize_rules(merged)


def preprocess(variants, rules, *, simplify: bool = True, max_states: int = 100_000):
    """Check consistency, optionally remove entailed absent activities.

    This preserves the reference miner's negative-pair and forced-absence
    strategy. Forced absence is proved from accepting DFA paths, rather than
    inferred from unsafe assumptions that an initial/final activity occurs only
    once. Activity labels mentioned only by rules remain part of the universe.
    """
    rules = normalize_rules(rules)
    variants = Counter(variants)
    if not rules:
        return variants, rules
    alphabet = {activity for trace in variants for activity in trace}
    alphabet.update(activity for rule in rules for activity in rule.args)
    product = product_automaton(rules, alphabet, max_states=max_states)
    if not product.final_states:  # Product construction includes reachable states only.
        raise InconsistentRulesError("The declarative rules admit no finite trace.")
    if not simplify:
        return variants, rules

    reverse = defaultdict(set)
    for source, transitions in product.transitions.items():
        for target in transitions.values():
            reverse[target].add(source)
    productive = set(product.final_states)
    queue = deque(productive)
    while queue:
        for source in reverse[queue.popleft()]:
            if source not in productive:
                productive.add(source)
                queue.append(source)
    possible = {
        activity
        for source in productive
        for activity, target in product.transitions[source].items()
        if target in productive
    }
    absent = alphabet - possible
    if absent:
        projected = Counter()
        for trace, count in variants.items():
            projected[
                tuple(activity for activity in trace if activity not in absent)
            ] += count
        variants = projected
    # Use the logically possible alphabet, including unobserved but required
    # activities, when deciding whether a rule is now vacuous.
    return variants, remove_tautologies(_merge_negative_pairs(rules), possible)
