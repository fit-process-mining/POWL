"""Local repairs used after all data-driven decompositions have been rejected."""

from __future__ import annotations

from collections import Counter
from enum import Enum
from heapq import heappop, heappush
from itertools import count

from .exceptions import InconsistentRulesError, RuleDiscoveryLimitError
from .preprocessing import product_automaton
from .rules import normalize_rules


class RepairMode(str, Enum):
    EVENT = "event_level"
    TRACE = "trace_level"
    EDIT_DISTANCE = "edit_distance"
    NONE = "none"


def repair_trace(trace, automaton, *, max_states: int = 100_000):
    """Minimum insertion/deletion repair against a rule-product DFA.

    Keeping a matching event costs zero; inserting or deleting costs one.
    A substitution therefore costs two, as in the reference method. Search is
    lazy, with stable tie-breaking, rather than materializing the edit graph.
    """
    initial = (0, automaton.initial_state)
    distances = {initial: 0}
    parents = {}
    serial = count()
    queue = [(0, next(serial), initial)]
    symbols = sorted(automaton.input_symbols)
    while queue:
        cost, _, state = heappop(queue)
        if cost != distances[state]:
            continue
        position, q = state
        if position == len(trace) and q in automaton.final_states:
            result = []
            cursor = state
            while cursor != initial:
                cursor, output = parents[cursor]
                if output is not None:
                    result.append(output)
            return tuple(reversed(result))
        successors = []
        if position < len(trace):
            activity = trace[position]
            target = automaton.transitions[q].get(activity)
            if target is not None:
                successors.append(((position + 1, target), activity, 0))
            successors.append(((position + 1, q), None, 1))
        for activity in symbols:
            target = automaton.transitions[q][activity]
            if target != q:  # Positive-cost self-insertions cannot improve a path.
                successors.append(((position, target), activity, 1))
        for target, output, extra in successors:
            new_cost = cost + extra
            if new_cost >= distances.get(target, float("inf")):
                continue
            if target not in distances and len(distances) >= max_states:
                raise RuleDiscoveryLimitError(
                    f"Edit-distance repair exceeded max_states={max_states}."
                )
            distances[target] = new_cost
            parents[target] = (state, output)
            heappush(queue, (new_cost, next(serial), target))
    raise InconsistentRulesError(
        "No accepting trace is reachable during edit-distance repair."
    )


def repair_log(
    variants, rejected_rules, rules, mode=RepairMode.EVENT, *, max_states=100_000
):
    """Return a changed (variant Counter, rules) pair, or None if no progress.

    Only the first rejected decomposition triggers a repair. Event and trace
    repair use its rejected rules; edit-distance repair uses their conjunction
    with every outstanding rule. Input multiplicities are preserved exactly.
    """
    mode = RepairMode(mode)
    rules = normalize_rules(rules)
    rejected_rules = normalize_rules(rejected_rules)
    if mode is RepairMode.NONE or not rejected_rules:
        return None
    repaired = Counter(variants)
    if mode is RepairMode.EVENT:
        for rule in rejected_rules:
            result = Counter()
            for trace, frequency in repaired.items():
                fixed = rule.repair_trace(trace)
                if fixed is not None:
                    result[fixed] += frequency
            if not result:
                return None
            repaired = result
    elif mode is RepairMode.TRACE:
        repaired = Counter(
            {
                trace: frequency
                for trace, frequency in repaired.items()
                if all(rule.accepts(trace) for rule in rejected_rules)
            }
        )
    else:
        alphabet = {activity for trace in variants for activity in trace}
        alphabet.update(activity for rule in rules for activity in rule.args)
        product = product_automaton(rules, alphabet, max_states=max_states)
        if not product.final_states:
            raise InconsistentRulesError(
                "Projected declarative obligations are inconsistent."
            )
        repaired = Counter()
        for trace, frequency in variants.items():
            fixed = (
                trace
                if product.accepts_input(trace)
                else repair_trace(trace, product, max_states=max_states)
            )
            repaired[fixed] += frequency
    # A later rule-based fallback or insertion repair may reintroduce an
    # activity removed here. Only logical preprocessing can prove absence;
    # disappearance from this particular log cannot discharge an obligation.
    remaining = rules
    if repaired == variants and remaining == rules:
        return None
    return repaired, remaining
