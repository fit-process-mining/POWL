"""POWL's data-based fall-throughs with rule checks and child projections."""

from __future__ import annotations

from collections import Counter

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.rule_based.cuts import (
    RULE_AWARE_CUTS,
    RuleAwareConcurrencyCutUVCL,
    RuleAwareLoopCutUVCL,
)
from powl.discovery.rule_based.dtypes import RuleDecomposition
from powl.discovery.total_order_based.inductive.fall_through.strict_tau_loop import (
    POWLStrictTauLoopUVCL,
)
from powl.discovery.total_order_based.inductive.fall_through.tau_loop import (
    POWLTauLoopUVCL,
)
from powl.discovery.total_order_based.inductive.modeling import PartialOrderSpec

from .utils import alphabet_of, project_alphabet


def _parallel_decomposition(obj, rules, candidate, parameters):
    groups = [{candidate}, alphabet_of(obj) - {candidate}]
    children = RuleAwareConcurrencyCutUVCL.project(obj, groups, parameters)
    projected = RuleAwareConcurrencyCutUVCL.project_rules(rules, groups)
    return RuleDecomposition(PartialOrderSpec(2), tuple(children), tuple(projected))


def activity_once(obj: IMDataStructureUVCL, rules=(), parameters=None):
    """Separate an activity occurring once in every trace, if rules allow it."""
    alphabet = alphabet_of(obj)
    if len(alphabet) <= 1 or not obj.data_structure:
        return None
    candidates = set(alphabet)
    for trace in obj.data_structure:
        counts = Counter(trace)
        candidates.intersection_update(a for a, n in counts.items() if n == 1)
    first_rejection = None
    for candidate in sorted(candidates):
        rejected = RuleAwareConcurrencyCutUVCL.check_rules(
            rules, [{candidate}, alphabet - {candidate}]
        )
        if rejected:
            if first_rejection is None:
                first_rejection = frozenset(rejected)
            continue
        return _parallel_decomposition(obj, rules, candidate, parameters)
    return first_rejection


def activity_concurrent(obj: IMDataStructureUVCL, rules=(), parameters=None):
    """Separate an activity whose removal reveals a structural IM cut."""
    alphabet = alphabet_of(obj)
    if len(alphabet) <= 1:
        return None
    first_rejection = None
    for candidate in sorted(alphabet):
        remainder = project_alphabet(obj, alphabet - {candidate})
        # The reference checks structural applicability in the remainder; its
        # rules will be checked when that child is mined recursively.
        if not any(cut.holds(remainder, parameters) for cut in RULE_AWARE_CUTS):
            continue
        rejected = RuleAwareConcurrencyCutUVCL.check_rules(
            rules, [{candidate}, alphabet - {candidate}]
        )
        if rejected:
            if first_rejection is None:
                first_rejection = frozenset(rejected)
            continue
        return _parallel_decomposition(obj, rules, candidate, parameters)
    return first_rejection


def _tau_loop(base, obj, rules, parameters):
    # The reference uses original start/end activities for tau fall-throughs,
    # even when cut detection has used a noise-filtered DFG.
    unfiltered = IMDataStructureUVCL(obj.data_structure)
    candidate = base.apply(unfiltered, parameters=parameters)
    if candidate is None:
        return None
    groups = [alphabet_of(obj), set()]
    rejected = RuleAwareLoopCutUVCL.check_rules(rules, groups)
    if rejected:
        return frozenset(rejected)
    projected = RuleAwareLoopCutUVCL.project_rules(rules, groups)
    # Preserve the source's rule-progress guard: splitting traces alone is
    # insufficient when exactly the same obligations return to the do child.
    if rules and set(projected[0]) == set(rules):
        return None
    spec, children = candidate
    return RuleDecomposition(spec, tuple(children), tuple(projected))


def strict_tau(obj: IMDataStructureUVCL, rules=(), parameters=None):
    return _tau_loop(POWLStrictTauLoopUVCL, obj, rules, parameters)


def tau(obj: IMDataStructureUVCL, rules=(), parameters=None):
    return _tau_loop(POWLTauLoopUVCL, obj, rules, parameters)


DATA_FALL_THROUGHS = (activity_once, activity_concurrent, strict_tau, tau)
