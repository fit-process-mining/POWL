"""Consistency, preprocessing, and weighted repair behavior for declarative rules."""

from collections import Counter
from itertools import product

import pytest

from powl.discovery.rule_based.exceptions import (
    InconsistentRulesError,
    RuleDiscoveryLimitError,
)
from powl.discovery.rule_based.preprocessing import (
    preprocess,
    product_automaton,
    remove_tautologies,
)
from powl.discovery.rule_based.repair import RepairMode, repair_log, repair_trace
from powl.discovery.rule_based.rules import (
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


def traces(alphabet, maximum_length):
    return [
        trace
        for size in range(maximum_length + 1)
        for trace in product(alphabet, repeat=size)
    ]


def insertion_deletion_distance(a, b):
    """An independent dynamic program; substitution has deletion+insertion cost."""
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        updated = [i]
        for j, right in enumerate(b, 1):
            updated.append(
                min(
                    row[j] + 1,
                    updated[j - 1] + 1,
                    row[j - 1] + (0 if left == right else 2),
                )
            )
        row = updated
    return row[-1]


@pytest.mark.parametrize(
    "rules",
    [
        (),
        (ExistenceRule("a"), AtMostOnceRule("a")),
        (InitializationRule("a"), EndRule("b")),
        (ResponseRule("a", "b"), PrecedenceRule("b", "c")),
        (ChainResponseRule("a", "b"), ChainPrecedenceRule("b", "c")),
        (CoExistenceRule("a", "b"), NotSuccessionRule("a", "c")),
        (RespondedExistenceRule("a", "b"), NotCoExistenceRule("b", "c")),
        (ExistenceRule("a"), ResponseRule("a", "b"), ResponseRule("b", "a")),
    ],
)
def test_reachable_product_accepts_exact_conjunction(rules):
    automaton = product_automaton(rules, "abc")
    for trace in traces("abc", 4):
        assert automaton.accepts_input(trace) == all(
            rule.accepts(trace) for rule in rules
        )


def test_empty_rule_product_accepts_empty_alphabet():
    automaton = product_automaton((), ())
    assert automaton.accepts_input(())
    assert len(automaton.states) == 1


def test_preprocess_preserves_input_and_required_unobserved_activity():
    variants = Counter({("a",): 6, (): 2})
    original = variants.copy()
    rule = ExistenceRule("b")
    projected, remaining = preprocess(variants, [rule])
    assert variants == original == projected
    assert variants is not projected
    assert remaining == (rule,)


@pytest.mark.parametrize(
    "rules,trace",
    [
        ((EndRule("a"), PrecedenceRule("a", "b")), ("a", "b", "a")),
        ((InitializationRule("b"), ResponseRule("a", "b")), ("b", "a", "b")),
        ((EndRule("a"), ChainPrecedenceRule("a", "b")), ("a", "b", "a")),
        ((InitializationRule("b"), ChainResponseRule("a", "b")), ("b", "a", "b")),
    ],
)
def test_boundary_rules_do_not_imply_single_occurrence_or_false_absence(rules, trace):
    assert all(rule.accepts(trace) for rule in rules)
    variants = Counter({trace: 7})
    projected, remaining = preprocess(variants, rules)
    assert projected == variants
    assert set(remaining) == set(rules)


@pytest.mark.parametrize(
    "rule_type", [ResponseRule, PrecedenceRule, ChainResponseRule, ChainPrecedenceRule]
)
def test_dependency_cycles_force_absence_and_accumulate_weights(rule_type):
    rules = [rule_type("a", "b"), rule_type("b", "a")]
    variants = Counter({("a", "b", "c"): 3, ("c",): 5, (): 2})
    projected, remaining = preprocess(variants, rules)
    assert projected == Counter({("c",): 8, (): 2})
    assert remaining == ()
    assert variants[("a", "b", "c")] == 3


def test_forced_absence_propagates_through_other_templates():
    rules = [ResponseRule("a", "b"), ResponseRule("b", "a"), CoExistenceRule("a", "c")]
    projected, remaining = preprocess(Counter({("a", "b", "c", "d"): 4}), rules)
    assert projected == Counter({("d",): 4})
    assert remaining == ()


def test_existence_and_exclusion_remove_only_entailed_absent_activity():
    rules = [ExistenceRule("a"), NotCoExistenceRule("a", "b")]
    projected, remaining = preprocess(Counter({("a", "b", "c"): 4}), rules)
    assert projected == Counter({("a", "c"): 4})
    assert remaining == (ExistenceRule("a"),)


def test_mutual_negative_succession_is_replaced_by_coexclusion():
    rules = [NotSuccessionRule("a", "b"), NotSuccessionRule("b", "a")]
    variants = Counter({("a",): 4, ("b",): 3})
    projected, remaining = preprocess(variants, rules)
    assert projected == variants
    assert remaining == (NotCoExistenceRule("a", "b"),)


@pytest.mark.parametrize("simplify", [True, False])
def test_contradictory_rules_fail_before_discovery(simplify):
    rules = [ExistenceRule("a"), ExistenceRule("b"), NotCoExistenceRule("a", "b")]
    with pytest.raises(InconsistentRulesError):
        preprocess(Counter({("a",): 1}), rules, simplify=simplify)


def test_disabled_simplification_still_retains_all_data_and_rules():
    rules = [ResponseRule("a", "b"), ResponseRule("b", "a")]
    variants = Counter({("a", "b", "c"): 5})
    projected, remaining = preprocess(variants, rules, simplify=False)
    assert projected == variants
    assert set(remaining) == set(rules)


def test_tautology_detection_preserves_missing_required_targets():
    rules = [
        ExistenceRule("missing"),
        InitializationRule("missing"),
        EndRule("missing"),
        AtMostOnceRule("missing"),
        ResponseRule("missing", "a"),
    ]
    assert set(remove_tautologies(rules, {"a"})) == set(rules[:3])


def test_product_state_limit_is_explicit():
    with pytest.raises(RuleDiscoveryLimitError, match="max_states=1"):
        product_automaton([ExistenceRule("a")], {"a"}, max_states=1)


def test_event_repair_aggregates_variants_without_expanding_or_mutating():
    rule = ResponseRule("a", "b")
    variants = Counter({("a", "b"): 5, ("a", "b", "a"): 2})
    original = variants.copy()
    repaired, remaining = repair_log(variants, [rule], [rule], RepairMode.EVENT)
    assert repaired == Counter({("a", "b"): 7})
    assert remaining == (rule,)
    assert variants == original


def test_local_repairs_never_drop_obligations_for_activities_that_can_reappear():
    response, existence = ResponseRule("a", "b"), ExistenceRule("a")
    repaired, remaining = repair_log(
        Counter({("a",): 3}), [response], [response, existence], RepairMode.EVENT
    )
    assert repaired == Counter({(): 3})
    assert set(remaining) == {response, existence}


def test_trace_repair_checks_all_rejected_rules_and_retains_weights():
    rules = [ResponseRule("a", "b"), PrecedenceRule("b", "c")]
    variants = Counter({("a", "b"): 3, ("a",): 4, ("c",): 2, ("b", "c"): 1})
    repaired, remaining = repair_log(variants, rules, rules, RepairMode.TRACE)
    assert repaired == Counter({("a", "b"): 3, ("b", "c"): 1})
    assert set(remaining) == set(rules)


def test_event_repair_does_not_return_all_dropped_traces_as_progress():
    rule = ExistenceRule("a")
    assert repair_log(Counter({("b",): 3}), [rule], [rule], RepairMode.EVENT) is None


def test_trace_repair_can_filter_all_traces_without_erasing_rules():
    rule = ExistenceRule("a")
    repaired, remaining = repair_log(
        Counter({("b",): 3}), [rule], [rule], RepairMode.TRACE
    )
    assert repaired == Counter()
    assert remaining == (rule,)


@pytest.mark.parametrize("mode", list(RepairMode))
def test_satisfied_or_disabled_repair_reports_no_progress(mode):
    rule = ResponseRule("a", "b")
    variants = Counter({("a", "b"): 2})
    assert repair_log(variants, [rule], [rule], mode) is None
    assert repair_log(variants, (), [rule], mode) is None
    assert repair_log(Counter({("a",): 1}), [rule], [rule], RepairMode.NONE) is None


def test_edit_distance_inserts_missing_required_labels_and_preserves_multiplicity():
    rules = [ExistenceRule("a"), ResponseRule("a", "b")]
    variants = Counter({("a",): 3, (): 2})
    repaired, remaining = repair_log(
        variants, [rules[1]], rules, RepairMode.EDIT_DISTANCE
    )
    assert repaired == Counter({("a", "b"): 5})
    assert set(remaining) == set(rules)
    assert variants == Counter({("a",): 3, (): 2})


def test_edit_distance_repairs_against_all_outstanding_rules():
    response = ResponseRule("a", "b")
    rules = [InitializationRule("a"), response, EndRule("c")]
    repaired, remaining = repair_log(
        Counter({("a",): 3}), [response], rules, RepairMode.EDIT_DISTANCE
    )
    assert repaired == Counter({("a", "b", "c"): 3})
    assert set(remaining) == set(rules)


def test_edit_distance_rejects_inconsistent_projected_obligations():
    rules = [ExistenceRule("a"), ExistenceRule("b"), NotCoExistenceRule("a", "b")]
    with pytest.raises(InconsistentRulesError):
        repair_log(Counter({("a",): 1}), rules, rules, RepairMode.EDIT_DISTANCE)


@pytest.mark.parametrize(
    "rules",
    [
        (ExistenceRule("a"),),
        (InitializationRule("a"), EndRule("b")),
        (ResponseRule("a", "b"), AtMostOnceRule("b")),
        (ExistenceRule("a"), ChainResponseRule("a", "b")),
        (PrecedenceRule("a", "b"), NotSuccessionRule("b", "a")),
        (CoExistenceRule("a", "b"), AtMostOnceRule("a"), AtMostOnceRule("b")),
    ],
)
def test_edit_repair_cost_matches_exhaustive_small_language(rules):
    automaton = product_automaton(rules, "ab")
    accepted = [
        candidate
        for candidate in traces("ab", 6)
        if all(rule.accepts(candidate) for rule in rules)
    ]
    for trace in traces("ab", 3):
        result = repair_trace(trace, automaton)
        assert all(rule.accepts(result) for rule in rules)
        optimum = min(
            insertion_deletion_distance(trace, candidate) for candidate in accepted
        )
        assert insertion_deletion_distance(trace, result) == optimum


def test_edit_search_state_limit_is_explicit():
    automaton = product_automaton([ExistenceRule("a")], {"a"})
    with pytest.raises(RuleDiscoveryLimitError, match="max_states=1"):
        repair_trace((), automaton, max_states=1)


def test_edit_repair_handles_unknown_input_label_by_deletion():
    automaton = product_automaton([ExistenceRule("a")], {"a"})
    assert repair_trace(("unknown",), automaton) == ("a",)
