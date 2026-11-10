"""The miner's language guard must reject violating *complete* model traces."""

from unittest.mock import patch

import pytest

from powl.discovery.rule_based.conformance import assert_model_satisfies_rules
from powl.discovery.rule_based.exceptions import (
    RuleConformanceError,
    RuleDiscoveryLimitError,
)
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
)
from powl.objects import Activity, PartialOrder, loop, sequence, silent_activity, xor


def test_sequence_satisfies_all_applicable_templates():
    model = sequence([Activity("A"), Activity("B")])
    rules = [
        ExistenceRule("A"),
        AtMostOnceRule("A"),
        InitializationRule("A"),
        EndRule("B"),
        ResponseRule("A", "B"),
        PrecedenceRule("A", "B"),
        ChainResponseRule("A", "B"),
        ChainPrecedenceRule("A", "B"),
        RespondedExistenceRule("A", "B"),
        CoExistenceRule("A", "B"),
        NotSuccessionRule("B", "A"),
        NotCoExistenceRule("A", "missing"),
    ]
    before = model.to_dict()
    assert assert_model_satisfies_rules(model, rules) is None
    assert model.to_dict() == before


def test_incomplete_prefix_is_not_a_counterexample():
    # Response is false after the A prefix, but B is mandatory before completion.
    assert_model_satisfies_rules(
        sequence([Activity("A"), Activity("B")]), [ResponseRule("A", "B")]
    )


def test_parallel_interleaving_produces_complete_violation_witness():
    model = PartialOrder(nodes=[Activity("A"), Activity("B")])
    rule = ResponseRule("A", "B")
    with pytest.raises(RuleConformanceError) as error:
        assert_model_satisfies_rules(model, [rule])
    assert error.value.rule == rule
    assert error.value.trace == ("B", "A")
    assert not rule.accepts(error.value.trace)


def test_chain_rule_checks_unrelated_interleavings():
    parallel = PartialOrder(nodes=[Activity("A"), Activity("C")])
    model = sequence([parallel, Activity("B")])
    assert_model_satisfies_rules(model, [ResponseRule("A", "B")])
    with pytest.raises(RuleConformanceError) as error:
        assert_model_satisfies_rules(model, [ChainResponseRule("A", "B")])
    assert error.value.trace == ("A", "C", "B")


def test_xor_respects_noncoexistence_and_detects_required_absence():
    model = xor([Activity("A"), Activity("B")])
    assert_model_satisfies_rules(model, [NotCoExistenceRule("A", "B")])
    with pytest.raises(RuleConformanceError) as error:
        assert_model_satisfies_rules(model, [ExistenceRule("A")])
    assert error.value.trace == ("B",)


def test_loop_exploration_has_no_trace_length_cutoff():
    body = sequence([Activity("A"), Activity("B")])
    model = loop(body, silent_activity())
    assert_model_satisfies_rules(
        model,
        [ResponseRule("A", "B"), ChainResponseRule("A", "B"), PrecedenceRule("A", "B")],
    )
    with pytest.raises(RuleConformanceError) as error:
        assert_model_satisfies_rules(model, [AtMostOnceRule("A")])
    assert error.value.trace == ("A", "B", "A", "B")


def test_normalized_frequency_tags_are_checked():
    repeated = loop(Activity("A"), silent_activity()).normalize()
    assert repeated.max_freq is None
    assert_model_satisfies_rules(repeated, [ExistenceRule("A")])
    with pytest.raises(RuleConformanceError) as error:
        assert_model_satisfies_rules(repeated, [AtMostOnceRule("A")])
    assert error.value.trace == ("A", "A")
    optional = xor([Activity("A"), silent_activity()]).normalize()
    assert optional.min_freq == 0
    with pytest.raises(RuleConformanceError) as error:
        assert_model_satisfies_rules(optional, [ExistenceRule("A")])
    assert error.value.trace == ()


def test_silent_model_and_silent_cycles():
    tau = silent_activity()
    assert_model_satisfies_rules(
        tau, [AtMostOnceRule("A"), ResponseRule("A", "B"), CoExistenceRule("A", "B")]
    )
    with pytest.raises(RuleConformanceError) as error:
        assert_model_satisfies_rules(tau, [InitializationRule("A")])
    assert error.value.trace == ()
    silent_loop = loop(silent_activity(), silent_activity())
    assert_model_satisfies_rules(silent_loop, [AtMostOnceRule("A")])


def test_unconstrained_flower_cannot_escape_rule_enforcement():
    flower = loop(silent_activity(), xor([Activity("A"), Activity("B")]))
    with pytest.raises(RuleConformanceError) as error:
        assert_model_satisfies_rules(flower, [ResponseRule("A", "B")])
    assert error.value.trace == ("A",)


def test_multicharacter_activity_labels_are_atomic_symbols():
    model = sequence([Activity("Receive order"), Activity("Ship order")])
    assert_model_satisfies_rules(
        model, [ChainResponseRule("Receive order", "Ship order")]
    )


def test_bound_failure_never_becomes_a_false_proof():
    model = sequence([Activity("A"), Activity("B")])
    with pytest.raises(RuleDiscoveryLimitError, match="not been proved"):
        assert_model_satisfies_rules(model, [ResponseRule("A", "B")], max_states=1)


@pytest.mark.parametrize("bound", [0, -1, 0.5, True, None])
def test_invalid_bound_rejected_even_without_rules(bound):
    with pytest.raises(ValueError, match="positive integer"):
        assert_model_satisfies_rules(silent_activity(), [], max_states=bound)


def test_no_rules_avoids_conversion():
    with patch(
        "powl.discovery.rule_based.conformance.to_petri_net.apply",
        side_effect=AssertionError("unnecessary conversion"),
    ):
        assert_model_satisfies_rules(Activity("A"), [])
