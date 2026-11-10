"""End-to-end contracts for the public rule-aware POWL discovery API."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pandas as pd
import pytest
from pm4py.objects.log.obj import Event, EventLog, Trace
from pm4py.objects.petri_net.obj import PetriNet

import powl
from powl.discovery.rule_based import (
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
from powl.discovery.rule_based.conformance import assert_model_satisfies_rules
from powl.discovery.rule_based.exceptions import (
    InconsistentRulesError,
    RuleConformanceError,
    RuleDiscoveryLimitError,
)
from powl.objects import Activity, TaggedPOWL


def visible_labels(model):
    if isinstance(model, Activity):
        return set() if model.label is None else {model.label}
    return set().union(*(visible_labels(child) for child in model.children))


def frame_from_traces(traces):
    rows = []
    for case, trace in enumerate(traces):
        rows.extend(
            {
                "case:concept:name": str(case),
                "concept:name": activity,
                "time:timestamp": pd.Timestamp("2025-01-01", tz="UTC")
                + pd.Timedelta(index, unit="s"),
            }
            for index, activity in enumerate(trace)
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "rule,traces,labels",
    [
        (ExistenceRule("A"), [["A", "B"], ["A", "C"]], {"A", "B", "C"}),
        (AtMostOnceRule("A"), [["A", "B"], ["A", "C"]], {"A", "B", "C"}),
        (InitializationRule("A"), [["A", "B"], ["A", "C"]], {"A", "B", "C"}),
        (EndRule("B"), [["A", "B"], ["C", "B"]], {"A", "B", "C"}),
        (ResponseRule("A", "B"), [["A", "B"], ["A", "C", "B"]], {"A", "B", "C"}),
        (PrecedenceRule("A", "B"), [["A", "B"], ["A", "C", "B"]], {"A", "B", "C"}),
        (ChainResponseRule("A", "B"), [["A", "B"], ["A", "B", "C"]], {"A", "B", "C"}),
        (ChainPrecedenceRule("A", "B"), [["A", "B"], ["C", "A", "B"]], {"A", "B", "C"}),
        (RespondedExistenceRule("A", "B"), [["A", "B"], ["B", "A"]], {"A", "B"}),
        (CoExistenceRule("A", "B"), [["A", "B"], ["B", "A"]], {"A", "B"}),
        (NotCoExistenceRule("A", "B"), [["A", "C"], ["B", "C"]], {"A", "B", "C"}),
        (NotSuccessionRule("A", "B"), [["B", "A"], ["B", "C", "A"]], {"A", "B", "C"}),
    ],
)
def test_each_template_discovers_a_native_powl_model(rule, traces, labels):
    original = deepcopy(traces)
    model = powl.discover_with_rules(traces, {rule})
    assert isinstance(model, TaggedPOWL)
    assert visible_labels(model) == labels
    assert traces == original
    # Run again on a normalized clone to catch any normalization-induced change.
    assert_model_satisfies_rules(model.normalize(), [rule])


@pytest.mark.parametrize(
    "repair_mode", ["event_level", "trace_level", "edit_distance", "none"]
)
def test_repair_modes_on_noisy_end_activity(repair_mode):
    log = [["A", "B"], ["A", "B", "X"]]
    rule = EndRule("B")
    model = powl.discover_with_rules(log, {rule}, repair_mode=repair_mode)
    assert isinstance(model, TaggedPOWL)
    assert {"A", "B"} <= visible_labels(model)
    assert_model_satisfies_rules(model, [rule])
    assert log == [["A", "B"], ["A", "B", "X"]]


@pytest.mark.parametrize("repair_mode", ["event_level", "edit_distance"])
def test_student_medical_running_example(repair_mode):
    log = [
        ["rg", "ct", "dg", "tr", "dc"],
        ["rg", "mr", "dg", "tr", "dc"],
        ["rg", "dg", "ct", "tr", "dc"],
        ["rg", "ct", "mr", "tr", "dg", "dc"],
    ]
    rules = {
        NotCoExistenceRule("ct", "mr"),
        ResponseRule("mr", "dg"),
        PrecedenceRule("dg", "tr"),
        ChainResponseRule("tr", "dc"),
    }
    model = powl.discover_with_rules(log, rules, repair_mode=repair_mode)
    assert isinstance(model, TaggedPOWL)
    assert {"rg", "dg", "tr", "dc"} <= visible_labels(model)
    assert_model_satisfies_rules(model, rules)


def test_variant_counter_keeps_frequencies_and_input_identity():
    log = Counter({("A", "B"): 1_000_000, ("A", "C", "B"): 2})
    rules = frozenset({ResponseRule("A", "B")})
    original = log.copy()
    model = powl.discover_with_rules(log, rules)
    assert visible_labels(model) == {"A", "B", "C"}
    assert log == original
    assert rules == frozenset({ResponseRule("A", "B")})
    (metrics,) = powl.validate_rules(log, rules)
    assert metrics.total_traces == 1_000_002
    assert metrics.support == metrics.confidence == 1.0


def test_native_powl_roundtrip_and_petri_net_conversion(tmp_path):
    rules = {ResponseRule("A", "B")}
    model = powl.discover_with_rules([["A", "B"], ["A", "C", "B"]], rules)
    destination = tmp_path / "rule_model.powl"
    powl.export(model, destination)
    restored = powl.read_powl_json(destination)
    assert model.same_structure(restored)
    assert_model_satisfies_rules(restored, rules)
    net, initial, final = powl.convert_to_petri_net(restored)
    assert isinstance(net, PetriNet)
    assert initial and final
    assert {transition.label for transition in net.transitions if transition.label} == {
        "A",
        "B",
        "C",
    }


def test_dataframe_custom_keys_stable_ties_and_completion_filter():
    frame = pd.DataFrame(
        {
            "case": ["c", "c", "c", "d", "d"],
            "task": ["B", "Start only", "A", "A", "B"],
            "clock": [2, 0, 1, 4, 4],
            "phase": ["complete", "start", "COMPLETE", "Complete", "complete"],
        }
    )
    original = frame.copy(deep=True)
    model = powl.discover_with_rules(
        frame,
        {ChainResponseRule("A", "B")},
        activity_key="task",
        case_id_key="case",
        timestamp_key="clock",
        lifecycle_key="phase",
        repair_mode="none",
    )
    assert visible_labels(model) == {"A", "B"}
    pd.testing.assert_frame_equal(frame, original)
    unfiltered = powl.discover_with_rules(
        frame,
        [],
        activity_key="task",
        case_id_key="case",
        timestamp_key="clock",
        lifecycle_key="phase",
        keep_only_completion_events=False,
    )
    assert "Start only" in visible_labels(unfiltered)


def test_completion_filter_preserves_log_when_no_completions_exist():
    frame = pd.DataFrame(
        {"case": ["c", "c"], "task": ["A", "B"], "phase": ["start", "start"]}
    )
    model = powl.discover_with_rules(
        frame, [], activity_key="task", case_id_key="case", lifecycle_key="phase"
    )
    assert visible_labels(model) == {"A", "B"}


def test_eventlog_keeps_empty_cases_and_custom_activity_key():
    log = EventLog([Trace(), Trace([Event({"task": "A"}), Event({"task": "B"})])])
    original = deepcopy(log)
    model = powl.discover_with_rules(log, {AtMostOnceRule("A")}, activity_key="task")
    assert isinstance(model, TaggedPOWL)
    assert visible_labels(model) == {"A", "B"}
    assert model.is_skippable()
    assert list(log) == list(original)
    (metrics,) = powl.validate_rules(log, {ExistenceRule("A")}, activity_key="task")
    assert metrics.total_traces == 2
    assert metrics.support == 0.5


def test_empty_log_and_empty_trace_remain_valid_inputs():
    for log in ([], [()], Counter({(): 7})):
        model = powl.discover_with_rules(log, [])
        assert isinstance(model, Activity)
        assert model.is_silent()


def test_public_variant_route_matches_direct_entry_point():
    frame = frame_from_traces([["A", "B"], ["A", "C", "B"]])
    rules = {ResponseRule("A", "B")}
    direct = powl.discover_with_rules(frame, rules, repair_mode="trace_level")
    via_variant = powl.discover(
        frame,
        variant=powl.POWLDiscoveryVariant.RULE_BASED,
        rules=rules,
        rule_parameters={"repair_mode": "trace_level"},
    )
    assert direct.same_structure(via_variant)


def test_rules_cannot_be_silently_ignored_by_another_variant():
    frame = frame_from_traces([["A", "B"]])
    with pytest.raises(ValueError):
        powl.discover(
            frame,
            variant=powl.POWLDiscoveryVariant.TREE,
            rules={ResponseRule("A", "B")},
        )


@pytest.mark.parametrize(
    "rules",
    [
        {InitializationRule("A"), InitializationRule("B")},
        {ExistenceRule("A"), ExistenceRule("B"), NotCoExistenceRule("A", "B")},
        {ExistenceRule("A"), ResponseRule("A", "B"), ResponseRule("B", "A")},
        {
            ExistenceRule("A"),
            ExistenceRule("B"),
            NotSuccessionRule("A", "B"),
            NotSuccessionRule("B", "A"),
        },
    ],
)
def test_inconsistent_rules_fail_before_mining(rules):
    with pytest.raises(InconsistentRulesError):
        powl.discover_with_rules([["A", "B"]], rules)


def test_disabling_rule_simplification_does_not_disable_consistency_check():
    with pytest.raises(InconsistentRulesError):
        powl.discover_with_rules(
            [["A", "B"]],
            {InitializationRule("A"), InitializationRule("B")},
            preprocess_rules=False,
            verify_rules=False,
        )


@pytest.mark.parametrize(
    "options",
    [
        {"repair_mode": "unknown"},
        {"noise_threshold": -0.1},
        {"noise_threshold": 1.0},
        {"max_states": 0},
        {"max_states": True},
        {"max_depth": 0},
    ],
)
def test_invalid_discovery_options_are_rejected(options):
    with pytest.raises((TypeError, ValueError)):
        powl.discover_with_rules([["A", "B"]], [], **options)


@pytest.mark.parametrize(
    "log",
    ["A,B", ["AB"], [[None]], [[""]], Counter({("A",): -1}), Counter({("A",): 0.5})],
)
def test_invalid_logs_are_rejected(log):
    with pytest.raises((TypeError, ValueError)):
        powl.discover_with_rules(log, [])


def test_missing_dataframe_column_and_nonrule_values_are_rejected():
    with pytest.raises(ValueError, match="Column"):
        powl.discover_with_rules(pd.DataFrame({"concept:name": ["A"]}), [])
    with pytest.raises(TypeError):
        powl.discover_with_rules([["A"]], ["Existence(A)"])


def test_explicit_exploration_and_recursion_limits_fail_closed():
    with pytest.raises(RuleDiscoveryLimitError):
        powl.discover_with_rules([["A", "B"]], {ResponseRule("A", "B")}, max_states=1)
    with pytest.raises(RuleDiscoveryLimitError):
        powl.discover_with_rules([["A", "B", "D"], ["A", "C", "D"]], [], max_depth=1)


def test_final_guard_rejects_a_real_unverified_fallback_candidate():
    # With no B observed and repair disabled, the reference fallbacks can still
    # yield behavior containing A. That candidate must not be silently endorsed.
    rule = ResponseRule("A", "B")
    candidate = powl.discover_with_rules(
        [["A"]], {rule}, repair_mode="none", verify_rules=False
    )
    assert isinstance(candidate, TaggedPOWL)
    with pytest.raises(RuleConformanceError) as explicit_check:
        assert_model_satisfies_rules(candidate, [rule])
    assert explicit_check.value.trace == ("A",)
    with pytest.raises(RuleConformanceError) as checked_discovery:
        powl.discover_with_rules([["A"]], {rule}, repair_mode="none")
    assert checked_discovery.value.rule == rule
    assert checked_discovery.value.trace == ("A",)


def test_edit_distance_can_introduce_an_unobserved_required_activity():
    rule = ExistenceRule("B")
    model = powl.discover_with_rules([["A"]], {rule}, repair_mode="edit_distance")
    assert "B" in visible_labels(model)
    assert_model_satisfies_rules(model, [rule])


def test_eventlog_completion_filter_matches_dataframe_policy():
    log = EventLog(
        [
            Trace(),
            Trace([Event({"task": "Start only", "phase": "start"})]),
            Trace(
                [
                    Event({"task": "A", "phase": "start"}),
                    Event({"task": "A", "phase": "complete"}),
                    Event({"task": "B", "phase": "COMPLETE"}),
                ]
            ),
        ]
    )
    model = powl.discover_with_rules(
        log, [], activity_key="task", lifecycle_key="phase"
    )
    assert visible_labels(model) == {"A", "B"}
    assert model.is_skippable()
    assert len(log) == 3 and len(log[2]) == 3


def test_discovery_rejects_custom_templates_but_validation_can_evaluate_them():
    class CustomResponse(ResponseRule):
        pass

    custom = CustomResponse("A", "B")
    (metrics,) = powl.validate_rules([["A", "B"]], [custom])
    assert metrics.support == 1.0
    with pytest.raises(TypeError, match="custom rule templates"):
        powl.discover_with_rules([["A", "B"]], [custom])


def test_low_level_variant_keeps_parameters_and_checks_normalized_result():
    from powl.discovery.total_order_based import algorithm

    log = frame_from_traces([["A", "B"], ["A", "C", "B"]])
    parameters = {
        "rules": [ResponseRule("A", "B")],
        "repair_mode": "trace_level",
        "verify_rules": True,
    }
    original = deepcopy(parameters)
    model = algorithm.apply(
        log,
        variant=powl.POWLDiscoveryVariant.RULE_BASED,
        parameters=parameters,
        time_ordering=False,
    )
    assert isinstance(model, TaggedPOWL)
    assert visible_labels(model) == {"A", "B", "C"}
    assert parameters == original
    assert parameters["verify_rules"] is True
    with pytest.raises(ValueError, match="time_ordering=False"):
        algorithm.apply(log, variant=powl.POWLDiscoveryVariant.RULE_BASED)


def test_low_level_generator_rules_cannot_evade_final_guard():
    from powl.discovery.total_order_based import algorithm

    parameters = {
        "rules": (rule for rule in [ResponseRule("A", "B")]),
        "repair_mode": "none",
    }
    with pytest.raises(RuleConformanceError):
        algorithm.apply(
            frame_from_traces([["A"]]),
            parameters=parameters,
            variant=powl.POWLDiscoveryVariant.RULE_BASED,
            time_ordering=False,
        )
