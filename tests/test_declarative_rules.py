"""Template semantics, immutable inputs, repairs, and empirical validation."""

from collections import Counter
from dataclasses import FrozenInstanceError
from itertools import product

import pytest

from powl.discovery.rule_based.rules import (
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
from powl.discovery.rule_based.validation import validate_rules

RULES = (
    AtMostOnceRule("A"),
    ChainPrecedenceRule("A", "B"),
    ChainResponseRule("A", "B"),
    CoExistenceRule("A", "B"),
    EndRule("A"),
    ExistenceRule("A"),
    InitializationRule("A"),
    NotCoExistenceRule("A", "B"),
    NotSuccessionRule("A", "B"),
    PrecedenceRule("A", "B"),
    RespondedExistenceRule("A", "B"),
    ResponseRule("A", "B"),
)


def traces(alphabet="ABC", max_length=4):
    for length in range(max_length + 1):
        yield from product(alphabet, repeat=length)


@pytest.mark.parametrize("rule", RULES)
def test_predicate_and_automaton_agree_on_every_short_trace(rule):
    automaton = rule.to_automaton({"A", "B", "C"})
    for trace in traces():
        assert rule.accepts(trace) == automaton.accepts_input(trace), (rule, trace)


@pytest.mark.parametrize("rule", RULES)
def test_automata_support_projected_alphabets(rule):
    for alphabet in (set(), {"A"}, {"B"}, {"C"}):
        automaton = rule.to_automaton(alphabet)
        for trace in traces(sorted(alphabet), 3):
            assert rule.accepts(trace) == automaton.accepts_input(trace), (rule, trace)


@pytest.mark.parametrize("rule", RULES)
def test_event_repair_produces_only_satisfying_subsequences(rule):
    for trace in traces():
        repaired = rule.repair_trace(trace)
        if repaired is not None:
            assert rule.accepts(repaired), (rule, trace, repaired)
            remaining = iter(trace)
            assert all(activity in remaining for activity in repaired)


@pytest.mark.parametrize(
    "rule,log,expected",
    [
        (AtMostOnceRule("A"), [["A", "B", "A", "A"]], [["A", "B"]]),
        (
            InitializationRule("A"),
            [[], ["C", "F", "G", "A", "B", "C"]],
            [["A", "B", "C"]],
        ),
        (EndRule("Z"), [["R"], ["A", "B", "Z", "Q"]], [["A", "B", "Z"]]),
        (ExistenceRule("A"), [["Q"], ["Z", "D", "A"]], [["Z", "D", "A"]]),
        (ChainPrecedenceRule("A", "B"), [["A", "B", "C", "B"]], [["A", "B", "C"]]),
        (PrecedenceRule("A", "B"), [["B", "A", "B", "C", "B"]], [["A", "B", "C", "B"]]),
        (
            ChainResponseRule("A", "B"),
            [["A", "B", "A", "D", "B"], []],
            [["A", "B", "D", "B"], []],
        ),
        (
            ResponseRule("A", "B"),
            [["A", "B", "A", "D", "B", "A"], []],
            [["A", "B", "A", "D", "B"], []],
        ),
        (RespondedExistenceRule("A", "B"), [["A", "C", "D", "C"]], [["C", "D", "C"]]),
        (
            CoExistenceRule("Z", "B"),
            [["Q", "N", "A"], ["Z", "P"]],
            [["Q", "N", "A"], ["P"]],
        ),
        (
            NotSuccessionRule("A", "B"),
            [["T", "R", "A", "L"], ["Q", "B", "A", "B"], []],
            [["T", "R", "A", "L"], ["Q", "B", "B"], []],
        ),
        (
            NotCoExistenceRule("A", "B"),
            [["Q", "B", "A", "C"], ["A", "B"], []],
            [["Q", "B", "C"], ["A"], []],
        ),
    ],
)
def test_student_repair_examples_are_preserved(rule, log, expected):
    copied = [trace[:] for trace in log]
    assert rule.repair(log) == expected
    assert log == copied


def test_response_checks_every_activation_and_coexistence_includes_vacuity():
    assert not ResponseRule("A", "B").accepts(("A", "B", "A"))
    assert ResponseRule("A", "B").accepts(("A", "A", "B"))
    assert CoExistenceRule("A", "B").accepts(())
    assert CoExistenceRule("A", "B").accepts(("C",))
    assert not CoExistenceRule("A", "B").accepts(("A",))


def test_rules_are_immutable_and_keep_semantic_identity():
    rule = ResponseRule("A", "B")
    assert isinstance(rule, AbstractRule)
    assert rule.args == ("A", "B")
    assert rule.activities == frozenset({"A", "B"})
    assert rule.name == "Response"
    assert str(rule) == "Response(A, B)"
    assert len({rule, ResponseRule("A", "B")}) == 1
    assert rule != ChainResponseRule("A", "B")
    assert isinstance(ChainResponseRule("A", "B"), ResponseRule)
    assert isinstance(ChainPrecedenceRule("A", "B"), PrecedenceRule)
    with pytest.raises(FrozenInstanceError):
        rule.activity_a = "C"
    with pytest.raises(TypeError):
        ExistenceRule(None)
    with pytest.raises(ValueError):
        ResponseRule("A", "A")
    with pytest.raises(TypeError):
        normalize_rules(["Response(A, B)"])


def test_filtering_and_repair_do_not_alias_input_traces():
    log = [["A", "B"], ["C"]]
    for output in (
        ResponseRule("A", "B").apply(log),
        ResponseRule("A", "B").repair(log),
    ):
        output[0].append("D")
    assert log == [["A", "B"], ["C"]]


def test_metrics_weight_trace_variants_and_count_empty_traces():
    rule = ResponseRule("A", "B")
    log = Counter({("A", "B"): 2, ("A", "B", "A"): 1, (): 3})
    (metrics,) = validate_rules(log, {rule})
    assert metrics.total_traces == 6
    assert metrics.satisfied_traces == 5
    assert metrics.violated_traces == 1
    assert metrics.activated_traces == 3
    assert metrics.satisfied_activated_traces == 2
    assert metrics.vacuously_satisfied_traces == 3
    assert metrics.support == pytest.approx(5 / 6)
    assert metrics.confidence == pytest.approx(2 / 3)
    assert metrics.violation_rate == pytest.approx(1 / 6)
    assert log == Counter({("A", "B"): 2, ("A", "B", "A"): 1, (): 3})


def test_negative_and_symmetric_confidence_uses_activated_trace_denominator():
    log = [[], ["A"], ["B"], ["A", "B"]]
    results = {
        result.rule: result
        for result in validate_rules(
            log,
            [
                CoExistenceRule("A", "B"),
                NotCoExistenceRule("A", "B"),
                NotSuccessionRule("A", "B"),
            ],
        )
    }
    coexist = results[CoExistenceRule("A", "B")]
    assert coexist.support == 0.5
    assert coexist.confidence == pytest.approx(1 / 3)
    not_coexist = results[NotCoExistenceRule("A", "B")]
    assert not_coexist.support == 0.75
    assert not_coexist.confidence == pytest.approx(2 / 3)
    not_succession = results[NotSuccessionRule("A", "B")]
    assert not_succession.support == 0.75
    assert not_succession.confidence == 0.5


def test_unary_metrics_and_zero_activation_conventions():
    (unary,) = validate_rules([[], ["A"]], [ExistenceRule("A")])
    assert unary.support == unary.confidence == 0.5
    assert unary.activated_traces == 2
    assert unary.vacuously_satisfied_traces == 0
    (absent,) = validate_rules([[], ["C"]], [ResponseRule("A", "B")])
    assert absent.support == absent.confidence == 1.0
    assert absent.activated_traces == 0
    assert absent.vacuously_satisfied_traces == 2
    assert validate_rules([], [ExistenceRule("A")])[0].confidence == 0.0
    empty = validate_rules([], [ResponseRule("A", "B")])[0]
    assert empty.support == empty.violation_rate == 0.0
    assert empty.confidence == 1.0
    assert empty.total_traces == 0


def test_validation_handles_duplicate_rules_without_mutation():
    rule = ResponseRule("A", "B")
    first = validate_rules([["A", "B"]], [rule, rule])
    second = validate_rules([["A"]], {rule})
    assert len(first) == len(second) == 1
    assert first[0].support == 1.0
    assert second[0].support == 0.0
    assert first[0].support == 1.0
    assert not hasattr(rule, "sup")


def test_validation_dataframe_keys_and_eventlog_empty_cases():
    import pandas as pd
    from pm4py.objects.log.obj import Event, EventLog, Trace

    frame = pd.DataFrame(
        {"case": ["case1", "case1"], "task": ["B", "A"], "clock": [2, 1]}
    )
    rule = ChainResponseRule("A", "B")
    (result,) = validate_rules(
        frame, [rule], activity_key="task", case_id_key="case", timestamp_key="clock"
    )
    assert result.support == result.confidence == 1.0
    assert frame["task"].tolist() == ["B", "A"]
    log = EventLog([Trace([Event({"task": "A"}), Event({"task": "B"})]), Trace()])
    (result,) = validate_rules(log, [ExistenceRule("A")], activity_key="task")
    assert result.total_traces == 2
    assert result.support == result.confidence == 0.5


def test_validation_handles_large_variant_multiplicities_without_expansion():
    (result,) = validate_rules(
        Counter({("A", "B"): 1_000_000_000, ("A",): 1}),
        [ResponseRule("A", "B")],
    )
    assert result.total_traces == 1_000_000_001
    assert result.satisfied_traces == 1_000_000_000
    assert result.violated_traces == 1
