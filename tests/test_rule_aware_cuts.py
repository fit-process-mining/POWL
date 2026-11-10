"""Contract and source-method regression tests for rule-aware POWL cuts."""

from collections import Counter

import pytest
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.rule_based.cuts import (
    RULE_AWARE_CUTS,
    RuleAwareConcurrencyCutUVCL,
    RuleAwareCut,
    RuleAwareExclusiveChoiceCutUVCL,
    RuleAwareLoopCutUVCL,
    RuleAwareSequenceCutUVCL,
    RuleAwareStrictSequenceCutUVCL,
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
from powl.discovery.total_order_based.inductive.cuts.concurrency import (
    POWLConcurrencyCutUVCL,
)
from powl.discovery.total_order_based.inductive.cuts.loop import POWLLoopCutUVCL
from powl.discovery.total_order_based.inductive.cuts.sequence import (
    POWLStrictSequenceCutUVCL,
)
from powl.discovery.total_order_based.inductive.cuts.xor import (
    POWLExclusiveChoiceCutUVCL,
)
from powl.discovery.total_order_based.inductive.modeling import (
    LoopSpec,
    PartialOrderSpec,
    SequenceSpec,
    XorSpec,
    build_model,
)
from powl.objects import Activity, TaggedPOWL


def uvcl(*traces):
    return IMDataStructureUVCL(Counter(tuple(trace) for trace in traces))


@pytest.mark.parametrize(
    "rule,rejected",
    [
        (AtMostOnceRule("a"), (False, False, False, True)),
        (ExistenceRule("a"), (True, False, False, False)),
        (InitializationRule("a"), (True, False, True, False)),
        (EndRule("a"), (True, True, True, False)),
        (ResponseRule("a", "b"), (True, False, True, True)),
        (ChainResponseRule("a", "b"), (True, False, True, True)),
        (PrecedenceRule("a", "b"), (True, False, True, False)),
        (ChainPrecedenceRule("a", "b"), (True, False, True, False)),
        (RespondedExistenceRule("a", "b"), (True, False, False, True)),
        (CoExistenceRule("a", "b"), (True, False, False, True)),
        (NotCoExistenceRule("a", "b"), (False, True, True, True)),
        (NotSuccessionRule("a", "b"), (False, True, True, True)),
        (ExistenceRule("b"), (True, False, False, True)),
        (InitializationRule("b"), (True, True, True, True)),
        (EndRule("b"), (True, False, True, True)),
        (ResponseRule("b", "a"), (True, True, True, True)),
        (ChainResponseRule("b", "a"), (True, True, True, True)),
        (PrecedenceRule("b", "a"), (True, True, True, True)),
        (ChainPrecedenceRule("b", "a"), (True, True, True, True)),
        (RespondedExistenceRule("b", "a"), (True, False, False, False)),
        (NotSuccessionRule("b", "a"), (False, False, True, True)),
    ],
)
def test_cross_group_rejection_matrix(rule, rejected):
    """Directions and mandatory/optional loop branches must not be swapped."""
    groups = [{"a"}, {"b"}]
    for cut, expected in zip(RULE_AWARE_CUTS, rejected):
        assert cut.check_rules([rule], groups) == ({rule} if expected else set())


@pytest.mark.parametrize(
    "rule", [ChainResponseRule("a", "b"), ChainPrecedenceRule("a", "b")]
)
def test_concurrency_rejects_chain_inside_one_branch(rule):
    assert RuleAwareConcurrencyCutUVCL.check_rules([rule], [{"a", "b"}, {"c"}]) == {
        rule
    }


@pytest.mark.parametrize(
    "rule", [NotSuccessionRule("a", "b"), NotCoExistenceRule("a", "b")]
)
def test_loop_rejects_negative_rule_even_inside_one_branch(rule):
    # One repetition can put otherwise separated activations in conflict.
    assert RuleAwareLoopCutUVCL.check_rules([rule], [{"a", "b"}, {"c"}]) == {rule}


@pytest.mark.parametrize(
    "rule,expected",
    [
        (ResponseRule("a", "b"), ((), (ExistenceRule("b"),))),
        (RespondedExistenceRule("a", "b"), ((), (ExistenceRule("b"),))),
        (PrecedenceRule("a", "b"), ((ExistenceRule("a"),), ())),
        (CoExistenceRule("a", "b"), ((ExistenceRule("a"),), (ExistenceRule("b"),))),
        (
            ChainResponseRule("a", "b"),
            ((AtMostOnceRule("a"), EndRule("a")), (InitializationRule("b"),)),
        ),
        (
            ChainPrecedenceRule("a", "b"),
            ((EndRule("a"),), (AtMostOnceRule("b"), InitializationRule("b"))),
        ),
        (NotSuccessionRule("b", "a"), ((), ())),
        (NotCoExistenceRule("a", "b"), ((), ())),
    ],
)
def test_cross_group_rule_projection(rule, expected):
    assert RuleAwareCut.project_rules([rule], [{"a"}, {"b"}]) == expected


def test_internal_and_missing_positive_obligations_are_retained():
    rules = (ResponseRule("a", "b"), ExistenceRule("missing"))
    assert RuleAwareCut.project_rules(rules, [{"a", "b"}, {"c"}]) == (rules, ())
    assert RuleAwareCut.project_rules([rules[0]], [{"a"}, {"c"}]) == ((rules[0],), ())
    negative = NotCoExistenceRule("a", "b")
    assert RuleAwareCut.project_rules([negative], [{"a"}, {"c"}]) == ((), ())


def test_chain_boundaries_must_be_adjacent():
    groups = [{"a"}, {"x"}, {"b"}]
    rules = [
        ResponseRule("a", "b"),
        PrecedenceRule("a", "b"),
        ChainResponseRule("a", "b"),
        ChainPrecedenceRule("a", "b"),
    ]
    assert RuleAwareSequenceCutUVCL.check_rules(rules, groups) == set(rules[2:])


def test_repeated_group_activities_keep_source_sequence_semantics():
    groups = [{"a"}, {"b"}, {"a"}, {"b"}]
    rules = (ChainResponseRule("a", "b"), ChainPrecedenceRule("a", "b"))
    assert not RuleAwareSequenceCutUVCL.check_rules(rules, groups)
    # Projection deliberately uses the last binary endpoints, as the student does.
    projected = RuleAwareSequenceCutUVCL.project_rules(rules[:1], groups)
    assert projected == (
        (),
        (),
        (AtMostOnceRule("a"), EndRule("a")),
        (InitializationRule("b"),),
    )


@pytest.mark.parametrize(
    "cut,base,log,spec_type",
    [
        (
            RuleAwareExclusiveChoiceCutUVCL,
            POWLExclusiveChoiceCutUVCL,
            [("a",), ("b",)],
            XorSpec,
        ),
        (
            RuleAwareStrictSequenceCutUVCL,
            POWLStrictSequenceCutUVCL,
            [("a", "b")],
            SequenceSpec,
        ),
        (
            RuleAwareConcurrencyCutUVCL,
            POWLConcurrencyCutUVCL,
            [("a", "b"), ("b", "a")],
            PartialOrderSpec,
        ),
        (RuleAwareLoopCutUVCL, POWLLoopCutUVCL, [("a", "b", "a")], LoopSpec),
    ],
)
def test_cut_extensions_build_native_powl(cut, base, log, spec_type):
    assert issubclass(cut, base)
    candidate = cut.evaluate(uvcl(*log))
    assert candidate is not None
    assert not candidate.rejected_rules
    assert isinstance(candidate.spec, spec_type)
    assert candidate.spec.size == len(candidate.children) == 2
    model = build_model(candidate.spec, [Activity(label="a"), Activity(label="b")])
    assert isinstance(model, TaggedPOWL)


def test_rejection_retains_trigger_without_projecting_or_mutating(monkeypatch):
    rule = ExistenceRule("a")
    data = uvcl(["a"], ["b"])
    original = data.data_structure.copy()

    def should_not_project(*args, **kwargs):
        raise AssertionError("rejected candidates must not project")

    monkeypatch.setattr(RuleAwareExclusiveChoiceCutUVCL, "project", should_not_project)
    result = RuleAwareExclusiveChoiceCutUVCL.evaluate(data, [rule])
    assert result.rejected_rules == frozenset([rule])
    assert result.children == result.projected_rules == ()
    assert data.data_structure == original
    assert RuleAwareExclusiveChoiceCutUVCL.apply(data, {"rules": [rule]}) is None


def test_child_rule_context_is_separate_from_shared_parameters():
    rule = ResponseRule("a", "b")
    parameters = {"rules": [rule]}
    result = RuleAwareStrictSequenceCutUVCL.evaluate(
        uvcl(["a", "b"]), parameters["rules"], parameters
    )
    assert result.projected_rules == ((), (ExistenceRule("b"),))
    assert parameters == {"rules": [rule]}


def test_xor_preserves_frequency_tie_break_and_branch_order():
    data = uvcl(["z", "y"], ["a"], ["b"])
    assert RuleAwareExclusiveChoiceCutUVCL.holds(data) == [{"y", "z"}, {"a"}, {"b"}]
    noisy = IMDataStructureUVCL(Counter({("a", "b"): 7, (): 2}))
    children = RuleAwareExclusiveChoiceCutUVCL.project(noisy, [{"a"}, {"b"}])
    assert children[0].data_structure == Counter()
    assert children[1].data_structure == Counter({("b",): 7, (): 2})


def test_sequence_projection_preserves_empty_segments_and_frequencies():
    data = IMDataStructureUVCL(Counter({("a", "b"): 3, ("b", "c"): 2}))
    children = RuleAwareSequenceCutUVCL.project(data, [{"a"}, {"b"}, {"c"}])
    assert [child.data_structure for child in children] == [
        Counter({("a",): 3, (): 2}),
        Counter({("b",): 5}),
        Counter({(): 3, ("c",): 2}),
    ]


def test_loop_combines_redo_components_and_counts_each_segment():
    data = IMDataStructureUVCL(Counter({("a", "b", "a"): 3, ("a", "c", "a"): 2}))
    candidate = RuleAwareLoopCutUVCL.evaluate(data)
    assert candidate.groups == (frozenset("a"), frozenset({"b", "c"}))
    assert [child.data_structure for child in candidate.children] == [
        Counter({("a",): 10}),
        Counter({("b",): 3, ("c",): 2}),
    ]


def test_relaxed_projection_claims_next_occurrence_when_optimal_segment_is_empty():
    data = IMDataStructureUVCL(Counter({("x", "x", "a", "b"): 4}))
    groups = [{"a"}, {"b"}, {"x"}]
    ordinary = RuleAwareSequenceCutUVCL.project(data, groups)
    relaxed = RuleAwareSequenceCutUVCL.relaxed_projection(data, groups)
    assert ordinary[0].data_structure == Counter({(): 4})
    assert [child.data_structure for child in relaxed] == [
        Counter({("a",): 4}),
        Counter({("b",): 4}),
        Counter({(): 4}),
    ]


def test_strict_sequence_rejects_nonprogress_partition(monkeypatch):
    monkeypatch.setattr(
        POWLStrictSequenceCutUVCL,
        "holds",
        classmethod(lambda cls, obj, parameters=None: [{"a", "b"}]),
    )
    assert RuleAwareStrictSequenceCutUVCL.holds(uvcl(["a", "b"])) is None
