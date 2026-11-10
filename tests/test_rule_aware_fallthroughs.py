"""Behavioral regression cases for the reference RBIM fall-through policies."""

from collections import Counter

import networkx as nx
import pytest
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.rule_based.dtypes import RuleDecomposition
from powl.discovery.rule_based.fall_through import (
    activity_concurrent,
    activity_once,
    flower,
    rule_po,
    rule_xor,
    strict_tau,
    tau,
)
from powl.discovery.rule_based.fall_through.po import (
    detect_rule_based_po,
    po_to_parallel_sequence_branches,
)
from powl.discovery.rule_based.fall_through.utils import alphabet_of, dfg_graph
from powl.discovery.rule_based.fall_through.xor import (
    detect_groups,
    normalize_not_succession_pairs,
)
from powl.discovery.rule_based.rules import (
    AtMostOnceRule,
    ChainResponseRule,
    CoExistenceRule,
    EndRule,
    ExistenceRule,
    InitializationRule,
    NotCoExistenceRule,
    NotSuccessionRule,
    PrecedenceRule,
    ResponseRule,
)
from powl.discovery.total_order_based.inductive.modeling import (
    LoopSpec,
    PartialOrderSpec,
)
from powl.objects.tagged_powl import Activity, ChoiceGraph, PartialOrder


def data(counts):
    return IMDataStructureUVCL(Counter(counts))


def test_once_fallthrough_preserves_trace_multiplicities_and_projects_coexistence():
    obj = data({("A", "B"): 7, ("B", "A"): 3})
    result = activity_once(obj, (CoExistenceRule("A", "B"),))
    assert isinstance(result, RuleDecomposition)
    assert result.spec == PartialOrderSpec(2)
    assert result.children[0].data_structure == Counter({("A",): 10})
    assert result.children[1].data_structure == Counter({("B",): 10})
    assert result.projected_rules == ((ExistenceRule("A"),), (ExistenceRule("B"),))


def test_once_fallthrough_reports_first_rule_rejection():
    rule = InitializationRule("A")
    result = activity_once(data({("A", "B"): 2, ("B", "A"): 1}), (rule,))
    assert result == frozenset((rule,))


def test_concurrent_fallthrough_finds_structured_remainder_without_once_requirement():
    obj = data({("A", "B", "A", "C"): 4, ("B", "C"): 2})
    assert activity_once(obj, (ResponseRule("B", "C"),)) is None or isinstance(
        activity_once(obj, (ResponseRule("B", "C"),)), frozenset
    )
    result = activity_concurrent(obj, (ResponseRule("B", "C"),))
    assert isinstance(result, RuleDecomposition)
    assert result.children[0].data_structure == Counter({("A", "A"): 4, (): 2})
    assert result.children[1].data_structure == Counter({("B", "C"): 6})
    assert result.projected_rules == ((), (ResponseRule("B", "C"),))


@pytest.mark.parametrize("fallthrough", (strict_tau, tau))
def test_tau_fallthrough_reuses_weighted_loop_projection(fallthrough):
    obj = data({("A", "B", "A", "B"): 5, ("A", "B"): 2})
    result = fallthrough(obj)
    assert isinstance(result, RuleDecomposition)
    assert result.spec == LoopSpec(2)
    assert result.children[0].data_structure == Counter({("A", "B"): 12})
    assert not result.children[1].data_structure
    assert obj.data_structure == Counter({("A", "B", "A", "B"): 5, ("A", "B"): 2})


@pytest.mark.parametrize("fallthrough", (strict_tau, tau))
def test_tau_fallthrough_rejects_at_most_once_and_keeps_rule_progress_guard(
    fallthrough,
):
    obj = data({("A", "B", "A", "B"): 1})
    rule = AtMostOnceRule("A")
    assert fallthrough(obj, (rule,)) == frozenset((rule,))
    assert fallthrough(obj, (ResponseRule("A", "B"),)) is None


def test_po_fallback_keeps_parallel_sequence_shape():
    obj = data({("A", "B", "C"): 2})
    rules = (ResponseRule("A", "B"),)
    po = detect_rule_based_po(rules, alphabet_of(obj), dfg_graph(obj))
    branches = po_to_parallel_sequence_branches(po, rules)
    assert {tuple(frozenset(group) for group in branch) for branch in branches} == {
        (frozenset(("A",)), frozenset(("B",))),
        (frozenset(("C",)),),
    }
    calls = []

    def mine(child, child_rules):
        calls.append((child.data_structure, child_rules))
        return Activity(label=next(iter(alphabet_of(child))))

    model = rule_po(obj, rules, mine)
    assert isinstance(model, PartialOrder)
    assert len(model.children) == 2
    assert not model.get_edges()
    sequence = next(
        child for child in model.children if isinstance(child, PartialOrder)
    )
    assert {(u.label, v.label) for u, v in sequence.get_edges()} == {("A", "B")}
    assert any(local == (ExistenceRule("B"),) for _, local in calls)


def test_chain_component_is_serialized_against_other_activity_by_dfg_weight():
    obj = data({("C", "A", "B"): 5})
    rules = (ChainResponseRule("A", "B"),)
    po = detect_rule_based_po(rules, alphabet_of(obj), dfg_graph(obj))
    branches = po_to_parallel_sequence_branches(po, rules)
    assert branches == [[{"C"}, {"A"}, {"B"}]]


def test_po_scc_label_splitting_preserves_shared_boundary():
    obj = data({("B", "A", "B"): 1})
    rules = (ResponseRule("A", "B"), PrecedenceRule("B", "A"))
    po = detect_rule_based_po(rules, alphabet_of(obj), dfg_graph(obj))
    assert po_to_parallel_sequence_branches(po, rules) == [[{"B"}, {"A"}, {"B"}]]


def test_initialization_and_end_on_same_label_use_two_boundary_occurrences():
    obj = data({("A", "B", "A"): 1})
    rules = (InitializationRule("A"), EndRule("A"))
    po = detect_rule_based_po(rules, alphabet_of(obj), dfg_graph(obj))
    assert po_to_parallel_sequence_branches(po, rules) == [[{"A"}, {"B"}, {"A"}]]


def test_rule_xor_preserves_reference_distinct_variant_projection():
    obj = data({("A", "B"): 9, ("B", "A"): 3})
    seen = []

    def mine(child, child_rules):
        seen.append((child.data_structure, child_rules))
        return Activity(label=next(iter(alphabet_of(child))))

    model = rule_xor(obj, (NotCoExistenceRule("A", "B"),), mine)
    assert isinstance(model, ChoiceGraph)
    assert {child.label for child in model.children} == {"A", "B"}
    assert len(seen) == 2
    assert {tuple(log.items()) for log, _ in seen} == {
        ((("A",), 1),),
        ((("B",), 1),),
    }
    assert all(not local for _, local in seen)


def test_rule_xor_forced_absence_retains_unrelated_activities():
    obj = data({("C", "A", "B"): 4})
    captured = []

    def mine(child, child_rules):
        captured.append((child.data_structure, child_rules))
        return Activity(label="A")

    rules = (ExistenceRule("A"), NotCoExistenceRule("A", "B"))
    rule_xor(obj, rules, mine)
    assert captured == [(Counter({("C", "A"): 4}), (ExistenceRule("A"),))]


def test_rule_xor_conflicting_component_is_inapplicable_and_pairs_normalize():
    positive, negative = nx.DiGraph(), nx.DiGraph()
    positive.add_edge("A", "B")
    negative.add_edge("A", "B")
    assert detect_groups(positive, negative) is None
    assert normalize_not_succession_pairs(
        (NotSuccessionRule("A", "B"), NotSuccessionRule("B", "A"))
    ) == (
        NotCoExistenceRule("A", "B"),
    )


def test_flower_honors_unary_obligations_with_synthetic_children():
    result = flower(
        data({("A", "B", "C"): 3}),
        (ExistenceRule("A"), AtMostOnceRule("B")),
    )
    assert isinstance(result, RuleDecomposition)
    assert result.spec == PartialOrderSpec(3)
    assert [child.data_structure for child in result.children] == [
        Counter({("A",): 1}),
        Counter({("B",): 1, (): 1}),
        Counter({("C",): 3}),
    ]
    assert result.projected_rules == ((ExistenceRule("A"),), (AtMostOnceRule("B"),), ())


def test_absent_at_most_once_does_not_invent_an_activity_and_empty_flower_is_silent():
    model = flower(data({(): 3}), (AtMostOnceRule("X"),))
    assert isinstance(model, Activity)
    assert model.label is None
    result = flower(data({("A",): 1}), (ExistenceRule("X"),))
    assert isinstance(result, RuleDecomposition)
    assert result.children[0].data_structure == Counter({("X",): 1})


def test_legacy_sentinel_strings_are_real_activities():
    labels = ("ArtificialNoneNode", "__ArtificialStart__", "__ArtificialEnd__")
    obj = data({labels: 2})
    graph = dfg_graph(obj)
    assert set(graph) == set(labels)
    assert graph.nodes[labels[0]]["start"] == 2
    assert graph[labels[0]][labels[1]]["weight"] == 2
