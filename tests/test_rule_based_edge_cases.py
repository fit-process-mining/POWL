"""Regressions for filtered empty traces, boundary labels, and singleton loops."""

from collections import Counter, deque

import pytest
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.objects.petri_net import semantics

import powl
from powl.discovery.rule_based import ExistenceRule
from powl.discovery.rule_based.fall_through.data import strict_tau, tau
from powl.discovery.rule_based.filtering import filter_data


def accepts_trace(model, trace):
    """Exact Petri-net replay of a fixed word, including silent transitions."""
    net, initial, final = powl.convert_to_petri_net(model)
    pending = deque([(initial, 0)])
    visited = {(initial, 0)}
    while pending:
        marking, position = pending.popleft()
        if marking == final and position == len(trace):
            return True
        for transition in net.transitions:
            if transition.label is not None and (
                position == len(trace) or transition.label != trace[position]
            ):
                continue
            following = semantics.execute(transition, net, marking)
            if following is None:
                continue
            target = (following, position + (transition.label is not None))
            if target not in visited:
                visited.add(target)
                pending.append(target)
    return False


def test_filter_drops_rare_empty_arc_without_changing_projection_data():
    variants = Counter({("A",): 10, (): 1})
    original = variants.copy()
    filtered, epsilon = filter_data(IMDataStructureUVCL(variants), 0.2)
    assert not epsilon
    assert filtered.data_structure == variants == original
    assert filtered.dfg.start_activities == Counter({"A": 10})
    assert filtered.dfg.end_activities == Counter({"A": 10})


def test_filter_keeps_empty_arc_at_exact_relative_threshold():
    filtered, epsilon = filter_data(
        IMDataStructureUVCL(Counter({("A",): 10, (): 2})), 0.2
    )
    assert epsilon
    assert filtered.data_structure[()] == 2


def test_filter_restores_rare_activity_connectivity_but_not_empty_arc():
    variants = Counter({("A", "B"): 10, ("A", "C"): 1, (): 1})
    filtered, epsilon = filter_data(IMDataStructureUVCL(variants), 0.2)
    assert not epsilon
    assert filtered.dfg.graph == Counter({("A", "B"): 10, ("A", "C"): 1})
    assert filtered.dfg.start_activities == Counter({"A": 11})
    assert filtered.dfg.end_activities == Counter({"B": 10, "C": 1})


def test_filter_uses_weighted_start_end_and_internal_frequencies():
    variants = Counter({("A", "B"): 10, ("B", "A"): 1, (): 1})
    filtered, epsilon = filter_data(IMDataStructureUVCL(variants), 0.2)
    assert not epsilon
    assert filtered.dfg.graph == Counter({("A", "B"): 10})
    assert filtered.dfg.start_activities == Counter({"A": 10})
    assert filtered.dfg.end_activities == Counter({"B": 10})
    assert filtered.data_structure == variants


@pytest.mark.parametrize(
    "variants,epsilon", [(Counter(), False), (Counter({(): 10}), True)]
)
def test_filter_accepts_empty_log_and_empty_only_log(variants, epsilon):
    filtered, observed = filter_data(IMDataStructureUVCL(variants), 0.9)
    assert observed is epsilon
    assert not filtered.dfg.graph
    assert not filtered.dfg.start_activities
    assert not filtered.dfg.end_activities
    assert filtered.data_structure == variants


def test_boundary_names_are_valid_activity_labels_during_filtering():
    labels = ("__START__", "__END__", "ArtificialNoneNode")
    variants = Counter({labels: 100, (): 1})
    filtered, epsilon = filter_data(IMDataStructureUVCL(variants), 0.5)
    assert not epsilon
    assert filtered.dfg.graph == Counter(
        {(labels[0], labels[1]): 100, (labels[1], labels[2]): 100}
    )
    assert filtered.dfg.start_activities == Counter({labels[0]: 100})
    assert filtered.dfg.end_activities == Counter({labels[2]: 100})
    model = powl.discover_with_rules(variants, [], noise_threshold=0.5)
    assert accepts_trace(model, labels)
    assert not accepts_trace(model, ())


@pytest.mark.parametrize("epsilon_count,skippable", [(1, False), (2, True)])
def test_miner_observes_filtered_empty_arc_before_its_empty_fallthrough(
    epsilon_count, skippable
):
    model = powl.discover_with_rules(
        Counter({("A",): 10, (): epsilon_count}), [], noise_threshold=0.2
    )
    assert accepts_trace(model, ("A",))
    assert accepts_trace(model, ()) is skippable


def test_filtered_empty_arc_does_not_make_singleton_loop_optional():
    model = powl.discover_with_rules(
        Counter({("A", "A"): 10, (): 1}), [], noise_threshold=0.2
    )
    assert accepts_trace(model, ("A",))
    assert accepts_trace(model, ("A", "A"))
    assert not accepts_trace(model, ())


@pytest.mark.parametrize("fall_through", [strict_tau, tau])
def test_tau_fallthroughs_use_original_trace_boundaries_after_filtering(fall_through):
    raw = IMDataStructureUVCL(Counter({("A", "B", "A"): 10, ("B", "A", "B"): 1}))
    filtered, _ = filter_data(raw, 0.2)
    assert (
        set(filtered.dfg.start_activities) == set(filtered.dfg.end_activities) == {"A"}
    )
    result = fall_through(filtered)
    assert result is not None
    assert result.children[0].data_structure == Counter({("A",): 21, ("B",): 12})


def test_required_singleton_loop_accepts_even_and_odd_positive_repetition():
    model = powl.discover_with_rules([["A", "A"]], {ExistenceRule("A")})
    assert not accepts_trace(model, ())
    for length in (1, 2, 3, 4):
        assert accepts_trace(model, ("A",) * length)
