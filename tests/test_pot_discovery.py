"""POT discovery must preserve event orders, multiplicities and recursive types."""

from collections import Counter, deque
from itertools import permutations

import pandas as pd
import pytest
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.objects.petri_net import semantics

from powl import POWLDiscoveryVariant
from powl.conversion.variants import to_petri_net
from powl.discovery.total_order_based.algorithm import apply, get_variant
from powl.discovery.total_order_based.inductive.cuts.concurrency import POWLConcurrencyCutPOT
from powl.discovery.total_order_based.inductive.cuts.factory import CutFactory
from powl.discovery.total_order_based.inductive.cuts.loop import POWLLoopCutPOT
from powl.discovery.total_order_based.inductive.cuts.sequence import (
    POWLSequenceCutPOT, POWLStrictSequenceCutPOT,
)
from powl.discovery.total_order_based.inductive.cuts.xor import POWLExclusiveChoiceCutPOT
from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT, PartialOrderTrace,
)
from powl.discovery.total_order_based.inductive.fall_through.activity_concurrent import POWLActivityConcurrentPOT
from powl.discovery.total_order_based.inductive.fall_through.activity_once_per_trace import POWLActivityOncePerTracePOT
from powl.discovery.total_order_based.inductive.fall_through.decision_graph.dfg_fall_through import DFGFallThroughPOT
from powl.discovery.total_order_based.inductive.fall_through.empty_traces import POWLEmptyTracesPOT
from powl.discovery.total_order_based.inductive.fall_through.factory import FallThroughFactory
from powl.discovery.total_order_based.inductive.fall_through.flower import POWLFlowerModelPOT
from powl.discovery.total_order_based.inductive.fall_through.strict_tau_loop import POWLStrictTauLoopPOT
from powl.discovery.total_order_based.inductive.fall_through.tau_loop import POWLTauLoopPOT
from powl.discovery.total_order_based.inductive.modeling import ChoiceGraphSpec, PartialOrderSpec, SequenceSpec
from powl.discovery.total_order_based.inductive.utils.filtering import (
    FILTERING_TYPE, FILTERING_THRESHOLD, FilteringType,
    filter_most_frequent_variants, filter_most_frequent_variants_with_decreasing_factor,
)
from powl.discovery.total_order_based.inductive.variants.brute_force.bf_partial_order_cut import BruteForcePartialOrderCutPOT
from powl.discovery.total_order_based.inductive.variants.decision_graph.cyclic_dg_cut import CyclicDecisionGraphCutPOT
from powl.discovery.total_order_based.inductive.variants.decision_graph.cyclic_dg_cut_strict import StrictCyclicDecisionGraphCutPOT
from powl.discovery.total_order_based.inductive.variants.decision_graph.dfg_cut import DFGCutPOT
from powl.discovery.total_order_based.inductive.variants.decision_graph.max_decision_graph_cut import MaximalDecisionGraphCutPOT
from powl.discovery.total_order_based.inductive.variants.dynamic_clustering.dynamic_clustering_partial_order_cut import DynamicClusteringPartialOrderCutPOT
from powl.discovery.total_order_based.inductive.variants.dynamic_clustering_frequency.dynamic_clustering_frequency_partial_order_cut import (
    DynamicClusteringFrequencyPartialOrderCutPOT, ORDER_FREQUENCY_RATIO,
    compute_efg_frequencies, compute_pot_efg_frequencies,
)
from powl.discovery.total_order_based.inductive.variants.im_dynamic_clustering import POWLInductiveMinerDynamicClustering
from powl.discovery.total_order_based.inductive.variants.maximal.maximal_partial_order_cut import MaximalPartialOrderCutPOT


VARIANTS = [v for v in POWLDiscoveryVariant if v is not POWLDiscoveryVariant.RULE_BASED]


def ordered(*activities):
    return PartialOrderTrace(activities, frozenset(
        (i, j) for i in range(len(activities)) for j in range(i + 1, len(activities))
    ))


def accepts(net_and_markings, word):
    net, initial, final = net_and_markings
    pending = deque([(initial, 0)])
    visited = {(initial, 0)}
    while pending:
        marking, position = pending.popleft()
        if marking == final and position == len(word):
            return True
        for transition in net.transitions:
            next_position = position
            if transition.label is not None:
                if position == len(word) or transition.label != word[position]:
                    continue
                next_position += 1
            following = semantics.execute(transition, net, marking)
            if following is not None and (following, next_position) not in visited:
                visited.add((following, next_position))
                pending.append((following, next_position))
    return False


def linearizations(trace):
    for indices in permutations(range(len(trace))):
        positions = {event: i for i, event in enumerate(indices)}
        if all(positions[i] < positions[j] for i, j in trace.order):
            yield tuple(trace.activities[i] for i in indices)


SAMPLES = [
    Counter(),
    Counter({PartialOrderTrace.empty(): 2}),
    Counter({ordered('A'): 3, PartialOrderTrace.empty(): 2}),
    Counter({ordered('A'): 2, ordered('B'): 3}),
    Counter({ordered('A', 'A', 'A'): 2}),
    Counter({PartialOrderTrace(('A', 'B')): 3}),
    # Index order deliberately differs from the actual event order.
    Counter({PartialOrderTrace(('C', 'B', 'A'), frozenset({(2, 1), (1, 0), (2, 0)})): 2}),
    Counter({ordered('A', 'B', 'A'): 3, ordered('A'): 2}),
    Counter({PartialOrderTrace(('A', 'B', 'C', 'D'), frozenset({
        (0, 1), (0, 2), (0, 3), (1, 3), (2, 3),
    })): 4}),
    Counter({PartialOrderTrace(('A', 'A', 'B'), frozenset({(0, 2)})): 2}),
]


@pytest.mark.parametrize('variant', VARIANTS, ids=lambda v: v.name)
@pytest.mark.parametrize('log', SAMPLES)
def test_every_structural_variant_accepts_all_input_linearizations(variant, log):
    model = get_variant(variant)().apply(IMDataStructurePOT(log))
    exported = to_petri_net.apply(model)
    for trace in log:
        for word in linearizations(trace):
            assert accepts(exported, word), (variant, trace, word, model.pretty())


@pytest.mark.parametrize('variant', VARIANTS, ids=lambda v: v.name)
def test_timestamped_public_algorithm_preserves_concurrency(variant):
    log = pd.DataFrame({
        'case:concept:name': ['1', '1'],
        'concept:name': ['A', 'B'],
        'time:timestamp': pd.to_datetime(['2026-01-01', '2026-01-01']),
    })
    model = apply(log, variant=variant, time_ordering=True)
    exported = to_petri_net.apply(model)
    assert accepts(exported, ('A', 'B'))
    assert accepts(exported, ('B', 'A'))
    assert not accepts(exported, ('A',))
    assert not accepts(exported, ('B',))


@pytest.mark.parametrize('cut', [
    POWLSequenceCutPOT, POWLStrictSequenceCutPOT, POWLConcurrencyCutPOT,
    BruteForcePartialOrderCutPOT, MaximalPartialOrderCutPOT,
    DynamicClusteringPartialOrderCutPOT, DynamicClusteringFrequencyPartialOrderCutPOT,
])
def test_combined_projections_preserve_orders_and_accumulate_frequencies(cut):
    first = PartialOrderTrace(('A', 'B', 'A', 'X'), frozenset({(2, 1), (2, 3)}))
    second = PartialOrderTrace(('A', 'B', 'A', 'Y'), first.order)
    log = Counter({first: 2, second: 3})
    children = cut.project(IMDataStructurePOT(log), [{'A', 'B'}, {'X'}])
    assert all(type(child) is IMDataStructurePOT for child in children)
    assert children[0].data_structure == Counter({
        PartialOrderTrace(('A', 'B', 'A'), frozenset({(2, 1)})): 5,
    })
    assert children[1].data_structure == Counter({ordered('X'): 2, PartialOrderTrace.empty(): 3})
    assert log == Counter({first: 2, second: 3})


def test_xor_projection_routes_each_trace_once_without_empty_branches():
    log = Counter({PartialOrderTrace(('A', 'B')): 3, ordered('C'): 2})
    result = POWLExclusiveChoiceCutPOT.apply(IMDataStructurePOT(log))
    assert result is not None
    children = {frozenset(t.alphabet): child.data_structure
                for child in result[1] for t in child.data_structure}
    assert children[frozenset({'A', 'B'})] == Counter({PartialOrderTrace(('A', 'B')): 3})
    assert children[frozenset({'C'})] == Counter({ordered('C'): 2})


@pytest.mark.parametrize('cut', [
    BruteForcePartialOrderCutPOT, MaximalPartialOrderCutPOT,
    DynamicClusteringPartialOrderCutPOT, DynamicClusteringFrequencyPartialOrderCutPOT,
])
def test_partial_order_detection_uses_pot_efg_instead_of_event_index_order(cut):
    trace = PartialOrderTrace(('B', 'A', 'C'), frozenset({(1, 0), (1, 2)}))
    spec, children = cut.apply(IMDataStructurePOT(Counter({trace: 2})))
    assert isinstance(spec, PartialOrderSpec)
    assert all(type(c) is IMDataStructurePOT for c in children)
    assert spec.edges  # A precedes the concurrent B/C block.


@pytest.mark.parametrize('cut', [
    POWLExclusiveChoiceCutPOT, POWLLoopCutPOT, MaximalDecisionGraphCutPOT,
    CyclicDecisionGraphCutPOT, StrictCyclicDecisionGraphCutPOT, DFGCutPOT,
])
def test_choice_graph_projections_split_repeated_visits(cut):
    log = Counter({ordered('A', 'X', 'A'): 3})
    children = cut.project(IMDataStructurePOT(log), [{'A'}, {'X'}])
    assert children[0].data_structure == Counter({ordered('A'): 6})
    assert children[1].data_structure == Counter({ordered('X'): 3})


def test_maximal_graph_projection_splits_visits_and_omits_empty_traces():
    log = Counter({ordered('A', 'X', 'A'): 3, ordered('X'): 2})
    children = MaximalDecisionGraphCutPOT.project(IMDataStructurePOT(log), [{'A'}, {'X'}])
    assert children[0].data_structure == Counter({ordered('A'): 6})
    assert children[1].data_structure == Counter({ordered('X'): 5})


def test_frequency_clustering_counts_trace_votes_with_concurrency_and_repetition():
    log = Counter({ordered('A', 'A', 'B'): 3, ordered('B', 'A'): 1,
                   PartialOrderTrace(('A', 'B')): 2})
    obj = IMDataStructurePOT(log)
    groups = [('A',), ('B',)]
    frequencies = compute_pot_efg_frequencies(obj, groups)
    assert frequencies[(groups[0], groups[1])] == 5
    assert frequencies[(groups[1], groups[0])] == 3
    spec, _ = DynamicClusteringFrequencyPartialOrderCutPOT.apply(
        obj, {ORDER_FREQUENCY_RATIO: 0.6},
    )
    assert len(spec.edges) == 1


def test_pot_frequency_votes_match_sequential_votes_for_total_orders():
    log = Counter({('A', 'B', 'A'): 3, ('B', 'A'): 2})
    groups = [('A',), ('B',)]
    assert compute_pot_efg_frequencies(
        IMDataStructurePOT(Counter({ordered(*t): f for t, f in log.items()})), groups,
    ) == compute_efg_frequencies(IMDataStructureUVCL(log), groups)


@pytest.mark.parametrize('fallthrough', [POWLStrictTauLoopPOT, POWLTauLoopPOT])
def test_tau_fallthrough_splits_iterations_with_frequencies(fallthrough):
    obj = IMDataStructurePOT(Counter({ordered('A', 'B', 'A', 'B'): 3}))
    _, children = fallthrough.apply(obj)
    assert children[0].data_structure == Counter({ordered('A', 'B'): 6})
    assert children[1].data_structure == Counter()


def test_general_tau_loop_also_splits_at_non_end_to_start_boundaries():
    obj = IMDataStructurePOT(Counter({ordered('A', 'B', 'A', 'C'): 3}))
    assert POWLStrictTauLoopPOT.apply(obj) is None
    _, children = POWLTauLoopPOT.apply(obj)
    assert children[0].data_structure == Counter({ordered('A', 'B'): 3, ordered('A', 'C'): 3})


@pytest.mark.parametrize('fallthrough', [POWLStrictTauLoopPOT, POWLTauLoopPOT])
def test_tau_loops_do_not_split_unordered_events(fallthrough):
    obj = IMDataStructurePOT(Counter({PartialOrderTrace(('A', 'B', 'A', 'B')): 3}))
    assert fallthrough.apply(obj) is None


@pytest.mark.parametrize('cut', [
    POWLExclusiveChoiceCutPOT, POWLLoopCutPOT, MaximalDecisionGraphCutPOT,
    CyclicDecisionGraphCutPOT, StrictCyclicDecisionGraphCutPOT, DFGCutPOT,
])
def test_choice_graph_projections_keep_unordered_events_in_one_fragment(cut):
    trace = PartialOrderTrace(('A', 'X', 'A'))
    children = cut.project(IMDataStructurePOT(Counter({trace: 3})), [{'A'}, {'X'}])
    assert children[0].data_structure == Counter({PartialOrderTrace(('A', 'A')): 3})


@pytest.mark.parametrize('fallthrough', [POWLActivityOncePerTracePOT, POWLActivityConcurrentPOT])
def test_activity_fallthrough_preserves_pot_children(fallthrough):
    obj = IMDataStructurePOT(Counter({ordered('X', 'A', 'B'): 2, ordered('A', 'B', 'X'): 3}))
    _, children = fallthrough.apply(obj)
    assert all(type(c) is IMDataStructurePOT for c in children)
    assert all(sum(c.data_structure.values()) == 5 for c in children)


def test_empty_trace_fallthrough_does_not_mutate_input():
    log = Counter({PartialOrderTrace.empty(): 2, ordered('A'): 3})
    _, children = POWLEmptyTracesPOT.apply(IMDataStructurePOT(log))
    assert children[0].data_structure == Counter()
    assert children[1].data_structure == Counter({ordered('A'): 3})
    assert log[PartialOrderTrace.empty()] == 2


@pytest.mark.parametrize('enable_dfg', [False, True])
def test_pot_fallback_obeys_dfg_setting_and_terminates(enable_dfg):
    obj = IMDataStructurePOT(Counter({ordered('A', 'B', 'A'): 2}))
    fallthroughs = FallThroughFactory.get_fall_throughs(obj, enable_dfg)
    expected = DFGFallThroughPOT if enable_dfg else POWLFlowerModelPOT
    assert fallthroughs[-1] is expected
    spec, children = expected.apply(obj)
    assert all(type(c) is IMDataStructurePOT for c in children)
    if enable_dfg:
        assert isinstance(spec, ChoiceGraphSpec)
        assert all(max(map(len, c.data_structure), default=0) < 3 for c in children)


def test_standard_factory_dispatches_pot_sequence():
    spec, children = CutFactory.find_cut(IMDataStructurePOT(Counter({ordered('A', 'B'): 2})))
    assert isinstance(spec, SequenceSpec)
    assert [c.data_structure for c in children] == [Counter({ordered('A'): 2}), Counter({ordered('B'): 2})]


def test_legacy_dynamic_clustering_also_supports_pot_recursion():
    model = POWLInductiveMinerDynamicClustering().apply(IMDataStructurePOT(SAMPLES[-2]))
    exported = to_petri_net.apply(model)
    assert all(accepts(exported, word) for word in linearizations(next(iter(SAMPLES[-2]))))


def test_variant_filters_keep_pot_type_even_when_all_variants_are_removed():
    log = Counter({ordered('A'): 3, ordered('B'): 3})
    filtered = filter_most_frequent_variants(log)
    assert type(filtered) is IMDataStructurePOT
    assert not filtered.data_structure
    filtered = filter_most_frequent_variants_with_decreasing_factor(log, 0.5)
    assert type(filtered) is IMDataStructurePOT
    assert filtered.data_structure == log


@pytest.mark.parametrize('filtering', list(FilteringType))
def test_recursive_filtering_keeps_pot_data_structures(filtering, monkeypatch):
    miner = get_variant(POWLDiscoveryVariant.TREE)()
    # Force the first iteration through filtering, then allow ordinary cuts.
    calls = []
    original_find_cut = miner.find_cut

    def find_cut(obj, parameters=None):
        calls.append(type(obj))
        return None if len(calls) == 1 else original_find_cut(obj, parameters)

    monkeypatch.setattr(miner, 'find_cut', find_cut)
    obj = IMDataStructurePOT(Counter({ordered('A', 'B'): 10, ordered('B', 'A'): 1}))
    miner.apply(obj, {FILTERING_TYPE: filtering, FILTERING_THRESHOLD: 0.5})
    assert len(calls) >= 2
    assert set(calls) == {IMDataStructurePOT}
