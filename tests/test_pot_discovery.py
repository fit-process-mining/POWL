"""POT discovery must preserve event orders, multiplicities and recursive types."""

from collections import Counter, deque
from datetime import timedelta
from itertools import combinations, groupby, permutations
from random import Random

import pandas as pd
import pytest
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.objects.petri_net import semantics

from powl import POWLDiscoveryVariant
from powl.conversion.variants import to_petri_net
from powl.discovery.total_order_based.algorithm import DEFAULT_POWL_MINER, apply, get_variant
from powl.discovery.total_order_based.inductive.cuts.concurrency import POWLConcurrencyCutPOT
from powl.discovery.total_order_based.inductive.cuts.factory import CutFactory
from powl.discovery.total_order_based.inductive.cuts.loop import POWLLoopCutPOT
from powl.discovery.total_order_based.inductive.cuts.sequence import (
    POWLSequenceCutPOT, POWLStrictSequenceCutPOT,
)
from powl.discovery.total_order_based.inductive.cuts.xor import POWLExclusiveChoiceCutPOT
from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT, PartialOrderTrace, discover_dfg_efg_pot,
)
from powl.discovery.total_order_based.inductive.fall_through.activity_concurrent import (
    POWLActivityConcurrentPOT, _candidate_dfgs,
)
from powl.discovery.total_order_based.inductive.fall_through.activity_once_per_trace import POWLActivityOncePerTracePOT
from powl.discovery.total_order_based.inductive.fall_through.decision_graph.dfg_fall_through import DFGFallThroughPOT
from powl.discovery.total_order_based.inductive.fall_through.empty_traces import POWLEmptyTracesPOT
from powl.discovery.total_order_based.inductive.fall_through.factory import FallThroughFactory
from powl.discovery.total_order_based.inductive.fall_through.flower import POWLFlowerModelPOT
from powl.discovery.total_order_based.inductive.fall_through.strict_tau_loop import (
    POWLStrictTauLoopPOT, POWLStrictTauLoopUVCL,
)
from powl.discovery.total_order_based.inductive.fall_through.tau_loop import POWLTauLoopPOT, POWLTauLoopUVCL
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


def expand(log):
    words = Counter()
    for trace, frequency in log.items():
        for word in linearizations(trace):
            words[word] += frequency
    return words


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


def test_strict_tau_boundary_across_enabled_layers():
    # B can wait until after C even though A and B are enabled first.
    trace = PartialOrderTrace(('A', 'B', 'C'), frozenset({(0, 2)}))
    obj = IMDataStructurePOT(Counter({trace: 1}))
    _, (body, _) = POWLStrictTauLoopPOT.apply(obj)
    assert set(expand(body.data_structure)) == {('A', 'B', 'C'), ('A', 'C'), ('B',)}


@pytest.mark.parametrize('fallthrough', [POWLStrictTauLoopPOT, POWLTauLoopPOT])
def test_tau_loops_split_every_event_when_all_adjacencies_are_boundaries(fallthrough):
    obj = IMDataStructurePOT(Counter({PartialOrderTrace(('A', 'B', 'A', 'B')): 3}))
    _, children = fallthrough.apply(obj)
    assert children[0].data_structure == Counter({ordered('A'): 6, ordered('B'): 6})


@pytest.mark.parametrize('cut', [
    POWLExclusiveChoiceCutPOT, POWLLoopCutPOT, MaximalDecisionGraphCutPOT,
    CyclicDecisionGraphCutPOT, StrictCyclicDecisionGraphCutPOT, DFGCutPOT,
])
def test_split_projections_include_separate_and_contiguous_visits(cut):
    trace = PartialOrderTrace(('A', 'X', 'A'))
    children = cut.project(IMDataStructurePOT(Counter({trace: 3})), [{'A'}, {'X'}])
    assert set(expand(children[0].data_structure)) == {('A',), ('A', 'A')}
    assert set(expand(children[1].data_structure)) == {('X',)}


def test_ordinary_loop_cut_matches_visits_when_redo_can_interleave_with_do():
    # A (A || S) R A has two extensions: AASRA visits SR together, while
    # ASARA visits S and R separately. The original loop cut lost both singletons.
    ranks = (0, 1, 1, 2, 3)
    trace = PartialOrderTrace(tuple('AASRA'), frozenset(
        (i, j) for i in range(5) for j in range(5) if ranks[i] < ranks[j]
    ))
    log = Counter({trace: 3})
    obj = IMDataStructurePOT(log)
    miner = get_variant(DEFAULT_POWL_MINER)()
    _, (body, redo) = miner.find_cut(obj)
    assert set(expand(body.data_structure)) == {('A',), ('A', 'A')}
    assert set(expand(redo.data_structure)) == {('R',), ('S',), ('S', 'R')}
    assert miner.apply(obj).normalize().same_structure(
        miner.apply(IMDataStructureUVCL(expand(log))).normalize()
    )


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


def random_pot(rng, size):
    # Generate a closed DAG with labels independent of event identity and rank.
    indices = rng.sample(range(size), size)
    edges = set()
    for j, event in enumerate(indices):
        for earlier in indices[:j]:
            if rng.random() < 0.35:
                edges.update((i, event) for i in indices[:j] if (i, earlier) in edges)
                edges.add((earlier, event))
    return PartialOrderTrace(tuple(rng.choices('ABCD', k=size)), frozenset(edges))


def test_default_miner_matches_full_linearization():
    rng = Random(327)
    logs = SAMPLES + [Counter({PartialOrderTrace(('A', 'A')): 2})]
    for _ in range(50):
        logs.append(Counter({random_pot(rng, rng.randrange(1, 7)): rng.randrange(1, 5)
                             for _ in range(rng.randrange(1, 4))}))
    miner = get_variant(DEFAULT_POWL_MINER)()
    for log in logs:
        actual = miner.apply(IMDataStructurePOT(log))
        reference = miner.apply(IMDataStructureUVCL(expand(log)))
        assert actual.same_structure(reference), log
        assert actual.normalize().same_structure(reference.normalize()), log


def test_projected_languages_match_sequential_projections_on_general_posets():
    rng = Random(831)
    for _ in range(60):
        log = Counter({random_pot(rng, rng.randrange(2, 7)): 3, ordered('A', 'C'): 2})
        obj = IMDataStructurePOT(log)
        words = expand(log)
        for pot, uvcl in [(POWLStrictTauLoopPOT, POWLStrictTauLoopUVCL),
                          (POWLTauLoopPOT, POWLTauLoopUVCL)]:
            projected = pot._get_projected_log(obj)
            expected = uvcl._get_projected_log(words)
            assert set(expand(projected.data_structure)) == set(expected), log
            assert pot.holds(obj) == (sum(expected.values()) > sum(words.values()))
        child = POWLLoopCutPOT.project(obj, [{'A', 'B'}, {'C', 'D'}])[0]
        visits = {tuple(run) for word in words for keep, run in groupby(word, key='AB'.__contains__) if keep}
        assert set(expand(child.data_structure)) == visits, log


def test_unit_artifact_weights_count_each_possible_event_pair_once():
    layered = PartialOrderTrace(tuple('AAABB'), frozenset((i, j) for i in range(3) for j in (3, 4)))
    rng = Random(52)
    log = Counter({random_pot(rng, 5): 2 for _ in range(15)})
    log[layered] = 5
    expected = [Counter() for _ in range(4)]
    for trace, freq in log.items():
        events = PartialOrderTrace(tuple(range(len(trace))), trace.order)
        orders = list(linearizations(events))
        starts = {word[0] for word in orders}
        ends = {word[-1] for word in orders}
        direct = {pair for word in orders for pair in zip(word, word[1:])}
        eventual = {pair for word in orders for pair in combinations(word, 2)}
        for counts, indices in zip(expected[:2], (starts, ends)):
            for i in indices:
                counts[trace.activities[i]] += freq
        for counts, pairs in zip(expected[2:], (direct, eventual)):
            for i, j in pairs:
                counts[trace.activities[i], trace.activities[j]] += freq
    dfg, efg = discover_dfg_efg_pot(log)
    assert [dfg.start_activities, dfg.end_activities, dfg.graph, efg] == expected


def test_activity_candidate_graphs_match_erasing_from_all_linearizations():
    rng = Random(184)
    for _ in range(30):
        # Singleton runs, tied layers of the same label, and arbitrary DAGs.
        log = Counter({ordered(*'BAAAC'): 1, random_pot(rng, 6): 3,
                       PartialOrderTrace(('B', 'B', 'C')): 2})
        words = expand(log)
        for candidate, dfg in _candidate_dfgs(IMDataStructurePOT(log)):
            erased = Counter(tuple(a for a in w if a != candidate) for w in words)
            reference = IMDataStructureUVCL(erased).dfg
            assert set(dfg.start_activities) == set(reference.start_activities)
            assert set(dfg.end_activities) == set(reference.end_activities)
            assert set(dfg.graph) == set(reference.graph)


def test_wide_tie_only_fixes_the_endpoint_that_decides_the_boundary():
    tied = tuple(f'E{i}' for i in range(20)) + ('X',)
    trace = PartialOrderTrace(tied + ('S',), frozenset((i, len(tied)) for i in range(len(tied))))
    parts = trace.split_project_by_start_boundaries('end_to_start', {'S'}, set(tied) - {'X'})
    # X last avoids the cut. Every other possible last event triggers it.
    by_size = Counter(map(len, parts))
    assert by_size == {1: 20, 21: 20, 22: 1}
    uncut, = [p for p in parts if len(p) == len(trace)]
    assert all(uncut.precedes(i, 20) for i in range(20))
    assert all(uncut.concurrent(i, j) for i, j in combinations(range(20), 2))


@pytest.mark.parametrize('window', [None, timedelta(seconds=1)], ids=['ties', 'overlapping-window'])
def test_timestamped_default_matches_standard_public_entry_point(window):
    rows = [('one', 'A', 0), ('one', 'B', 1), ('one', 'A', 2), ('one', 'C', 2),
            ('two', 'C', 0), ('two', 'C', 0)]
    df = pd.DataFrame(rows, columns=['case:concept:name', 'concept:name', 'time:timestamp'])
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], unit='s')
    # Construct the reference order from the defining timestamp inequality.
    traces = []
    for _, case in df.groupby('case:concept:name'):
        times = list(case['time:timestamp'])
        edges = {(i, j) for i, a in enumerate(times) for j, b in enumerate(times)
                 if a + (window or timedelta(0)) < b}
        traces.append(PartialOrderTrace(tuple(case['concept:name']), frozenset(edges)))
    words = [word for trace in traces for word in linearizations(trace)]
    linear = pd.DataFrame([(str(case), a, pd.Timestamp(i, unit='s'))
                           for case, word in enumerate(words) for i, a in enumerate(word)],
                          columns=df.columns)
    assert apply(df, time_ordering=True, time_window=window).same_structure(
        apply(linear, time_ordering=False)
    )


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
