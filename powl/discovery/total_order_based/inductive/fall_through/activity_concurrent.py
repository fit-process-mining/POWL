from collections import Counter, defaultdict
from itertools import chain, product
from multiprocessing import Manager, Pool
from typing import Any, Dict, List, Optional, Tuple

from pm4py.algo.discovery.inductive.cuts.factory import CutFactory
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.algo.discovery.inductive.fall_through.activity_concurrent import (
    ActivityConcurrentUVCL,
    FallThrough,
)
from pm4py.algo.discovery.inductive.variants.instances import IMInstance
from pm4py.objects.dfg import util as dfg_utils
from pm4py.objects.dfg.obj import DFG

from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT, combined_project_pot_on_groups,
)
from powl.discovery.total_order_based.inductive.modeling import PartialOrderSpec


class POWLActivityConcurrentUVCL(ActivityConcurrentUVCL):
    @classmethod
    def apply(
        cls,
        obj: IMDataStructureUVCL,
        pool: Pool = None,
        manager: Manager = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[PartialOrderSpec, List[IMDataStructureUVCL]]]:
        candidate = cls._get_candidate(obj, pool, manager, parameters)
        if candidate is None:
            return None
        log = obj.data_structure
        l_a = Counter()
        l_other = Counter()
        for t in log:
            l_a.update({tuple(filter(lambda e: e == candidate, t)): log[t]})
            l_other.update({tuple(filter(lambda e: e != candidate, t)): log[t]})
        return PartialOrderSpec(2), [
            IMDataStructureUVCL(l_a),
            IMDataStructureUVCL(l_other),
        ]


# Virtual endpoint for the beginning or end of a trace.
_BOUNDARY = object()


def _removal_bridges(trace):
    """Yield (removed activity, left, right) for newly possible adjacencies."""
    if trace.is_layered:
        layers = [frozenset(trace.activities[i] for i in block) for block in trace.serial_blocks]
        # Only a run of layers consisting entirely of c can disappear when c
        # is removed. Its two surviving neighbors then become adjacent.
        pos = 0
        while pos < len(layers):
            start = pos
            layer = layers[pos]
            pos += 1
            if len(layer) != 1:
                continue
            while pos < len(layers) and layers[pos] == layer:
                pos += 1
            left = layers[start - 1] - layer if start else {_BOUNDARY}
            right = layers[pos] - layer if pos < len(layers) else {_BOUNDARY}
            removed, = layer
            for a, b in product(left, right):
                yield removed, a, b
        return

    # In a general poset, i < j becomes a cover iff every intervening
    # event was removed. The same test with virtual endpoints gives starts
    # and ends. Integer bitsets avoid allocating an event set per interval.
    labels = trace.activities
    bits = [1 << i for i in range(len(trace))]
    by_label = defaultdict(int)
    predecessors = [0] * len(trace)
    successors = [0] * len(trace)
    for i, label in enumerate(labels):
        by_label[label] |= bits[i]
    for i, j in trace.order:
        predecessors[j] |= bits[i]
        successors[i] |= bits[j]
    intervals = chain(
        ((_BOUNDARY, a, predecessors[i]) for i, a in enumerate(labels)),
        ((a, _BOUNDARY, successors[i]) for i, a in enumerate(labels)),
        ((labels[i], labels[j], successors[i] & predecessors[j]) for i, j in trace.order),
    )
    for a, b, events in intervals:
        if events:
            first = (events & -events).bit_length() - 1
            removed = labels[first]
            if removed != a and removed != b and (events & by_label[removed]) == events:
                yield removed, a, b


def _candidate_dfgs(obj):
    """DFG support after erasing each candidate, sharing one pass over the log."""
    base = obj.unfiltered_dfg
    if len(dfg_utils.get_vertices(base)) <= 2:
        return  # Removing any candidate leaves too few activities for an IM cut.
    bridges = defaultdict(set)
    for trace in obj.data_structure:
        for removed, a, b in _removal_bridges(trace):
            bridges[removed].add((a, b))

    for candidate in sorted(dfg_utils.get_vertices(obj.dfg)):
        dfg = DFG()
        dfg.start_activities.update(a for a in base.start_activities if a != candidate)
        dfg.end_activities.update(a for a in base.end_activities if a != candidate)
        dfg.graph.update((a, b) for a, b in base.graph if candidate not in (a, b))
        for a, b in bridges[candidate]:
            if a is _BOUNDARY:
                if b is not _BOUNDARY:
                    dfg.start_activities[b] = 1
            elif b is _BOUNDARY:
                dfg.end_activities[a] = 1
            else:
                dfg.graph[a, b] = 1
        yield candidate, dfg


class POWLActivityConcurrentPOT(FallThrough[IMDataStructurePOT]):
    @classmethod
    def _get_candidate(
        cls,
        obj: IMDataStructurePOT,
        pool: Pool = None,
        manager: Manager = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        # These are the same DFG-only predicates used by ActivityConcurrentUVCL.
        # Their child projections are needed only for the selected candidate.
        for candidate, dfg in _candidate_dfgs(obj):
            probe = IMDataStructureUVCL(Counter(), dfg=dfg)
            if any(cut.holds(probe, parameters) is not None
                   for cut in CutFactory.get_cuts(probe, IMInstance.IM, parameters)):
                return candidate
        return None

    @classmethod
    def holds(
        cls,
        obj: IMDataStructurePOT,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return cls._get_candidate(obj, None, None, parameters) is not None

    @classmethod
    def apply(
        cls,
        obj: IMDataStructurePOT,
        pool: Pool = None,
        manager: Manager = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[PartialOrderSpec, List[IMDataStructurePOT]]]:
        candidate = cls._get_candidate(obj, pool, manager, parameters)
        if candidate is None:
            return None
        alphabet = set(dfg_utils.get_vertices(obj.dfg))
        children = combined_project_pot_on_groups(
            obj.data_structure,
            [{candidate}, alphabet.difference({candidate})],
            keep_empty=True,
        )
        return PartialOrderSpec(2), children
