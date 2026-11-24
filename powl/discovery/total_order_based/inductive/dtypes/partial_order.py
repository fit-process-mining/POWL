from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from itertools import chain, combinations, groupby, pairwise
from operator import ne
from typing import Any, Collection, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Literal, Union

import pandas as pd
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureLog
from pm4py.objects.dfg.obj import DFG

from powl.discovery.total_order_based.inductive.dtypes.pot_projection import split_at_boundaries

ArtifactWeighting = Literal["normalized", "unit"]
BoundaryMode = Literal["end_to_start", "any_to_start"]

IndexPair = Tuple[int, int]


@dataclass(frozen=True, init=False)
class PartialOrderTrace:
    """Event-indexed partially ordered trace variant.

    ``activities`` is an ordered list of activity labels. The position in this
    list is the event identity, therefore duplicate labels are allowed.

    ``order`` is a strict partial order over event indices. It is assumed to be
    transitive and irreflexive. For example::

        activities = ("a", "b", "c", "a")
        order = {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)}

    represents a trace where the first ``a`` happens before ``b`` and ``c``,
    ``b`` and ``c`` are unordered, and both happen before the final ``a``.
    """

    activities: Tuple[Any, ...]
    order: FrozenSet[IndexPair] = frozenset()

    def __init__(self, activities: Iterable[Any], order: Iterable[IndexPair] = frozenset()):
        object.__setattr__(self, "activities", tuple(activities))
        object.__setattr__(self, "order", frozenset((int(i), int(j)) for i, j in order))

    def __len__(self) -> int:
        return len(self.activities)

    @property
    def labels(self) -> Tuple[Any, ...]:
        return self.activities

    @property
    def is_empty(self) -> bool:
        return len(self.activities) == 0

    @property
    def alphabet(self) -> set:
        return set(self.activities)

    @cached_property
    def predecessors(self) -> Tuple[FrozenSet[int], ...]:
        preds: List[set] = [set() for _ in range(len(self))]
        for i, j in self.order:
            preds[j].add(i)
        return tuple(frozenset(p) for p in preds)

    @cached_property
    def successors(self) -> Tuple[FrozenSet[int], ...]:
        succs: List[set] = [set() for _ in range(len(self))]
        for i, j in self.order:
            succs[i].add(j)
        return tuple(frozenset(s) for s in succs)

    @cached_property
    def transitive_reduction_edges(self) -> FrozenSet[IndexPair]:
        return frozenset(
            (i, j)
            for i, j in self.order
            if self.successors[i].isdisjoint(self.predecessors[j])
        )

    @cached_property
    def serial_blocks(self) -> tuple[tuple[int, ...], ...]:
        """Finest blocks such that every event in a block precedes all later blocks."""
        earliest = [0] * len(self)
        latest = [len(self) - 1] * len(self)
        for before, after in self.order:
            earliest[after] += 1
            latest[before] -= 1
        topological = sorted(range(len(self)), key=earliest.__getitem__)
        blocks = []
        start = 0
        last_position = -1
        for position, event in enumerate(topological):
            # The prefix is fixed as a set iff none of its events can occur later.
            last_position = max(last_position, latest[event])
            if position == last_position:
                blocks.append(tuple(topological[start:position + 1]))
                start = position + 1
        return tuple(blocks)

    @cached_property
    def is_layered(self) -> bool:
        """Whether each serial block is an antichain (includes timestamp ties)."""
        cross_pairs = 0
        earlier = 0
        for block in self.serial_blocks:
            cross_pairs += earlier * len(block)
            earlier += len(block)
        return cross_pairs == len(self.order)

    @cached_property
    def minimal_indices(self) -> FrozenSet[int]:
        """Events with no predecessors."""
        return frozenset(i for i in range(len(self)) if not self.predecessors[i])

    @cached_property
    def maximal_indices(self) -> FrozenSet[int]:
        """Events with no successors."""
        return frozenset(i for i in range(len(self)) if not self.successors[i])

    @classmethod
    def empty(cls) -> "PartialOrderTrace":
        return cls(tuple(), frozenset())

    @classmethod
    def from_timestamped_events(
        cls,
        activities: Sequence[Any],
        timestamps: Sequence[Any],
        time_window: Optional[timedelta | pd.Timedelta],
    ) -> "PartialOrderTrace":
        """Create a general POT from timestamped events.

        Event ``i`` precedes event ``j`` iff::

            timestamp_i + time_window < timestamp_j

        Hence, exactly equal timestamps are unordered for ``time_window=0``;
        with a positive window, events whose timestamps are too close are also
        unordered.

        """
        if len(activities) != len(timestamps):
            raise ValueError("activities and timestamps must have the same length")
        if len(activities) == 0:
            return cls.empty()

        if time_window is not None:
            time_window = _normalize_time_window(time_window)

        return cls._from_timestamp_key(*_timestamp_key(activities, timestamps, time_window))

    @classmethod
    def _from_timestamp_key(cls, labels, successors):
        edges = ((i, j) for i, first in enumerate(successors) for j in range(first, len(labels)))
        return cls(labels, edges)

    def precedes(self, i: int, j: int) -> bool:
        return (i, j) in self.order

    def concurrent(self, i: int, j: int) -> bool:
        return i != j and (i, j) not in self.order and (j, i) not in self.order

    def combined_project(self, group: Collection[Any]) -> "PartialOrderTrace":
        """Combined projection for partial-order traces.

        All events belonging to the group will be combined into a single sub-POT.
        """
        group_set = group if isinstance(group, (set, frozenset)) else set(group)
        indices = [i for i, activity in enumerate(self.activities) if activity in group_set]
        return self.induced_subtrace(indices)

    def split_project(self, group: Collection[Any]) -> List["PartialOrderTrace"]:
        """Maximal nonempty visits to the group across all linear extensions."""
        selected = frozenset(group)
        return [part for part in split_at_boundaries(self, selected.__contains__, ne)
                if part.activities[0] in selected]

    def split_project_by_start_boundaries(
        self,
        boundary_mode: BoundaryMode,
        start_activities: Collection[Any],
        end_activities: Collection[Any] = frozenset(),
    ) -> List["PartialOrderTrace"]:
        classify, boundary = _tau_boundary(boundary_mode, start_activities, end_activities)
        return list(split_at_boundaries(self, classify, boundary)) if self else [self]

    def induced_subtrace(self, indices: Sequence[int]) -> "PartialOrderTrace":
        """Keep the given event indices in their supplied indexing order."""
        if len(indices) == len(self) and all(i == j for i, j in enumerate(indices)):
            return self
        activities = tuple(self.activities[i] for i in indices)
        if len(indices) < 2:
            return PartialOrderTrace(activities)
        # For a short visit, probing retained pairs avoids scanning (or copying)
        # the full relation. Sparse parents are cheaper to scan by edge.
        if len(indices) ** 2 < len(self.order):
            order = ((i, j) for i, a in enumerate(indices) for j, b in enumerate(indices)
                     if (a, b) in self.order)
        else:
            old_to_new = {old: new for new, old in enumerate(indices)}
            order = ((old_to_new[a], old_to_new[b]) for a, b in self.order
                     if a in old_to_new and b in old_to_new)
        return PartialOrderTrace(activities, order)


def _normalize_time_window(window: timedelta | pd.Timedelta) -> pd.Timedelta:
    """Normalize the timestamp tolerance window.

    We intentionally require a real timedelta object instead of accepting
    numbers or strings, because numeric windows are ambiguous and can hide
    mistakes.
    """

    if isinstance(window, timedelta):
        window = pd.Timedelta(window)
    elif not isinstance(window, pd.Timedelta):
        raise TypeError(
            "time_window must be None, datetime.timedelta, or pandas.Timedelta"
        )

    if window < pd.Timedelta(0):
        raise ValueError("time_window must be non-negative")

    return window


def _timestamp_key(activities, timestamps, window):
    # A timestamp order is determined by the first successor of each event
    # in time order. Absolute times need not be kept in the variant key.
    events = sorted(zip(map(pd.Timestamp, timestamps), activities))
    times = [time for time, _ in events]
    labels = tuple(label for _, label in events)
    successors = tuple(bisect_right(times, t if window is None else t + window) for t in times)
    return labels, successors


def log_to_pot_variants(
    df: pd.DataFrame,
    activity_key: str,
    timestamp_key: str,
    case_id_key: str,
    time_window: Optional[Union[timedelta, pd.Timedelta]],
) -> Counter:
    if time_window is not None:
        time_window = _normalize_time_window(time_window)
    counts = Counter(
        _timestamp_key(case[activity_key], case[timestamp_key], time_window)
        for _, case in df.groupby(case_id_key, sort=False)
    )
    return Counter({PartialOrderTrace._from_timestamp_key(*key): freq for key, freq in counts.items()})


def get_partial_order_variants(traces: Iterable[PartialOrderTrace]) -> Counter:
    return Counter(traces)


def discover_dfg_efg_pot(
        log: Counter[PartialOrderTrace],
        weighting: ArtifactWeighting = "unit",
) -> Tuple[DFG, Counter]:
    """Compute native start/end, directly-follows and eventually-follows weights."""
    if weighting == "unit":
        return _unit_dfg(log), _unit_efg(log)
    if weighting != "normalized":
        raise ValueError("weighting must be 'unit' or 'normalized'")
    dfg = DFG()
    efg = Counter()
    for trace, freq in log.items():
        start, end, direct, eventual = _trace_normalized_expanded_counters(trace)
        for total, local in ((dfg.start_activities, start), (dfg.end_activities, end),
                             (dfg.graph, direct), (efg, eventual)):
            for key, value in local.items():
                total[key] += freq * value
    return dfg, efg


def _unit_pairs(trace, *, direct):
    """Yield label pairs and the number of event pairs contributing unit weight."""
    labels = trace.activities
    if trace.is_layered:
        blocks = trace.serial_blocks
        if direct and len(blocks) == len(trace):
            for (i,), (j,) in pairwise(blocks):
                yield (labels[i], labels[j]), 1
            return
        # Covers span neighboring layers. Eventual precedence spans all earlier
        # layers. Distinct events in the same layer contribute both directions.
        earlier = Counter()
        for block in blocks:
            current = Counter(labels[i] for i in block)
            for b, n in current.items():
                for a, count in earlier.items():
                    yield (a, b), count * n
                for a, count in current.items():
                    pairs = (count - (a == b)) * n
                    if pairs:
                        yield (a, b), pairs
            if direct:
                earlier = current
            else:
                earlier.update(current)
    else:
        ordered_pairs = trace.transitive_reduction_edges if direct else trace.order
        for i, j in ordered_pairs:
            yield (labels[i], labels[j]), 1
        for i, j in combinations(range(len(trace)), 2):
            if trace.concurrent(i, j):
                yield (labels[i], labels[j]), 1
                yield (labels[j], labels[i]), 1


def _unit_dfg(log):
    dfg = DFG()
    for trace, freq in log.items():
        if not trace:
            continue
        if trace.is_layered:
            sources, sinks = trace.serial_blocks[0], trace.serial_blocks[-1]
        else:
            sources, sinks = trace.minimal_indices, trace.maximal_indices
        for i in sources:
            dfg.start_activities[trace.activities[i]] += freq
        for i in sinks:
            dfg.end_activities[trace.activities[i]] += freq
        for pair, count in _unit_pairs(trace, direct=True):
            dfg.graph[pair] += freq * count
    return dfg


def _unit_efg(log):
    efg = Counter()
    for trace, freq in log.items():
        for pair, count in _unit_pairs(trace, direct=False):
            efg[pair] += freq * count
    return efg



def _trace_normalized_expanded_counters(
    trace: PartialOrderTrace,
) -> Tuple[Counter, Counter, Counter, Counter]:
    """Trace-normalized expanded start/end/DFG/EFG counters.

    Semantics:

    DFG/start/end:
    * Add a hidden SOURCE with edges to all minimal events.
    * Add a hidden SINK with edges from all maximal events.
    * Add cover relations from the transitive reduction.
    * Add concurrency as bidirectional local possibilities.
    * Normalize outgoing mass per source node/event inside this trace.

    EFG:
    * For ordered event pairs i < j, add full mass 1 to label_i -> label_j.
    * For concurrent event pairs i || j, split pair mass equally:
      label_i -> label_j += 0.5
      label_j -> label_i += 0.5
    """
    labels = trace.activities
    n = len(trace)

    start = Counter()
    end = Counter()
    dfg = Counter()
    efg = Counter()

    if n == 0:
        return start, end, dfg, efg

    source = -1
    sink = n

    # ------------------------------------------------------------------
    # DFG + start/end via hidden SOURCE/SINK and local outgoing normalization
    # ------------------------------------------------------------------
    expanded_edges = set()

    # SOURCE -> minimal events
    for i in trace.minimal_indices:
        expanded_edges.add((source, i))

    # maximal events -> SINK
    for i in trace.maximal_indices:
        expanded_edges.add((i, sink))

    # Direct causal edges
    expanded_edges.update(trace.transitive_reduction_edges)

    order = trace.order

    # ------------------------------------------------------------------
    # EFG via pairwise eventual-order evidence
    # Concurrent events as bidirectional local possibilities for the DFG
    # ------------------------------------------------------------------
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in order:
                efg[(labels[i], labels[j])] += 1.0

            elif (j, i) in order:
                efg[(labels[j], labels[i])] += 1.0

            else:
                # i || j: split the unordered pair mass equally
                efg[(labels[i], labels[j])] += 0.5
                efg[(labels[j], labels[i])] += 0.5
                # add DFG expanded edges
                expanded_edges.add((i, j))
                expanded_edges.add((j, i))

    outgoing = {}
    for src, tgt in expanded_edges:
        outgoing.setdefault(src, []).append(tgt)

    for src, targets in outgoing.items():
        weight = 1.0 / len(targets)

        for tgt in targets:
            if src == source:
                # Hidden SOURCE -> activity
                start[labels[tgt]] += weight

            elif tgt == sink:
                # activity -> hidden SINK
                end[labels[src]] += weight

            else:
                # activity -> activity
                dfg[(labels[src], labels[tgt])] += weight

    return start, end, dfg, efg


class IMDataStructurePOT(IMDataStructureLog[Counter[PartialOrderTrace]]):
    """Inductive-miner data structure for partially ordered trace variants."""

    def __init__(
        self,
        obj: Counter[PartialOrderTrace],
        dfg: Optional[DFG] = None,
        efg: Optional[Counter] = None,
    ):

        super().__init__(obj)

        if (dfg is None) != (efg is None):
            raise ValueError("dfg and efg must be provided together")

        self._dfg = dfg
        self._efg = efg

    @cached_property
    def unfiltered_dfg(self) -> DFG:
        return _unit_dfg(self.data_structure)

    @property
    def dfg(self) -> DFG:
        return self.unfiltered_dfg if self._dfg is None else self._dfg

    @cached_property
    def efg(self) -> Counter:
        return _unit_efg(self.data_structure) if self._efg is None else self._efg


def is_pot_data_structure(obj: Any) -> bool:
    return isinstance(obj, IMDataStructurePOT)


def combined_project_pot_on_groups(
    log: Counter[PartialOrderTrace],
    groups: List[Collection[Any]],
    keep_empty: bool = True,
) -> List[IMDataStructurePOT]:
    projected_logs = [Counter() for _ in groups]
    for trace, freq in log.items():
        for i, group in enumerate(groups):
            projected = trace.combined_project(group)
            if keep_empty or len(projected) > 0:
                projected_logs[i][projected] += freq
    return [IMDataStructurePOT(projected_log) for projected_log in projected_logs]


def split_project_pot_on_groups(
        log: Counter[PartialOrderTrace],
        groups: List[Collection[Any]]
) -> List[IMDataStructurePOT]:
    groups = [frozenset(g) for g in groups]
    group_of = {a: i for i, group in enumerate(groups) for a in group}
    if len(group_of) != sum(map(len, groups)):
        # Cuts partition the alphabet. Overlapping projections are independent.
        return [split_project_pot_on_groups(log, [group])[0] for group in groups]
    projected_logs = [Counter() for _ in groups]
    for trace, freq in log.items():
        membership = tuple(map(group_of.get, trace.activities))
        blocks = trace.serial_blocks
        if all(all(membership[i] == membership[block[0]] for i in block) for block in blocks):
            # In a cyclic decision-graph cut, incomparable events necessarily
            # belong to the same group: they witness both directly-follows edges.
            for group_id, run in groupby(chain.from_iterable(blocks), key=membership.__getitem__):
                if group_id is not None:
                    projected_logs[group_id][trace.induced_subtrace(sorted(run))] += freq
        else:
            for part in split_at_boundaries(trace, group_of.get, ne):
                group_id = group_of.get(part.activities[0])
                if group_id is not None:
                    projected_logs[group_id][part] += freq
    return [IMDataStructurePOT(projected_log) for projected_log in projected_logs]


def _tau_boundary(mode, start_activities, end_activities):
    starts = frozenset(start_activities)
    if mode == "any_to_start":
        return starts.__contains__, lambda previous, current: current
    if mode == "end_to_start":
        ends = frozenset(end_activities)

        def classify(label):
            return label in starts, label in ends

        return classify, lambda previous, current: previous[1] and current[0]
    raise ValueError("boundary_mode must be 'end_to_start' or 'any_to_start'")


def split_project_pot_by_start_boundaries(
    log: Counter[PartialOrderTrace],
    boundary_mode: BoundaryMode,
    start_activities: Collection[Any],
    end_activities: Collection[Any] = frozenset(),
) -> IMDataStructurePOT:
    classify, boundary = _tau_boundary(boundary_mode, start_activities, end_activities)
    projected_log = Counter()
    for trace, freq in log.items():
        if not trace:
            projected_log[trace] += freq
            continue
        for projected in split_at_boundaries(trace, classify, boundary):
            projected_log[projected] += freq

    return IMDataStructurePOT(projected_log)
