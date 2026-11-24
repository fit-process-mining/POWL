"""Exact support of maximal runs separated by an adjacent-label predicate."""

from __future__ import annotations

from collections import Counter
from itertools import chain, combinations
from typing import TYPE_CHECKING, Callable, Iterator

if TYPE_CHECKING:
    from .partial_order import PartialOrderTrace


def _fix_endpoint(trace, block, candidates, *, first):
    """Partition the extensions by the first (or last) event of a serial block."""
    for pivot in candidates:
        edges = ((pivot, i) if first else (i, pivot) for i in block if i != pivot)
        # All events outside the block already precede or follow the entire block.
        yield type(trace)(trace.activities, trace.order.union(edges))


def _order_pair(trace, left, right):
    """Partition on two incomparable events, maintaining the transitive closure."""
    for before, after in ((left, right), (right, left)):
        edges = ((i, j) for i in trace.predecessors[before] | {before}
                 for j in trace.successors[after] | {after})
        yield type(trace)(trace.activities, trace.order.union(edges))


def _internal_cuts(trace, block, kinds, boundary):
    """Possible split decisions for consecutive events inside a serial block."""
    counts = Counter(kinds[i] for i in block)
    decisions = {boundary(a, b) for a in counts for b in counts if a != b or counts[a] > 1}
    if trace.is_layered or len(decisions) == 1:
        return decisions

    decisions = set()
    for i in block:
        for j in block:
            # Possible consecutive events are exactly covers and incomparable pairs.
            if i == j or (j, i) in trace.order:
                continue
            if (i, j) in trace.order and (i, j) not in trace.transitive_reduction_edges:
                continue
            decisions.add(boundary(kinds[i], kinds[j]))
            if len(decisions) == 2:
                return decisions
    return decisions


def _extrema(trace, block, offset, *, first):
    if len(block) == 1 or trace.is_layered:
        return block
    neighbors = trace.predecessors if first else trace.successors
    return tuple(i for i in block if len(neighbors[i]) == offset)


def _split_between_blocks(trace, kinds, boundary):
    """Factor at guaranteed boundaries, otherwise resolve an ambiguous interface."""
    blocks = trace.serial_blocks
    cuts = [0]
    uncertain = None
    offset = 0
    for k, (left, right) in enumerate(zip(blocks, blocks[1:]), 1):
        offset += len(left)
        if len(left) == len(right) == 1:
            if boundary(kinds[left[0]], kinds[right[0]]):
                cuts.append(k)
            continue
        lasts = _extrema(trace, left, len(trace) - offset, first=False)
        firsts = _extrema(trace, right, offset, first=True)
        # Each possible last can be adjacent to each possible first.
        first_kinds = frozenset(kinds[i] for i in firsts)
        # For each possible last kind, which first kinds start a new fragment?
        last_kinds = {kinds[i] for i in lasts}
        cut_targets = {frozenset(b for b in first_kinds if boundary(a, b)) for a in last_kinds}
        if cut_targets == {first_kinds}:
            cuts.append(k)
        elif any(cut_targets) and uncertain is None:
            if len(cut_targets) > 1:
                uncertain = _fix_endpoint(trace, left, lasts, first=False)
            else:
                uncertain = _fix_endpoint(trace, right, firsts, first=True)

    if len(cuts) > 1:
        cuts.append(len(blocks))
        return (trace.induced_subtrace(sorted(chain.from_iterable(blocks[a:b])))
                for a, b in zip(cuts, cuts[1:]))
    return uncertain


def _split_within_block(trace, kinds, boundary):
    blocks = trace.serial_blocks
    offset = 0
    for block in blocks:
        before = offset
        offset += len(block)
        if len(block) == 1:
            continue
        decisions = _internal_cuts(trace, block, kinds, boundary)
        if True not in decisions:
            continue
        if False not in decisions:
            if len(blocks) == 1:
                return (trace.induced_subtrace([i]) for i in block)
            # Only an endpoint of this block can join a neighboring fragment.
            first = before > 0
            outside = before if first else len(trace) - offset
            candidates = _extrema(trace, block, outside, first=first)
            return _fix_endpoint(trace, block, candidates, first=first)

        # Serial blocks are connected components of the incomparability graph.
        # A mixed block therefore contains an incomparable pair of unlike kinds.
        i, j = next((i, j) for i, j in combinations(block, 2)
                    if kinds[i] != kinds[j] and trace.concurrent(i, j))
        return _order_pair(trace, i, j)
    return None


def split_at_boundaries(
    trace: PartialOrderTrace, classify: Callable, boundary: Callable,
) -> Iterator[PartialOrderTrace]:
    """Yield POTs whose union of linearizations is exactly the projected language.

    Mandatory cuts factor independently. An ambiguous cut partitions the possible
    orders into exhaustive alternatives. Each refinement adds a strict precedence,
    so the search terminates even on non-layered posets. The number of distinct
    fragments/refinements can be exponential. Each occurrence is emitted once
    per visited alternative, independently of its linear-extension count.
    """
    if not trace:
        return
    pending = [iter((trace,))]
    while pending:
        current = next(pending[-1], None)
        if current is None:
            pending.pop()
            continue
        kinds = tuple(map(classify, current.activities))
        parts = _split_between_blocks(current, kinds, boundary)
        if parts is None:
            parts = _split_within_block(current, kinds, boundary)
        if parts is None:
            yield current
        else:
            pending.append(parts)
