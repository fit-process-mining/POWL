from __future__ import annotations

from typing import Iterable, Optional

from .activity import Activity
from .base import TaggedPOWL
from .choice_graph import ChoiceGraph
from .partial_order import PartialOrder


def silent_activity(*, min_freq: int = 1, max_freq: Optional[int] = 1) -> Activity:
    return Activity(label=None, min_freq=min_freq, max_freq=max_freq)


def sequence(
    children: Iterable[TaggedPOWL],
    *,
    min_freq: int = 1,
    max_freq: Optional[int] = 1,
) -> PartialOrder:
    ordered_children = list(children)
    return PartialOrder(
        nodes=ordered_children,
        edges=[(ordered_children[i], ordered_children[i + 1]) for i in range(len(ordered_children) - 1)],
        min_freq=min_freq,
        max_freq=max_freq,
    )


def xor(
    children: Iterable[TaggedPOWL],
    *,
    min_freq: int = 1,
    max_freq: Optional[int] = 1,
) -> ChoiceGraph:
    choices = list(children)
    return ChoiceGraph(
        nodes=choices,
        start_nodes=choices,
        end_nodes=choices,
        min_freq=min_freq,
        max_freq=max_freq,
    )


def loop(
    do: TaggedPOWL,
    redo: TaggedPOWL,
    *,
    min_freq: int = 1,
    max_freq: Optional[int] = 1,
) -> ChoiceGraph:
    return ChoiceGraph(
        nodes=[do, redo],
        edges=[(do, redo), (redo, do)],
        start_nodes=[do],
        end_nodes=[do],
        min_freq=min_freq,
        max_freq=max_freq,
    )
