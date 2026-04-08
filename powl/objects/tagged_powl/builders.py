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


def expand_frequency_tags(
    model: TaggedPOWL,
    *,
    expand_activities: bool = True,
) -> TaggedPOWL:
    if isinstance(model, Activity):
        if not expand_activities:
            return model.clone(deep=True)
        body = model.clone(deep=True)
        body.set_freqs(min_freq=1, max_freq=1)
        return _wrap_frequency_tags(body, model)

    if isinstance(model, PartialOrder):
        node_map = {
            node: expand_frequency_tags(node, expand_activities=expand_activities)
            for node in model.children
        }
        body = PartialOrder(
            nodes=[node_map[node] for node in model.children],
            edges=[(node_map[u], node_map[v]) for (u, v) in model.get_edges()],
            min_freq=1,
            max_freq=1,
        )
        return _wrap_frequency_tags(body, model)

    if isinstance(model, ChoiceGraph):
        node_map = {
            node: expand_frequency_tags(node, expand_activities=expand_activities)
            for node in model.children
        }
        body = ChoiceGraph(
            nodes=[node_map[node] for node in model.children],
            edges=[(node_map[u], node_map[v]) for (u, v) in model.get_edges()],
            start_nodes=[node_map[node] for node in model.start_nodes()],
            end_nodes=[node_map[node] for node in model.end_nodes()],
            min_freq=1,
            max_freq=1,
        )
        return _wrap_frequency_tags(body, model)

    raise TypeError(f"Unsupported TaggedPOWL type: {type(model).__name__}")


def _wrap_frequency_tags(body: TaggedPOWL, original: TaggedPOWL) -> TaggedPOWL:
    if original.is_repeatable():
        if original.is_skippable():
            return loop(silent_activity(), body)
        return loop(body, silent_activity())
    if original.is_skippable():
        return xor([body, silent_activity()])
    return body
