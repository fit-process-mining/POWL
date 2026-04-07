from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeAlias

from powl.objects.tagged_powl import Activity, ChoiceGraph, PartialOrder, TaggedPOWL
from powl.objects.tagged_powl.builders import loop, sequence, xor


@dataclass(frozen=True)
class SequenceSpec:
    size: int


@dataclass(frozen=True)
class XorSpec:
    size: int


@dataclass(frozen=True)
class LoopSpec:
    size: int


@dataclass(frozen=True)
class PartialOrderSpec:
    size: int
    edges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ChoiceGraphSpec:
    size: int
    edges: tuple[tuple[int, int], ...] = ()
    start_nodes: tuple[int, ...] = ()
    end_nodes: tuple[int, ...] = ()
    min_freq: int = 1
    max_freq: Optional[int] = 1


InductiveSpec: TypeAlias = (
    SequenceSpec | XorSpec | LoopSpec | PartialOrderSpec | ChoiceGraphSpec
)
InductiveModel: TypeAlias = TaggedPOWL | InductiveSpec


def build_model(spec: InductiveModel, children: list[TaggedPOWL]) -> TaggedPOWL:
    if isinstance(spec, TaggedPOWL):
        return spec

    if isinstance(spec, SequenceSpec):
        return sequence(children)
    if isinstance(spec, XorSpec):
        if not children:
            return Activity(label=None)
        return xor(children)

    if isinstance(spec, LoopSpec):
        if not children:
            return Activity(label=None)
        do = children[0]
        redo_children = children[1:]
        if not redo_children:
            redo = Activity(label=None)
        elif len(redo_children) == 1:
            redo = redo_children[0]
        else:
            redo = xor(redo_children)
        return loop(do, redo)

    if isinstance(spec, PartialOrderSpec):
        po = PartialOrder(
            nodes=children,
            edges=[(children[i], children[j]) for (i, j) in spec.edges],
        )
        return po

    if isinstance(spec, ChoiceGraphSpec):
        cg = ChoiceGraph(
            nodes=children,
            edges=[(children[i], children[j]) for (i, j) in spec.edges],
            start_nodes=[children[i] for i in spec.start_nodes],
            end_nodes=[children[i] for i in spec.end_nodes],
            min_freq=spec.min_freq,
            max_freq=spec.max_freq,
        )
        return cg

    raise TypeError(f"Unsupported inductive spec: {type(spec)}")
