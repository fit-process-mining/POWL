"""Recursive data passed between rule-aware discovery components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.total_order_based.inductive.modeling import InductiveModel

if TYPE_CHECKING:
    from .rules import AbstractRule


@dataclass(frozen=True)
class RuleDecomposition:
    """A native POWL construct and an independent rule set for each child."""

    spec: InductiveModel
    children: tuple[IMDataStructureUVCL, ...]
    projected_rules: tuple[tuple[AbstractRule, ...], ...]

    def __post_init__(self) -> None:
        if len(self.children) != len(self.projected_rules):
            raise ValueError("Each projected sublog must have its own rule set.")
