"""Rule-aware extensions of POWL's XOR, sequence, concurrency, and loop cuts."""

from .base import RuleAwareCut, RuleAwareCutResult
from .concurrency import RuleAwareConcurrencyCutUVCL
from .loop import RuleAwareLoopCutUVCL
from .sequence import RuleAwareSequenceCutUVCL, RuleAwareStrictSequenceCutUVCL
from .xor import RuleAwareExclusiveChoiceCutUVCL

RULE_AWARE_CUTS = (
    RuleAwareExclusiveChoiceCutUVCL,
    RuleAwareStrictSequenceCutUVCL,
    RuleAwareConcurrencyCutUVCL,
    RuleAwareLoopCutUVCL,
)

__all__ = [
    "RuleAwareCut",
    "RuleAwareCutResult",
    "RuleAwareExclusiveChoiceCutUVCL",
    "RuleAwareSequenceCutUVCL",
    "RuleAwareStrictSequenceCutUVCL",
    "RuleAwareConcurrencyCutUVCL",
    "RuleAwareLoopCutUVCL",
    "RULE_AWARE_CUTS",
]
