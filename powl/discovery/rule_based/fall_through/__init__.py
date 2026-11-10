"""Rule-aware fall-throughs, ordered as in the reference RBIM implementation."""

from .data import (
    DATA_FALL_THROUGHS,
    activity_concurrent,
    activity_once,
    strict_tau,
    tau,
)
from .flower import flower
from .po import rule_po
from .xor import rule_xor

RULE_FALL_THROUGHS = (rule_po, rule_xor)

__all__ = [
    "DATA_FALL_THROUGHS",
    "RULE_FALL_THROUGHS",
    "activity_once",
    "activity_concurrent",
    "strict_tau",
    "tau",
    "rule_po",
    "rule_xor",
    "flower",
]
