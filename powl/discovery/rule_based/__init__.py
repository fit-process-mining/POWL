"""Declarative rule-aware discovery, using POWL's inductive framework."""

from .algorithm import discover_with_rules
from .exceptions import (
    InconsistentRulesError,
    RuleConformanceError,
    RuleDiscoveryError,
    RuleDiscoveryLimitError,
)
from .repair import RepairMode
from .rules import (
    AbstractRule,
    AtMostOnceRule,
    ChainPrecedenceRule,
    ChainResponseRule,
    CoExistenceRule,
    EndRule,
    ExistenceRule,
    InitializationRule,
    NotCoExistenceRule,
    NotSuccessionRule,
    PrecedenceRule,
    RespondedExistenceRule,
    ResponseRule,
)
from .validation import RuleValidationResult, validate_rules

__all__ = [
    "discover_with_rules",
    "validate_rules",
    "RuleValidationResult",
    "RepairMode",
    "AbstractRule",
    "AtMostOnceRule",
    "ChainPrecedenceRule",
    "ChainResponseRule",
    "CoExistenceRule",
    "EndRule",
    "ExistenceRule",
    "InitializationRule",
    "NotCoExistenceRule",
    "NotSuccessionRule",
    "PrecedenceRule",
    "RespondedExistenceRule",
    "ResponseRule",
    "RuleDiscoveryError",
    "InconsistentRulesError",
    "RuleConformanceError",
    "RuleDiscoveryLimitError",
]
