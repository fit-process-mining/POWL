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
from .validation import RuleValidationResult, RuleViolationExample, validate_rules
from .serialization import rule_catalog, rule_from_dict, rule_to_dict, validate_rule_consistency

__all__ = [
    "discover_with_rules",
    "validate_rules",
    "RuleValidationResult",
    "RuleViolationExample",
    "rule_catalog",
    "rule_from_dict",
    "rule_to_dict",
    "validate_rule_consistency",
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
