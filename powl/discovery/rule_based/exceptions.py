"""Explicit failure modes for declarative rule discovery."""


class RuleDiscoveryError(ValueError):
    """The supplied rules could not be represented by the discovery method."""


class InconsistentRulesError(RuleDiscoveryError):
    """No finite trace can satisfy the supplied rule set."""


class RuleConformanceError(RuleDiscoveryError):
    """The discovered model allows a trace that violates a supplied rule."""

    def __init__(self, rule, trace):
        self.rule = rule
        self.trace = tuple(trace)
        super().__init__(
            f"The discovered model does not enforce {rule}; counterexample: {self.trace!r}. "
            "Try repair_mode='edit_distance' or revise the rules/log."
        )


class RuleDiscoveryLimitError(RuleDiscoveryError):
    """An explicit exploration bound prevented a definitive result."""
