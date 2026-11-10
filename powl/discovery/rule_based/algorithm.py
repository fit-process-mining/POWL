"""Public entry point for discovery from user-supplied declarative rules."""

from __future__ import annotations

import pandas as pd
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.objects.log.obj import EventLog

from powl.objects.tagged_powl.base import TaggedPOWL

from .conformance import assert_model_satisfies_rules
from .log_utils import to_variants
from .miner import RuleAwarePOWLInductiveMiner
from .repair import RepairMode
from .rules import normalize_rules


def _completion_events(log, activity_key, lifecycle_key):
    if lifecycle_key is None:
        return log
    completed = {"complete", "COMPLETE", "Complete"}
    if isinstance(log, pd.DataFrame) and lifecycle_key in log.columns:
        filtered = log[log[lifecycle_key].isin(completed)]
        return filtered if not filtered.empty else log
    if isinstance(log, EventLog):
        # As with powl.discover, retain the original log when the filter would
        # remove every event. Explicit empty EventLog traces remain explicit.
        filtered = [
            tuple(
                event[activity_key]
                for event in trace
                if event.get(lifecycle_key) in completed
            )
            for trace in log
        ]
        if any(filtered):
            return [
                trace for trace, original in zip(filtered, log) if trace or not original
            ]
    return log


def discover_with_rules(
    log,
    rules,
    *,
    repair_mode: RepairMode | str = RepairMode.EVENT,
    noise_threshold: float = 0.0,
    preprocess_rules: bool = True,
    verify_rules: bool = True,
    max_states: int = 100_000,
    max_depth: int = 200,
    activity_key: str = "concept:name",
    case_id_key: str = "case:concept:name",
    timestamp_key: str | None = "time:timestamp",
    lifecycle_key: str | None = "lifecycle:transition",
    keep_only_completion_events: bool = True,
    simplify: bool = True,
) -> TaggedPOWL:
    """Discover a native POWL model using supplied declarative constraints.

    ``log`` can be a DataFrame, PM4Py EventLog, iterable of activity-label traces,
    or a mapping from trace tuples to nonnegative integer frequencies. Inputs
    are not mutated. DataFrames use stable timestamp order when the named column
    exists, otherwise existing row order. Explicit empty cases need EventLog,
    iterable traces, or a weighted variant mapping.

    The miner tries XOR, strict sequence, concurrency and loop cuts, then the
    reference data fall-throughs. Rejected rule obligations drive local event,
    trace, or minimum-edit repair. Rule-directed PO/XOR and flower fall-throughs
    follow. Sequence and concurrency are represented natively as POWL orders.

    ``preprocess_rules`` simplifies proved absent activities and equivalent
    negative pairs; consistency is always checked. ``verify_rules`` proves the
    returned model against the original rules, including all loop executions
    and parallel interleavings. A heuristic result that violates a rule raises
    RuleConformanceError with a counterexample. Set it to False only when an
    explicitly unverified candidate is wanted for experimentation.

    ``max_states`` bounds reachable rule-product, edit-search, and per-rule
    model-verification states. ``max_depth`` bounds recursive decomposition.
    Exhaustion raises RuleDiscoveryLimitError rather than returning an unchecked
    model. These finite-state searches may be exponential in model/rule size.

    No rules are mined from the log. Call ``powl.validate_rules`` separately to
    inspect empirical support/confidence; discovery does not filter input rules
    by these metrics.
    """
    if not isinstance(keep_only_completion_events, bool) or not isinstance(
        simplify, bool
    ):
        raise TypeError("keep_only_completion_events and simplify must be bools.")
    supplied = normalize_rules(rules)
    if keep_only_completion_events:
        log = _completion_events(log, activity_key, lifecycle_key)
    variants = to_variants(
        log,
        activity_key=activity_key,
        case_id_key=case_id_key,
        timestamp_key=timestamp_key,
    )
    parameters = {
        "rules": supplied,
        "repair_mode": repair_mode,
        "noise_threshold": noise_threshold,
        "preprocess_rules": preprocess_rules,
        # Verify the final normalized object, so exported behavior is covered.
        "verify_rules": False,
        "max_states": max_states,
        "max_depth": max_depth,
    }
    if not isinstance(verify_rules, bool):
        raise TypeError("verify_rules must be a bool.")
    miner = RuleAwarePOWLInductiveMiner(parameters)
    model = miner.apply(IMDataStructureUVCL(variants), parameters)
    if simplify:
        model = model.normalize()
    if verify_rules:
        assert_model_satisfies_rules(model, supplied, max_states=max_states)
    return model


# Conventional discovery-module alias, consistent with POWL's other packages.
apply = discover_with_rules
