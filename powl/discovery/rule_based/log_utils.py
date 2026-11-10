"""Lossless conversion of supported event logs to weighted trace variants."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from numbers import Integral

import pandas as pd
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.objects.log.obj import EventLog

Trace = tuple[str, ...]
Variants = Counter[Trace]


def _trace(value: Iterable[str]) -> Trace:
    if isinstance(value, (str, bytes)):
        raise TypeError("A trace must be a sequence of activity labels, not a string.")
    result = tuple(value)
    if any(not isinstance(activity, str) or not activity for activity in result):
        raise ValueError(
            "Activity labels must be nonempty strings; use () for an empty trace."
        )
    return result


def to_variants(
    log,
    *,
    activity_key: str = "concept:name",
    case_id_key: str = "case:concept:name",
    timestamp_key: str | None = "time:timestamp",
) -> Variants:
    """Return a fresh Counter without expanding multiplicities.

    DataFrames are stably ordered within each case by ``timestamp_key`` when
    that column exists. Otherwise row order is used. EventLog and iterable
    traces retain their supplied event order. DataFrames cannot encode empty
    cases; supply those as explicit empty traces or a variant Counter.
    """
    if isinstance(log, IMDataStructureUVCL):
        log = log.data_structure
    if isinstance(log, pd.DataFrame):
        for column in (activity_key, case_id_key):
            if column not in log.columns:
                raise ValueError(f"Column {column!r} not found in the event log.")
            if log[column].isna().any():
                raise ValueError(f"Column {column!r} contains missing values.")
        data = log
        if timestamp_key is not None and timestamp_key in data.columns:
            if data[timestamp_key].isna().any():
                raise ValueError(f"Column {timestamp_key!r} contains missing values.")
            data = data.sort_values(timestamp_key, kind="stable")
        return Counter(
            _trace(group[activity_key])
            for _, group in data.groupby(case_id_key, sort=False, observed=True)
        )
    if isinstance(log, EventLog):
        return Counter(_trace(event[activity_key] for event in trace) for trace in log)
    if isinstance(log, Mapping):
        result: Variants = Counter()
        for trace, count in log.items():
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
                raise ValueError("Variant frequencies must be nonnegative integers.")
            value = _trace(trace)
            if count:
                result[value] += int(count)
        return result
    if isinstance(log, (str, bytes)) or not isinstance(log, Iterable):
        raise TypeError(
            "Expected a DataFrame, EventLog, iterable of traces, or variant Counter."
        )
    return Counter(_trace(trace) for trace in log)


def alphabet(obj: IMDataStructureUVCL) -> set[str]:
    return {activity for trace in obj.data_structure for activity in trace}


def project_variants(variants: Variants, activities: set[str]) -> Variants:
    result: Variants = Counter()
    for trace, count in variants.items():
        result[tuple(activity for activity in trace if activity in activities)] += count
    return result
