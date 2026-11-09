import warnings
from bisect import bisect_left
from collections import defaultdict
from typing import Any, List, Optional, Set

import pandas as pd

from powl.discovery.partial_order_based.utils.constants import VARIANT_FREQUENCY_KEY
from powl.discovery.partial_order_based.utils.simplified_objects import (
    ActivityInstance,
    Graph,
)
from powl.general_utils.time_utils import should_parse_column_as_date


def generate_interval_df_fifo(
    df: pd.DataFrame,
    case_id_col: str,
    activity_col: str,
    ordering_col: str,
    lifecycle_col: Optional[str],
    start_transitions: Set[str],
    complete_transitions: Set[str],
) -> pd.DataFrame:

    cols_to_keep = [case_id_col, activity_col, ordering_col]
    if lifecycle_col:
        cols_to_keep.append(lifecycle_col)
    df_filtered = df[cols_to_keep].copy()

    if lifecycle_col:
        df_filtered = df_filtered.sort_values([case_id_col, ordering_col])

        activity_col_idx = df_filtered.columns.get_loc(activity_col)
        timestamp_col_idx = df_filtered.columns.get_loc(ordering_col)
        lifecycle_col_idx = df_filtered.columns.get_loc(lifecycle_col)

        all_intervals = []
        grouped_by_case = df_filtered.groupby(case_id_col, sort=False)

        for case_id, case_df in grouped_by_case:
            pending_starts = defaultdict(list)
            activity_instance_counter = defaultdict(int)

            for event_tuple in case_df.itertuples(index=False, name=None):
                activity_name = event_tuple[activity_col_idx]
                timestamp = event_tuple[timestamp_col_idx]
                lifecycle = event_tuple[lifecycle_col_idx]

                if lifecycle in start_transitions:
                    pending_starts[activity_name].append(timestamp)

                elif lifecycle in complete_transitions:
                    if pending_starts[activity_name]:
                        start_timestamp = pending_starts[activity_name].pop(0)
                    else:
                        start_timestamp = timestamp
                    activity_instance_counter[activity_name] += 1
                    all_intervals.append(
                        {
                            case_id_col: case_id,
                            "activity": ActivityInstance(
                                activity_name, activity_instance_counter[activity_name]
                            ),
                            "start_timestamp": start_timestamp,
                            "end_timestamp": timestamp,
                        }
                    )

        interval_df = pd.DataFrame(all_intervals) if all_intervals else pd.DataFrame()

    else:
        # No lifecycle: each event is an atomic interval
        df_filtered = df_filtered.sort_values([case_id_col, ordering_col])
        group_cols = [case_id_col, activity_col]
        df_filtered["instance_id"] = df_filtered.groupby(group_cols).cumcount()

        df_filtered["activity"] = [
            ActivityInstance(act, inst + 1)
            for act, inst in zip(df_filtered[activity_col], df_filtered["instance_id"])
        ]
        df_filtered["start_timestamp"] = df_filtered[ordering_col]
        df_filtered["end_timestamp"] = df_filtered[ordering_col]

        interval_df = df_filtered[
            [case_id_col, "activity", "start_timestamp", "end_timestamp"]
        ]

    if interval_df.empty:
        return pd.DataFrame()

    # Sort for consistent output and to aid the next processing step
    interval_df = interval_df.sort_values(
        [case_id_col, "start_timestamp", "end_timestamp"]
    ).reset_index(drop=True)
    interval_df["event_instance_id"] = interval_df.index
    print(
        f"Successfully created {len(interval_df)} activity intervals using FIFO logic."
    )
    return interval_df


def apply(
    df: pd.DataFrame,
    case_id_col: str,
    activity_col: str,
    ordering_col: str,
    lifecycle_col: Optional[str],
    start_transitions: Set[str],
    complete_transitions: Set[str],
) -> List[Any]:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'log' must be a Pandas DataFrame.")

    for col in [activity_col, ordering_col, case_id_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the event table!")

    if lifecycle_col:
        if lifecycle_col in df.columns:
            complete_log = df[df[lifecycle_col].isin(complete_transitions)]
            if len(complete_log) == 0:
                lifecycle_col = None
                warnings.warn(
                    message=f"The event log does not contain any completion events! All events will be considered during discovery!",
                    category=UserWarning,
                )
        else:
            missing_column = lifecycle_col
            lifecycle_col = None
            warnings.warn(
                f'The event log does not contain a lifecycle column {missing_column!r}; '
                'all events will be considered during discovery.',
                category=UserWarning,
            )

    if should_parse_column_as_date(df, ordering_col):
        df = df.copy()
        df[ordering_col] = pd.to_datetime(df[ordering_col])

    interval_df = generate_interval_df_fifo(
        df,
        case_id_col,
        activity_col,
        ordering_col,
        lifecycle_col,
        start_transitions,
        complete_transitions,
    )

    if interval_df.empty:
        raise Exception("Interval DataFrame is empty, no variants to generate.")

    variants_key_to_frequency = defaultdict(int)
    interval_df = interval_df.sort_values(
        [case_id_col, "start_timestamp", "end_timestamp"]
    ).reset_index(drop=True)
    grouped_intervals = interval_df.groupby(case_id_col, sort=False)

    for case_id, trace_df in grouped_intervals:
        trace_activities_multiset = frozenset(trace_df["activity"].tolist())

        trace_events = list(
            trace_df[["activity", "start_timestamp", "end_timestamp"]].itertuples(
                index=False, name=None
            )
        )
        events_sorted_by_end = sorted(trace_events, key=lambda x: x[2])
        end_timestamps = [event[2] for event in events_sorted_by_end]

        edges = []
        for act_j, start_j, _ in trace_events:
            pos = bisect_left(end_timestamps, start_j)
            for i in range(pos):
                act_i = events_sorted_by_end[i][0]
                if act_i != act_j:
                    edges.append((act_i, act_j))

        trace_edges = frozenset(edges)
        variant_key = (trace_activities_multiset, trace_edges)
        variants_key_to_frequency[variant_key] += 1

    print(f"Found {len(variants_key_to_frequency)} unique variants.")

    output_list = [
        Graph(variant_key[0], variant_key[1], {VARIANT_FREQUENCY_KEY: freq})
        for variant_key, freq in variants_key_to_frequency.items()
    ]

    output_list.sort(
        key=lambda x: x.additional_information[VARIANT_FREQUENCY_KEY], reverse=True
    )

    return output_list
