from collections import Counter
from enum import auto, Enum

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT,
    PartialOrderTrace,
)


class FilteringType(Enum):
    DYNAMIC = auto()
    DECREASING_FACTOR = auto()
    DFG_FREQUENCY = auto()


DEFAULT_FILTERING_TYPE = FilteringType.DECREASING_FACTOR
FILTERING_THRESHOLD = "weight_factor_filtering_threshold"
FILTERING_TYPE = "filtering_type"


def _wrap_filtered_log_like(original_log, filtered_log):
    if any(isinstance(variant, PartialOrderTrace) for variant in original_log):
        return IMDataStructurePOT(filtered_log)
    return IMDataStructureUVCL(filtered_log)


def filter_most_frequent_variants(log):
    to_remove_freq = min([freq for var, freq in log.items()])
    new_log = Counter()
    for var, freq in log.items():
        if freq == to_remove_freq:
            continue
        new_log[var] = freq

    return _wrap_filtered_log_like(log, new_log)


def filter_most_frequent_variants_with_decreasing_factor(log, decreasing_factor):
    sorted_variants = sorted(log, key=log.get, reverse=True)
    new_log = Counter()

    already_added_sum = 0
    prev_var_count = -1

    for variant in sorted_variants:
        frequency = log[variant]
        if already_added_sum == 0 or frequency > decreasing_factor * prev_var_count:
            new_log[variant] = frequency
            already_added_sum = already_added_sum + frequency
            prev_var_count = frequency
        else:
            break

    return _wrap_filtered_log_like(log, new_log)
