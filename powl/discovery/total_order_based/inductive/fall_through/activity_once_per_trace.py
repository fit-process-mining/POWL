from collections import Counter
from typing import Any, Optional

from pm4py.algo.discovery.inductive.fall_through.activity_once_per_trace import (
    ActivityOncePerTraceUVCL,
)

from powl.discovery.total_order_based.inductive.dtypes.partial_order import IMDataStructurePOT
from powl.discovery.total_order_based.inductive.fall_through.activity_concurrent import (
    POWLActivityConcurrentPOT,
    POWLActivityConcurrentUVCL,
)


class POWLActivityOncePerTraceUVCL(
    ActivityOncePerTraceUVCL, POWLActivityConcurrentUVCL
):
    pass
class POWLActivityOncePerTracePOT(POWLActivityConcurrentPOT):
    @classmethod
    def _get_candidate(
        cls,
        obj: IMDataStructurePOT,
        pool=None,
        manager=None,
        parameters = None,
    ) -> Optional[Any]:
        candidates = set()
        first_trace = True

        for trace in obj.data_structure:
            activity_counts = Counter(trace.activities)
            activities_once = {
                activity
                for activity, count in activity_counts.items()
                if count == 1
            }
            if first_trace:
                candidates = activities_once
                first_trace = False
            else:
                candidates &= activities_once
            if not candidates:
                return None

        return min(candidates) if candidates else None
