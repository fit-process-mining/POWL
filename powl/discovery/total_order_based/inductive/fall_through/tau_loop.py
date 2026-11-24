from collections import Counter
from typing import Any, Dict, Optional

from pm4py.algo.discovery.inductive.fall_through.tau_loop import TauLoopUVCL
from pm4py.util.compression import util as comut
from pm4py.util.compression.dtypes import UVCL

from powl.discovery.total_order_based.inductive.fall_through.strict_tau_loop import (
    POWLStrictTauLoopPOT,
    POWLStrictTauLoopUVCL,
)


class POWLTauLoopUVCL(POWLStrictTauLoopUVCL, TauLoopUVCL):
    @classmethod
    def _get_projected_log(
        cls, log: UVCL, parameters: Optional[Dict[str, Any]] = None
    ) -> UVCL:
        start_activities = comut.get_start_activities(log)
        proj = Counter()
        for t in log:
            x = 0
            for i in range(1, len(t)):
                if t[i] in start_activities:
                    proj.update({t[x:i]: log[t]})
                    x = i
            proj.update({t[x : len(t)]: log[t]})
        return proj


class POWLTauLoopPOT(POWLStrictTauLoopPOT):
    boundary_mode = "any_to_start"
