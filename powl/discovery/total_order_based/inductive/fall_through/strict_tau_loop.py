from collections import Counter
from multiprocessing import Manager, Pool
from typing import Any, Dict, List, Optional, Tuple

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.algo.discovery.inductive.fall_through.strict_tau_loop import (
    StrictTauLoopUVCL,
)

from powl.discovery.total_order_based.inductive.modeling import LoopSpec


class POWLStrictTauLoopUVCL(StrictTauLoopUVCL):
    @classmethod
    def apply(
        cls,
        obj: IMDataStructureUVCL,
        pool: Pool = None,
        manager: Manager = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[LoopSpec, List[IMDataStructureUVCL]]]:
        log = obj.data_structure
        proj = cls._get_projected_log(log)
        if sum(proj.values()) > sum(log.values()):
            return LoopSpec(2), [
                IMDataStructureUVCL(proj),
                IMDataStructureUVCL(Counter()),
            ]
