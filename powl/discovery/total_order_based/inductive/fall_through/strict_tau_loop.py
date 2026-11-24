from collections import Counter
from multiprocessing import Manager, Pool
from typing import Any, Dict, List, Optional, Tuple

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.algo.discovery.inductive.fall_through.strict_tau_loop import (
    FallThrough,
    StrictTauLoopUVCL,
)

from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    BoundaryMode, IMDataStructurePOT, split_project_pot_by_start_boundaries,
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


class POWLStrictTauLoopPOT(FallThrough[IMDataStructurePOT]):
    boundary_mode: BoundaryMode = "end_to_start"

    @classmethod
    def _get_projected_log(
        cls, im_data_struct: IMDataStructurePOT, parameters: Optional[Dict[str, Any]] = None
    ) -> IMDataStructurePOT:
        return split_project_pot_by_start_boundaries(
            im_data_struct.data_structure, cls.boundary_mode,
            im_data_struct.dfg.start_activities, im_data_struct.dfg.end_activities,
        )

    @classmethod
    def holds(
        cls,
        obj: IMDataStructurePOT,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        # A DFG edge witnesses consecutive events in some linear extension.
        # Counting POT alternatives would mistake refinements for new iterations.
        dfg = obj.dfg
        return any(b in dfg.start_activities and
                   (cls.boundary_mode == "any_to_start" or a in dfg.end_activities)
                   for a, b in obj.unfiltered_dfg.graph)

    @classmethod
    def apply(
        cls,
        obj: IMDataStructurePOT,
        pool=None,
        manager=None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[LoopSpec, List[IMDataStructurePOT]]]:
        if cls.holds(obj, parameters):
            return LoopSpec(2), [cls._get_projected_log(obj), IMDataStructurePOT(Counter())]
        return None
