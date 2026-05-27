from collections import Counter
from multiprocessing import Manager, Pool
from typing import Any, Dict, List, Optional, Tuple

from pm4py.algo.discovery.inductive.dtypes.im_dfg import InductiveDFG

from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureDFG,
    IMDataStructureUVCL,
)
from pm4py.algo.discovery.inductive.fall_through.abc import FallThrough
from pm4py.algo.discovery.inductive.fall_through.flower import (
    FlowerModelDFG,
    FlowerModelUVCL,
)
from pm4py.objects.dfg.obj import DFG
from pm4py.util.compression import util as comut
from pm4py.util.compression.dtypes import UVCL

from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT,
    PartialOrderTrace,
)
from powl.discovery.total_order_based.inductive.modeling import LoopSpec


class POWLFlowerModelUVCL(FlowerModelUVCL):
    @classmethod
    def apply(
        cls,
        obj: IMDataStructureUVCL,
        pool: Pool = None,
        manager: Manager = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[LoopSpec, List[IMDataStructureUVCL]]]:
        log = obj.data_structure
        uvcl_redo = UVCL()
        for a in sorted(list(comut.get_alphabet(log))):
            uvcl_redo[(a,)] = 1
        uvcl_do = UVCL()
        im_uvcl_do = IMDataStructureUVCL(uvcl_do)
        im_uvcl_redo = IMDataStructureUVCL(uvcl_redo)
        return LoopSpec(2), [im_uvcl_do, im_uvcl_redo]


class POWLFlowerModelDFG(FlowerModelDFG):
    @classmethod
    def apply(
        cls,
        obj: IMDataStructureDFG,
        pool: Pool = None,
        manager: Manager = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[LoopSpec, List[IMDataStructureDFG]]]:
        activities = (
            set(obj.dfg.start_activities)
            .union(set(obj.dfg.end_activities))
            .union(set(x[0] for x in obj.dfg.graph))
            .union(set(x[1] for x in obj.dfg.graph))
        )
        dfg_redo = DFG()
        for a in activities:
            dfg_redo.start_activities[a] = 1
            dfg_redo.end_activities[a] = 1
        dfg_do = DFG()
        im_dfg_do = IMDataStructureDFG(InductiveDFG(dfg_do))
        im_dfg_redo = IMDataStructureDFG(InductiveDFG(dfg_redo))
        return LoopSpec(2), [im_dfg_do, im_dfg_redo]


class POWLFlowerModelPOT(FallThrough[IMDataStructurePOT]):
    @classmethod
    def holds(
        cls,
        obj: IMDataStructurePOT,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return not any(len(t) == 0 for t in obj.data_structure)

    @classmethod
    def apply(
        cls,
        obj: IMDataStructurePOT,
        pool: Pool = None,
        manager: Manager = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[LoopSpec, List[IMDataStructurePOT]]]:
        redo_log = Counter()
        activities = (
            set(obj.dfg.start_activities)
            .union(set(obj.dfg.end_activities))
            .union(set(x[0] for x in obj.dfg.graph))
            .union(set(x[1] for x in obj.dfg.graph))
        )
        for activity in sorted(activities, key=lambda a: a.__str__()):
            redo_log[PartialOrderTrace((activity,), frozenset())] = 1
        return LoopSpec(2), [IMDataStructurePOT(Counter()), IMDataStructurePOT(redo_log)]
