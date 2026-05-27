from typing import Any, Dict, List, Optional, Tuple, Type

from pm4py.algo.discovery.inductive.cuts.factory import S, T
from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructure,
    IMDataStructureDFG,
    IMDataStructureUVCL,
)

from powl.discovery.total_order_based.inductive.cuts.concurrency import (
    POWLConcurrencyCutDFG,
    POWLConcurrencyCutPOT,
    POWLConcurrencyCutUVCL,
)
from powl.discovery.total_order_based.inductive.cuts.loop import (
    POWLLoopCutDFG,
    POWLLoopCutPOT,
    POWLLoopCutUVCL,
)
from powl.discovery.total_order_based.inductive.cuts.sequence import (
    POWLStrictSequenceCutDFG,
    POWLStrictSequenceCutPOT,
    POWLStrictSequenceCutUVCL,
)
from powl.discovery.total_order_based.inductive.cuts.xor import (
    POWLExclusiveChoiceCutDFG,
    POWLExclusiveChoiceCutPOT,
    POWLExclusiveChoiceCutUVCL,
)
from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT,
)
from powl.discovery.total_order_based.inductive.modeling import InductiveModel


class CutFactory:
    @classmethod
    def get_cuts(
        cls, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Type[S]]:
        if type(obj) is IMDataStructureUVCL:
            return [
                POWLExclusiveChoiceCutUVCL,
                POWLStrictSequenceCutUVCL,
                POWLConcurrencyCutUVCL,
                POWLLoopCutUVCL,
            ]
        elif type(obj) is IMDataStructureDFG:
            return [
                POWLExclusiveChoiceCutDFG,
                POWLStrictSequenceCutDFG,
                POWLConcurrencyCutDFG,
                POWLLoopCutDFG,
            ]
        elif type(obj) is IMDataStructurePOT:
            return [
                POWLExclusiveChoiceCutPOT,
                POWLStrictSequenceCutPOT,
                POWLConcurrencyCutPOT,
                POWLLoopCutPOT,
            ]
        else:
            return []

    @classmethod
    def find_cut(
        cls, obj: IMDataStructure, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[InductiveModel, List[T]]]:
        for c in CutFactory.get_cuts(obj):
            r = c.apply(obj, parameters)
            if r is not None:
                return r
        return None
