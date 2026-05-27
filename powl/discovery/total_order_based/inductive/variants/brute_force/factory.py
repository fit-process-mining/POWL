from typing import Any, Dict, List, Optional, Tuple, Type

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.total_order_based.inductive.cuts.concurrency import (
    POWLConcurrencyCutPOT,
    POWLConcurrencyCutUVCL,
)
from powl.discovery.total_order_based.inductive.cuts.factory import CutFactory, S, T
from powl.discovery.total_order_based.inductive.cuts.loop import (
    POWLLoopCutPOT,
    POWLLoopCutUVCL,
)
from powl.discovery.total_order_based.inductive.cuts.sequence import (
    POWLStrictSequenceCutPOT,
    POWLStrictSequenceCutUVCL,
)
from powl.discovery.total_order_based.inductive.cuts.xor import (
    POWLExclusiveChoiceCutPOT,
    POWLExclusiveChoiceCutUVCL,
)
from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT,
)
from powl.discovery.total_order_based.inductive.variants.brute_force.bf_partial_order_cut import (
    BruteForcePartialOrderCutPOT,
    BruteForcePartialOrderCutUVCL,
)
from powl.discovery.total_order_based.inductive.modeling import InductiveModel


class CutFactoryPOWLBruteForce(CutFactory):
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
                BruteForcePartialOrderCutUVCL,
            ]
        elif type(obj) is IMDataStructurePOT:
            return [
                POWLExclusiveChoiceCutPOT,
                POWLStrictSequenceCutPOT,
                POWLConcurrencyCutPOT,
                POWLLoopCutPOT,
                BruteForcePartialOrderCutPOT,
            ]
        return list()

    @classmethod
    def find_cut(
        cls, obj: IMDataStructureUVCL, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[InductiveModel, List[T]]]:
        for c in CutFactoryPOWLBruteForce.get_cuts(obj):
            r = c.apply(obj, parameters)
            if r is not None:
                return r
        return None
