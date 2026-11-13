from typing import Any, Dict, List, Optional, Tuple, Type

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.total_order_based.inductive.dtypes.partial_order import IMDataStructurePOT
from powl.discovery.total_order_based.inductive.cuts.concurrency import (
    POWLConcurrencyCutUVCL,
    POWLConcurrencyCutPOT,
)
from powl.discovery.total_order_based.inductive.cuts.factory import CutFactory, S, T
from powl.discovery.total_order_based.inductive.cuts.loop import (
    POWLLoopCutUVCL,
    POWLLoopCutPOT,
)
from powl.discovery.total_order_based.inductive.cuts.sequence import (
    POWLStrictSequenceCutUVCL,
    POWLStrictSequenceCutPOT,
)
from powl.discovery.total_order_based.inductive.cuts.xor import (
    POWLExclusiveChoiceCutUVCL,
    POWLExclusiveChoiceCutPOT,
)
from powl.discovery.total_order_based.inductive.variants.brute_force.bf_partial_order_cut import (
    BruteForcePartialOrderCutUVCL,
    BruteForcePartialOrderCutPOT,
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
