from typing import Any, Dict, List, Optional, Tuple, Type

from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureDFG,
    IMDataStructureUVCL,
)
from pm4py.objects.dfg import util as dfu

from powl.discovery.total_order_based.inductive.cuts.concurrency import (
    POWLConcurrencyCutDFG,
    POWLConcurrencyCutUVCL,
)
from powl.discovery.total_order_based.inductive.cuts.factory import CutFactory, S, T
from powl.discovery.total_order_based.inductive.cuts.loop import (
    POWLLoopCutDFG,
    POWLLoopCutUVCL,
)
from powl.discovery.total_order_based.inductive.cuts.sequence import (
    POWLStrictSequenceCutDFG,
    POWLStrictSequenceCutUVCL,
)
from powl.discovery.total_order_based.inductive.cuts.xor import (
    POWLExclusiveChoiceCutDFG,
    POWLExclusiveChoiceCutUVCL,
)
from powl.discovery.total_order_based.inductive.variants.maximal.maximal_partial_order_cut import (
    MaximalPartialOrderCutDFG,
    MaximalPartialOrderCutUVCL,
)
from powl.discovery.total_order_based.inductive.modeling import InductiveModel


class CutFactoryPOWLMaximal(CutFactory):
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
                MaximalPartialOrderCutUVCL,
            ]
        elif type(obj) is IMDataStructureDFG:
            return [
                POWLExclusiveChoiceCutDFG,
                POWLStrictSequenceCutDFG,
                POWLConcurrencyCutDFG,
                POWLLoopCutDFG,
                MaximalPartialOrderCutDFG,
            ]
        else:
            return []

    @classmethod
    def find_cut(
        cls, obj: IMDataStructureUVCL, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[InductiveModel, List[T]]]:

        alphabet = sorted(dfu.get_vertices(obj.dfg), key=lambda g: g.__str__())
        if len(alphabet) < 2:
            return None
        for c in CutFactoryPOWLMaximal.get_cuts(obj):
            r = c.apply(obj, parameters)
            if r is not None:
                return r
        return None
