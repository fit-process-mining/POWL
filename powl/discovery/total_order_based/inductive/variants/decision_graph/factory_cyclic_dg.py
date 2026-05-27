from typing import Any, Dict, List, Optional, Tuple

from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructure,
    IMDataStructureDFG,
    IMDataStructureUVCL,
)
from pm4py.objects.dfg import util as dfu

from powl.discovery.total_order_based.inductive.cuts.concurrency import (
    POWLConcurrencyCutUVCL, POWLConcurrencyCutDFG, POWLConcurrencyCutPOT,
)
from powl.discovery.total_order_based.inductive.cuts.factory import CutFactory, T
from powl.discovery.total_order_based.inductive.cuts.loop import POWLLoopCutUVCL, POWLLoopCutDFG, POWLLoopCutPOT
from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT,
)
from powl.discovery.total_order_based.inductive.variants.decision_graph.cyclic_dg_cut import (
    CyclicDecisionGraphCutUVCL, CyclicDecisionGraphCutDFG, CyclicDecisionGraphCutPOT,
)
from powl.discovery.total_order_based.inductive.variants.maximal.maximal_partial_order_cut import (
    MaximalPartialOrderCutUVCL, MaximalPartialOrderCutDFG, MaximalPartialOrderCutPOT,
)
from powl.discovery.total_order_based.inductive.modeling import InductiveModel


class CutFactoryCyclicDecisionGraph(CutFactory):
    @classmethod
    def get_cuts(cls, obj, parameters=None):

        if type(obj) is IMDataStructureUVCL:
            return [
                CyclicDecisionGraphCutUVCL,
                MaximalPartialOrderCutUVCL,
                POWLConcurrencyCutUVCL,
                POWLLoopCutUVCL,
            ]
        elif type(obj) is IMDataStructureDFG:
            return [
                CyclicDecisionGraphCutDFG,
                MaximalPartialOrderCutDFG,
                POWLConcurrencyCutDFG,
                POWLLoopCutDFG,
            ]
        elif type(obj) is IMDataStructurePOT:
            return [
                CyclicDecisionGraphCutPOT,
                MaximalPartialOrderCutPOT,
                POWLConcurrencyCutPOT,
                POWLLoopCutPOT,
            ]
        else:
            return []

    @classmethod
    def find_cut(
        cls, obj: IMDataStructure, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[InductiveModel, List[T]]]:
        alphabet = sorted(dfu.get_vertices(obj.dfg), key=lambda g: g.__str__())
        if len(alphabet) < 2:
            return None
        for c in CutFactoryCyclicDecisionGraph.get_cuts(obj):
            r = c.apply(obj, parameters)
            if r is not None:
                return r
        return None
