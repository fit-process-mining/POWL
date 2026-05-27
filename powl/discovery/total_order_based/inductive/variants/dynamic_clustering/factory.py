from typing import Any, Dict, List, Optional, Tuple

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructure
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.objects.dfg import util as dfu

from powl.discovery.total_order_based.inductive.cuts.factory import CutFactory, T
from powl.discovery.total_order_based.inductive.cuts.loop import (
    POWLLoopCutPOT,
    POWLLoopCutUVCL,
)
from powl.discovery.total_order_based.inductive.cuts.xor import (
    POWLExclusiveChoiceCutPOT,
    POWLExclusiveChoiceCutUVCL,
)
from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT,
)
from powl.discovery.total_order_based.inductive.variants.dynamic_clustering.dynamic_clustering_partial_order_cut import (
    DynamicClusteringPartialOrderCutPOT,
    DynamicClusteringPartialOrderCutUVCL,
)
from powl.discovery.total_order_based.inductive.modeling import InductiveModel


class CutFactoryPOWLDynamicClustering(CutFactory):
    @classmethod
    def get_cuts(cls, obj, parameters=None):
        if type(obj) is IMDataStructureUVCL:
            return [
                POWLExclusiveChoiceCutUVCL,
                POWLLoopCutUVCL,
                DynamicClusteringPartialOrderCutUVCL,
            ]
        elif type(obj) is IMDataStructurePOT:
            return [
                POWLExclusiveChoiceCutPOT,
                POWLLoopCutPOT,
                DynamicClusteringPartialOrderCutPOT,
            ]
        return []

    @classmethod
    def find_cut(
        cls, obj: IMDataStructure, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[InductiveModel, List[T]]]:
        alphabet = sorted(dfu.get_vertices(obj.dfg), key=lambda g: g.__str__())
        if len(alphabet) < 2:
            return None
        for c in CutFactoryPOWLDynamicClustering.get_cuts(obj):
            r = c.apply(obj, parameters)
            if r is not None:
                return r
        return None
