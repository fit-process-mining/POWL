from typing import Any, Dict, List, Optional, Tuple

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructure
from pm4py.objects.dfg import util as dfu

from powl.discovery.total_order_based.inductive.cuts.factory import CutFactory, T
from powl.discovery.total_order_based.inductive.cuts.loop import POWLLoopCutUVCL
from powl.discovery.total_order_based.inductive.variants.decision_graph.max_decision_graph_cut import (
    MaximalDecisionGraphCutUVCL,
)
from powl.discovery.total_order_based.inductive.variants.dynamic_clustering.dynamic_clustering_partial_order_cut import (
    DynamicClusteringPartialOrderCutUVCL,
)
from powl.discovery.total_order_based.inductive.modeling import InductiveModel


class CutFactoryPOWLDecisionGraphClustering(CutFactory):
    @classmethod
    def get_cuts(cls, obj, parameters=None):
        return [
            MaximalDecisionGraphCutUVCL,
            POWLLoopCutUVCL,
            DynamicClusteringPartialOrderCutUVCL,
        ]

    @classmethod
    def find_cut(
        cls, obj: IMDataStructure, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[InductiveModel, List[T]]]:
        alphabet = sorted(dfu.get_vertices(obj.dfg), key=lambda g: g.__str__())
        if len(alphabet) < 2:
            return None
        for c in CutFactoryPOWLDecisionGraphClustering.get_cuts(obj):
            r = c.apply(obj, parameters)
            if r is not None:
                return r
        return None
