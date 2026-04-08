from typing import Any, Dict, List, Optional, Tuple

from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructure,
    IMDataStructureDFG,
    IMDataStructureUVCL,
)
from pm4py.objects.dfg import util as dfu

from powl.discovery.total_order_based.inductive.cuts.concurrency import (
    POWLConcurrencyCutDFG,
)
from powl.discovery.total_order_based.inductive.cuts.factory import CutFactory, T
from powl.discovery.total_order_based.inductive.cuts.loop import POWLLoopCutUVCL
from powl.discovery.total_order_based.inductive.cuts.sequence import (
    POWLStrictSequenceCutDFG,
)
from powl.discovery.total_order_based.inductive.cuts.xor import (
    POWLExclusiveChoiceCutDFG,
    POWLExclusiveChoiceCutUVCL,
)
from powl.discovery.total_order_based.inductive.variants.dynamic_clustering_frequency.dynamic_clustering_frequency_partial_order_cut import (
    DynamicClusteringFrequencyPartialOrderCutDFG,
    DynamicClusteringFrequencyPartialOrderCutUVCL,
)
from powl.discovery.total_order_based.inductive.modeling import InductiveModel


class CutFactoryPOWLDynamicClusteringFrequency(CutFactory):
    @classmethod
    def get_cuts(cls, obj, parameters=None):
        if type(obj) is IMDataStructureUVCL:
            return [
                POWLExclusiveChoiceCutUVCL,
                POWLLoopCutUVCL,
                DynamicClusteringFrequencyPartialOrderCutUVCL,
            ]
        elif type(obj) is IMDataStructureDFG:
            return [
                POWLExclusiveChoiceCutDFG,
                POWLStrictSequenceCutDFG,
                POWLConcurrencyCutDFG,
                POWLLoopCutUVCL,
                DynamicClusteringFrequencyPartialOrderCutDFG,
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
        for c in CutFactoryPOWLDynamicClusteringFrequency.get_cuts(obj):
            r = c.apply(obj, parameters)
            if r is not None:
                return r
        return None
