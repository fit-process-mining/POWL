from abc import ABC
from collections import Counter
from itertools import combinations
from typing import Any, Collection, Dict, List, Optional

from pm4py.algo.discovery.inductive.cuts import utils as cut_util
from pm4py.algo.discovery.inductive.cuts.abc import T
from pm4py.algo.discovery.inductive.dtypes.im_dfg import InductiveDFG
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL, IMDataStructureDFG
from pm4py.objects.dfg.obj import DFG
from pm4py.util import exec_utils
from pm4py.algo.discovery.inductive.variants.imf import IMFParameters

from powl.discovery.total_order_based.inductive.utils.filtering import FILTERING_TYPE, FilteringType
from powl.discovery.total_order_based.inductive.variants.decision_graph.max_decision_graph_cut import (
    MaximalDecisionGraphCut
)
from powl.discovery.total_order_based.inductive.variants.maximal.maximal_partial_order_cut import \
    MaximalPartialOrderCutDFG


class CyclicDecisionGraphCut(MaximalDecisionGraphCut[T], ABC):
    @classmethod
    def holds(
        cls, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Any]]:

        dfg = obj.dfg
        alphabet = parameters["alphabet"]

        groups = [frozenset([a]) for a in alphabet]

        for a, b in combinations(alphabet, 2):
            if (a, b) in dfg.graph and (b, a) in dfg.graph:
                groups = cut_util.merge_groups_based_on_activities(a, b, groups)

        if len(groups) < 2:
            return None

        return groups


class CyclicDecisionGraphCutUVCL(CyclicDecisionGraphCut[IMDataStructureUVCL], ABC):
    @classmethod
    def project(
        cls,
        obj: IMDataStructureUVCL,
        groups: List[Collection[Any]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[IMDataStructureUVCL]:

        filtering = False
        if FILTERING_TYPE in parameters.keys():
            filtering_type = parameters[FILTERING_TYPE]
            if filtering_type is FilteringType.DFG_FREQUENCY:
                noise_threshold = exec_utils.get_param_value(
                    IMFParameters.NOISE_THRESHOLD, parameters, 0.0
                )
                if noise_threshold > 0.0:
                    filtering = True

        logs = [Counter() for _ in groups]
        for t, freq in obj.data_structure.items():
            for i, group in enumerate(groups):
                seg = []
                last = None
                for e in t:
                    if e in group:
                        if len(seg) > 0 and last not in group:
                            if not filtering or obj.dfg.graph.get((last, e), 0) > obj.dfg.graph.get((seg[-1], e), 0):
                                logs[i][tuple(seg)] += freq
                                seg = []
                        seg.append(e)
                    last = e
                if len(seg) > 0:
                    logs[i][tuple(seg)] += freq

        return [IMDataStructureUVCL(l) for l in logs]


class CyclicDecisionGraphCutDFG(CyclicDecisionGraphCut[IMDataStructureDFG], ABC):
    @classmethod
    def project(
            cls,
            obj: IMDataStructureDFG,
            groups: List[Collection[Any]],
            parameters: Optional[Dict[str, Any]] = None,
    ) -> List[IMDataStructureDFG]:
        return MaximalPartialOrderCutDFG.project(obj, groups, parameters=parameters)
