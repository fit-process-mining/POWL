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

        base_dfg = obj.dfg
        dfg_map = {group: DFG() for group in groups}

        activity_to_group_map = {}
        for group in groups:
            for activity in group:
                activity_to_group_map[activity] = group

        for (a, b) in base_dfg.graph:
            group_a = activity_to_group_map[a]
            group_b = activity_to_group_map[b]
            freq = base_dfg.graph[(a, b)]
            if group_a == group_b:
                dfg_map[group_a].graph[(a, b)] = freq
            else:
                dfg_map[group_a].end_activities[a] += freq
                dfg_map[group_b].start_activities[b] += freq
        for a in base_dfg.start_activities:
            group_a = activity_to_group_map[a]
            dfg_map[group_a].start_activities[a] += base_dfg.start_activities[a]
        for a in base_dfg.end_activities:
            group_a = activity_to_group_map[a]
            dfg_map[group_a].end_activities[a] += base_dfg.end_activities[a]

        return list(
            map(
                lambda g: IMDataStructureDFG(InductiveDFG(dfg=dfg_map[g], skip=False)),
                groups,
            )
        )
