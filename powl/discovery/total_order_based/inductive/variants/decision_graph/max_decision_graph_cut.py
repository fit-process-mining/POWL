from abc import ABC
from collections import Counter
from itertools import combinations
from typing import Any, Collection, Dict, Generic, List, Optional, Tuple

from pm4py.algo.discovery.inductive.cuts import utils as cut_util

from pm4py.algo.discovery.inductive.cuts.abc import Cut, T
from pm4py.algo.discovery.inductive.dtypes.im_dfg import InductiveDFG
from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureDFG,
    IMDataStructureUVCL,
)
from pm4py.objects.dfg import util as dfu
from pm4py.objects.dfg.obj import DFG

from powl.discovery.total_order_based.inductive.modeling import ChoiceGraphSpec


class MaximalDecisionGraphCut(Cut[T], ABC, Generic[T]):
    @classmethod
    def operator(cls, parameters: Optional[Dict[str, Any]] = None):
        return None

    @classmethod
    def holds(
        cls, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Any]]:

        alphabet = parameters["alphabet"]
        transitive_successors = parameters["transitive_successors"]

        groups = [frozenset([a]) for a in alphabet]

        for a, b in combinations(alphabet, 2):
            if b in transitive_successors[a] and a in transitive_successors[b]:
                groups = cut_util.merge_groups_based_on_activities(a, b, groups)


        if len(groups) < 2:
            return None

        return groups

    @classmethod
    def apply(
        cls, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[ChoiceGraphSpec, List[T]]]:

        dfg = obj.dfg

        start_acts = set(obj.dfg.start_activities.keys())
        end_acts = set(obj.dfg.end_activities.keys())

        alphabet = sorted(dfu.get_vertices(dfg), key=lambda g: g.__str__())
        transitive_predecessors, transitive_successors = dfu.get_transitive_relations(
            dfg
        )

        parameters["alphabet"] = alphabet
        parameters["transitive_predecessors"] = transitive_predecessors
        parameters["transitive_successors"] = transitive_successors

        groups = cls.holds(obj, parameters)
        if groups is None:
            return groups

        # cache start/end flags per group
        group_has_start = [any(a in start_acts for a in g) for g in groups]
        group_has_end = [any(a in end_acts for a in g) for g in groups]

        # cache group->group connectivity (directed)
        n = len(groups)
        group_conn = [[False] * n for _ in range(n)]

        # efficient: build act->group index
        act_to_group = {}
        for i, g in enumerate(groups):
            for a in g:
                act_to_group[a] = i

        for (a, b), freq in dfg.graph.items():
            if freq <= 0:
                continue
            ga = act_to_group.get(a)
            gb = act_to_group.get(b)
            if ga != gb:
                group_conn[ga][gb] = True

        parameters["_mdgc_act_to_group"] = act_to_group
        parameters["_mdgc_group_has_start"] = group_has_start
        parameters["_mdgc_group_has_end"] = group_has_end
        parameters["_mdgc_group_conn"] = group_conn

        children = cls.project(obj, groups, parameters)

        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and group_conn[i][j]:
                    edges.append((i, j))

        start_nodes = tuple(i for i in range(n) if group_has_start[i])
        end_nodes = tuple(i for i in range(n) if group_has_end[i])

        return ChoiceGraphSpec(
            size=len(children),
            edges=tuple(edges),
            start_nodes=start_nodes,
            end_nodes=end_nodes,
        ), children


class MaximalDecisionGraphCutUVCL(MaximalDecisionGraphCut[IMDataStructureUVCL], ABC):
    @classmethod
    def project(
        cls,
        obj: IMDataStructureUVCL,
        groups: List[Collection[Any]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[IMDataStructureUVCL]:

        logs = [Counter() for _ in groups]
        for t, freq in obj.data_structure.items():
            for i, group in enumerate(groups):
                seg = []
                for e in t:
                    if e in group:
                        seg.append(e)
                if len(seg) > 0:
                    logs[i][tuple(seg)] += freq

        return [IMDataStructureUVCL(l) for l in logs]


class MaximalDecisionGraphCutDFG(MaximalDecisionGraphCut[IMDataStructureDFG], ABC):
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
