from abc import ABC
from collections.abc import Collection
from typing import Any, Dict, Generic, List, Optional

from pm4py.algo.discovery.inductive.cuts.loop import LoopCut, LoopCutDFG, LoopCutUVCL, T
from pm4py.algo.discovery.inductive.dtypes.im_dfg import InductiveDFG
from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureDFG,
    IMDataStructureUVCL,
)
from pm4py.objects.dfg.obj import DFG

from powl.discovery.total_order_based.inductive.modeling import LoopSpec


class POWLLoopCut(LoopCut, ABC, Generic[T]):
    @classmethod
    def operator(cls, parameters: Optional[Dict[str, Any]] = None) -> LoopSpec:
        return LoopSpec(0)


class POWLLoopCutUVCL(LoopCutUVCL, POWLLoopCut[IMDataStructureUVCL]):
    pass


class POWLLoopCutDFG(LoopCutDFG, POWLLoopCut[IMDataStructureDFG]):
    @classmethod
    def holds(
        cls, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Collection[Any]]]:
        """
        This method finds a loop cut in the dfg.
        Implementation follows function LoopCut on page 190 of
        "Robust Process Mining with Guarantees" by Sander J.J. Leemans (ISBN: 978-90-386-4257-4)

        Basic Steps:
        1. merge all start and end activities in one group ('do' group)
        2. remove start/end activities from the dfg
        3. detect connected components in (undirected representative) of the reduced graph
        4. check if each component meets the start/end criteria of the loop cut definition (merge with the 'do' group if not)
        5. return the cut if at least two groups remain

        """
        dfg = obj.dfg
        start_activities = set(dfg.start_activities.keys())
        end_activities = set(dfg.end_activities.keys())
        if len(dfg.graph) == 0:
            return None

        groups = [start_activities.union(end_activities)]
        for c in cls._compute_connected_components(
            dfg, start_activities, end_activities
        ):
            groups.append(set(c.nodes))

        groups = cls._exclude_sets_non_reachable_from_start(
            dfg, start_activities, end_activities, groups
        )
        groups = cls._exclude_sets_no_reachable_from_end(
            dfg, start_activities, end_activities, groups
        )
        groups = cls._check_start_completeness(
            dfg, start_activities, end_activities, groups
        )
        groups = cls._check_end_completeness(
            dfg, start_activities, end_activities, groups
        )

        groups = list(filter(lambda g: len(g) > 0, groups))

        if len(groups) <= 1:
            return None

        return groups

    @classmethod
    def project(
        cls,
        obj: IMDataStructureDFG,
        groups: List[Collection[Any]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[IMDataStructureDFG]:
        dfg = obj.dfg

        do_group = groups[0]
        dfgs = [DFG() for _ in groups]
        skippable = [False for _ in groups]

        activity_to_group_idx = {}
        for i, group in enumerate(groups):
            for activity in group:
                activity_to_group_idx[activity] = i

        for activity, freq in dfg.start_activities.items():
            if activity not in do_group:
                raise Exception("Start activities must be in the do part of the loop!")
            dfgs[0].start_activities[activity] += freq

        for activity, freq in dfg.end_activities.items():
            if activity not in do_group:
                raise Exception("End activities must be in the do part of the loop!")
            dfgs[0].end_activities[activity] += freq

        for (a, b), freq in dfg.graph.items():
            i = activity_to_group_idx[a]
            j = activity_to_group_idx[b]

            if i == 0 and j == 0:
                dfgs[0].graph[(a, b)] = freq

            elif i > 0 and j > 0:
                if i != j:
                    raise Exception("Direct edges between different redo groups!")
                dfgs[i].graph[(a, b)] = freq

            elif i == 0 and j > 0:
                dfgs[0].end_activities[a] += freq
                dfgs[j].start_activities[b] += freq

            elif i > 0 and j == 0:
                dfgs[i].end_activities[a] += freq
                dfgs[0].start_activities[b] += freq

            else:
                raise Exception("We should never reach here!")

        return [
            IMDataStructureDFG(InductiveDFG(dfg=dfg, skip=skip))
            for dfg, skip in zip(dfgs, skippable)
        ]
