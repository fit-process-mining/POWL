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

        skippable = [False for i in range(len(groups))]

        do = groups[0]

        dfgs = [DFG() for i in range(len(groups))]
        activity_to_group_id = {}
        for i in range(len(groups)):
            for a in groups[i]:
                activity_to_group_id[a] = i

        for a in dfg.start_activities:
            if a not in do:
                raise Exception("Start activities must be in the do part of the loop!")
            dfgs[0].start_activities[a] = 1

        for a in dfg.end_activities:
            if a not in do:
                raise Exception("End activities must be in the do part of the loop!")
            dfgs[0].end_activities[a] = 1

        for (a, b) in dfg.graph:

            i = activity_to_group_id[a]
            j = activity_to_group_id[b]

            if i == 0 and j == 0:
                dfgs[0].graph[(a, b)] = dfg.graph[(a, b)]
                # No need for this: if we have a case where the do part is followed by another execution of the do part, then we won't be able to split the execution during projection, and therefore, there this will be modeled as a self-loop in the do part
                # if a in dfg.end_activities and b in dfg.start_activities:
                # skippable[1] = True
            elif i > 0 and j > 0:
                if i == j:
                    dfgs[i].graph[(a, b)] = dfg.graph[(a, b)]
                else:
                    raise Exception("Direct edges between different redo groups!")
            elif (i == 0 and j > 0) or (i > 0 and j == 0):
                dfgs[i].end_activities[a] = 1
                dfgs[j].start_activities[b] = 1
            else:
                raise Exception("We should never reach here!")

        return [
            IMDataStructureDFG(InductiveDFG(dfg=dfgs[i], skip=skippable[i]))
            for i in range(len(dfgs))
        ]
