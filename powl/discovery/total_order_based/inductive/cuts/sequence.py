from abc import ABC
from collections import Counter
from typing import Any, Collection, Dict, Generic, List, Optional, Tuple

from pm4py.algo.discovery.inductive.base_case.abc import T
from pm4py.algo.discovery.inductive.cuts.sequence import (
    SequenceCut,
    SequenceCutDFG,
    SequenceCutUVCL,
    StrictSequenceCut,
    StrictSequenceCutDFG,
    StrictSequenceCutUVCL,
)
from pm4py.algo.discovery.inductive.dtypes.im_dfg import InductiveDFG
from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureDFG,
    IMDataStructureUVCL,
)
from pm4py.objects.dfg import util as dfu
from pm4py.objects.dfg.obj import DFG

from powl.discovery.total_order_based.inductive.modeling import SequenceSpec


class POWLSequenceCut(SequenceCut, ABC, Generic[T]):
    @classmethod
    def operator(cls, parameters: Optional[Dict[str, Any]] = None) -> SequenceSpec:
        return SequenceSpec(0)

    @classmethod
    def apply(
        cls, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[SequenceSpec, List[T]]]:
        g = cls.holds(obj, parameters)
        if g is None:
            return g
        children = cls.project(obj, g, parameters)
        return SequenceSpec(len(children)), children


class POWLStrictSequenceCut(POWLSequenceCut[T], StrictSequenceCut, ABC):
    pass


class POWLSequenceCutUVCL(SequenceCutUVCL, POWLSequenceCut[IMDataStructureUVCL]):
    pass


class POWLStrictSequenceCutUVCL(
    StrictSequenceCutUVCL, StrictSequenceCut[IMDataStructureUVCL], POWLSequenceCutUVCL
):
    pass


class POWLSequenceCutDFG(SequenceCutDFG, POWLSequenceCut[IMDataStructureDFG]):
    @classmethod
    def project(
        cls,
        obj: IMDataStructureDFG,
        groups: List[Collection[Any]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[IMDataStructureDFG]:
        dfg = obj.dfg
        start_activities = []
        end_activities = []
        activities = []
        dfgs = []
        skippable = []
        for g in groups:
            skippable.append(False)
        activities_idx = {}
        for gind, g in enumerate(groups):
            for act in g:
                activities_idx[act] = int(gind)
        i = 0
        while i < len(groups):
            to_succ_arcs = Counter()
            from_prev_arcs = Counter()
            if i < len(groups) - 1:
                for (a, b) in dfg.graph:
                    if a in groups[i] and b in groups[i + 1]:
                        to_succ_arcs[a] += dfg.graph[(a, b)]

            if i > 0:
                for (a, b) in dfg.graph:
                    if a in groups[i - 1] and b in groups[i]:
                        from_prev_arcs[b] += dfg.graph[(a, b)]

            if i == 0:
                start_activities.append({})
                for a in dfg.start_activities:
                    if a in groups[i]:
                        start_activities[i][a] = dfg.start_activities[a]
                    else:
                        j = i
                        while j < activities_idx[a]:
                            skippable[j] = True
                            j = j + 1
            else:
                start_activities.append(from_prev_arcs)

            if i == len(groups) - 1:
                end_activities.append({})
                for a in dfg.end_activities:
                    if a in groups[i]:
                        end_activities[i][a] = dfg.end_activities[a]
                    else:
                        j = activities_idx[a] + 1
                        while j <= i:
                            skippable[j] = True
                            j = j + 1
            else:
                end_activities.append(to_succ_arcs)

            activities.append({})
            act_count = dfu.get_vertex_frequencies(dfg)
            for a in groups[i]:
                activities[i][a] = act_count[a]
            dfgs.append({})
            for (a, b) in dfg.graph:
                if a in groups[i] and b in groups[i]:
                    dfgs[i][(a, b)] = dfg.graph[(a, b)]
            i = i + 1
        i = 0
        while i < len(dfgs):
            dfi = DFG()
            [dfi.graph.update({(a, b): dfgs[i][(a, b)]}) for (a, b) in dfgs[i]]
            [
                dfi.start_activities.update({a: start_activities[i][a]})
                for a in start_activities[i]
            ]
            [
                dfi.end_activities.update({a: end_activities[i][a]})
                for a in end_activities[i]
            ]
            dfgs[i] = dfi
            i = i + 1
        for (a, b) in dfg.graph:
            z = activities_idx[b]
            j = activities_idx[a] + 1
            while j < z:
                skippable[j] = True
                j = j + 1

        return [
            IMDataStructureDFG(InductiveDFG(dfg=dfgs[i], skip=skippable[i]))
            for i in range(len(dfgs))
        ]


class POWLStrictSequenceCutDFG(
    StrictSequenceCutDFG, StrictSequenceCut[IMDataStructureDFG], POWLSequenceCutDFG
):
    pass
