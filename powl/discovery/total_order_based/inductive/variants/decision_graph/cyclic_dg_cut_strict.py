from abc import ABC
from itertools import combinations
from typing import Any, Collection, Dict, List, Optional

from pm4py.algo.discovery.inductive.cuts import utils as cut_util
from pm4py.objects.dfg import util as dfu
from pm4py.algo.discovery.inductive.cuts.abc import T
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT,
)
from powl.discovery.total_order_based.inductive.variants.decision_graph.cyclic_dg_cut import (
    CyclicDecisionGraphCut,
    CyclicDecisionGraphCutPOT,
    CyclicDecisionGraphCutUVCL,
)


class StrictCyclicDecisionGraphCut(CyclicDecisionGraphCut[T], ABC):
    @classmethod
    def holds(
        cls, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Any]]:

        dfg = obj.dfg
        alphabet = parameters["alphabet"]
        groups = [frozenset([a]) for a in alphabet]

        def _get_group(activity):
            for g in groups:
                if activity in g:
                    return g
            raise Exception(f"activity {activity} not in {groups}")

        changed = True

        while changed:
            changed = False
            for (a, b), (c, d) in combinations(dfg.graph, 2):
                if {a, b, c, d}.issubset(alphabet):
                    if (
                        _get_group(a) != _get_group(b)
                        and _get_group(a) == _get_group(d)
                        and _get_group(b) == _get_group(c)
                    ):
                        groups = cut_util.merge_groups_based_on_activities(a, b, groups)
                        changed = True

        if len(groups) < 2:
            return None

        return groups


class StrictCyclicDecisionGraphCutUVCL(
    StrictCyclicDecisionGraphCut[IMDataStructureUVCL], ABC
):
    @classmethod
    def project(
        cls,
        obj: IMDataStructureUVCL,
        groups: List[Collection[Any]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[IMDataStructureUVCL]:

        return CyclicDecisionGraphCutUVCL.project(obj, groups, parameters)


class StrictCyclicDecisionGraphCutPOT(
    StrictCyclicDecisionGraphCut[IMDataStructurePOT], ABC
):
    @classmethod
    def project(
        cls,
        obj: IMDataStructurePOT,
        groups: List[Collection[Any]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[IMDataStructurePOT]:
        return CyclicDecisionGraphCutPOT.project(obj, groups, parameters)
