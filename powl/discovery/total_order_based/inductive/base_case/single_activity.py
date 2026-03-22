from typing import Any, Dict, Optional

from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureDFG,
    IMDataStructureUVCL,
)

from powl.discovery.total_order_based.inductive.base_case.abc import BaseCase

from powl.objects.tagged_powl.activity import Activity


class SingleActivityBaseCaseUVCL(BaseCase[IMDataStructureUVCL]):
    @classmethod
    def holds(
        cls, obj=IMDataStructureUVCL, parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        if len(obj.data_structure.keys()) != 1:
            return False
        if len(list(obj.data_structure.keys())[0]) > 1:
            return False
        return True

    @classmethod
    def leaf(
        cls, obj=IMDataStructureUVCL, parameters: Optional[Dict[str, Any]] = None
    ) -> Activity:
        for t in obj.data_structure:
            if t:
                return Activity(label=t[0])
            else:
                return Activity(label=None)


class SingleActivityBaseCaseDFG(BaseCase[IMDataStructureDFG]):
    @classmethod
    def holds(
        cls,
        obj=IMDataStructureDFG,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if len(obj.dfg.graph) == 0:
            if set(obj.dfg.start_activities) == set(obj.dfg.end_activities):
                return len(obj.dfg.start_activities) == 1
            else:
                raise Exception(
                    "Invalid DFG: non-start/end activities are not involved in any edges!"
                )
        return False

    @classmethod
    def leaf(
        cls,
        obj=IMDataStructureDFG,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Activity:
        return Activity(label=list(obj.dfg.start_activities)[0])
