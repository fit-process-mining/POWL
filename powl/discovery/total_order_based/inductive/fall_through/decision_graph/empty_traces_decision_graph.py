from copy import copy
from multiprocessing import Manager, Pool
from typing import Any, Dict, List, Optional, Tuple

from pm4py.algo.discovery.inductive.dtypes.im_dfg import InductiveDFG

from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureDFG,
    IMDataStructureUVCL,
)
from pm4py.algo.discovery.inductive.fall_through.empty_traces import (
    EmptyTracesDFG,
    EmptyTracesUVCL,
)

from powl.discovery.total_order_based.inductive.modeling import ChoiceGraphSpec


class POWLEmptyTracesDecisionGraphUVCL(EmptyTracesUVCL):
    @classmethod
    def apply(
        cls,
        obj: IMDataStructureUVCL,
        pool: Pool = None,
        manager: Manager = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[ChoiceGraphSpec, List[IMDataStructureUVCL]]]:
        if cls.holds(obj, parameters):
            data_structure = copy(obj.data_structure)
            del data_structure[()]
            children = [IMDataStructureUVCL(data_structure)]
            return ChoiceGraphSpec(
                size=1,
                start_nodes=(0,),
                end_nodes=(0,),
                min_freq=0,
                max_freq=1,
            ), children
        else:
            return None


class POWLEmptyTracesDecisionGraphDFG(EmptyTracesDFG):
    @classmethod
    def apply(
        cls,
        obj: IMDataStructureDFG,
        pool=None,
        manager=None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[ChoiceGraphSpec, List[IMDataStructureDFG]]]:
        if cls.holds(obj, parameters):
            children = [IMDataStructureDFG(InductiveDFG(obj.data_structure.dfg))]
            return ChoiceGraphSpec(
                size=1,
                start_nodes=(0,),
                end_nodes=(0,),
                min_freq=0,
                max_freq=1,
            ), children
        return None
