from collections import Counter
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
from pm4py.objects.dfg.obj import DFG

from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT,
)
from powl.discovery.total_order_based.inductive.modeling import XorSpec


class POWLEmptyTracesUVCL(EmptyTracesUVCL):
    @classmethod
    def holds(
        cls,
        obj: IMDataStructureUVCL,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if isinstance(obj, IMDataStructurePOT):
            return any(len(t) == 0 for t in obj.data_structure)
        return EmptyTracesUVCL.holds(obj, parameters)

    @classmethod
    def apply(
        cls,
        obj: IMDataStructureUVCL,
        pool: Pool = None,
        manager: Manager = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[XorSpec, List[IMDataStructureUVCL]]]:
        if cls.holds(obj, parameters):
            if isinstance(obj, IMDataStructurePOT):
                data_structure = copy(obj.data_structure)
                for trace in list(data_structure.keys()):
                    if len(trace) == 0:
                        del data_structure[trace]
                return XorSpec(2), [
                    IMDataStructurePOT(Counter()),
                    IMDataStructurePOT(data_structure),
                ]
            data_structure = copy(obj.data_structure)
            del data_structure[()]
            return XorSpec(2), [
                IMDataStructureUVCL(Counter()),
                IMDataStructureUVCL(data_structure),
            ]
        else:
            return None


class POWLEmptyTracesDFG(EmptyTracesDFG):
    @classmethod
    def apply(
        cls,
        obj: IMDataStructureDFG,
        pool=None,
        manager=None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[XorSpec, List[IMDataStructureDFG]]]:
        if cls.holds(obj, parameters):
            return XorSpec(2), [
                IMDataStructureDFG(InductiveDFG(DFG())),
                IMDataStructureDFG(InductiveDFG(obj.data_structure.dfg)),
            ]
        return None
