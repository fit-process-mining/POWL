from multiprocessing import Manager, Pool
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from pm4py.algo.discovery.inductive.cuts.abc import Cut, T

from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureDFG,
    IMDataStructureUVCL,
)
from pm4py.algo.discovery.inductive.fall_through.abc import FallThrough

from powl.discovery.total_order_based.inductive.fall_through.activity_concurrent import (
    POWLActivityConcurrentUVCL,
)
from powl.discovery.total_order_based.inductive.fall_through.activity_once_per_trace import (
    POWLActivityOncePerTraceUVCL,
)
from powl.discovery.total_order_based.inductive.fall_through.decision_graph.dfg_fall_through import (
    DFGFallThroughUVCL,
)
from powl.discovery.total_order_based.inductive.fall_through.flower import (
    POWLFlowerModelDFG,
    POWLFlowerModelUVCL,
)
from powl.discovery.total_order_based.inductive.fall_through.strict_tau_loop import (
    POWLStrictTauLoopUVCL,
)
from powl.discovery.total_order_based.inductive.fall_through.tau_loop import (
    POWLTauLoopUVCL,
)
from powl.discovery.total_order_based.inductive.modeling import InductiveModel

S = TypeVar("S", bound=FallThrough | Cut)


class FallThroughFactory:
    @classmethod
    def get_fall_throughs(
        cls,
        obj: T,
        enable_dfg_fall_through,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Type[S]]:
        if type(obj) is IMDataStructureUVCL:
            if enable_dfg_fall_through:
                return [
                    POWLActivityOncePerTraceUVCL,
                    POWLActivityConcurrentUVCL,
                    POWLStrictTauLoopUVCL,
                    POWLTauLoopUVCL,
                    DFGFallThroughUVCL,
                ]
            else:
                return [
                    POWLActivityOncePerTraceUVCL,
                    POWLActivityConcurrentUVCL,
                    POWLStrictTauLoopUVCL,
                    POWLTauLoopUVCL,
                    POWLFlowerModelUVCL,
                ]
        elif type(obj) is IMDataStructureDFG:
            return [POWLFlowerModelDFG]
        return list()

    @classmethod
    def fall_through(
        cls,
        obj: T,
        pool: Pool,
        manager: Manager,
        enable_dfg_fall_through,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[InductiveModel, List[T]]]:
        for f in FallThroughFactory.get_fall_throughs(obj, enable_dfg_fall_through):
            r = f.apply(obj, pool, manager, parameters)
            if r is not None:
                return r
        return None
