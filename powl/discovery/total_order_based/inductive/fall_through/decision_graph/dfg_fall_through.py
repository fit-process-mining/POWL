from typing import Any, Dict, List, Optional, Tuple

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.algo.discovery.inductive.fall_through.abc import FallThrough, T

from powl.discovery.total_order_based.inductive.variants.decision_graph.dfg_cut import (
    DFGCutUVCL,
)
from powl.discovery.total_order_based.inductive.modeling import ChoiceGraphSpec


class DFGFallThroughUVCL(FallThrough[IMDataStructureUVCL]):
    @classmethod
    def holds(
        cls,
        obj: IMDataStructureUVCL,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return True

    @classmethod
    def apply(
        cls,
        obj: T,
        pool=None,
        manager=None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[ChoiceGraphSpec, List[T]]]:
        return DFGCutUVCL.apply(obj, parameters)
