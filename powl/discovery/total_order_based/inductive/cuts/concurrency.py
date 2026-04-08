from abc import ABC
from typing import Any, Dict, Generic, Optional

from pm4py.algo.discovery.inductive.cuts.concurrency import (
    ConcurrencyCut,
    ConcurrencyCutDFG,
    ConcurrencyCutUVCL,
    T,
)
from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureDFG,
    IMDataStructureUVCL,
)

from powl.discovery.total_order_based.inductive.modeling import PartialOrderSpec


class POWLConcurrencyCut(ConcurrencyCut, ABC, Generic[T]):
    @classmethod
    def operator(
        cls, parameters: Optional[Dict[str, Any]] = None
    ) -> PartialOrderSpec:
        return PartialOrderSpec(0)


class POWLConcurrencyCutUVCL(
    ConcurrencyCutUVCL, POWLConcurrencyCut[IMDataStructureUVCL]
):
    pass


class POWLConcurrencyCutDFG(ConcurrencyCutDFG, POWLConcurrencyCut[IMDataStructureDFG]):
    pass
