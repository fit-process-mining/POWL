from abc import ABC
from typing import Any, Dict, Generic, Optional, Collection, List

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
from powl.discovery.total_order_based.inductive.variants.maximal.maximal_partial_order_cut import \
    MaximalPartialOrderCutDFG


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
    @classmethod
    def project(
            cls,
            obj: IMDataStructureDFG,
            groups: List[Collection[Any]],
            parameters: Optional[Dict[str, Any]] = None,
    ) -> List[IMDataStructureDFG]:
        return MaximalPartialOrderCutDFG.project(obj, groups, parameters=parameters)
