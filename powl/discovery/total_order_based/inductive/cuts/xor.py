from abc import ABC
from typing import Any, Collection, Dict, Generic, List, Optional

from pm4py.algo.discovery.inductive.cuts.xor import (
    ExclusiveChoiceCut,
    ExclusiveChoiceCutDFG,
    ExclusiveChoiceCutUVCL,
    T,
)
from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructureDFG,
    IMDataStructureUVCL,
)

from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT,
    split_project_pot_on_groups,
)
from powl.discovery.total_order_based.inductive.modeling import XorSpec


class POWLExclusiveChoiceCut(ExclusiveChoiceCut, ABC, Generic[T]):
    @classmethod
    def operator(cls, parameters: Optional[Dict[str, Any]] = None) -> XorSpec:
        return XorSpec(0)


class POWLExclusiveChoiceCutUVCL(
    ExclusiveChoiceCutUVCL, POWLExclusiveChoiceCut[IMDataStructureUVCL], ABC
):
    pass


class POWLExclusiveChoiceCutDFG(
    ExclusiveChoiceCutDFG, POWLExclusiveChoiceCut[IMDataStructureDFG], ABC
):
    pass


class POWLExclusiveChoiceCutPOT(
    POWLExclusiveChoiceCut[IMDataStructurePOT], ABC
):
    @classmethod
    def project(
        cls,
        obj: IMDataStructurePOT,
        groups: List[Collection[Any]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[IMDataStructurePOT]:
        return split_project_pot_on_groups(obj.data_structure, groups)
