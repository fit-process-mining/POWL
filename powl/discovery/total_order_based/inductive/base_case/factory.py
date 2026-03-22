from typing import Any, Dict, List as TList, Optional, Type, TypeVar

from pm4py.algo.discovery.inductive.dtypes.im_ds import (
    IMDataStructure,
    IMDataStructureDFG,
    IMDataStructureUVCL,
)

from powl.discovery.total_order_based.inductive.base_case.abc import BaseCase
from powl.discovery.total_order_based.inductive.base_case.empty_log import (
    EmptyLogBaseCaseDFG,
    EmptyLogBaseCaseUVCL,
)
from powl.discovery.total_order_based.inductive.base_case.single_activity import (
    SingleActivityBaseCaseDFG,
    SingleActivityBaseCaseUVCL,
)

from powl.objects.tagged_powl.base import TaggedPOWL

T = TypeVar("T", bound=IMDataStructure)
S = TypeVar("S", bound=BaseCase)


class BaseCaseFactory:
    @classmethod
    def get_base_cases(
        cls, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> TList[Type[S]]:
        if type(obj) is IMDataStructureUVCL:
            return [EmptyLogBaseCaseUVCL, SingleActivityBaseCaseUVCL]
        elif type(obj) is IMDataStructureDFG:
            return [EmptyLogBaseCaseDFG, SingleActivityBaseCaseDFG]
        else:
            return []

    @classmethod
    def apply_base_cases(
        cls, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[TaggedPOWL]:
        for b in BaseCaseFactory.get_base_cases(obj):
            r = b.apply(obj, parameters)
            if r is not None:
                return r
        return None
