from collections import Counter
from multiprocessing import Manager, Pool
from typing import Any, Dict, List, Optional, Tuple

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.algo.discovery.inductive.fall_through.abc import FallThrough

from pm4py.algo.discovery.inductive.fall_through.activity_concurrent import (
    ActivityConcurrentUVCL,
)
from pm4py.objects.dfg import util as dfu

from powl.discovery.total_order_based.inductive.cuts.factory import CutFactory
from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT,
    combined_project_pot_on_groups,
)
from powl.discovery.total_order_based.inductive.modeling import PartialOrderSpec


class POWLActivityConcurrentUVCL(ActivityConcurrentUVCL):
    @classmethod
    def apply(
        cls,
        obj: IMDataStructureUVCL,
        pool: Pool = None,
        manager: Manager = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[PartialOrderSpec, List[IMDataStructureUVCL]]]:
        candidate = cls._get_candidate(obj, pool, manager, parameters)
        if candidate is None:
            return None
        log = obj.data_structure
        l_a = Counter()
        l_other = Counter()
        for t in log:
            l_a.update({tuple(filter(lambda e: e == candidate, t)): log[t]})
            l_other.update({tuple(filter(lambda e: e != candidate, t)): log[t]})
        return PartialOrderSpec(2), [
            IMDataStructureUVCL(l_a),
            IMDataStructureUVCL(l_other),
        ]


class POWLActivityConcurrentPOT(FallThrough[IMDataStructurePOT]):
    @classmethod
    def _process_candidate(
        cls,
        candidate: Any,
        obj: IMDataStructurePOT,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        remaining_activities = set(dfu.get_vertices(obj.dfg)).difference({candidate})
        projected = combined_project_pot_on_groups(
            obj.data_structure,
            [remaining_activities],
            keep_empty=True,
        )[0]
        return CutFactory.find_cut(projected, parameters=parameters)

    @classmethod
    def _get_candidate(
        cls,
        obj: IMDataStructurePOT,
        pool: Pool = None,
        manager: Manager = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        candidates = sorted(dfu.get_vertices(obj.dfg), key=lambda a: a.__str__())
        for activity in candidates:
            if cls._process_candidate(activity, obj, parameters=parameters) is not None:
                return activity
        return None

    @classmethod
    def holds(
        cls,
        obj: IMDataStructurePOT,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return cls._get_candidate(obj, None, None, parameters) is not None

    @classmethod
    def apply(
        cls,
        obj: IMDataStructurePOT,
        pool: Pool = None,
        manager: Manager = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[PartialOrderSpec, List[IMDataStructurePOT]]]:
        candidate = cls._get_candidate(obj, pool, manager, parameters)
        if candidate is None:
            return None
        alphabet = set(dfu.get_vertices(obj.dfg))
        children = combined_project_pot_on_groups(
            obj.data_structure,
            [{candidate}, alphabet.difference({candidate})],
            keep_empty=True,
        )
        return PartialOrderSpec(2), children
