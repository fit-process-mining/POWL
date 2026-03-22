from typing import Any, Dict, List, Optional, Tuple

from powl.discovery.total_order_based.inductive.variants.decision_graph.factory_cyclic_dg import (
    CutFactoryCyclicDecisionGraph,
)
from powl.discovery.total_order_based.inductive.variants.decision_graph.factory_cyclic_dg_strict import (
    CutFactoryCyclicDecisionGraphStrict,
)
from powl.discovery.total_order_based.inductive.variants.im_decision_graph_maximal import (
    POWLInductiveMinerDecisionGraphMaximal,
)
from powl.discovery.total_order_based.inductive.variants.im_tree import T
from powl.discovery.total_order_based.inductive.variants.powl_discovery_varaints import (
    POWLDiscoveryVariant,
)
from powl.discovery.total_order_based.inductive.modeling import InductiveModel


class POWLInductiveMinerDecisionGraphCyclic(POWLInductiveMinerDecisionGraphMaximal):
    def instance(self) -> POWLDiscoveryVariant:
        return POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC

    def find_cut(
        self, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[InductiveModel, List[T]]]:
        res = CutFactoryCyclicDecisionGraph.find_cut(obj, parameters=parameters)
        return res


class POWLInductiveMinerDecisionGraphCyclicStrict(
    POWLInductiveMinerDecisionGraphMaximal
):
    def instance(self) -> POWLDiscoveryVariant:
        return POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC_STRICT

    def find_cut(
        self, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[InductiveModel, List[T]]]:
        res = CutFactoryCyclicDecisionGraphStrict.find_cut(obj, parameters=parameters)
        return res
