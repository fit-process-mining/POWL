from typing import Any, Dict, List, Optional, Tuple, Type

from pm4py.algo.discovery.inductive.fall_through.empty_traces import EmptyTracesDFG

from powl.discovery.dfg_based.variants.dfg_im_tree import DFGIMBasePOWL, T
from powl.discovery.total_order_based.inductive.fall_through.decision_graph.empty_traces_decision_graph import (
    POWLEmptyTracesDecisionGraphDFG,
)
from powl.discovery.total_order_based.inductive.variants.decision_graph.factory_dg_max import (
    CutFactoryPOWLDecisionGraphMaximal,
)
from powl.discovery.total_order_based.inductive.variants.powl_discovery_varaints import (
    POWLDiscoveryVariant,
)
from powl.discovery.total_order_based.inductive.modeling import InductiveModel


class DFGPOWLInductiveMinerDecisionGraphMaximal(DFGIMBasePOWL):
    def instance(self) -> POWLDiscoveryVariant:
        return POWLDiscoveryVariant.DECISION_GRAPH_MAX

    def empty_traces_cut(self) -> Type[EmptyTracesDFG]:
        return POWLEmptyTracesDecisionGraphDFG

    def find_cut(
        self, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[InductiveModel, List[T]]]:
        res = CutFactoryPOWLDecisionGraphMaximal.find_cut(obj, parameters=parameters)
        return res
