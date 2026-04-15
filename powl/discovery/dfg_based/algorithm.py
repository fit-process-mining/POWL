from typing import Any, Dict, Optional, Type

from pm4py.algo.discovery.inductive.dtypes.im_dfg import InductiveDFG
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureDFG
from pm4py.objects.dfg.obj import DFG

from powl.discovery.dfg_based.variants.dfg_im_decision_graph_cyclic import DFGPOWLInductiveMinerDecisionGraphCyclic
from powl.discovery.dfg_based.variants.dfg_im_decision_graph_maximal import (
    DFGPOWLInductiveMinerDecisionGraphMaximal,
)
from powl.discovery.dfg_based.variants.dfg_im_maximal import (
    DFGPOWLInductiveMinerMaximalOrder,
)
from powl.discovery.dfg_based.variants.dfg_im_tree import DFGIMBasePOWL
from powl.discovery.dfg_based.variants.im_dynamic_clustering_frequencies import (
    DFGPOWLInductiveMinerDynamicClusteringFrequency,
)
from powl.discovery.total_order_based.inductive.variants.powl_discovery_varaints import (
    POWLDiscoveryVariant,
)
from powl.objects.tagged_powl.base import TaggedPOWL


def get_variant(variant: POWLDiscoveryVariant) -> Type[DFGIMBasePOWL]:
    if variant == POWLDiscoveryVariant.TREE:
        return DFGIMBasePOWL
    elif variant == POWLDiscoveryVariant.MAXIMAL:
        return DFGPOWLInductiveMinerMaximalOrder
    elif variant == POWLDiscoveryVariant.DYNAMIC_CLUSTERING:
        return DFGPOWLInductiveMinerDynamicClusteringFrequency
    elif variant == POWLDiscoveryVariant.DECISION_GRAPH_MAX:
        return DFGPOWLInductiveMinerDecisionGraphMaximal
    elif variant == POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC:
        return DFGPOWLInductiveMinerDecisionGraphCyclic
    else:
        raise Exception("Invalid Variant!")


def apply(
    dfg: DFG,
    parameters: Optional[Dict[Any, Any]] = None,
    variant=POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC,
) -> TaggedPOWL:
    if parameters is None:
        parameters = {}

    im_dfg = InductiveDFG(dfg=dfg, skip=False)

    algorithm = get_variant(variant)
    im = algorithm(parameters)
    res = im.apply(IMDataStructureDFG(im_dfg), parameters)
    res = res.normalize()

    return res
