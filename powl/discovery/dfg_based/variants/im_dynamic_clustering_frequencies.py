from typing import Any, Dict, List, Optional, Tuple

from powl.discovery.dfg_based.variants.dfg_im_tree import DFGIMBasePOWL, T
from powl.discovery.total_order_based.inductive.variants.dynamic_clustering_frequency.factory import (
    CutFactoryPOWLDynamicClusteringFrequency,
)
from powl.discovery.total_order_based.inductive.variants.powl_discovery_varaints import (
    POWLDiscoveryVariant,
)
from powl.discovery.total_order_based.inductive.modeling import InductiveModel


class DFGPOWLInductiveMinerDynamicClusteringFrequency(DFGIMBasePOWL):
    def instance(self) -> POWLDiscoveryVariant:
        return POWLDiscoveryVariant.DYNAMIC_CLUSTERING

    def find_cut(
        self, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[InductiveModel, List[T]]]:
        res = CutFactoryPOWLDynamicClusteringFrequency.find_cut(
            obj, parameters=parameters
        )
        return res
