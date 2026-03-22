from typing import Any, Dict, List, Optional, Tuple

from powl.discovery.total_order_based.inductive.variants.im_tree import IMBasePOWL, T

from powl.discovery.total_order_based.inductive.variants.maximal.factory import (
    CutFactoryPOWLMaximal,
)
from powl.discovery.total_order_based.inductive.variants.powl_discovery_varaints import (
    POWLDiscoveryVariant,
)
from powl.discovery.total_order_based.inductive.modeling import InductiveModel


class POWLInductiveMinerMaximalOrder(IMBasePOWL):
    def instance(self) -> POWLDiscoveryVariant:
        return POWLDiscoveryVariant.MAXIMAL

    def find_cut(
        self, obj: T, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[InductiveModel, List[T]]]:
        res = CutFactoryPOWLMaximal.find_cut(obj, parameters=parameters)
        return res
