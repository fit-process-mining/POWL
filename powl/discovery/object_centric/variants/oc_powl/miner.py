from typing import Any, Dict, Optional

from pm4py.objects.ocel.obj import OCEL

from powl.discovery.dfg_based.algorithm import apply as discover_from_dfg

from powl.discovery.object_centric.variants.oc_powl.utils.divergence_free_graph import (
    get_divergence_free_graph,
)
from powl.discovery.object_centric.variants.oc_powl.utils.filtering import (
    keep_most_frequent_activities,
)
from powl.discovery.object_centric.variants.oc_powl.utils.interaction_properties import (
    get_interaction_patterns,
)
from powl.discovery.object_centric.variants.oc_powl.utils.ocpn_conversion import (
    convert_ocpowl_to_ocpn,
)
from powl.discovery.total_order_based.inductive.variants.powl_discovery_varaints import (
    POWLDiscoveryVariant,
)
from powl.objects.oc_powl import load_oc_powl


def apply(
    oc_log: OCEL,
    powl_miner_variant=POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC,
    activity_coverage_threshold: float = 1.0,
    parameters: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:

    relations = oc_log.relations
    relations = keep_most_frequent_activities(
        relations, coverage=activity_coverage_threshold
    )

    div, con, rel, defi = get_interaction_patterns(relations)
    df2_graph, divergence_matrices = get_divergence_free_graph(relations, div, rel)

    powl_model = discover_from_dfg(df2_graph, variant=powl_miner_variant)
    import powl

    powl.view(powl_model)
    oc_powl = load_oc_powl(powl_model, rel, div, con, defi)
    ocpn = convert_ocpowl_to_ocpn(oc_powl, divergence_matrices)

    return ocpn
