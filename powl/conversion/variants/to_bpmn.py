from __future__ import annotations

import networkx as nx

from powl.conversion.variants.to_petri_net import apply as to_petri_net
from powl.objects.tagged_powl.base import TaggedPOWL
from powl.visualization.powl.variants.net import to_bpmn as petri_net_to_bpmn


def apply(powl: TaggedPOWL):
    """
    Convert a tagged POWL model to BPMN through the Petri-net translation.

    Returns the BPMN model and placeholder graph/mapping values to preserve the
    historical return shape of this API.
    """

    net, initial_marking, final_marking = to_petri_net(powl)
    bpmn = petri_net_to_bpmn(net, initial_marking, final_marking)
    return bpmn, nx.DiGraph(), {}
