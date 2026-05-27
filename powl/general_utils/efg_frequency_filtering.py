from typing import TypeVar

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.statistics.eventually_follows.uvcl.get import apply as to_efg
from pm4py.objects.dfg import util as dfu

from powl.discovery.total_order_based.inductive.dtypes.partial_order import (
    IMDataStructurePOT,
)

T = TypeVar("T", bound=IMDataStructureUVCL)


def filter_efg_based_on_filtered_dfg(obj, alphabet, dfg, filtering_threshold):
    efg = obj.efg if isinstance(obj, IMDataStructurePOT) else to_efg(obj)
    if filtering_threshold is None:
        return efg
    filtered_efg = {}
    transitive_predecessors, transitive_successors = dfu.get_transitive_relations(
        dfg
    )
    for a in alphabet:
        for b in alphabet:
            if (a, b) in efg:
                if b in transitive_successors[a]:
                    filtered_efg[(a, b)] = efg[(a, b)]
    return filtered_efg
