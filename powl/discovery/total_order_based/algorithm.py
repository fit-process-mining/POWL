from typing import Any, Dict, Optional, Type, Union
from datetime import timedelta
from collections import Counter

import pandas as pd
import pm4py
from pm4py import util
from pm4py.algo.discovery.inductive.algorithm import Parameters
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.objects.log.obj import EventLog
from pm4py.util import exec_utils, xes_constants as xes_util
from pm4py.util.compression import util as comut

from powl.discovery.total_order_based.inductive.dtypes.partial_order import IMDataStructurePOT, log_to_pot_variants
from powl.discovery.total_order_based.inductive.variants.im_brute_force import (
    POWLInductiveMinerBruteForce,
)
from powl.discovery.total_order_based.inductive.variants.im_decision_graph_clustering import (
    POWLInductiveMinerDecisionGraphClustering,
)

from powl.discovery.total_order_based.inductive.variants.im_decision_graph_cyclic import (
    POWLInductiveMinerDecisionGraphCyclic,
    POWLInductiveMinerDecisionGraphCyclicStrict,
)
from powl.discovery.total_order_based.inductive.variants.im_decision_graph_maximal import (
    POWLInductiveMinerDecisionGraphMaximal,
)
from powl.discovery.total_order_based.inductive.variants.im_dynamic_clustering_frequencies import (
    POWLInductiveMinerDynamicClusteringFrequency,
)
from powl.discovery.total_order_based.inductive.variants.im_maximal import (
    POWLInductiveMinerMaximalOrder,
)
from powl.discovery.total_order_based.inductive.variants.im_tree import IMBasePOWL
from powl.discovery.total_order_based.inductive.variants.powl_discovery_varaints import (
    POWLDiscoveryVariant,
)
from powl.objects.tagged_powl.base import TaggedPOWL

DEFAULT_POWL_MINER = POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC


def get_variant(variant: POWLDiscoveryVariant) -> Type[IMBasePOWL]:
    if variant == POWLDiscoveryVariant.RULE_BASED:
        from powl.discovery.rule_based.miner import RuleAwarePOWLInductiveMiner

        return RuleAwarePOWLInductiveMiner
    elif variant == POWLDiscoveryVariant.TREE:
        return IMBasePOWL
    elif variant == POWLDiscoveryVariant.BRUTE_FORCE:
        return POWLInductiveMinerBruteForce
    elif variant == POWLDiscoveryVariant.MAXIMAL:
        return POWLInductiveMinerMaximalOrder
    elif variant == POWLDiscoveryVariant.DYNAMIC_CLUSTERING:
        return POWLInductiveMinerDynamicClusteringFrequency
    elif variant == POWLDiscoveryVariant.DECISION_GRAPH_MAX:
        return POWLInductiveMinerDecisionGraphMaximal
    elif variant == POWLDiscoveryVariant.DECISION_GRAPH_CLUSTERING:
        return POWLInductiveMinerDecisionGraphClustering
    elif variant == POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC:
        return POWLInductiveMinerDecisionGraphCyclic
    elif variant == POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC_STRICT:
        return POWLInductiveMinerDecisionGraphCyclicStrict
    else:
        raise Exception("Invalid Variant!")


def apply(
        obj: Union[EventLog, pd.DataFrame],
        parameters: Optional[Dict[Any, Any]] = None,
        variant=DEFAULT_POWL_MINER,
        simplify=True,
        time_ordering: bool = True,
        time_window: Optional[Union[timedelta, pd.Timedelta]] = None,
) -> TaggedPOWL:
    if parameters is None:
        parameters = {}
    if variant == POWLDiscoveryVariant.RULE_BASED and time_ordering:
        raise ValueError("RULE_BASED requires total-order traces; pass time_ordering=False.")
    ack = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY, parameters, xes_util.DEFAULT_NAME_KEY
    )
    tk = exec_utils.get_param_value(
        Parameters.TIMESTAMP_KEY, parameters, xes_util.DEFAULT_TIMESTAMP_KEY
    )
    cidk = exec_utils.get_param_value(
        Parameters.CASE_ID_KEY, parameters, util.constants.CASE_CONCEPT_NAME
    )

    if time_ordering:
        if isinstance(obj, EventLog):
            obj = pm4py.convert_to_dataframe(obj)
        pot_variants = log_to_pot_variants(
            obj,
            activity_key=ack,
            timestamp_key=tk,
            case_id_key=cidk,
            time_window=time_window,
        )
        data_structure = IMDataStructurePOT(pot_variants)

    else:
        uvcl = comut.get_variants(
            comut.project_univariate(
                obj, key=ack, df_glue=cidk, df_sorting_criterion_key=tk
            )
        )

        data_structure = IMDataStructureUVCL(uvcl)

    algorithm = get_variant(variant)
    miner_parameters = parameters
    if variant == POWLDiscoveryVariant.RULE_BASED:
        from powl.discovery.rule_based.rules import normalize_rules

        # Check the final normalized result below, without mutating the caller's
        # parameters or performing the expensive proof twice.
        miner_parameters = dict(parameters)
        miner_parameters["rules"] = normalize_rules(parameters.get("rules", ()))
        miner_parameters["verify_rules"] = False
    im = algorithm(miner_parameters)
    res = im.apply(data_structure, miner_parameters)

    if simplify:
        res = res.normalize()

    if variant == POWLDiscoveryVariant.RULE_BASED:
        from powl.discovery.rule_based.conformance import assert_model_satisfies_rules

        if not isinstance(parameters.get("verify_rules", True), bool):
            raise TypeError("verify_rules must be a bool.")
        if parameters.get("verify_rules", True):
            assert_model_satisfies_rules(
                res, miner_parameters["rules"],
                max_states=parameters.get("max_states", 100_000),
            )

    return res
