import warnings

import pandas as pd
import pm4py
from pm4py.algo.discovery.inductive.variants.imf import IMFParameters
from pm4py.objects.bpmn.layout import layouter as bpmn_layouter
from pm4py.objects.dfg.obj import DFG
from pm4py.objects.ocel.obj import OCEL
from pm4py.utils import get_properties

from powl.conversion.converter import apply as powl_converter
from powl.conversion.to_powl.from_pn.converter import convert_workflow_net_to_powl
from powl.conversion.variants.to_bpmn import apply as to_bpmn
from powl.discovery.dfg_based.algorithm import apply as dfg_discovery
from powl.discovery.object_centric.algorithm import apply as oc_discovery
from powl.discovery.total_order_based.inductive.utils.filtering import (
    FILTERING_THRESHOLD,
)
from powl.discovery.total_order_based.inductive.variants.dynamic_clustering_frequency.dynamic_clustering_frequency_partial_order_cut import (
    ORDER_FREQUENCY_RATIO,
)
from powl.discovery.total_order_based.inductive.variants.powl_discovery_varaints import (
    POWLDiscoveryVariant,
)
from powl.objects.tagged_powl.base import TaggedPOWL
from powl.visualization.powl.visualizer import POWLVisualizationVariants
from pm4py import PetriNet
from powl.discovery.object_centric.algorithm import Variants as OC_Variants


def import_ocel(path: str) -> OCEL:
    try:
        return pm4py.read_ocel2(path)
    except BaseException as first_error:
        try:
            return pm4py.read_ocel(path)
        except BaseException as second_error:
            raise Exception(
                f"Could not import OCEL file via OCEL 2 or generic reader: {first_error}; fallback failed: {second_error}"
            ) from second_error


def import_event_log(path: str, timestamp_key=None) -> pd.DataFrame:
    import rustxes

    if path.endswith(".xes") or path.endswith(".xes.gz"):
        [xes, _] = rustxes.import_xes(path)
        df = xes.to_pandas()
    elif path.endswith(".csv"):
        cols_to_parse = []
        df_sample = pd.read_csv(path, nrows=0)

        if timestamp_key:
            if timestamp_key not in df_sample.columns:
                raise ValueError("Timestamp key not found in table!")
        elif "time:timestamp" in df_sample.columns:
            timestamp_key = "time:timestamp"
            warnings.warn(
                f"No column given as a timestamp key! Default value 'time:timestamp' is used."
            )
        else:
            warnings.warn(
                "No column given as a timestamp key! Timestamp column parsing is skipped!"
            )

        if timestamp_key:
            cols_to_parse = [timestamp_key]

        df = pd.read_csv(path, keep_default_na=False, parse_dates=cols_to_parse)
    else:
        raise ValueError("Unsupported file type!")
    return df


def discover(
    log: pd.DataFrame,
    variant=POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC,
    filtering_weight_factor: float = None,
    order_graph_filtering_threshold: float = None,
    dfg_frequency_filtering_threshold: float = None,
    activity_key: str = "concept:name",
    timestamp_key: str = "time:timestamp",
    case_id_key: str = "case:concept:name",
    lifecycle_key: str = "lifecycle:transition",
    keep_only_completion_events: bool = True,
    simplify=True,
) -> TaggedPOWL:
    """
    Discovers a POWL model from an event log.

    Reference paper:
    H Kourani, G Park, WMP van der Aalst. "Unlocking Non-Block-Structured Decisions: Inductive Mining with Choice Graphs" arXiv preprint arXiv:2505.07052.

    :param keep_only_completion_events:
    :param lifecycle_key:
    :param log: event log / Pandas dataframe
    :param variant: variant of the algorithm
    :param filtering_weight_factor: accepts values 0 <= x < 1
    :param order_graph_filtering_threshold: accepts values 0.5 < x <= 1
    :param dfg_frequency_filtering_threshold: accepts values 0 <= x < 1
    :param activity_key: attribute to be used for the activity
    :param timestamp_key: attribute to be used for the timestamp
    :param case_id_key: attribute to be used as case identifier
    :param simplify: whether to simplify or not the POWL model
    :rtype: ``POWL``
    """

    log = log.sort_values([case_id_key, timestamp_key])
    properties = get_properties(
        log, activity_key=activity_key, timestamp_key=timestamp_key
    )

    if keep_only_completion_events and lifecycle_key in log.columns:
        filtered_log = log[
            log["lifecycle:transition"].isin(["complete", "COMPLETE", "Complete"])
        ]
        if len(filtered_log) > 0:
            log = filtered_log

    from powl.discovery.total_order_based.inductive.utils.filtering import (
        FILTERING_TYPE,
        FilteringType,
    )

    num_filters = 0
    if order_graph_filtering_threshold is not None:
        if not variant is POWLDiscoveryVariant.DYNAMIC_CLUSTERING:
            raise Exception(
                "The order graph filtering threshold can only be used for the variant DYNAMIC_CLUSTERING!"
            )
        properties[ORDER_FREQUENCY_RATIO] = order_graph_filtering_threshold
        properties[FILTERING_TYPE] = FilteringType.DYNAMIC
        num_filters += 1
    if dfg_frequency_filtering_threshold is not None and dfg_frequency_filtering_threshold > 0:
        properties[IMFParameters.NOISE_THRESHOLD] = dfg_frequency_filtering_threshold
        properties[FILTERING_TYPE] = FilteringType.DFG_FREQUENCY
        num_filters += 1
    if filtering_weight_factor is not None:
        properties[FILTERING_THRESHOLD] = filtering_weight_factor
        properties[FILTERING_TYPE] = FilteringType.DECREASING_FACTOR
        num_filters += 1

    if num_filters > 1:
        raise Exception(
            "The algorithm can only be used with one filtering threshold at a time!"
        )

    from powl.discovery.total_order_based import algorithm as powl_discovery

    return powl_discovery.apply(
        log, variant=variant, parameters=properties, simplify=simplify
    )


def discover_petri_net_from_ocel(ocel: OCEL, variant=OC_Variants.OC_POWL, parameters=None):
    return oc_discovery(ocel, variant=variant, parameters=parameters)


def discover_from_dfg(dfg: DFG, variant=POWLDiscoveryVariant.MAXIMAL, parameters=None):
    return dfg_discovery(dfg, variant=variant, parameters=parameters)


def discover_from_partially_ordered_log(
    log: pd.DataFrame,
    activity_key: str = "concept:name",
    order_key: str = "time:timestamp",
    case_id_key: str = "case:concept:name",
    lifecycle_key: str | None = "lifecycle:transition",
) -> TaggedPOWL:
    """
    Discovers a POWL model from a partially ordered event log.

    Reference paper:
    H Kourani, G Park, WMP van der Aalst. "Revealing Inherent Concurrency in Event Data: A Partial Order Approach to Process Discovery"

    :param log: event log / Pandas dataframe
    :param activity_key: attribute to be used for the activity
    :param order_key: attribute to be used for ordering events within traces
    :param case_id_key: attribute to be used as case identifier
    :param lifecycle_key: attribute to be used as lifecycle identifier
    :rtype: ``POWL``
    """
    complete_tags = {"complete", "COMPLETE", "Complete"}
    start_tags = {"start", "START", "Start"}

    from powl.discovery.partial_order_based.utils import log_to_partial_orders

    partial_orders = log_to_partial_orders.apply(
        log,
        case_id_col=case_id_key,
        activity_col=activity_key,
        ordering_col=order_key,
        lifecycle_col=lifecycle_key,
        start_transitions=start_tags,
        complete_transitions=complete_tags,
    )
    from powl.discovery.partial_order_based.variants.base import miner

    powl = miner.apply(partial_orders)
    return powl


def view(powl: TaggedPOWL, use_frequency_tags=True):
    from powl.visualization.powl import visualizer as powl_visualizer

    gviz = powl_visualizer.apply(
        powl, variant=POWLVisualizationVariants.BASIC, frequency_tags=use_frequency_tags
    )
    powl_visualizer.view(gviz)


def view_net(powl: TaggedPOWL, use_frequency_tags=True):
    from powl.visualization.powl import visualizer as powl_visualizer

    gviz = powl_visualizer.apply(
        powl, variant=POWLVisualizationVariants.NET, frequency_tags=use_frequency_tags
    )
    powl_visualizer.view(gviz)


def view_ocpn(object_centric_pn, selected_object_types=None):
    if selected_object_types is not None:
        from powl.discovery.object_centric.utils.filter_ocpn import filter_ocpn_by_object_types
        object_centric_pn = filter_ocpn_by_object_types(object_centric_pn, selected_object_types)
    return pm4py.view_ocpn(object_centric_pn, format="SVG")


def save_visualization(powl: TaggedPOWL, file_path: str, use_frequency_tags=True):
    file_path = str(file_path)
    from powl.visualization.powl import visualizer as powl_visualizer

    gviz = powl_visualizer.apply(
        powl, variant=POWLVisualizationVariants.BASIC, frequency_tags=use_frequency_tags
    )
    return powl_visualizer.save(gviz, file_path)


def save_visualization_net(powl: TaggedPOWL, file_path: str, use_frequency_tags=True):
    file_path = str(file_path)
    from powl.visualization.powl import visualizer as powl_visualizer

    gviz = powl_visualizer.apply(
        powl, variant=POWLVisualizationVariants.NET, frequency_tags=use_frequency_tags
    )
    return powl_visualizer.save(gviz, file_path)


def convert_to_petri_net(powl: TaggedPOWL):
    return powl_converter(powl)

def convert_from_workflow_net(net: PetriNet) -> TaggedPOWL:
    return convert_workflow_net_to_powl(net)

def convert_to_bpmn(powl: TaggedPOWL):
    bpmn, _, _ = to_bpmn(powl)
    bpmn = bpmn_layouter.apply(bpmn)
    return bpmn
