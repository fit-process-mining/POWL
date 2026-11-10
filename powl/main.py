from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
from powl.conversion.variants.to_bpmn_with_resources import (
    apply as to_bpmn_with_resources,
)
from powl.discovery.dfg_based.algorithm import apply as dfg_discovery
from powl.discovery.object_centric.algorithm import apply as oc_discovery
from powl.discovery.rule_based.algorithm import discover_with_rules
from powl.discovery.rule_based.validation import validate_rules
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
from powl.io.powl_json import write_powl_json
from powl.visualization.powl.visualizer import POWLVisualizationVariants
from powl.visualization.bpmn.layout import save_bpmn as _save_bpmn
from pm4py import PetriNet
from powl.discovery.object_centric.algorithm import Variants as OC_Variants


def import_ocel(path: str | Path) -> OCEL:
    """Read OCEL 2.0, falling back to the legacy OCEL reader."""
    path = str(path)
    try:
        return pm4py.read_ocel2(path)
    except Exception as first_error:
        try:
            return pm4py.read_ocel(path)
        except Exception as second_error:
            raise ValueError(
                "Could not import the OCEL file with either reader: "
                f"OCEL 2.0: {first_error}; legacy: {second_error}"
            ) from second_error


def import_event_log(path: str | Path, timestamp_key: str | None = None) -> pd.DataFrame:
    """Read a CSV, XES, or compressed XES event log as a DataFrame.

    For CSV files, the supplied timestamp column is parsed as a datetime.
    If omitted, ``time:timestamp`` is used when present.
    """
    path = str(path)
    if path.endswith((".xes", ".xes.gz")):
        import rustxes

        xes, _ = rustxes.import_xes(path)
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
                "No column given as a timestamp key! Default value 'time:timestamp' is used."
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
    rules=None,
    rule_parameters: Mapping[str, Any] | None = None,
) -> TaggedPOWL:
    """
    Discovers a POWL model from an event log.

    Reference paper:
    H Kourani, G Park, WMP van der Aalst. "Unlocking Non-Block-Structured Decisions: Inductive Mining with Choice Graphs" arXiv preprint arXiv:2505.07052.

    :param keep_only_completion_events: use completion events when available
    :param lifecycle_key: name of the lifecycle column
    :param log: Pandas DataFrame of events
    :param variant: variant of the algorithm
    :param filtering_weight_factor: accepts values 0 <= x < 1
    :param order_graph_filtering_threshold: accepts values 0.5 < x <= 1
    :param dfg_frequency_filtering_threshold: accepts values 0 <= x < 1
    :param activity_key: attribute to be used for the activity
    :param timestamp_key: attribute to be used for the timestamp
    :param case_id_key: attribute to be used as case identifier
    :param simplify: whether to simplify or not the POWL model
    :param rules: declarative rules for the RULE_BASED variant
    :param rule_parameters: RULE_BASED options: repair_mode, preprocess_rules,
        verify_rules, max_states, max_depth; see discover_with_rules
    :return: Discovered POWL model.
    """

    if variant is POWLDiscoveryVariant.RULE_BASED:
        if filtering_weight_factor is not None or order_graph_filtering_threshold is not None:
            raise ValueError("RULE_BASED supports only DFG frequency filtering.")
        options = dict(rule_parameters or {})
        allowed = {"repair_mode", "preprocess_rules", "verify_rules", "max_states", "max_depth"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unknown rule_parameters: {sorted(unknown)!r}.")
        return discover_with_rules(
            log, rules,
            noise_threshold=dfg_frequency_filtering_threshold if dfg_frequency_filtering_threshold is not None else 0.0,
            activity_key=activity_key, timestamp_key=timestamp_key, case_id_key=case_id_key,
            lifecycle_key=lifecycle_key, keep_only_completion_events=keep_only_completion_events,
            simplify=simplify, **options,
        )
    if rules is not None or rule_parameters is not None:
        raise ValueError("rules and rule_parameters require variant=POWLDiscoveryVariant.RULE_BASED.")

    for column in (case_id_key, activity_key, timestamp_key):
        if column not in log.columns:
            raise ValueError(f"Column {column!r} not found in the event log.")

    if filtering_weight_factor is not None and not (0 <= filtering_weight_factor < 1):
        raise ValueError("filtering_weight_factor must satisfy 0 <= x < 1.")
    if order_graph_filtering_threshold is not None and not (
        0.5 < order_graph_filtering_threshold <= 1
    ):
        raise ValueError("order_graph_filtering_threshold must satisfy 0.5 < x <= 1.")
    if dfg_frequency_filtering_threshold is not None and not (
        0 <= dfg_frequency_filtering_threshold < 1
    ):
        raise ValueError("dfg_frequency_filtering_threshold must satisfy 0 <= x < 1.")

    log = log.sort_values([case_id_key, timestamp_key])
    properties = get_properties(
        log, activity_key=activity_key, timestamp_key=timestamp_key,
        case_id_key=case_id_key,
    )

    if keep_only_completion_events and lifecycle_key in log.columns:
        filtered_log = log[
            log[lifecycle_key].isin(["complete", "COMPLETE", "Complete"])
        ]
        if len(filtered_log) > 0:
            log = filtered_log

    from powl.discovery.total_order_based.inductive.utils.filtering import (
        FILTERING_TYPE,
        FilteringType,
    )

    num_filters = 0
    if order_graph_filtering_threshold is not None:
        if variant is not POWLDiscoveryVariant.DYNAMIC_CLUSTERING:
            raise ValueError(
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
        raise ValueError(
            "The algorithm can only be used with one filtering threshold at a time!"
        )

    from powl.discovery.total_order_based import algorithm as powl_discovery

    return powl_discovery.apply(
        log, variant=variant, parameters=properties, simplify=simplify, time_ordering=False,
    )


def discover_petri_net_from_ocel(ocel: OCEL, variant=OC_Variants.OC_POWL, parameters=None):
    return oc_discovery(ocel, variant=variant, parameters=parameters)


def discover_from_dfg(dfg: DFG, variant=POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC, parameters=None):
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


def view_ocpn(object_centric_pn, selected_object_types=None, projected_views=False):
    if selected_object_types is not None:
        from powl.discovery.object_centric.utils.filter_ocpn import filter_ocpn_by_object_types
        object_centric_pn = filter_ocpn_by_object_types(object_centric_pn, selected_object_types)
    if projected_views:
        for ot in object_centric_pn["object_types"]:
            pn, im, fm = object_centric_pn["petri_nets"][ot]
            pm4py.view_petri_net(pn, im, fm, graph_title=ot, format="SVG")
    pm4py.view_ocpn(object_centric_pn, format="SVG")


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


def convert_to_bpmn_with_resources(
    activity_to_pool_lane: dict[str, tuple[str, str]], powl: TaggedPOWL
) -> str:
    """
    Convert a POWL model to BPMN 2.0 XML with resources (pools & lanes),
    and return cleanly layouted BPMN 2.0 XML string.
    """
    return to_bpmn_with_resources(activity_to_pool_lane, powl)


def export(
    model: TaggedPOWL,
    file_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    activity_to_pool_lane: Mapping[str, tuple[str, str]] | None = None,
) -> None:
    """Export a POWL model to POWL JSON (``.powl``) or laid-out BPMN (``.bpmn``).

    The file extension chooses the format. BPMN export automatically includes
    diagram coordinates; callers never need to convert or lay out BPMN manually.
    Optional ``activity_to_pool_lane`` assignments are supported for BPMN only,
    and POWL JSON ``metadata`` is supported for .powl files only.

    The destination's parent directory must already exist. Existing files are
    overwritten.
    """
    if not isinstance(model, TaggedPOWL):
        raise TypeError("model must be a POWL model (TaggedPOWL)")

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".powl":
        if activity_to_pool_lane is not None:
            raise ValueError("activity_to_pool_lane is only supported for .bpmn export")
        write_powl_json(model, path, metadata=metadata)
    elif extension == ".bpmn":
        if metadata is not None:
            raise ValueError("metadata is only supported for .powl export")
        if activity_to_pool_lane is None:
            bpmn, _, _ = to_bpmn(model)
            _save_bpmn(bpmn, path)
        else:
            # The collaboration converter already generates laid-out BPMN XML.
            bpmn_xml = to_bpmn_with_resources(dict(activity_to_pool_lane), model)
            _save_bpmn(bpmn_xml, path, layout=False)
    else:
        raise ValueError("Unsupported export format: use a .powl or .bpmn file path")
