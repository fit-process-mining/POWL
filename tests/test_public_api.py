import warnings

import pandas as pd
import pm4py
import pytest
from pandas.api.types import is_datetime64_any_dtype
from pm4py.algo.discovery.inductive.variants.imf import IMFParameters
from pm4py.objects.dfg.obj import DFG

import powl
import powl.main
from powl.discovery.total_order_based import algorithm as total_order_algorithm
from powl.discovery.total_order_based.inductive.utils.filtering import (
    FILTERING_THRESHOLD,
)
from powl.discovery.total_order_based.inductive.variants.dynamic_clustering_frequency.dynamic_clustering_frequency_partial_order_cut import (
    ORDER_FREQUENCY_RATIO,
)
from powl.discovery.total_order_based.inductive.variants.powl_discovery_varaints import (
    POWLDiscoveryVariant,
)
from powl.objects.tagged_powl.partial_order import PartialOrder


def test_import_event_log_warns_and_parses_default_timestamp(running_example_path):
    with pytest.warns(
        UserWarning, match=r"Default value 'time:timestamp' is used"
    ):
        log = powl.import_event_log(str(running_example_path))

    assert not log.empty
    assert is_datetime64_any_dtype(log["time:timestamp"])


def test_import_event_log_rejects_missing_timestamp_key(running_example_path):
    with pytest.raises(ValueError, match="Timestamp key not found"):
        powl.import_event_log(str(running_example_path), timestamp_key="missing-column")


def test_import_event_log_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text("unsupported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        powl.import_event_log(str(path))


def test_import_ocel_falls_back_to_legacy_reader(monkeypatch):
    sentinel = object()

    def fail_read_ocel2(path):
        raise RuntimeError("ocel2 not available")

    def read_ocel(path):
        assert path == "dummy-path"
        return sentinel

    monkeypatch.setattr(powl.main.pm4py, "read_ocel2", fail_read_ocel2)
    monkeypatch.setattr(powl.main.pm4py, "read_ocel", read_ocel)

    assert powl.import_ocel("dummy-path") is sentinel


def test_discover_filters_completion_events_before_delegating(monkeypatch):
    captured = {}

    def fake_apply(log, variant=None, parameters=None, simplify=None):
        captured["log"] = log.copy()
        captured["variant"] = variant
        captured["parameters"] = parameters
        captured["simplify"] = simplify
        return "sentinel-model"

    monkeypatch.setattr(total_order_algorithm, "apply", fake_apply)

    log = pd.DataFrame(
        {
            "case:concept:name": ["1", "1", "1", "1"],
            "concept:name": ["A", "A", "B", "B"],
            "time:timestamp": pd.to_datetime(
                [
                    "2024-01-01T10:00:00",
                    "2024-01-01T10:01:00",
                    "2024-01-01T10:02:00",
                    "2024-01-01T10:03:00",
                ]
            ),
            "lifecycle:transition": ["start", "complete", "start", "complete"],
        }
    )

    result = powl.discover(log, simplify=False)

    assert result == "sentinel-model"
    assert captured["variant"] == POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC
    assert captured["simplify"] is False
    assert list(captured["log"]["lifecycle:transition"]) == ["complete", "complete"]


def test_discover_keeps_original_log_if_completion_filter_would_empty_it(monkeypatch):
    captured = {}

    def fake_apply(log, variant=None, parameters=None, simplify=None):
        captured["rows"] = len(log)
        return "sentinel-model"

    monkeypatch.setattr(total_order_algorithm, "apply", fake_apply)

    log = pd.DataFrame(
        {
            "case:concept:name": ["1", "1"],
            "concept:name": ["A", "B"],
            "time:timestamp": pd.to_datetime(
                ["2024-01-01T10:00:00", "2024-01-01T10:01:00"]
            ),
            "lifecycle:transition": ["start", "start"],
        }
    )

    assert powl.discover(log) == "sentinel-model"
    assert captured["rows"] == 2


def test_discover_passes_filtering_threshold_parameters(monkeypatch):
    captured = {}

    def fake_apply(log, variant=None, parameters=None, simplify=None):
        captured["parameters"] = parameters
        return "sentinel-model"

    monkeypatch.setattr(total_order_algorithm, "apply", fake_apply)

    log = pd.DataFrame(
        {
            "case:concept:name": ["1", "1"],
            "concept:name": ["A", "B"],
            "time:timestamp": pd.to_datetime(
                ["2024-01-01T10:00:00", "2024-01-01T10:01:00"]
            ),
        }
    )

    powl.discover(log, filtering_weight_factor=0.3, keep_only_completion_events=False)
    assert captured["parameters"][FILTERING_THRESHOLD] == 0.3


def test_discover_passes_dfg_noise_threshold(monkeypatch):
    captured = {}

    def fake_apply(log, variant=None, parameters=None, simplify=None):
        captured["parameters"] = parameters
        return "sentinel-model"

    monkeypatch.setattr(total_order_algorithm, "apply", fake_apply)

    log = pd.DataFrame(
        {
            "case:concept:name": ["1", "1"],
            "concept:name": ["A", "B"],
            "time:timestamp": pd.to_datetime(
                ["2024-01-01T10:00:00", "2024-01-01T10:01:00"]
            ),
        }
    )

    powl.discover(
        log,
        dfg_frequency_filtering_threshold=0.2,
        keep_only_completion_events=False,
    )

    assert captured["parameters"][IMFParameters.NOISE_THRESHOLD] == 0.2


def test_discover_passes_order_graph_threshold_for_dynamic_clustering(monkeypatch):
    captured = {}

    def fake_apply(log, variant=None, parameters=None, simplify=None):
        captured["variant"] = variant
        captured["parameters"] = parameters
        return "sentinel-model"

    monkeypatch.setattr(total_order_algorithm, "apply", fake_apply)

    log = pd.DataFrame(
        {
            "case:concept:name": ["1", "1"],
            "concept:name": ["A", "B"],
            "time:timestamp": pd.to_datetime(
                ["2024-01-01T10:00:00", "2024-01-01T10:01:00"]
            ),
        }
    )

    powl.discover(
        log,
        variant=POWLDiscoveryVariant.DYNAMIC_CLUSTERING,
        order_graph_filtering_threshold=0.8,
        keep_only_completion_events=False,
    )

    assert captured["variant"] == POWLDiscoveryVariant.DYNAMIC_CLUSTERING
    assert captured["parameters"][ORDER_FREQUENCY_RATIO] == 0.8


def test_discover_rejects_multiple_filter_thresholds(running_example_log):
    with pytest.raises(Exception, match="one filtering threshold at a time"):
        powl.discover(
            running_example_log,
            filtering_weight_factor=0.2,
            dfg_frequency_filtering_threshold=0.3,
        )


def test_discover_rejects_order_graph_threshold_for_non_dynamic_variant(
    running_example_log,
):
    with pytest.raises(Exception, match="only be used for the variant DYNAMIC_CLUSTERING"):
        powl.discover(
            running_example_log,
            order_graph_filtering_threshold=0.8,
            variant=POWLDiscoveryVariant.MAXIMAL,
        )


def test_discover_running_example_and_convert(running_example_model):
    assert isinstance(running_example_model, PartialOrder)
    assert len(running_example_model.get_nodes()) == 3
    assert len(running_example_model.get_edges()) == 2

    net, initial_marking, final_marking = powl.convert_to_petri_net(running_example_model)
    assert any(t.label == "register request" for t in net.transitions)
    assert len(net.places) > 0
    assert len(initial_marking) == 1
    assert len(final_marking) == 1

    bpmn = powl.convert_to_bpmn(running_example_model)
    assert any(getattr(node, "name", None) == "register request" for node in bpmn.get_nodes())
    assert len(bpmn.get_flows()) > 0


def test_discover_from_dfg_smoke():
    dfg = DFG()
    dfg.graph[("A", "B")] = 4
    dfg.start_activities["A"] = 4
    dfg.end_activities["B"] = 4

    model = powl.discover_from_dfg(dfg, variant=POWLDiscoveryVariant.MAXIMAL)

    assert isinstance(model, PartialOrder)
    assert len(model.get_nodes()) == 2
    assert len(model.get_edges()) == 1


def test_discover_partially_ordered_log_and_convert(hospital_example_model):
    assert isinstance(hospital_example_model, PartialOrder)
    assert len(hospital_example_model.get_nodes()) == 5
    assert len(hospital_example_model.get_edges()) == 6
    assert any(
        getattr(node, "label", None) == "Blood Test"
        for node in hospital_example_model.get_nodes()
    )

    net, initial_marking, final_marking = powl.convert_to_petri_net(
        hospital_example_model
    )
    assert len(net.transitions) > 0
    assert len(initial_marking) == 1
    assert len(final_marking) == 1


def test_discover_petri_net_from_ocel_delegates(monkeypatch):
    sentinel = object()

    def fake_oc_discovery(ocel, parameters=None):
        assert ocel == "ocel"
        assert parameters == {"alpha": 1}
        return sentinel

    monkeypatch.setattr(powl.main, "oc_discovery", fake_oc_discovery)

    assert powl.discover_petri_net_from_ocel("ocel", parameters={"alpha": 1}) is sentinel


def test_convert_from_workflow_net_example(workflow_net_path):
    net, _, _ = pm4py.read_pnml(str(workflow_net_path))

    model = powl.convert_from_workflow_net(net)

    assert isinstance(model, PartialOrder)
    assert len(model.get_nodes()) == 2
    roundtrip_net, _, _ = powl.convert_to_petri_net(model)
    assert len(roundtrip_net.transitions) > 0
