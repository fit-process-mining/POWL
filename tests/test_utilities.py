from collections import Counter

import networkx as nx
import pandas as pd
import pytest

import powl
from powl.objects.utils.graph_sequentialization import split_graph_into_stages
from powl.objects.utils.relation import get_transitive_closure_from_counter
from powl.general_utils.time_utils import should_parse_column_as_date
from powl.visualization.bpmn.layout import layout_bpmn, layout_bpmn_xml
from powl.visualization.bpmn import visualizer as bpmn_visualizer


def test_should_parse_column_as_date_detects_valid_values():
    df = pd.DataFrame({"candidate": ["2024-01-01T10:00:00", ""]})

    assert should_parse_column_as_date(df, "candidate") is True


def test_should_parse_column_as_date_rejects_non_dates():
    df = pd.DataFrame({"candidate": ["not-a-date", "still-not-a-date"]})

    assert should_parse_column_as_date(df, "candidate") is False


def test_should_parse_column_as_date_rejects_empty_columns():
    df = pd.DataFrame({"candidate": [None, None]})

    assert should_parse_column_as_date(df, "candidate") is False


def test_get_transitive_closure_from_counter_ignores_zero_weight_edges():
    relation = Counter({("A", "B"): 2, ("B", "C"): 1, ("A", "C"): 0})

    closure = get_transitive_closure_from_counter(relation)

    assert closure["A"] == {"B", "C"}
    assert closure["B"] == {"C"}
    assert closure["C"] == set()


def test_split_graph_into_stages_for_linear_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([("START", "A"), ("A", "B"), ("B", "END")])

    stages, skippable = split_graph_into_stages(graph, "START", "END")

    assert stages == [{"START"}, {"A"}, {"B"}, {"END"}]
    assert skippable == [False, False, False, False]


def test_split_graph_into_stages_marks_skippable_region():
    graph = nx.DiGraph()
    graph.add_edges_from([("START", "A"), ("A", "END"), ("START", "END")])

    stages, skippable = split_graph_into_stages(graph, "START", "END")

    assert stages == [{"START"}, {"A"}, {"END"}]
    assert skippable == [False, True, False]


def test_split_graph_into_stages_merges_cycle_into_single_stage():
    graph = nx.DiGraph()
    graph.add_edges_from(
        [("START", "A"), ("A", "B"), ("B", "A"), ("B", "END")]
    )

    stages, skippable = split_graph_into_stages(graph, "START", "END")

    assert stages == [{"START"}, {"A", "B"}, {"END"}]
    assert skippable == [False, False, False]


def test_layout_bpmn_rejects_string_input():
    with pytest.raises(ValueError, match="must be a PM4Py BPMN object"):
        layout_bpmn("<bpmn />")


def test_layout_bpmn_generates_bpmn_xml(running_example_model):
    bpmn = powl.convert_to_bpmn(running_example_model)

    xml = layout_bpmn(bpmn)

    assert xml.startswith('<?xml version="1.0"')
    assert "<bpmn:definitions" in xml


def test_layout_bpmn_xml_handles_raw_bpmn(examples_dir):
    diagram_1_path = examples_dir / "diagram_1.bpmn"
    if diagram_1_path.exists():
        raw_xml = diagram_1_path.read_text(encoding="utf-8")
        result = layout_bpmn_xml(raw_xml)
        assert "<bpmndi:BPMNDiagram" in result
        assert "<bpmndi:BPMNShape" in result
        assert "<bpmndi:BPMNEdge" in result


def test_convert_to_bpmn_with_resources_generates_collaboration(running_example_model):
    activity_to_pool_lane = {
        "register request": ("P1", "Lane1"),
        "reinitiate request": ("P2", "Lane1"),
        "pay compensation": ("P1", "Lane1"),
        "reject request": ("P1", "Lane1"),
        "examine casually": ("P2", "Lane2"),
        "examine thoroughly": ("P1", "Lane2"),
        "check ticket": ("P2", "Lane3"),
        "decide": ("P1", "Lane3"),
    }
    xml = powl.convert_to_bpmn_with_resources(activity_to_pool_lane, running_example_model)
    assert "<bpmn:collaboration" in xml
    assert 'name="P1"' in xml
    assert 'name="P2"' in xml
    assert 'name="Lane1"' in xml
    assert "<bpmn:messageFlow" in xml


def test_bpmn_visualizer_apply(running_example_model):
    bpmn = powl.convert_to_bpmn(running_example_model)
    gviz = bpmn_visualizer.apply(bpmn)
    assert gviz is not None

