from collections import Counter

import networkx as nx
import pandas as pd
import pytest

import powl
from powl.objects.utils.graph_sequentialization import split_graph_into_stages
from powl.objects.utils.relation import get_transitive_closure_from_counter
from powl.general_utils.time_utils import should_parse_column_as_date
from powl.visualization.bpmn.layout import layout_bpmn


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
