from collections import Counter
import xml.etree.ElementTree as ET

import networkx as nx
import pandas as pd
import pytest

import powl
from powl.objects.utils.graph_sequentialization import split_graph_into_stages
from powl.objects.utils.relation import get_transitive_closure_from_counter
from powl.general_utils.time_utils import should_parse_column_as_date
from powl.visualization.bpmn.layout import layout_bpmn, layout_bpmn_xml, save_bpmn
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


def test_save_bpmn_from_pm4py_model_layouts_and_writes(tmp_path, running_example_model):
    model = powl.convert_to_bpmn(running_example_model)
    path = tmp_path / "laid-out.bpmn"

    result = save_bpmn(model, path)

    assert result is None
    xml = path.read_text(encoding="utf-8")
    assert xml.startswith('<?xml version="1.0"')
    root = ET.fromstring(xml)
    ns = {"bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI"}
    assert root.findall(".//bpmndi:BPMNShape", ns)


def test_save_bpmn_from_xml_layouts_and_writes(tmp_path):
    xml = """<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
    id="Definitions_1" targetNamespace="urn:test"><bpmn:process id="Process_1">
    <bpmn:startEvent id="start"/><bpmn:endEvent id="end"/>
    <bpmn:sequenceFlow id="f1" sourceRef="start" targetRef="end"/>
    </bpmn:process></bpmn:definitions>"""
    path = tmp_path / "layout.bpmn"

    save_bpmn(xml, str(path))

    root = ET.fromstring(path.read_text(encoding="utf-8"))
    ns = {"bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI"}
    assert len(root.findall(".//bpmndi:BPMNShape", ns)) == 2


def test_save_bpmn_can_skip_layout_for_existing_xml(tmp_path, monkeypatch):
    xml = '<?xml version="1.0"?><definitions/>'

    def fail_layout(_xml):
        raise AssertionError("Graphviz must not be called")

    monkeypatch.setattr("powl.visualization.bpmn.layout.layout_bpmn_xml", fail_layout)
    path = tmp_path / "as-is.bpmn"
    save_bpmn(xml, path, layout=False)
    assert path.read_text(encoding="utf-8") == xml


def test_save_bpmn_rejects_other_input_types(tmp_path):
    with pytest.raises(TypeError, match="PM4Py BPMN object or a BPMN XML string"):
        save_bpmn(123, tmp_path / "invalid.bpmn")
    assert not (tmp_path / "invalid.bpmn").exists()


def test_layout_bpmn_rejects_string_input():
    with pytest.raises(ValueError, match="must be a PM4Py BPMN object"):
        layout_bpmn("<bpmn />")


def test_layout_bpmn_generates_bpmn_xml(running_example_model):
    bpmn = powl.convert_to_bpmn(running_example_model)

    xml = layout_bpmn(bpmn)

    assert xml.startswith('<?xml version="1.0"')
    assert "<bpmn:definitions" in xml


def test_layout_bpmn_xml_handles_raw_bpmn():
    raw_xml = """<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
      id="Definitions_1" targetNamespace="urn:test">
      <bpmn:process id="Process_1">
        <bpmn:startEvent id="start"/>
        <bpmn:task id="task" name="Review"/>
        <bpmn:endEvent id="end"/>
        <bpmn:sequenceFlow id="f1" sourceRef="start" targetRef="task"/>
        <bpmn:sequenceFlow id="f2" sourceRef="task" targetRef="end"/>
      </bpmn:process>
    </bpmn:definitions>"""
    result = layout_bpmn_xml(raw_xml)
    assert result.startswith('<?xml version="1.0"')
    ns = {
        "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
        "dc": "http://www.omg.org/spec/DD/20100524/DC",
        "di": "http://www.omg.org/spec/DD/20100524/DI",
    }
    root = ET.fromstring(result)
    shapes = root.findall(".//bpmndi:BPMNShape", ns)
    edges = root.findall(".//bpmndi:BPMNEdge", ns)
    assert len(shapes) == 3
    assert len(edges) == 2
    assert all(len(edge.findall("di:waypoint", ns)) >= 2 for edge in edges)
    assert all(float(shape.find("dc:Bounds", ns).attrib["width"]) > 0 for shape in shapes)


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



def test_convert_to_bpmn_with_resources_preserves_input_map(running_example_model):
    mapping = {
        "register request": ("P1", None),
        "examine casually": ("P2", "Review"),
    }
    original_mapping = dict(mapping)
    xml = powl.convert_to_bpmn_with_resources(mapping, running_example_model)
    assert mapping == original_mapping
    assert '<bpmn:collaboration' in xml
    assert 'name="DefaultLane"' in xml


def test_partially_ordered_log_uses_custom_lifecycle_and_preserves_dataframe():
    from powl.discovery.partial_order_based.utils.log_to_partial_orders import apply

    log = pd.DataFrame(
        {
            "case": ["1", "1", "1", "1"],
            "activity": ["A", "A", "B", "B"],
            "when": [
                "2024-01-01T10:00:00",
                "2024-01-01T10:01:00",
                "2024-01-01T10:02:00",
                "2024-01-01T10:03:00",
            ],
            "phase": ["start", "complete", "start", "complete"],
        }
    )
    original = log.copy(deep=True)
    variants = apply(
        log,
        case_id_col="case",
        activity_col="activity",
        ordering_col="when",
        lifecycle_col="phase",
        start_transitions={"start"},
        complete_transitions={"complete"},
    )
    assert len(variants) == 1
    assert {activity.label for activity in variants[0].nodes} == {"A", "B"}
    pd.testing.assert_frame_equal(log, original)
