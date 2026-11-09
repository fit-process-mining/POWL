"""Public, POWL-centric model export contract."""

import xml.etree.ElementTree as ET

import pytest

import powl


BPMNDI = "http://www.omg.org/spec/BPMN/20100524/DI"
BPMN = "http://www.omg.org/spec/BPMN/20100524/MODEL"


def test_public_api_exposes_powl_export_not_generic_bpmn_layout():
    assert callable(powl.export)
    assert "export" in powl.__all__
    for name in ("save_bpmn", "layout_bpmn", "layout_bpmn_xml"):
        assert name not in powl.__all__
        assert not hasattr(powl, name)


def test_export_powl_roundtrip_with_metadata(tmp_path):
    model = powl.sequence([powl.Activity("Receive"), powl.Activity("Check")])
    output = tmp_path / "model.powl"

    assert powl.export(model, output, metadata={"name": "Example"}) is None
    restored = powl.read_powl_json(output)

    assert output.exists()
    assert model.same_structure(restored)
    assert '"name": "Example"' in output.read_text(encoding="utf-8")


def test_export_bpmn_writes_laid_out_diagram(tmp_path):
    model = powl.sequence([powl.Activity("Receive"), powl.Activity("Check")])
    output = tmp_path / "model.bpmn"

    assert powl.export(model, output) is None
    root = ET.parse(output).getroot()
    shapes = root.findall(f".//{{{BPMNDI}}}BPMNShape")
    tasks = root.findall(f".//{{{BPMN}}}task")
    assert len(tasks) == 2
    assert len(shapes) >= 4  # two tasks and two events


def test_export_bpmn_with_pools_and_lanes(tmp_path):
    model = powl.sequence([powl.Activity("Receive"), powl.Activity("Check")])
    assignments = {"Receive": ("Customer", "Requester"), "Check": ("Team", "Reviewer")}
    original = dict(assignments)
    output = tmp_path / "collaboration.bpmn"

    powl.export(model, output, activity_to_pool_lane=assignments)

    root = ET.parse(output).getroot()
    assert root.find(f".//{{{BPMN}}}collaboration") is not None
    assert len(root.findall(f".//{{{BPMN}}}participant")) == 2
    assert root.findall(f".//{{{BPMNDI}}}BPMNShape")
    assert assignments == original


@pytest.mark.parametrize("suffix", ["", ".xml", ".json", ".svg"])
def test_export_rejects_other_formats(tmp_path, suffix):
    model = powl.Activity("A")
    with pytest.raises(ValueError, match=".powl or .bpmn"):
        powl.export(model, tmp_path / ("model" + suffix))


def test_export_rejects_non_powl_input(tmp_path):
    with pytest.raises(TypeError, match="POWL model"):
        powl.export('<bpmn:definitions />', tmp_path / 'model.bpmn')


def test_export_rejects_options_for_wrong_format(tmp_path):
    model = powl.Activity("A")
    with pytest.raises(ValueError, match="only supported for .powl"):
        powl.export(model, tmp_path / "model.bpmn", metadata={"name": "Example"})
    with pytest.raises(ValueError, match="only supported for .bpmn"):
        powl.export(model, tmp_path / "model.powl", activity_to_pool_lane={})
