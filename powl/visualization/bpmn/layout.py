from __future__ import annotations

import sys
from pathlib import Path

# Ensure resource_utils is in sys.path so bare imports inside resource_utils resolve seamlessly
_resource_utils_dir = str((Path(__file__).parent / "resource_utils").resolve())
if _resource_utils_dir not in sys.path:
    sys.path.insert(0, _resource_utils_dir)

import pm4py
from pm4py.objects.bpmn.exporter.variants.etree import get_xml_string

from input_parser import parse_from_string
from layouter import layout_model
from layout_post_processing import post_process_layout
from output_serializer import parse_to_string


def layout_bpmn_xml(xml_content: str) -> str:
    """
    Takes an unlayouted BPMN 2.0 XML string and returns a cleanly layouted BPMN XML string.
    Runs parsing, Graphviz LR layout, pool ordering, lane sizing, orthogonal routing,
    and serialization.
    """
    model = parse_from_string(xml_content)
    layout_model(model)
    post_process_layout(model)
    xml_str = parse_to_string(model)
    if xml_str.startswith("<?xml version='1.0'"):
        xml_str = '<?xml version="1.0"' + xml_str[len("<?xml version='1.0'"):]
    return xml_str


def layout_bpmn(bpmn_model: pm4py.objects.bpmn.obj.BPMN) -> str:
    """
    Apply layout to a BPMN model.

    :param bpmn_model: PM4Py BPMN object (for raw XML strings, use layout_bpmn_xml)
    :return: String representation of the layouted BPMN model
    """
    if isinstance(bpmn_model, str):
        raise ValueError("Input must be a PM4Py BPMN object, not a string.")
    bpmn_xml = get_xml_string(bpmn_model).decode("utf-8")
    return layout_bpmn_xml(bpmn_xml)
