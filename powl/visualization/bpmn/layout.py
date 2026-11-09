from __future__ import annotations

from pathlib import Path

import pm4py
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.bpmn.exporter.variants.etree import get_xml_string

from .resource_utils.input_parser import parse_from_string
from .resource_utils.layouter import layout_model
from .resource_utils.layout_post_processing import post_process_layout
from .resource_utils.output_serializer import parse_to_string


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


def save_bpmn(
    bpmn_model: BPMN | str, file_path: str | Path, *, layout: bool = True
) -> None:
    """Save a BPMN model or BPMN XML string as UTF-8 XML.

    Applies the Graphviz BPMN layout by default. Use ``layout=False`` when
    the XML already contains the desired layout, such as output from
    ``convert_to_bpmn_with_resources``. Parent directories must exist.

    :param bpmn_model: PM4Py BPMN object or BPMN XML content (not a file path)
    :param file_path: destination .bpmn or .xml file (overwritten if present)
    :param layout: whether to apply Graphviz layout before writing
    """
    if isinstance(bpmn_model, str):
        xml = layout_bpmn_xml(bpmn_model) if layout else bpmn_model
    elif isinstance(bpmn_model, BPMN):
        xml = (
            layout_bpmn(bpmn_model)
            if layout
            else get_xml_string(bpmn_model).decode("utf-8")
        )
    else:
        raise TypeError("bpmn_model must be a PM4Py BPMN object or a BPMN XML string")

    Path(file_path).write_text(xml, encoding="utf-8")
