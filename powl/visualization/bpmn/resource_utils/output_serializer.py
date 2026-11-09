"""BPMN Output Serializer.
Serializes the internal BPMN representation and computed layout geometry into a valid BPMN 2.0 XML file
with complete bpmndi:BPMNDiagram, bpmndi:BPMNPlane, bpmndi:BPMNShape, and bpmndi:BPMNEdge elements.
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import Optional

from constants import NAMESPACES
from input_parser import (
    BPMNModel,
    Process,
    Lane,
    FlowNode,
    FlowEdge,
    Participant,
    TextAnnotation,
    Bounds,
    Waypoint,
)


def _register_all_namespaces():
    """Register standard BPMN namespaces with ElementTree."""
    for prefix, uri in NAMESPACES.items():
        ET.register_namespace(prefix, uri)


def _format_num(val: float) -> str:
    """Format float coordinates to integer string if whole, else clean float string."""
    if val.is_integer():
        return str(int(val))
    return f"{val:.1f}"


def _build_or_update_diagram(model: BPMNModel) -> ET.Element:
    """Build or update BPMN XML tree with generated bpmndi:BPMNDiagram structure."""
    _register_all_namespaces()

    root = model.root_element
    if root is None:
        # Construct fresh definitions element
        root = ET.Element(
            f"{{{NAMESPACES['bpmn']}}}definitions",
            attrib={
                "id": model.definitions_id,
                "targetNamespace": model.target_namespace,
            },
        )
        model.root_element = root

    # Remove any existing BPMNDiagram elements
    for child in list(root):
        if child.tag.endswith("BPMNDiagram"):
            root.remove(child)

    # 1. Main BPMNDiagram
    diag_elem = ET.SubElement(
        root,
        f"{{{NAMESPACES['bpmndi']}}}BPMNDiagram",
        attrib={"id": "BPMNDiagram_1"},
    )

    plane_elem_id = model.collaboration.id if model.collaboration else (model.processes[0].id if model.processes else "Process_1")
    plane_elem = ET.SubElement(
        diag_elem,
        f"{{{NAMESPACES['bpmndi']}}}BPMNPlane",
        attrib={"id": "BPMNPlane_1", "bpmnElement": plane_elem_id},
    )

    # 2. Add Collaboration Participants (Pools)
    if model.collaboration:
        for part in model.collaboration.participants:
            part_shape = ET.SubElement(
                plane_elem,
                f"{{{NAMESPACES['bpmndi']}}}BPMNShape",
                attrib={
                    "id": f"{part.id}_di",
                    "bpmnElement": part.id,
                    "isHorizontal": "true",
                },
            )
            ET.SubElement(
                part_shape,
                f"{{{NAMESPACES['dc']}}}Bounds",
                attrib={
                    "x": _format_num(part.bounds.x),
                    "y": _format_num(part.bounds.y),
                    "width": _format_num(part.bounds.width),
                    "height": _format_num(part.bounds.height),
                },
            )
            ET.SubElement(part_shape, f"{{{NAMESPACES['bpmndi']}}}BPMNLabel")

    # 3. Add Lanes
    for proc in model.processes:
        for lane in proc.lanes:
            lane_shape = ET.SubElement(
                plane_elem,
                f"{{{NAMESPACES['bpmndi']}}}BPMNShape",
                attrib={
                    "id": f"{lane.id}_di",
                    "bpmnElement": lane.id,
                    "isHorizontal": "true",
                },
            )
            ET.SubElement(
                lane_shape,
                f"{{{NAMESPACES['dc']}}}Bounds",
                attrib={
                    "x": _format_num(lane.bounds.x),
                    "y": _format_num(lane.bounds.y),
                    "width": _format_num(lane.bounds.width),
                    "height": _format_num(lane.bounds.height),
                },
            )
            ET.SubElement(lane_shape, f"{{{NAMESPACES['bpmndi']}}}BPMNLabel")

    # 4. Add Flow Nodes (Activities, Gateways, Events)
    for proc in model.processes:
        for node_id, node in proc.flow_nodes.items():
            attribs = {
                "id": f"{node.id}_di",
                "bpmnElement": node.id,
            }
            if node.element_type in ("exclusiveGateway", "inclusiveGateway"):
                attribs["isMarkerVisible"] = "true"

            node_shape = ET.SubElement(
                plane_elem,
                f"{{{NAMESPACES['bpmndi']}}}BPMNShape",
                attrib=attribs,
            )
            ET.SubElement(
                node_shape,
                f"{{{NAMESPACES['dc']}}}Bounds",
                attrib={
                    "x": _format_num(node.bounds.x),
                    "y": _format_num(node.bounds.y),
                    "width": _format_num(node.bounds.width),
                    "height": _format_num(node.bounds.height),
                },
            )
            if node.name:
                label = ET.SubElement(node_shape, f"{{{NAMESPACES['bpmndi']}}}BPMNLabel")
                if "Event" in node.element_type or "Gateway" in node.element_type:
                    if node.label_bounds is not None:
                        lb = node.label_bounds
                    else:
                        lb = Bounds(
                            x=node.bounds.x - 15.0,
                            y=node.bounds.y + node.bounds.height + 6.0,
                            width=node.bounds.width + 30.0,
                            height=14.0,
                        )
                    ET.SubElement(
                        label,
                        f"{{{NAMESPACES['dc']}}}Bounds",
                        attrib={
                            "x": _format_num(lb.x),
                            "y": _format_num(lb.y),
                            "width": _format_num(lb.width),
                            "height": _format_num(lb.height),
                        },
                    )
            else:
                ET.SubElement(node_shape, f"{{{NAMESPACES['bpmndi']}}}BPMNLabel")

    # 5. Add Text Annotations (from collaboration and/or processes)
    for annot in model.get_all_text_annotations():
        annot_shape = ET.SubElement(
            plane_elem,
            f"{{{NAMESPACES['bpmndi']}}}BPMNShape",
            attrib={
                "id": f"{annot.id}_di",
                "bpmnElement": annot.id,
            },
        )
        ET.SubElement(
            annot_shape,
            f"{{{NAMESPACES['dc']}}}Bounds",
            attrib={
                "x": _format_num(annot.bounds.x),
                "y": _format_num(annot.bounds.y),
                "width": _format_num(annot.bounds.width),
                "height": _format_num(annot.bounds.height),
            },
        )
        ET.SubElement(annot_shape, f"{{{NAMESPACES['bpmndi']}}}BPMNLabel")

    # 6. Add Sequence Flow Edges
    for proc in model.processes:
        for sf in proc.sequence_flows:
            edge_elem = ET.SubElement(
                plane_elem,
                f"{{{NAMESPACES['bpmndi']}}}BPMNEdge",
                attrib={
                    "id": f"{sf.id}_di",
                    "bpmnElement": sf.id,
                },
            )
            for wp in sf.waypoints:
                ET.SubElement(
                    edge_elem,
                    f"{{{NAMESPACES['di']}}}waypoint",
                    attrib={"x": _format_num(wp.x), "y": _format_num(wp.y)},
                )

    # 7. Add Message Flows
    if model.collaboration:
        for mf in model.collaboration.message_flows:
            mf_elem = ET.SubElement(
                plane_elem,
                f"{{{NAMESPACES['bpmndi']}}}BPMNEdge",
                attrib={
                    "id": f"{mf.id}_di",
                    "bpmnElement": mf.id,
                },
            )
            for wp in mf.waypoints:
                ET.SubElement(
                    mf_elem,
                    f"{{{NAMESPACES['di']}}}waypoint",
                    attrib={"x": _format_num(wp.x), "y": _format_num(wp.y)},
                )
            if mf.name:
                mf_label = ET.SubElement(mf_elem, f"{{{NAMESPACES['bpmndi']}}}BPMNLabel")
                if len(mf.waypoints) >= 2:
                    mid_wp = mf.waypoints[len(mf.waypoints) // 2]
                    ET.SubElement(
                        mf_label,
                        f"{{{NAMESPACES['dc']}}}Bounds",
                        attrib={
                            "x": _format_num(mid_wp.x - 20.0),
                            "y": _format_num(mid_wp.y - 10.0),
                            "width": _format_num(max(40.0, len(mf.name) * 7.0)),
                            "height": "14",
                        },
                    )
            else:
                ET.SubElement(mf_elem, f"{{{NAMESPACES['bpmndi']}}}BPMNLabel")

    # 8. Add Associations (from collaboration and/or processes)
    for assoc in model.get_all_associations():
        assoc_elem = ET.SubElement(
            plane_elem,
            f"{{{NAMESPACES['bpmndi']}}}BPMNEdge",
            attrib={
                "id": f"{assoc.id}_di",
                "bpmnElement": assoc.id,
            },
        )
        for wp in assoc.waypoints:
            ET.SubElement(
                assoc_elem,
                f"{{{NAMESPACES['di']}}}waypoint",
                attrib={"x": _format_num(wp.x), "y": _format_num(wp.y)},
            )

    # 9. Add Sub-Process Planes if any sub-processes exist
    subproc_idx = 0
    for proc in model.processes:
        for node in proc.flow_nodes.values():
            if node.element_type in ("subProcess", "adHocSubProcess"):
                subproc_idx += 1
                sp_diag = ET.SubElement(
                    root,
                    f"{{{NAMESPACES['bpmndi']}}}BPMNDiagram",
                    attrib={"id": f"BPMNDiagram_sub_{subproc_idx}"},
                )
                ET.SubElement(
                    sp_diag,
                    f"{{{NAMESPACES['bpmndi']}}}BPMNPlane",
                    attrib={
                        "id": f"BPMNPlane_sub_{subproc_idx}",
                        "bpmnElement": node.id,
                    },
                )

    return root


def parse_to_string(model: BPMNModel) -> str:
    """Serialize the internal BPMN model into a valid BPMN 2.0 XML string."""
    root = _build_or_update_diagram(model)
    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8")


def parse_to_file_path(model: BPMNModel, file_path: str) -> None:
    """Serialize the internal BPMN model into a valid BPMN 2.0 XML file at the given path."""
    xml_str = parse_to_string(model)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
        if not xml_str.endswith("\n"):
            f.write("\n")
