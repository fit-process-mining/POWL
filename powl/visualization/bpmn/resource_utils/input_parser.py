"""Input parser for BPMN 2.0 XML files.
Parses semantic BPMN elements (collaboration, processes, lanes, activities, gateways,
events, annotations, markers, and flow relationships) into a structured internal representation.
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from .constants import (
    NAMESPACES,
    ELEMENT_BOUNDS,
    DEFAULT_NODE_BOUNDS,
    GATEWAY_TAGS,
    TASK_TAGS,
    EVENT_TAGS,
)


@dataclass
class Bounds:
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class Waypoint:
    x: float
    y: float

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass
class FlowNode:
    id: str
    name: str = ""
    element_type: str = "task"
    lane_id: Optional[str] = None
    incoming: List[str] = field(default_factory=list)
    outgoing: List[str] = field(default_factory=list)
    markers: List[str] = field(default_factory=list)
    event_definitions: List[str] = field(default_factory=list)
    bounds: Bounds = field(default_factory=Bounds)
    label_bounds: Optional[Bounds] = None
    xml_element: Optional[ET.Element] = None


@dataclass
class FlowEdge:
    id: str
    name: str = ""
    edge_type: str = "sequenceFlow"  # sequenceFlow, messageFlow, association
    source_ref: str = ""
    target_ref: str = ""
    waypoints: List[Waypoint] = field(default_factory=list)
    xml_element: Optional[ET.Element] = None


@dataclass
class TextAnnotation:
    id: str
    text: str = ""
    bounds: Bounds = field(default_factory=Bounds)
    xml_element: Optional[ET.Element] = None


@dataclass
class Lane:
    id: str
    name: str = ""
    flow_node_refs: List[str] = field(default_factory=list)
    bounds: Bounds = field(default_factory=Bounds)
    xml_element: Optional[ET.Element] = None


@dataclass
class Participant:
    id: str
    name: str = ""
    process_ref: Optional[str] = None
    bounds: Bounds = field(default_factory=Bounds)
    xml_element: Optional[ET.Element] = None

    @property
    def is_black_box(self) -> bool:
        return self.process_ref is None or len(self.process_ref.strip()) == 0


@dataclass
class Process:
    id: str
    name: str = ""
    is_executable: bool = True
    lanes: List[Lane] = field(default_factory=list)
    flow_nodes: Dict[str, FlowNode] = field(default_factory=dict)
    sequence_flows: List[FlowEdge] = field(default_factory=list)
    text_annotations: List[TextAnnotation] = field(default_factory=list)
    associations: List[FlowEdge] = field(default_factory=list)
    xml_element: Optional[ET.Element] = None


@dataclass
class Collaboration:
    id: str
    participants: List[Participant] = field(default_factory=list)
    message_flows: List[FlowEdge] = field(default_factory=list)
    text_annotations: List[TextAnnotation] = field(default_factory=list)
    associations: List[FlowEdge] = field(default_factory=list)
    xml_element: Optional[ET.Element] = None


@dataclass
class BPMNModel:
    definitions_id: str = "Definitions_1"
    target_namespace: str = "http://bpmn.io/schema/bpmn"
    collaboration: Optional[Collaboration] = None
    processes: List[Process] = field(default_factory=list)
    root_element: Optional[ET.Element] = None

    def get_flow_node(self, node_id: str) -> Optional[FlowNode]:
        for proc in self.processes:
            if node_id in proc.flow_nodes:
                return proc.flow_nodes[node_id]
        return None

    def get_process(self, process_id: str) -> Optional[Process]:
        for proc in self.processes:
            if proc.id == process_id:
                return proc
        return None

    def get_participant(self, participant_id: str) -> Optional[Participant]:
        if self.collaboration:
            for p in self.collaboration.participants:
                if p.id == participant_id:
                    return p
        return None

    def get_participant_for_process(self, process_id: str) -> Optional[Participant]:
        if self.collaboration:
            for p in self.collaboration.participants:
                if p.process_ref == process_id:
                    return p
        return None

    def get_process_for_node(self, node_id: str) -> Optional[Process]:
        for proc in self.processes:
            if node_id in proc.flow_nodes:
                return proc
        return None

    def get_all_message_flows(self) -> List[FlowEdge]:
        flows = []
        if self.collaboration:
            flows.extend(self.collaboration.message_flows)
        return flows

    def get_element_bounds_and_kind(self, elem_id: str) -> Tuple[Optional[Bounds], str]:
        """Lookup bounds and element kind (node, blackbox_participant, process_participant, annotation, unknown)."""
        node = self.get_flow_node(elem_id)
        if node:
            return node.bounds, "node"

        part = self.get_participant(elem_id)
        if part:
            return part.bounds, ("blackbox_participant" if part.is_black_box else "process_participant")

        proc_part = self.get_participant_for_process(elem_id)
        if proc_part:
            return proc_part.bounds, "process_participant"

        for annot in self.get_all_text_annotations():
            if annot.id == elem_id:
                return annot.bounds, "annotation"

        return None, "unknown"

    def get_all_text_annotations(self) -> List[TextAnnotation]:
        annots = []
        if self.collaboration:
            annots.extend(self.collaboration.text_annotations)
        for proc in self.processes:
            annots.extend(proc.text_annotations)
        return annots

    def get_all_associations(self) -> List[FlowEdge]:
        assocs = []
        if self.collaboration:
            assocs.extend(self.collaboration.associations)
        for proc in self.processes:
            annocs = proc.associations
            assocs.extend(annocs)
        return assocs


def estimate_annotation_bounds(text: str) -> Bounds:
    """Calculate realistic width and height for text annotations based on text content."""
    if not text or not text.strip():
        return Bounds(width=104.0, height=40.0)

    text = text.strip()
    width = 120.0
    chars_per_line = max(12, int(width / 7.0))  # ~17 chars/line

    total_lines = 0
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            total_lines += 1
            continue
        cur_len = 0
        p_lines = 1
        for w in words:
            if cur_len + len(w) + 1 > chars_per_line:
                p_lines += 1
                cur_len = len(w)
            else:
                cur_len += len(w) + 1
        total_lines += p_lines

    line_height = 16.0
    height = max(40.0, total_lines * line_height + 16.0)
    return Bounds(width=round(width, 1), height=round(height, 1))


def _strip_ns(tag: str) -> str:
    """Strip XML namespace prefix from tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_child_by_tag(element: ET.Element, local_tag: str) -> Optional[ET.Element]:
    """Find immediate child element matching local tag name regardless of namespace."""
    for child in element:
        if _strip_ns(child.tag) == local_tag:
            return child
    return None


def _find_all_by_tag(element: ET.Element, local_tag: str) -> List[ET.Element]:
    """Find all immediate children matching local tag name regardless of namespace."""
    return [child for child in element if _strip_ns(child.tag) == local_tag]


def _parse_task_markers(elem: ET.Element) -> List[str]:
    """Extract loop / multi-instance markers from task or subprocess elements."""
    markers = []
    for child in elem:
        tag_local = _strip_ns(child.tag)
        if tag_local == "multiInstanceLoopCharacteristics":
            is_seq = child.attrib.get("isSequential", "").lower() in ("true", "1")
            if is_seq:
                markers.append("multiInstanceSequential")
            else:
                markers.append("multiInstanceParallel")
        elif tag_local == "standardLoopCharacteristics":
            markers.append("standardLoop")
    return markers


def _parse_event_definitions(elem: ET.Element) -> List[str]:
    """Extract event definitions (timer, message, etc.) from an event element."""
    defs = []
    for child in elem:
        tag_local = _strip_ns(child.tag)
        if tag_local.endswith("EventDefinition"):
            defs.append(tag_local)
    return defs


def _parse_collaboration(collab_elem: ET.Element, root: Optional[ET.Element] = None) -> Collaboration:
    """Parse bpmn:collaboration into Collaboration dataclass."""
    collab = Collaboration(
        id=collab_elem.attrib.get("id", "Collaboration_1"),
        xml_element=collab_elem,
    )

    # Build map of parsed bounds from input XML if present
    parsed_bounds_map = {}
    if root is not None:
        for shape in root.findall(f".//{{{NAMESPACES['bpmndi']}}}BPMNShape"):
            elem_id = shape.attrib.get("bpmnElement")
            b_elem = shape.find(f"{{{NAMESPACES['dc']}}}Bounds")
            if b_elem is not None and elem_id:
                try:
                    w = float(b_elem.attrib.get("width", 0))
                    h = float(b_elem.attrib.get("height", 0))
                    if w > 0 and h > 0:
                        parsed_bounds_map[elem_id] = Bounds(width=w, height=h)
                except (ValueError, TypeError):
                    pass

    for child in collab_elem:
        tag_local = _strip_ns(child.tag)
        child_id = child.attrib.get("id", "")
        child_name = child.attrib.get("name", "")

        if tag_local == "participant":
            process_ref = child.attrib.get("processRef")
            participant = Participant(
                id=child_id,
                name=child_name,
                process_ref=process_ref,
                xml_element=child,
            )
            collab.participants.append(participant)

        elif tag_local == "messageFlow":
            m_flow = FlowEdge(
                id=child_id,
                name=child_name,
                edge_type="messageFlow",
                source_ref=child.attrib.get("sourceRef", ""),
                target_ref=child.attrib.get("targetRef", ""),
                xml_element=child,
            )
            collab.message_flows.append(m_flow)

        elif tag_local == "textAnnotation":
            text_elem = _find_child_by_tag(child, "text")
            text_content = text_elem.text if text_elem is not None and text_elem.text else ""
            b = parsed_bounds_map.get(child_id) or estimate_annotation_bounds(text_content)
            annotation = TextAnnotation(
                id=child_id,
                text=text_content,
                bounds=b,
                xml_element=child,
            )
            collab.text_annotations.append(annotation)

        elif tag_local == "association":
            assoc = FlowEdge(
                id=child_id,
                name=child_name,
                edge_type="association",
                source_ref=child.attrib.get("sourceRef", ""),
                target_ref=child.attrib.get("targetRef", ""),
                xml_element=child,
            )
            collab.associations.append(assoc)

    return collab


def _parse_process(proc_elem: ET.Element, root: Optional[ET.Element] = None) -> Process:
    """Parse bpmn:process into Process dataclass."""
    process = Process(
        id=proc_elem.attrib.get("id", "Process_1"),
        name=proc_elem.attrib.get("name", ""),
        is_executable=proc_elem.attrib.get("isExecutable", "true").lower() == "true",
        xml_element=proc_elem,
    )

    parsed_bounds_map = {}
    if root is not None:
        for shape in root.findall(f".//{{{NAMESPACES['bpmndi']}}}BPMNShape"):
            elem_id = shape.attrib.get("bpmnElement")
            b_elem = shape.find(f"{{{NAMESPACES['dc']}}}Bounds")
            if b_elem is not None and elem_id:
                try:
                    w = float(b_elem.attrib.get("width", 0))
                    h = float(b_elem.attrib.get("height", 0))
                    if w > 0 and h > 0:
                        parsed_bounds_map[elem_id] = Bounds(width=w, height=h)
                except (ValueError, TypeError):
                    pass

    # 1. Parse LaneSet / Lanes
    lane_sets = _find_all_by_tag(proc_elem, "laneSet")
    for lane_set in lane_sets:
        lanes = _find_all_by_tag(lane_set, "lane")
        for l_elem in lanes:
            lane_id = l_elem.attrib.get("id", "")
            lane_name = l_elem.attrib.get("name", "")
            node_refs = []
            for ref_elem in _find_all_by_tag(l_elem, "flowNodeRef"):
                if ref_elem.text:
                    node_refs.append(ref_elem.text.strip())
            lane = Lane(
                id=lane_id,
                name=lane_name,
                flow_node_refs=node_refs,
                xml_element=l_elem,
            )
            process.lanes.append(lane)

    # Build node_id -> lane_id mapping
    node_to_lane = {}
    for lane in process.lanes:
        for ref in lane.flow_node_refs:
            node_to_lane[ref] = lane.id

    # 2. Parse Flow Nodes, Sequence Flows, Text Annotations, Associations
    all_flow_node_tags = GATEWAY_TAGS | TASK_TAGS | EVENT_TAGS

    for child in proc_elem:
        tag_local = _strip_ns(child.tag)
        child_id = child.attrib.get("id", "")
        child_name = child.attrib.get("name", "")

        if tag_local in all_flow_node_tags:
            incomings = [e.text.strip() for e in _find_all_by_tag(child, "incoming") if e.text]
            outgoings = [e.text.strip() for e in _find_all_by_tag(child, "outgoing") if e.text]
            markers = _parse_task_markers(child)
            event_defs = _parse_event_definitions(child)

            default_b = ELEMENT_BOUNDS.get(tag_local, DEFAULT_NODE_BOUNDS)
            bounds = Bounds(width=default_b["width"], height=default_b["height"])

            node = FlowNode(
                id=child_id,
                name=child_name,
                element_type=tag_local,
                lane_id=node_to_lane.get(child_id),
                incoming=incomings,
                outgoing=outgoings,
                markers=markers,
                event_definitions=event_defs,
                bounds=bounds,
                xml_element=child,
            )
            process.flow_nodes[child_id] = node

        elif tag_local == "sequenceFlow":
            s_flow = FlowEdge(
                id=child_id,
                name=child_name,
                edge_type="sequenceFlow",
                source_ref=child.attrib.get("sourceRef", ""),
                target_ref=child.attrib.get("targetRef", ""),
                xml_element=child,
            )
            process.sequence_flows.append(s_flow)

        elif tag_local == "textAnnotation":
            text_elem = _find_child_by_tag(child, "text")
            text_content = text_elem.text if text_elem is not None and text_elem.text else ""
            b = parsed_bounds_map.get(child_id) or estimate_annotation_bounds(text_content)
            annotation = TextAnnotation(
                id=child_id,
                text=text_content,
                bounds=b,
                xml_element=child,
            )
            process.text_annotations.append(annotation)

        elif tag_local == "association":
            assoc = FlowEdge(
                id=child_id,
                name=child_name,
                edge_type="association",
                source_ref=child.attrib.get("sourceRef", ""),
                target_ref=child.attrib.get("targetRef", ""),
                xml_element=child,
            )
            process.associations.append(assoc)

    return process


def parse_from_string(xml_str: str) -> BPMNModel:
    """Parse a BPMN 2.0 XML string into an internal BPMNModel representation."""
    root = ET.fromstring(xml_str)
    model = BPMNModel(
        definitions_id=root.attrib.get("id", "Definitions_1"),
        target_namespace=root.attrib.get("targetNamespace", "http://bpmn.io/schema/bpmn"),
        root_element=root,
    )

    # Check for collaboration
    collab_elem = _find_child_by_tag(root, "collaboration")
    if collab_elem is not None:
        model.collaboration = _parse_collaboration(collab_elem, root)

    # Parse all processes
    for proc_elem in _find_all_by_tag(root, "process"):
        process = _parse_process(proc_elem, root)
        model.processes.append(process)

    return preprocess_model(model)


def parse_from_file_path(file_path: str) -> BPMNModel:
    """Parse a BPMN 2.0 XML file from file path into an internal BPMNModel representation."""
    with open(file_path, "r", encoding="utf-8") as f:
        xml_str = f.read()
    return parse_from_string(xml_str)


def preprocess_model(model: BPMNModel) -> BPMNModel:
    """Preprocess model to validate cross-references and ensure lane memberships."""
    for proc in model.processes:
        for node_id, node in proc.flow_nodes.items():
            if not node.lane_id and proc.lanes:
                for lane in proc.lanes:
                    if node_id in lane.flow_node_refs:
                        node.lane_id = lane.id
                        break
                if not node.lane_id and proc.lanes:
                    node.lane_id = proc.lanes[0].id
                    proc.lanes[0].flow_node_refs.append(node_id)

    return model