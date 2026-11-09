"""Constants for BPMN Visualizer: standard bounds, XML namespaces, and styling parameters."""

# XML Namespaces
NAMESPACES = {
    'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
    'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
    'dc': 'http://www.omg.org/spec/DD/20100524/DC',
    'di': 'http://www.omg.org/spec/DD/20100524/DI',
    'modeler': 'http://camunda.org/schema/modeler/1.0',
}

# Standard element dimensions (width, height) in pixels
ELEMENT_BOUNDS = {
    'task': {'width': 100, 'height': 80},
    'userTask': {'width': 100, 'height': 80},
    'serviceTask': {'width': 100, 'height': 80},
    'sendTask': {'width': 100, 'height': 80},
    'receiveTask': {'width': 100, 'height': 80},
    'manualTask': {'width': 100, 'height': 80},
    'businessRuleTask': {'width': 100, 'height': 80},
    'scriptTask': {'width': 100, 'height': 80},
    'callActivity': {'width': 100, 'height': 80},
    'subProcess': {'width': 100, 'height': 80},
    'adHocSubProcess': {'width': 100, 'height': 80},
    'transaction': {'width': 100, 'height': 80},
    'parallelGateway': {'width': 50, 'height': 50},
    'exclusiveGateway': {'width': 50, 'height': 50},
    'inclusiveGateway': {'width': 50, 'height': 50},
    'eventBasedGateway': {'width': 50, 'height': 50},
    'complexGateway': {'width': 50, 'height': 50},
    'startEvent': {'width': 36, 'height': 36},
    'endEvent': {'width': 36, 'height': 36},
    'intermediateCatchEvent': {'width': 36, 'height': 36},
    'intermediateThrowEvent': {'width': 36, 'height': 36},
    'boundaryEvent': {'width': 36, 'height': 36},
    'textAnnotation': {'width': 104, 'height': 40},
    'blackBoxParticipant': {'width': 600, 'height': 60},
}

# Default bounds fallback
DEFAULT_NODE_BOUNDS = {'width': 100, 'height': 80}

# Layout padding & margins
POOL_HEADER_WIDTH = 30
LANE_HEADER_WIDTH = 30
POOL_MARGIN_X = 50
POOL_MARGIN_Y = 50
LANE_PADDING_TOP = 40
LANE_PADDING_BOTTOM = 40
LANE_PADDING_LEFT = 50
LANE_PADDING_RIGHT = 50
BLACKBOX_MARGIN_TOP = 80
ANNOTATION_OFFSET_X = 20
ANNOTATION_OFFSET_Y = -60

# External label layout margins for Events and Gateways
LABEL_CARDINAL_GAP = 11.0       # Gap for cardinal labels (top, bottom, right, left)
LABEL_CORNER_MARGIN_X = 0.0    # Horizontal margin between node and corner labels (top-left, top-right, bottom-left, bottom-right)
LABEL_CORNER_MARGIN_Y = 0.0    # Vertical margin between node and corner labels (top-left, top-right, bottom-left, bottom-right)

# Gateway element tag names
GATEWAY_TAGS = {
    'parallelGateway',
    'exclusiveGateway',
    'inclusiveGateway',
    'eventBasedGateway',
    'complexGateway',
}

# Task & Activity tag names
TASK_TAGS = {
    'task',
    'userTask',
    'serviceTask',
    'sendTask',
    'receiveTask',
    'manualTask',
    'businessRuleTask',
    'scriptTask',
    'callActivity',
    'subProcess',
    'adHocSubProcess',
    'transaction',
}

# Event tag names
EVENT_TAGS = {
    'startEvent',
    'endEvent',
    'intermediateCatchEvent',
    'intermediateThrowEvent',
    'boundaryEvent',
}