from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import networkx as nx

from powl.objects.tagged_powl.activity import Activity
from powl.objects.tagged_powl.base import TaggedPOWL
from powl.objects.tagged_powl.choice_graph import ChoiceGraph
from powl.objects.tagged_powl.partial_order import PartialOrder

FORMAT = "powl-json"
FORMAT_VERSION = "1.0"
START_ID = "@start"
END_ID = "@end"
FILE_EXTENSION = ".powl"
ENCODING = "utf-8"
MAX_JSON_BYTES = 5 * 1024 * 1024
MAX_MODEL_NODES = 10_000
MAX_MODEL_DEPTH = 64

_TOP_LEVEL_FIELDS = {"format", "format_version", "metadata", "model"}
_METADATA_FIELDS = {"name", "description", "creator", "created_at", "tool_name", "tool_version"}
_COMMON_MODEL_FIELDS = {"id", "type", "skippable", "repeatable", "attributes"}
_ACTIVITY_FIELDS = _COMMON_MODEL_FIELDS | {"label"}
_COMPOSITE_FIELDS = _COMMON_MODEL_FIELDS | {"nodes", "edges"}
_COMMON_ATTRIBUTE_FIELDS = {"name", "description"}
_ACTIVITY_ATTRIBUTE_FIELDS = _COMMON_ATTRIBUTE_FIELDS | {"resource", "role", "cost", "lifecycle"}
_RESERVED_IDS = {START_ID, END_ID}
_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))


class PowlJsonValidationError(ValueError):
    """Raised when a POWL JSON document is structurally or semantically invalid."""


@dataclass(frozen=True)
class PowlJsonDocument:
    """A parsed POWL JSON document including non-semantic file metadata."""

    model: TaggedPOWL
    metadata: dict[str, Any]
    format: str = FORMAT
    format_version: str = FORMAT_VERSION

def read_powl_json(path: str | Path) -> TaggedPOWL:
    """Read a POWL JSON file and return its root POWL model."""
    return read_powl_json_document(path).model


def read_powl_json_document(path: str | Path) -> PowlJsonDocument:
    """Read a POWL JSON file and return the root model plus file metadata."""
    path = _validate_powl_file_extension(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise PowlJsonValidationError(f"POWL JSON exceeds the {MAX_JSON_BYTES}-byte size limit.")
    with Path(path).open("r", encoding=ENCODING) as file:
        return loads_powl_json_document(file.read())


def loads_powl_json(text: str) -> TaggedPOWL:
    """Parse a POWL JSON string and return its root POWL model."""
    return loads_powl_json_document(text).model


def loads_powl_json_document(text: str) -> PowlJsonDocument:
    """Parse a POWL JSON string and return the root model plus file metadata."""
    if not isinstance(text, str):
        raise PowlJsonValidationError("POWL JSON input must be text.")
    if len(text.encode(ENCODING)) > MAX_JSON_BYTES:
        raise PowlJsonValidationError(f"POWL JSON exceeds the {MAX_JSON_BYTES}-byte size limit.")
    try:
        data = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise PowlJsonValidationError(f"Invalid JSON: {exc}") from exc
    return from_powl_json_dict(data)


def write_powl_json(
    model: TaggedPOWL,
    path: str | Path,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
) -> None:
    """Write a POWL model to a POWL JSON file."""
    path = _validate_powl_file_extension(path)
    text = dumps_powl_json(
        model,
        metadata=metadata,
        indent=indent,
        ensure_ascii=ensure_ascii,
    )
    path.write_text(text, encoding=ENCODING)


def dumps_powl_json(
    model: TaggedPOWL,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
) -> str:
    """Serialize a POWL model to a POWL JSON string."""
    return json.dumps(
        to_powl_json_dict(model, metadata=metadata),
        indent=indent,
        ensure_ascii=ensure_ascii,
        allow_nan=False,
    )


def from_powl_json_dict(data: Mapping[str, Any]) -> PowlJsonDocument:
    """Build a POWL model from a decoded top-level POWL JSON object."""
    _require_mapping(data, "$")
    _reject_unknown_fields(data, _TOP_LEVEL_FIELDS, "$")

    if data.get("format") != FORMAT:
        raise PowlJsonValidationError(f"$.format must be {FORMAT!r}.")
    if data.get("format_version") != FORMAT_VERSION:
        raise PowlJsonValidationError(
            f"Unsupported $.format_version {data.get('format_version')!r}; "
            f"supported versions: {FORMAT_VERSION!r}."
        )
    if "model" not in data:
        raise PowlJsonValidationError("$.model is required.")

    _validate_model_size(data["model"])
    metadata = _validate_metadata(data.get("metadata", {}), "$.metadata")
    model = _decode_model(data["model"], path="$.model", require_id=False)
    return PowlJsonDocument(model=model, metadata=metadata)


def to_powl_json_dict(
    model: TaggedPOWL,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Convert a POWL model to the decoded top-level POWL JSON representation."""
    if not isinstance(model, TaggedPOWL):
        raise TypeError(f"model must be a TaggedPOWL instance, got {type(model).__name__}")

    out: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
    }
    if metadata is not None:
        out["metadata"] = _validate_metadata(metadata, "metadata")
    out["model"] = _encode_model(model, node_id=None, path="model")
    return out


def _validate_powl_file_extension(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.suffix.lower() != FILE_EXTENSION:
        raise PowlJsonValidationError(
            f"POWL JSON files must use the {FILE_EXTENSION!r} file extension; "
            f"got {resolved.suffix!r} for {str(resolved)!r}."
        )
    return resolved


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _decode_model(data: Any, *, path: str, require_id: bool) -> TaggedPOWL:
    _require_mapping(data, path)

    if "type" not in data:
        raise PowlJsonValidationError(f"{path}.type is required.")
    model_type = data["type"]
    if model_type not in {"activity", "partial_order", "choice_graph"}:
        raise PowlJsonValidationError(
            f"Unsupported model type at {path}.type: {model_type!r}."
        )

    if require_id:
        _validate_child_id(data, path)
    elif "id" in data and not isinstance(data["id"], str):
        raise PowlJsonValidationError(f"{path}.id must be a string when present.")

    min_freq, max_freq = _frequencies_from_flags(data, path)

    if model_type == "activity":
        _reject_unknown_fields(data, _ACTIVITY_FIELDS, path)
        return _decode_activity(data, path=path, min_freq=min_freq, max_freq=max_freq)

    _reject_unknown_fields(data, _COMPOSITE_FIELDS, path)
    if model_type == "partial_order":
        return _decode_partial_order(data, path=path, min_freq=min_freq, max_freq=max_freq)
    return _decode_choice_graph(data, path=path, min_freq=min_freq, max_freq=max_freq)


def _decode_activity(
    data: Mapping[str, Any],
    *,
    path: str,
    min_freq: int,
    max_freq: Optional[int],
) -> Activity:
    if "label" not in data:
        raise PowlJsonValidationError(f"{path}.label is required for activity models.")
    label = data["label"]
    if label is not None and not isinstance(label, str):
        raise PowlJsonValidationError(f"{path}.label must be a string or null.")

    attributes = _validate_attributes(
        data.get("attributes", {}),
        allowed=_ACTIVITY_ATTRIBUTE_FIELDS,
        path=f"{path}.attributes",
    )
    activity = Activity(
        label=label,
        organization=attributes.get("resource"),
        role=attributes.get("role"),
        min_freq=min_freq,
        max_freq=max_freq,
        attributes=attributes,
    )
    return activity


def _decode_partial_order(
    data: Mapping[str, Any],
    *,
    path: str,
    min_freq: int,
    max_freq: Optional[int],
) -> PartialOrder:
    attributes = _validate_attributes(
        data.get("attributes", {}),
        allowed=_COMMON_ATTRIBUTE_FIELDS,
        path=f"{path}.attributes",
    )
    children, child_by_id = _decode_children(data, path=path)
    edges = _decode_edges(data, path=path, allowed_ids=set(child_by_id))

    graph = nx.DiGraph()
    graph.add_nodes_from(child_by_id)
    graph.add_edges_from(edges)
    _validate_partial_order_graph(graph, path=f"{path}.edges")

    model = PartialOrder(
        nodes=children,
        edges=[(child_by_id[source], child_by_id[target]) for source, target in edges],
        min_freq=min_freq,
        max_freq=max_freq,
    )
    model.attributes = attributes
    return model


def _decode_choice_graph(
    data: Mapping[str, Any],
    *,
    path: str,
    min_freq: int,
    max_freq: Optional[int],
) -> ChoiceGraph:
    attributes = _validate_attributes(
        data.get("attributes", {}),
        allowed=_COMMON_ATTRIBUTE_FIELDS,
        path=f"{path}.attributes",
    )
    children, child_by_id = _decode_children(data, path=path)
    allowed_ids = set(child_by_id) | {START_ID, END_ID}
    edges = _decode_edges(data, path=path, allowed_ids=allowed_ids)
    _validate_choice_graph_edges(edges, child_ids=set(child_by_id), path=f"{path}.edges")

    model = ChoiceGraph(
        nodes=children,
        edges=[
            (child_by_id[source], child_by_id[target])
            for source, target in edges
            if source not in _RESERVED_IDS and target not in _RESERVED_IDS
        ],
        start_nodes=[child_by_id[target] for source, target in edges if source == START_ID],
        end_nodes=[child_by_id[source] for source, target in edges if target == END_ID],
        min_freq=min_freq,
        max_freq=max_freq,
    )
    model.attributes = attributes
    return model


def _decode_children(
    data: Mapping[str, Any],
    *,
    path: str,
) -> tuple[list[TaggedPOWL], dict[str, TaggedPOWL]]:
    if "nodes" not in data:
        raise PowlJsonValidationError(f"{path}.nodes is required for composite models.")
    nodes_data = data["nodes"]
    if not isinstance(nodes_data, list):
        raise PowlJsonValidationError(f"{path}.nodes must be an array.")
    if len(nodes_data) < 2:
        raise PowlJsonValidationError(f"{path}.nodes must contain at least two child models.")

    children: list[TaggedPOWL] = []
    child_by_id: dict[str, TaggedPOWL] = {}
    for index, child_data in enumerate(nodes_data):
        child_path = f"{path}.nodes[{index}]"
        _validate_child_id(child_data, child_path)
        child_id = child_data["id"]
        if child_id in child_by_id:
            raise PowlJsonValidationError(f"Duplicate child id {child_id!r} in {path}.nodes.")
        child = _decode_model(child_data, path=child_path, require_id=True)
        children.append(child)
        child_by_id[child_id] = child
    return children, child_by_id


def _decode_edges(
    data: Mapping[str, Any],
    *,
    path: str,
    allowed_ids: set[str],
) -> list[tuple[str, str]]:
    if "edges" not in data:
        raise PowlJsonValidationError(f"{path}.edges is required for composite models.")
    edges_data = data["edges"]
    if not isinstance(edges_data, list):
        raise PowlJsonValidationError(f"{path}.edges must be an array.")

    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, edge in enumerate(edges_data):
        edge_path = f"{path}.edges[{index}]"
        _require_mapping(edge, edge_path)
        _reject_unknown_fields(edge, {"source", "target"}, edge_path)
        if "source" not in edge or "target" not in edge:
            raise PowlJsonValidationError(f"{edge_path} must contain source and target.")
        source = edge["source"]
        target = edge["target"]
        if not isinstance(source, str) or not isinstance(target, str):
            raise PowlJsonValidationError(f"{edge_path}.source and target must be strings.")
        if source not in allowed_ids:
            raise PowlJsonValidationError(f"{edge_path}.source refers to unknown id {source!r}.")
        if target not in allowed_ids:
            raise PowlJsonValidationError(f"{edge_path}.target refers to unknown id {target!r}.")
        pair = (source, target)
        if pair in seen:
            raise PowlJsonValidationError(f"Duplicate edge {source!r} -> {target!r} at {edge_path}.")
        seen.add(pair)
        edges.append(pair)
    return edges


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _encode_model(model: TaggedPOWL, *, node_id: Optional[str], path: str) -> dict[str, Any]:
    if isinstance(model, Activity):
        return _encode_activity(model, node_id=node_id, path=path)
    if isinstance(model, PartialOrder):
        return _encode_partial_order(model, node_id=node_id, path=path)
    if isinstance(model, ChoiceGraph):
        return _encode_choice_graph(model, node_id=node_id, path=path)
    raise TypeError(f"Unsupported POWL model at {path}: {type(model).__name__}")


def _encode_common(model: TaggedPOWL, *, model_type: str, node_id: Optional[str]) -> dict[str, Any]:
    skippable, repeatable = _flags_from_frequencies(model)
    out: dict[str, Any] = {}
    if node_id is not None:
        out["id"] = node_id
    out["type"] = model_type
    if skippable:
        out["skippable"] = True
    if repeatable:
        out["repeatable"] = True
    return out


def _encode_activity(model: Activity, *, node_id: Optional[str], path: str) -> dict[str, Any]:
    out = _encode_common(model, model_type="activity", node_id=node_id)
    if model.label is not None and not isinstance(model.label, str):
        raise PowlJsonValidationError(f"{path}.label must be a string or None.")
    out["label"] = model.label

    attributes = _model_attributes(model)
    if model.organization is not None:
        _merge_attribute(attributes, "resource", model.organization, path)
    if model.role is not None:
        _merge_attribute(attributes, "role", model.role, path)
    attributes = _validate_attributes(
        attributes,
        allowed=_ACTIVITY_ATTRIBUTE_FIELDS,
        path=f"{path}.attributes",
    )
    if attributes:
        out["attributes"] = _ordered_attributes(attributes, _ACTIVITY_ATTRIBUTE_FIELDS)
    return out


def _encode_partial_order(model: PartialOrder, *, node_id: Optional[str], path: str) -> dict[str, Any]:
    children = _ordered_children(model)
    _validate_composite_child_count(children, path)
    ids = _make_local_ids(children)

    graph = nx.DiGraph()
    graph.add_nodes_from(children)
    graph.add_edges_from(model.get_edges())
    _validate_partial_order_for_writing(graph, children, path)
    reduced = nx.transitive_reduction(graph)

    out = _encode_common(model, model_type="partial_order", node_id=node_id)
    attributes = _validate_attributes(
        _model_attributes(model),
        allowed=_COMMON_ATTRIBUTE_FIELDS,
        path=f"{path}.attributes",
    )
    if attributes:
        out["attributes"] = _ordered_attributes(attributes, _COMMON_ATTRIBUTE_FIELDS)
    out["nodes"] = [
        _encode_model(child, node_id=ids[child], path=f"{path}.nodes[{index}]")
        for index, child in enumerate(children)
    ]
    out["edges"] = [
        {"source": ids[source], "target": ids[target]}
        for source, target in _ordered_model_edges(reduced.edges(), children)
    ]
    return out


def _encode_choice_graph(model: ChoiceGraph, *, node_id: Optional[str], path: str) -> dict[str, Any]:
    children = _ordered_children(model)
    _validate_composite_child_count(children, path)
    ids = _make_local_ids(children)
    child_set = set(children)

    internal_edges = list(model.get_edges())
    start_nodes = list(model.start_nodes())
    end_nodes = list(model.end_nodes())

    if any(node not in child_set for node in start_nodes):
        raise PowlJsonValidationError(f"{path} has a start node that is not a child model.")
    if any(node not in child_set for node in end_nodes):
        raise PowlJsonValidationError(f"{path} has an end node that is not a child model.")
    if any(source == target for source, target in internal_edges):
        raise PowlJsonValidationError(f"{path} contains a choice-graph self-loop, which v1.0 rejects.")

    edge_ids: list[tuple[str, str]] = []
    edge_ids.extend((START_ID, ids[node]) for node in _ordered_nodes(start_nodes, children))
    edge_ids.extend(
        (ids[source], ids[target])
        for source, target in _ordered_model_edges(internal_edges, children)
    )
    edge_ids.extend((ids[node], END_ID) for node in _ordered_nodes(end_nodes, children))
    _validate_choice_graph_edges(edge_ids, child_ids=set(ids.values()), path=f"{path}.edges")

    out = _encode_common(model, model_type="choice_graph", node_id=node_id)
    attributes = _validate_attributes(
        _model_attributes(model),
        allowed=_COMMON_ATTRIBUTE_FIELDS,
        path=f"{path}.attributes",
    )
    if attributes:
        out["attributes"] = _ordered_attributes(attributes, _COMMON_ATTRIBUTE_FIELDS)
    out["nodes"] = [
        _encode_model(child, node_id=ids[child], path=f"{path}.nodes[{index}]")
        for index, child in enumerate(children)
    ]
    out["edges"] = [{"source": source, "target": target} for source, target in edge_ids]
    return out


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PowlJsonValidationError(f"Duplicate JSON object field: {key!r}.")
        result[key] = value
    return result


def _reject_constant(value):
    raise PowlJsonValidationError(f"Non-finite JSON number is not supported: {value}.")


def _validate_model_size(model):
    stack = [(model, 0, frozenset())]
    count = 0
    while stack:
        node, depth, ancestors = stack.pop()
        count += 1
        if count > MAX_MODEL_NODES:
            raise PowlJsonValidationError(f"POWL JSON exceeds the {MAX_MODEL_NODES}-node limit.")
        if depth > MAX_MODEL_DEPTH:
            raise PowlJsonValidationError(f"POWL JSON exceeds the nesting limit of {MAX_MODEL_DEPTH}.")
        if isinstance(node, Mapping):
            if id(node) in ancestors:
                raise PowlJsonValidationError("POWL JSON contains a cyclic model reference.")
            children = node.get("nodes", ())
            if isinstance(children, (list, tuple)):
                trail = ancestors | {id(node)}
                stack.extend((child, depth + 1, trail) for child in children)


def _validate_metadata(value: Any, path: str) -> dict[str, Any]:
    _require_mapping(value, path)
    _reject_unknown_fields(value, _METADATA_FIELDS, path)
    metadata = dict(value)
    for key, item in metadata.items():
        if not isinstance(item, _JSON_SCALAR_TYPES) or isinstance(item, bool):
            raise PowlJsonValidationError(
                f"{path}.{key} must be a JSON string, number, or null."
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise PowlJsonValidationError(f"{path}.{key} must be finite.")
    return metadata


def _validate_attributes(value: Any, *, allowed: set[str], path: str) -> dict[str, Any]:
    _require_mapping(value, path)
    _reject_unknown_fields(value, allowed, path)
    attributes = dict(value)

    for key in ("name", "description"):
        if key in attributes and not isinstance(attributes[key], str):
            raise PowlJsonValidationError(f"{path}.{key} must be a string.")
    for key in ("resource", "role", "lifecycle"):
        if key in attributes and not isinstance(attributes[key], str):
            raise PowlJsonValidationError(f"{path}.{key} must be a string.")
    if "cost" in attributes:
        cost = attributes["cost"]
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            raise PowlJsonValidationError(f"{path}.cost must be a number.")
        if (isinstance(cost, float) and not math.isfinite(cost)) or cost < 0:
            raise PowlJsonValidationError(f"{path}.cost must be finite and non-negative.")
    return attributes


def _validate_partial_order_graph(graph: nx.DiGraph, *, path: str) -> None:
    try:
        reduced_graph = nx.transitive_reduction(graph)
    except nx.NetworkXError as exc:
        raise PowlJsonValidationError(
            f"{path} must be acyclic."
        ) from exc

    redundant_edges = set(graph.edges()) - set(reduced_graph.edges())

    if redundant_edges:
        source, target = next(iter(redundant_edges))
        raise PowlJsonValidationError(
            f"{path} is not transitively reduced; edge "
            f"{source!r} -> {target!r} is implied by another path."
        )


def _validate_partial_order_for_writing(
    graph: nx.DiGraph,
    children: Sequence[TaggedPOWL],
    path: str,
) -> None:
    child_set = set(children)
    if set(graph.nodes()) != child_set:
        raise PowlJsonValidationError(f"{path} contains partial-order edges with non-child endpoints.")
    if any(source == target for source, target in graph.edges()):
        raise PowlJsonValidationError(f"{path} contains a partial-order self-loop.")
    if not nx.is_directed_acyclic_graph(graph):
        try:
            cycle = nx.find_cycle(graph, orientation="original")
        except nx.NetworkXNoCycle:
            cycle = "unknown"
        raise PowlJsonValidationError(f"{path} must be acyclic; found cycle {cycle}.")


def _validate_choice_graph_edges(
    edges: Iterable[tuple[str, str]],
    *,
    child_ids: set[str],
    path: str,
) -> None:
    edges = list(edges)
    seen: set[tuple[str, str]] = set()
    graph = nx.DiGraph()
    graph.add_nodes_from(child_ids | {START_ID, END_ID})

    for source, target in edges:
        if (source, target) in seen:
            raise PowlJsonValidationError(f"{path} contains duplicate edge {source!r} -> {target!r}.")
        seen.add((source, target))

        if source not in child_ids | {START_ID, END_ID}:
            raise PowlJsonValidationError(f"{path} contains unknown source {source!r}.")
        if target not in child_ids | {START_ID, END_ID}:
            raise PowlJsonValidationError(f"{path} contains unknown target {target!r}.")
        if source == target:
            raise PowlJsonValidationError(f"{path} contains self-loop {source!r} -> {target!r}.")
        if target == START_ID:
            raise PowlJsonValidationError(f"{path} may not contain an edge entering {START_ID}.")
        if source == END_ID:
            raise PowlJsonValidationError(f"{path} may not contain an edge leaving {END_ID}.")
        if source == START_ID and target == END_ID:
            raise PowlJsonValidationError(f"{path} may not contain a direct {START_ID} -> {END_ID} edge.")
        graph.add_edge(source, target)

    for child_id in child_ids:
        if not nx.has_path(graph, START_ID, child_id):
            raise PowlJsonValidationError(f"{path}: child {child_id!r} is not reachable from {START_ID}.")
        if not nx.has_path(graph, child_id, END_ID):
            raise PowlJsonValidationError(f"{path}: {END_ID} is not reachable from child {child_id!r}.")


def _frequencies_from_flags(data: Mapping[str, Any], path: str) -> tuple[int, Optional[int]]:
    skippable = data.get("skippable", False)
    repeatable = data.get("repeatable", False)
    if not isinstance(skippable, bool):
        raise PowlJsonValidationError(f"{path}.skippable must be a boolean when present.")
    if not isinstance(repeatable, bool):
        raise PowlJsonValidationError(f"{path}.repeatable must be a boolean when present.")
    min_freq = 0 if skippable else 1
    max_freq = None if repeatable else 1
    return min_freq, max_freq


def _flags_from_frequencies(model: TaggedPOWL) -> tuple[bool, bool]:
    if model.min_freq not in (0, 1):
        raise PowlJsonValidationError(
            "POWL JSON v1.0 can only represent min_freq 0 or 1 "
            f"({model!r} has min_freq={model.min_freq!r})."
        )
    if model.max_freq not in (1, None):
        raise PowlJsonValidationError(
            "POWL JSON v1.0 can only represent max_freq 1 or None/unbounded "
            f"({model!r} has max_freq={model.max_freq!r})."
        )
    return model.min_freq == 0, model.max_freq is None


def _validate_child_id(data: Any, path: str) -> None:
    _require_mapping(data, path)
    if "id" not in data:
        raise PowlJsonValidationError(f"{path}.id is required for child models.")
    child_id = data["id"]
    if not isinstance(child_id, str):
        raise PowlJsonValidationError(f"{path}.id must be a string.")
    if child_id in _RESERVED_IDS:
        raise PowlJsonValidationError(f"{path}.id may not be reserved id {child_id!r}.")
    if child_id == "":
        raise PowlJsonValidationError(f"{path}.id may not be empty.")


def _require_mapping(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise PowlJsonValidationError(f"{path} must be a JSON object.")


def _reject_unknown_fields(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        fields = ", ".join(sorted(repr(field) for field in unknown))
        raise PowlJsonValidationError(f"Unsupported field(s) at {path}: {fields}.")


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _model_attributes(model: TaggedPOWL) -> dict[str, Any]:
    return dict(getattr(model, "attributes", {}) or {})


def _merge_attribute(attributes: dict[str, Any], key: str, value: Any, path: str) -> None:
    if key in attributes and attributes[key] != value:
        raise PowlJsonValidationError(
            f"{path} has conflicting {key!r} values in attributes and legacy object fields."
        )
    attributes[key] = value


def _ordered_attributes(attributes: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
    preferred_order = ["name", "description", "resource", "role", "cost", "lifecycle"]
    return {
        key: attributes[key]
        for key in preferred_order
        if key in allowed and key in attributes
    }


def _ordered_children(model: Any) -> list[TaggedPOWL]:
    children = list(getattr(model, "children", []))
    if not children:
        children = list(model.get_nodes())
    return children


def _validate_composite_child_count(children: Sequence[TaggedPOWL], path: str) -> None:
    if len(children) < 2:
        raise PowlJsonValidationError(f"{path} must contain at least two child models for POWL JSON v1.0.")


def _ordered_nodes(nodes: Iterable[TaggedPOWL], order: Sequence[TaggedPOWL]) -> list[TaggedPOWL]:
    node_set = set(nodes)
    index = {node: position for position, node in enumerate(order)}
    return sorted(node_set, key=lambda node: index[node])


def _ordered_model_edges(
    edges: Iterable[tuple[TaggedPOWL, TaggedPOWL]],
    order: Sequence[TaggedPOWL],
) -> list[tuple[TaggedPOWL, TaggedPOWL]]:
    index = {node: position for position, node in enumerate(order)}
    return sorted(set(edges), key=lambda edge: (index[edge[0]], index[edge[1]]))


def _make_local_ids(children: Sequence[TaggedPOWL]) -> dict[TaggedPOWL, str]:
    used: set[str] = set()
    ids: dict[TaggedPOWL, str] = {}
    for child in children:
        base = _suggest_id(child)
        candidate = base
        suffix = 2
        while candidate in used or candidate in _RESERVED_IDS:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        ids[child] = candidate
    return ids


def _suggest_id(model: TaggedPOWL) -> str:
    if isinstance(model, Activity):
        if model.label:
            base = model.label
        else:
            base = "tau"
    elif isinstance(model, PartialOrder):
        base = "partial_order"
    elif isinstance(model, ChoiceGraph):
        base = "choice_graph"
    else:
        base = "model"
    base = re.sub(r"[^0-9A-Za-z_]+", "_", str(base).strip().lower())
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "model"
