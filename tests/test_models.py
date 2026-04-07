import pytest

from powl.objects.tagged_powl.activity import Activity
from powl.objects.tagged_powl.builders import (
    expand_frequency_tags,
    loop,
    sequence,
    silent_activity,
    xor,
)
from powl.objects.tagged_powl.choice_graph import ChoiceGraph
from powl.objects.tagged_powl.partial_order import PartialOrder


def test_activity_frequency_helpers_and_serialization():
    activity = Activity("Review", organization="Org", role="Analyst", min_freq=0, max_freq=None)

    assert activity.is_skippable() is True
    assert activity.is_repeatable() is True
    assert activity.is_unbounded() is True
    assert activity.freq_range() == (0, None)

    restored = Activity.from_dict(activity.to_dict())
    assert restored.same_structure(activity)


def test_silent_activity_builder_creates_tau():
    activity = silent_activity(min_freq=0, max_freq=1)

    assert activity.is_silent() is True
    assert activity.freq_range() == (0, 1)


def test_sequence_builder_preserves_order():
    a = Activity("A")
    b = Activity("B")
    c = Activity("C")

    model = sequence([a, b, c], min_freq=0, max_freq=None)

    assert isinstance(model, PartialOrder)
    assert model.freq_range() == (0, None)
    assert model.get_edges() == {(a, b), (b, c)}


def test_xor_builder_marks_all_children_as_start_and_end():
    a = Activity("A")
    b = Activity("B")

    model = xor([a, b])

    assert isinstance(model, ChoiceGraph)
    assert model.start_nodes() == {a, b}
    assert model.end_nodes() == {a, b}
    assert model.get_edges() == set()


def test_loop_builder_connects_do_and_redo():
    do = Activity("Do")
    redo = Activity("Redo")

    model = loop(do, redo)

    assert isinstance(model, ChoiceGraph)
    assert model.start_nodes() == {do}
    assert model.end_nodes() == {do}
    assert model.get_edges() == {(do, redo), (redo, do)}


def test_expand_frequency_tags_wraps_optional_activity_in_choice():
    model = expand_frequency_tags(Activity("A", min_freq=0, max_freq=1))

    assert isinstance(model, ChoiceGraph)
    assert len(model.get_nodes()) == 2
    assert any(isinstance(node, Activity) and node.is_silent() for node in model.get_nodes())
    assert model.start_nodes() == model.get_nodes()
    assert model.end_nodes() == model.get_nodes()


def test_expand_frequency_tags_wraps_repeatable_activity_in_loop():
    model = expand_frequency_tags(Activity("A", min_freq=1, max_freq=None))

    assert isinstance(model, ChoiceGraph)
    assert len(model.get_nodes()) == 2
    assert len(model.get_edges()) == 2
    assert any(isinstance(node, Activity) and node.is_silent() for node in model.get_nodes())


def test_partial_order_validate_rejects_cycles():
    a = Activity("A")
    b = Activity("B")
    model = PartialOrder(nodes=[a, b], edges=[(a, b), (b, a)])

    with pytest.raises(ValueError, match="must be acyclic"):
        model.validate()


def test_partial_order_normalize_bypasses_simple_silent_activity():
    a = Activity("A")
    tau = Activity(None)
    b = Activity("B")
    model = PartialOrder(nodes=[a, tau, b], edges=[(a, tau), (tau, b)])

    normalized = model.normalize()

    assert isinstance(normalized, PartialOrder)
    assert {node.label for node in normalized.get_nodes()} == {"A", "B"}
    assert {
        (source.label, target.label) for source, target in normalized.get_edges()
    } == {("A", "B")}


def test_choice_graph_validate_connectivity_detects_detached_nodes():
    a = Activity("A")
    b = Activity("B")
    detached = Activity("C")
    model = ChoiceGraph(nodes=[a, b, detached], edges=[(a, b)], start_nodes=[a], end_nodes=[b])

    with pytest.raises(ValueError, match="every user node must lie on a path"):
        model.validate_connectivity()
