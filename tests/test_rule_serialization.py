"""Application contracts for reviewable, data-only guided discovery."""

from collections import Counter
from copy import deepcopy

import pytest

from powl.discovery.rule_based import (
    EndRule,
    ExistenceRule,
    InconsistentRulesError,
    InitializationRule,
    NotCoExistenceRule,
    ResponseRule,
    RuleDiscoveryLimitError,
    rule_catalog,
    rule_from_dict,
    rule_to_dict,
    validate_rule_consistency,
    validate_rules,
)


@pytest.mark.parametrize("template", rule_catalog())
def test_catalog_templates_roundtrip_without_code(template):
    spec = {"template": template["template"], "activities": ["A", "B"][: template["arity"]]}
    original = deepcopy(spec)
    rule = rule_from_dict(spec, activities=["A", "B"])
    assert rule_to_dict(rule) == spec
    assert spec == original
    assert template["description"]


@pytest.mark.parametrize(
    "spec",
    [
        {"template": "Response", "activities": ["A"]},
        {"template": "Response", "activities": ["A", "A"]},
        {"template": "End", "activities": "A"},
        {"template": "End", "activities": [None]},
        {"template": "End", "activities": [""]},
        {"template": "End", "activities": ["A"], "ignored": True},
        {"template": "__import__", "activities": ["A"]},
        {"template": [], "activities": ["A"]},
        {"template": "End", "activities": ["Unknown"]},
        [],
    ],
)
def test_invalid_specs_fail_explicitly(spec):
    with pytest.raises((TypeError, ValueError)):
        rule_from_dict(spec, activities=["A", "B"])


def test_label_escaping_and_init_alias():
    label = "Manager's \"approval\"\ncomplete"
    rule = rule_from_dict({"template": "Init", "activities": [label]}, activities=[label])
    assert rule == InitializationRule(label)
    assert rule_to_dict(rule) == {"template": "Initialization", "activities": [label]}


def test_catalog_results_are_not_shared_mutable_state():
    catalog = rule_catalog()
    catalog[0]["template"] = "Invalid"
    assert rule_catalog()[0]["template"] == "Existence"


def test_consistency_returns_a_shortest_accepting_witness():
    rules = [InitializationRule("A"), ResponseRule("A", "B"), EndRule("C")]
    witness = validate_rule_consistency(rules, activities=["A", "B", "C"])
    assert witness == ("A", "B", "C")
    assert all(rule.accepts(witness) for rule in rules)


def test_vacuous_consistency_is_explicit():
    assert validate_rule_consistency([ResponseRule("A", "B")]) == ()
    assert validate_rule_consistency([]) == ()


def test_contradiction_unknown_labels_and_limits_are_distinct():
    with pytest.raises(InconsistentRulesError):
        validate_rule_consistency([ExistenceRule("A"), ExistenceRule("B"), NotCoExistenceRule("A", "B")])
    with pytest.raises(ValueError, match="Unknown activity"):
        validate_rule_consistency([ExistenceRule("A")], activities=["B"])
    with pytest.raises(RuleDiscoveryLimitError):
        validate_rule_consistency([ExistenceRule("A")], max_states=1)
    with pytest.raises(ValueError, match="positive integer"):
        validate_rule_consistency([], max_states=False)


def test_weighted_violation_examples_are_ordered_bounded_and_nonmutating():
    log = Counter({("A",): 12, ("A", "C"): 3, ("A", "B"): 5, ("C", "A"): 3})
    original = deepcopy(log)
    (result,) = validate_rules(log, [ResponseRule("A", "B")], max_examples=2)
    assert result.total_traces == 23
    assert result.violated_traces == 18
    assert [(x.activities, x.frequency) for x in result.violation_examples] == [(("A",), 12), (("A", "C"), 3)]
    assert log == original
    assert validate_rules(log, [ResponseRule("A", "B")])[0].violation_examples == ()


@pytest.mark.parametrize("value", [True, -1, 2.5, "3"])
def test_examples_require_a_nonnegative_integer(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        validate_rules([["A"]], [], max_examples=value)
