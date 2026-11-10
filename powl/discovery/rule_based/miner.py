"""Rule-aware recursion extending POWL's tree inductive miner."""

from __future__ import annotations

from collections import Counter

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL
from pm4py.algo.discovery.inductive.variants.imf import IMFParameters
from pm4py.util import exec_utils

from powl.discovery.total_order_based.inductive.modeling import XorSpec, build_model
from powl.discovery.total_order_based.inductive.variants.im_tree import IMBasePOWL
from powl.discovery.total_order_based.inductive.utils.filtering import (
    FILTERING_TYPE,
    FilteringType,
)
from powl.discovery.total_order_based.inductive.variants.powl_discovery_varaints import (
    POWLDiscoveryVariant,
)
from powl.objects.tagged_powl import Activity, TaggedPOWL
from powl.objects.tagged_powl.builders import loop, xor

from .conformance import assert_model_satisfies_rules
from .cuts import RULE_AWARE_CUTS, RuleAwareExclusiveChoiceCutUVCL
from .dtypes import RuleDecomposition
from .exceptions import RuleDiscoveryError, RuleDiscoveryLimitError
from .fall_through import DATA_FALL_THROUGHS, RULE_FALL_THROUGHS, flower
from .filtering import filter_data
from .log_utils import alphabet, to_variants
from .preprocessing import preprocess
from .repair import RepairMode, repair_log
from .rules import (
    AtMostOnceRule,
    EndRule,
    ExistenceRule,
    InitializationRule,
    SUPPORTED_RULE_TYPES,
    normalize_rules,
)


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class RuleAwarePOWLInductiveMiner(IMBasePOWL[IMDataStructureUVCL]):
    """Use POWL cuts/models with the reference miner's rule-aware control flow.

    The recursive protocol is the only specialized orchestration: it remembers
    rejected candidates, supplies child-local rule tuples, and attempts local
    repair before the two rule-directed fall-throughs. Structural IM cuts,
    weighted data structures, tau fall-throughs and POWL construction are reused.
    """

    def __init__(self, parameters=None):
        self._parameters = dict(parameters or {})
        # Rule context is deliberately local and deterministic; the inherited
        # multiprocessing machinery is unnecessary for these fall-throughs.
        super().__init__({"multiprocessing": False})

    def instance(self):
        return POWLDiscoveryVariant.RULE_BASED

    def apply(self, obj, parameters=None, second_iteration=False):
        if type(obj) is not IMDataStructureUVCL:
            raise TypeError(
                "Rule-aware discovery requires totally ordered trace variants."
            )
        options = dict(self._parameters)
        options.update(parameters or {})
        if (
            FILTERING_TYPE in options
            and options[FILTERING_TYPE] is not FilteringType.DFG_FREQUENCY
        ):
            raise ValueError("RULE_BASED supports only DFG frequency filtering.")
        self._repair_mode = RepairMode(options.get("repair_mode", RepairMode.EVENT))
        self._max_states = _positive_integer(
            "max_states", options.get("max_states", 100_000)
        )
        self._max_depth = _positive_integer("max_depth", options.get("max_depth", 200))
        self._noise = options.get(
            "noise_threshold",
            exec_utils.get_param_value(IMFParameters.NOISE_THRESHOLD, options, 0.0),
        )
        if (
            isinstance(self._noise, bool)
            or not isinstance(self._noise, (int, float))
            or not 0 <= self._noise < 1
        ):
            raise ValueError("noise_threshold must satisfy 0 <= x < 1.")
        for option in ("preprocess_rules", "verify_rules"):
            if not isinstance(options.get(option, True), bool):
                raise TypeError(f"{option} must be a bool.")
        original_rules = normalize_rules(options.get("rules", ()))
        unsupported = [
            rule for rule in original_rules if type(rule) not in SUPPORTED_RULE_TYPES
        ]
        if unsupported:
            raise TypeError(
                "Rule-aware discovery does not support custom rule templates: "
                + ", ".join(type(rule).__name__ for rule in unsupported)
            )
        variants, rules = preprocess(
            to_variants(obj),
            original_rules,
            simplify=options.get("preprocess_rules", True),
            max_states=self._max_states,
        )
        self._active = set()
        model = self._mine(IMDataStructureUVCL(variants), rules, 0)
        if options.get("verify_rules", True):
            assert_model_satisfies_rules(
                model, original_rules, max_states=self._max_states
            )
        return model

    def _mine(self, obj, rules, depth):
        rules = normalize_rules(rules)
        if depth >= self._max_depth:
            raise RuleDiscoveryLimitError(
                f"Rule-aware discovery exceeded max_depth={self._max_depth}."
            )
        signature = (tuple(sorted(obj.data_structure.items())), rules)
        if signature in self._active:
            raise RuleDiscoveryError(
                "A rule-directed decomposition repeated the same log and obligations; "
                "discovery stopped instead of recursing without progress. "
                "Try repair_mode='edit_distance'."
            )
        self._active.add(signature)
        try:
            filtered, empty_visible = filter_data(obj, self._noise)
            return self._mine_problem(filtered, rules, depth, empty_visible)
        finally:
            self._active.remove(signature)

    def _recurse(self, powl, objs, parameters=None, *, projected_rules=None, depth=0):
        """Extend the base recursion with explicit independent rule contexts."""
        child_rules = (
            projected_rules if projected_rules is not None else ((),) * len(objs)
        )
        if len(objs) != len(child_rules):
            raise ValueError("Every child sublog needs a projected rule set.")
        children = [
            self._mine(child, rules, depth + 1)
            for child, rules in zip(objs, child_rules)
        ]
        return build_model(powl, children)

    def _decompose(self, decomposition, depth):
        return self._recurse(
            decomposition.spec,
            decomposition.children,
            projected_rules=decomposition.projected_rules,
            depth=depth,
        )

    def _empty_traces(self, obj, rules, empty_visible):
        if not empty_visible:
            return None
        acts = alphabet(obj)
        groups = [set(), acts]
        rejected = RuleAwareExclusiveChoiceCutUVCL.check_rules(rules, groups)
        if rejected:
            return frozenset(rejected)
        nonempty = Counter(
            {trace: count for trace, count in obj.data_structure.items() if trace}
        )
        if not nonempty:
            # Empty-only logs still have to honor non-vacuous unary rules.
            violated = frozenset(rule for rule in rules if not rule.accepts(()))
            return violated or Activity(label=None)
        return RuleDecomposition(
            XorSpec(2),
            (IMDataStructureUVCL(Counter()), IMDataStructureUVCL(nonempty)),
            RuleAwareExclusiveChoiceCutUVCL.project_rules(rules, groups),
        )

    def _base_case(self, obj, rules, empty_visible):
        acts = alphabet(obj)
        if len(acts) > 1:
            return None
        required = {
            rule.target_activity
            for rule in rules
            if isinstance(rule, (ExistenceRule, InitializationRule, EndRule))
        }
        if not acts:
            if not required:
                return Activity(label=None)
            if len(required) > 1:
                return frozenset(
                    rule
                    for rule in rules
                    if isinstance(rule, (ExistenceRule, InitializationRule, EndRule))
                )
            activity = next(iter(required))
            candidate_traces = ((activity,),)
            model = Activity(label=activity)
        else:
            activity = next(iter(acts))
            at_most_once = any(
                isinstance(rule, AtMostOnceRule) and rule.target_activity == activity
                for rule in rules
            )
            exists = activity in required
            repeats = (activity, activity) in obj.dfg.graph
            if at_most_once and not exists:
                # Preserve the source's optional singleton for AtMostOnce.
                model = xor([Activity(label=None), Activity(label=activity)])
                candidate_traces = ((), (activity,))
            elif repeats and not at_most_once:
                model = loop(Activity(label=activity), Activity(label=None))
                candidate_traces = ((activity,), (activity, activity))
                if empty_visible and not exists:
                    model = loop(Activity(label=None), Activity(label=activity))
                    candidate_traces += ((),)
            else:
                model = Activity(label=activity)
                candidate_traces = ((activity,),)
        # These are exhaustive language representatives for the twelve rule
        # templates on a singleton alphabet, including arbitrary repetition.
        rejected = frozenset(
            rule
            for rule in rules
            if any(not rule.accepts(trace) for trace in candidate_traces)
        )
        return rejected or model

    def _mine_problem(self, obj, rules, depth, empty_visible):
        rejected = []
        empty = self._empty_traces(obj, rules, empty_visible)
        if isinstance(empty, frozenset):
            rejected.append(empty)
        elif isinstance(empty, RuleDecomposition):
            return self._decompose(empty, depth)
        elif isinstance(empty, TaggedPOWL):
            return empty

        base = self._base_case(obj, rules, empty_visible)
        if isinstance(base, frozenset):
            rejected.append(base)
        elif base is not None:
            return base

        for cut in RULE_AWARE_CUTS:
            candidate = cut.evaluate(obj, rules)
            if candidate is None:
                continue
            if candidate.rejected_rules:
                rejected.append(candidate.rejected_rules)
            else:
                return self._decompose(candidate, depth)

        for fall_through in DATA_FALL_THROUGHS:
            candidate = fall_through(obj, rules)
            if isinstance(candidate, frozenset):
                rejected.append(candidate)
            elif candidate is not None:
                return self._decompose(candidate, depth)

        if rejected:
            repaired = repair_log(
                obj.data_structure,
                rejected[0],
                rules,
                self._repair_mode,
                max_states=self._max_states,
            )
            if repaired is not None:
                variants, remaining_rules = repaired
                return self._mine(
                    IMDataStructureUVCL(variants), remaining_rules, depth + 1
                )

        def mine(child, obligations):
            return self._mine(child, obligations, depth + 1)

        for fall_through in RULE_FALL_THROUGHS:
            model = fall_through(obj, rules, mine)
            if model is not None:
                return model
        candidate = flower(obj, rules)
        return (
            self._decompose(candidate, depth)
            if isinstance(candidate, RuleDecomposition)
            else candidate
        )
