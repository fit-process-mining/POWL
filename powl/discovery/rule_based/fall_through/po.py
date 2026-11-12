"""Rule-directed parallel/sequence fall-through from the reference RBIM.

The dependency graph is an intermediate heuristic. Its weak components become
parallel branches, and each branch is serialized
into sequence layers. It deliberately does not introduce a more permissive
partial-order language than the reference process-tree construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple

import networkx as nx
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL

from powl.discovery.rule_based.cuts import (
    RuleAwareConcurrencyCutUVCL,
    RuleAwareSequenceCutUVCL,
)
from powl.discovery.rule_based.rules import (
    AbstractRule,
    ChainPrecedenceRule,
    ChainResponseRule,
    NotCoExistenceRule,
    PrecedenceRule,
    ResponseRule,
)
from powl.discovery.total_order_based.inductive.modeling import (
    PartialOrderSpec,
    SequenceSpec,
    build_model,
)
from powl.objects.tagged_powl.base import TaggedPOWL

from .utils import (
    ARTIFICIAL_END,
    ARTIFICIAL_START,
    alphabet_of,
    build_cdg,
    dfg_graph,
    supported_rules_alphabet,
)

ENABLE_LABEL_SPLITTING = True


@dataclass(frozen=True)
class RuleBasedPO:
    groups: List[Set[str]]
    edges: Set[Tuple[int, int]]
    non_refinable: List[Set[str]]


def abstract_dfg(dfg: nx.DiGraph, components: List[Set[str]]) -> nx.DiGraph:
    """
    Abstract an activity-level DFG into a component-level DFG.

    DFG format:
      - normal edges: a -> b with edge attribute weight
      - start counts: dfg.nodes[a]["start"]
      - end counts: dfg.nodes[a]["end"]

    Abstracted DFG format:
      - nodes are component indices: 0, 1, ...
      - start/end counts are summed onto component nodes
      - edge weights are summed between component indices
    """
    new_dfg = nx.DiGraph()
    new_dfg.add_nodes_from(range(len(components)))
    component_of = {}
    for i, component in enumerate(components):
        for activity in component:
            component_of[activity] = i
    for activity, data in dfg.nodes(data=True):
        if activity not in component_of:
            continue
        component = component_of[activity]
        start_count = data.get("start", 0)
        end_count = data.get("end", 0)
        if start_count:
            new_dfg.nodes[component]["start"] = (
                new_dfg.nodes[component].get("start", 0) + start_count
            )
        if end_count:
            new_dfg.nodes[component]["end"] = (
                new_dfg.nodes[component].get("end", 0) + end_count
            )
    for source, target, data in dfg.edges(data=True):
        if source not in component_of or target not in component_of:
            continue
        source_component = component_of[source]
        target_component = component_of[target]
        weight = data.get("weight", 1)
        if source_component == target_component:
            continue
        if new_dfg.has_edge(source_component, target_component):
            new_dfg[source_component][target_component]["weight"] += weight
        else:
            new_dfg.add_edge(source_component, target_component, weight=weight)
    return new_dfg


def get_chain_components(
    rules: List[AbstractRule], alphabet: Set[str]
) -> List[Set[str]]:
    chain_graph = nx.Graph()
    chain_graph.add_nodes_from(sorted(alphabet))
    for rule in rules or []:
        if isinstance(rule, (ChainPrecedenceRule, ChainResponseRule)):
            if rule.activity_a in alphabet and rule.activity_b in alphabet:
                chain_graph.add_edge(rule.activity_a, rule.activity_b)
    return [set(component) for component in nx.connected_components(chain_graph)]


def merge_components_by_not_coexistence(
    components: List[Set[str]], rules: List[AbstractRule]
) -> List[Set[str]]:
    """
    Keep activities connected by NotCoExistence in the same PO branch.

    This does not model XOR by itself. It only prevents the PO fall-through
    from treating the two activities as unrelated parallel branches.
    """
    component_of = {}
    for i, component in enumerate(components):
        for activity in component:
            component_of[activity] = i
    merge_graph = nx.Graph()
    merge_graph.add_nodes_from(range(len(components)))
    for rule in rules or []:
        if not isinstance(rule, NotCoExistenceRule):
            continue
        a = rule.activity_a
        b = rule.activity_b
        if a not in component_of or b not in component_of:
            continue
        ca = component_of[a]
        cb = component_of[b]
        if ca != cb:
            merge_graph.add_edge(ca, cb)
    merged_components = []
    for component_ids in nx.connected_components(merge_graph):
        merged = set().union(*(components[i] for i in component_ids))
        merged_components.append(merged)
    return merged_components


def merge_groups_connected_by_chain_rules(
    groups: List[Set[str]], rules: List[AbstractRule]
) -> List[Set[str]]:
    """
    If a chain rule still connects activities in different sequence layers,
    merge the entire interval between them.

    This avoids producing A -> something -> B for ChainResponse(A, B) or
    ChainPrecedence(A, B).
    """
    rules = rules or []
    groups = [set(group) for group in groups]
    changed = True
    while changed:
        changed = False
        position = {}
        for i, group in enumerate(groups):
            for activity in group:
                position[activity] = i
        for rule in rules:
            if not isinstance(rule, (ChainPrecedenceRule, ChainResponseRule)):
                continue
            a = rule.activity_a
            b = rule.activity_b
            if a not in position or b not in position:
                continue
            i = position[a]
            j = position[b]
            if i == j:
                continue
            low, high = sorted((i, j))
            if high - low == 1:
                continue
            merged_group = set().union(*groups[low : high + 1])
            groups = groups[:low] + [merged_group] + groups[high + 1 :]
            changed = True
            break
    return groups


def merge_components_by_scc(
    components: List[Set[str]], activity_graph: nx.DiGraph
) -> List[Set[str]]:
    component_of = {
        activity: i for i, component in enumerate(components) for activity in component
    }
    merge_graph = nx.Graph()
    merge_graph.add_nodes_from(range(len(components)))
    for scc in nx.strongly_connected_components(activity_graph):
        component_ids = {
            component_of[activity] for activity in scc if activity in component_of
        }
        for component_id in component_ids:
            merge_graph.add_node(component_id)
        component_ids = sorted(component_ids)
        for i in range(len(component_ids) - 1):
            merge_graph.add_edge(component_ids[i], component_ids[i + 1])
    return [
        set().union(*(components[i] for i in component_ids))
        for component_ids in nx.connected_components(merge_graph)
    ]


def merge_groups_connected_by_not_coexistence(
    groups: List[Set[str]], rules: List[AbstractRule]
) -> List[Set[str]]:
    """
    Merge sequence layers connected by NotCoExistence.

    Since the groups form an ordered sequence, merging non-adjacent groups
    requires merging the entire interval between them. The internal
    NotCoExistence relation is then handled by recursive mining.
    """
    groups = [set(group) for group in groups]
    rules = rules or []
    changed = True
    while changed:
        changed = False
        position = {
            activity: index for index, group in enumerate(groups) for activity in group
        }
        for rule in rules:
            if not isinstance(rule, NotCoExistenceRule):
                continue
            a = rule.activity_a
            b = rule.activity_b
            if a not in position or b not in position:
                continue
            i = position[a]
            j = position[b]
            if i == j:
                continue
            low, high = sorted((i, j))
            merged_group = set().union(*groups[low : high + 1])
            groups = groups[:low] + [merged_group] + groups[high + 1 :]
            changed = True
            break
    return groups


def calculate_weight(dfg, source, target) -> float:
    """
    Runs Dijkstra, treates DFG weights as inverses
    """
    if source == target:
        return 0
    if source not in dfg.nodes or target not in dfg.nodes:
        return 0
    cost_graph = nx.DiGraph()
    cost_graph.add_nodes_from(dfg.nodes)
    for a, b, data in dfg.edges(data=True):
        if a == b:
            continue
        weight = data.get("weight", 1)
        if weight <= 0:
            continue
        cost_graph.add_edge(a, b, cost=1 / weight)
    try:
        path = nx.dijkstra_path(cost_graph, source, target, weight="cost")
    except nx.NetworkXNoPath:
        return 0
    total_weight = 0
    for a, b in zip(path, path[1:]):
        total_weight += dfg[a][b].get("weight", 1)
    return total_weight


def handle_chain_components(
    po: RuleBasedPO, rules: Set[AbstractRule], dfg: nx.DiGraph
) -> Optional[RuleBasedPO]:
    components_with_chain_rules = set()
    for r in rules:
        if isinstance(r, ChainPrecedenceRule) or isinstance(r, ChainResponseRule):
            act_a, act_b = (r.activity_a, r.activity_b)
            for i, group in enumerate(po.groups):
                if act_a in group or act_b in group:
                    components_with_chain_rules.add(i)
    if not components_with_chain_rules:
        return po
    g = nx.DiGraph()
    g.add_nodes_from(range(len(po.groups)))
    g.add_edges_from(po.edges)
    if len(g.nodes) == 1:
        return po
    if not nx.is_directed_acyclic_graph(g):
        return None
    closure = nx.transitive_closure_dag(g)
    abstracted_dfg = abstract_dfg(dfg, po.groups)

    def comparable(i: int, j: int) -> bool:
        return i == j or closure.has_edge(i, j) or closure.has_edge(j, i)

    new_edges = set(po.edges)
    for cmp in sorted(components_with_chain_rules):
        for other_cmp in g.nodes:
            if other_cmp == cmp:
                continue
            if not comparable(cmp, other_cmp):
                cost_cmp_o = calculate_weight(abstracted_dfg, cmp, other_cmp)
                cost_o_cmp = calculate_weight(abstracted_dfg, other_cmp, cmp)
                (
                    new_edges.add((other_cmp, cmp))
                    if cost_o_cmp > cost_cmp_o
                    else new_edges.add((cmp, other_cmp))
                )
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(po.groups)))
    graph.add_edges_from(new_edges)
    sccs = list(nx.strongly_connected_components(graph))
    node_to_scc = {
        node: scc_index for scc_index, scc in enumerate(sccs) for node in scc
    }
    merged_groups = [set().union(*(po.groups[node] for node in scc)) for scc in sccs]
    merged_edges = {
        (node_to_scc[source], node_to_scc[target])
        for source, target in graph.edges
        if node_to_scc[source] != node_to_scc[target]
    }
    condensed_graph = nx.DiGraph()
    condensed_graph.add_nodes_from(range(len(merged_groups)))
    condensed_graph.add_edges_from(merged_edges)
    try:
        reduced_graph = nx.transitive_reduction(condensed_graph)
        return RuleBasedPO(
            groups=merged_groups,
            edges=set(reduced_graph.edges),
            non_refinable=[
                group
                for group in merged_groups
                if any((protected <= group for protected in po.non_refinable))
            ],
        )
    except nx.NetworkXAlgorithmError:
        return None


def detect_label_splitting(
    alphabet: Set[str], boundary_nodes: Set[str]
) -> Optional[RuleBasedPO]:
    if len(boundary_nodes) != 1 or not ENABLE_LABEL_SPLITTING:
        return None
    label_splitting = set(boundary_nodes)
    remaining_acts = alphabet - label_splitting
    return RuleBasedPO(
        [label_splitting, remaining_acts, label_splitting],
        edges={(0, 1), (1, 2)},
        non_refinable=[],
    )


def detect_rule_based_po(
    rules: List[AbstractRule], alphabet: Set[str], dfg: nx.DiGraph
) -> Optional[RuleBasedPO]:
    """
    Detect a rule-induced partial order between activity blocks.

    This does not directly perform layering. It returns:
        groups = block activities
        edges = partial-order edges between block indices
    """
    if not alphabet:
        return None
    rules = rules or []
    cdg = build_cdg(rules, alphabet)
    activity_graph = cdg.subgraph(alphabet).copy()
    start_nodes = {
        activity for activity in alphabet if cdg.has_edge(ARTIFICIAL_START, activity)
    }
    end_nodes = {
        activity for activity in alphabet if cdg.has_edge(activity, ARTIFICIAL_END)
    }
    if start_nodes & end_nodes:
        return detect_label_splitting(alphabet, start_nodes & end_nodes)
    if len(start_nodes) > 1 or len(end_nodes) > 1:
        return None
    components = get_chain_components(rules, alphabet)
    components = merge_components_by_not_coexistence(components, rules)
    merged_components = components.copy()
    components = merge_components_by_scc(components, activity_graph)
    components_not_to_refine = [c for c in components if c not in merged_components]
    component_of = {}
    for i, component in enumerate(components):
        for activity in component:
            component_of[activity] = i
    block_graph = nx.DiGraph()
    block_graph.add_nodes_from(range(len(components)))
    for source, target in activity_graph.edges:
        source_component = component_of[source]
        target_component = component_of[target]
        if source_component != target_component:
            block_graph.add_edge(source_component, target_component)
    start_components = {
        component_of[activity] for activity in start_nodes if activity in component_of
    }
    end_components = {
        component_of[activity] for activity in end_nodes if activity in component_of
    }
    if start_components & end_components:
        return None
    for start_component in start_components:
        for other in block_graph.nodes:
            if other != start_component:
                block_graph.add_edge(start_component, other)
    for end_component in end_components:
        for other in block_graph.nodes:
            if other != end_component:
                block_graph.add_edge(other, end_component)
    if not nx.is_directed_acyclic_graph(block_graph):
        cyclic_components = set()
        for scc in nx.strongly_connected_components(block_graph):
            if len(scc) > 1:
                cyclic_components.update(scc)
        cyclic_activities = set().union(*(components[i] for i in cyclic_components))
        boundary_nodes = (start_nodes | end_nodes) & cyclic_activities
        if len(boundary_nodes) == 1:
            return detect_label_splitting(alphabet, boundary_nodes)
        return None
    if block_graph.number_of_nodes() == 0:
        return None
    po = RuleBasedPO(
        groups=components,
        edges=set(block_graph.edges),
        non_refinable=components_not_to_refine,
    )
    return handle_chain_components(po, rules, dfg)


def topological_layers_for_nodes(
    graph: nx.DiGraph, nodes: Set[int]
) -> Optional[List[Set[int]]]:
    remaining = set(nodes)
    layers: List[Set[int]] = []
    while remaining:
        layer = {
            node
            for node in remaining
            if all(
                (
                    predecessor not in remaining
                    for predecessor in graph.predecessors(node)
                )
            )
        }
        if not layer:
            return None
        layers.append(layer)
        remaining -= layer
    return layers


def split_group_by_rules(group: Set[str], rules: List[AbstractRule]) -> List[Set[str]]:
    if len(group) <= 1:
        return [set(group)]
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(group))
    for rule in rules or []:
        if not hasattr(rule, "activity_a"):
            continue
        a = rule.activity_a
        b = rule.activity_b
        if a not in group or b not in group:
            continue
        if isinstance(rule, (ChainResponseRule, ChainPrecedenceRule)):
            graph.add_edge(a, b)
        elif isinstance(rule, NotCoExistenceRule):
            graph.add_edge(a, b)
            graph.add_edge(b, a)
    if graph.number_of_edges() == 0:
        return [set(group)]
    condensed = nx.condensation(graph)
    return [
        set().union(*(condensed.nodes[node]["members"] for node in generation))
        for generation in nx.topological_generations(condensed)
    ]


def try_label_splitting(
    group: Set[str], rules: List[AbstractRule]
) -> Optional[List[Set[str]]]:
    if not ENABLE_LABEL_SPLITTING:
        return None
    before = set()
    after = set()
    for rule in rules or []:
        if not hasattr(rule, "activity_a"):
            continue
        a = rule.activity_a
        b = rule.activity_b
        if a not in group or b not in group or a == b:
            continue
        if isinstance(rule, (ResponseRule, ChainResponseRule)):
            after.add(b)
        elif isinstance(rule, (PrecedenceRule, ChainPrecedenceRule)):
            before.add(a)
    split_candidates = before & after & group
    if not split_candidates:
        return None
    if len(split_candidates) != 1:
        return None
    boundary = next(iter(split_candidates))
    middle = group - {boundary}
    if not middle:
        return None
    return [{boundary}, middle, {boundary}]


def po_to_parallel_sequence_branches(
    po: RuleBasedPO, rules: List[AbstractRule]
) -> Optional[List[List[Set[str]]]]:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(po.groups)))
    graph.add_edges_from(po.edges)
    if not nx.is_directed_acyclic_graph(graph):
        return None
    branches: List[List[Set[str]]] = []
    for weak_nodes in nx.connected_components(graph.to_undirected()):
        weak_nodes = set(weak_nodes)
        local_layers = topological_layers_for_nodes(graph, weak_nodes)
        if local_layers is None:
            return None
        branch: List[Set[str]] = []
        for layer in local_layers:
            for group_idx in sorted(layer):
                group = set(po.groups[group_idx])
                if group in po.non_refinable:
                    branch.append(group)
                else:
                    branch.extend(split_group_by_rules(group, rules))
        if not branch:
            continue
        branch_alphabet = set().union(*branch)
        branch_rules = supported_rules_alphabet(branch_alphabet, rules)
        branch = merge_groups_connected_by_not_coexistence(branch, branch_rules)
        branch = merge_groups_connected_by_chain_rules(branch, branch_rules)
        if len(branch) == 1:
            group = branch[0]
            split = try_label_splitting(group, branch_rules)
            if split is not None:
                branch = split
        if branch:
            branches.append(branch)
    if not branches:
        return None
    if len(branches) == 1 and len(branches[0]) <= 1:
        return None
    return branches


def mine_sequence_branch(
    obj: IMDataStructureUVCL,
    branch_rules,
    groups,
    mine: Callable[[IMDataStructureUVCL, tuple[AbstractRule, ...]], TaggedPOWL],
    parameters=None,
):
    if not groups:
        return None
    branch_alphabet = set().union(*groups)
    branch_rules = supported_rules_alphabet(branch_alphabet, branch_rules)
    if RuleAwareSequenceCutUVCL.check_rules(branch_rules, groups):
        return None
    children = RuleAwareSequenceCutUVCL.relaxed_projection(obj, groups, parameters)
    projected_rules = RuleAwareSequenceCutUVCL.project_rules(branch_rules, groups)
    if len(children) != len(groups):
        return None
    models = [
        mine(child, tuple(child_rules))
        for child, child_rules in zip(children, projected_rules)
    ]
    if any(model is None for model in models):
        return None
    if len(models) == 1:
        return models[0]
    return build_model(SequenceSpec(len(models)), models)


def rule_po(obj: IMDataStructureUVCL, rules, mine, parameters=None):
    """Mine the reference rule-induced parallel/sequence decomposition."""
    if not rules:
        return None
    graph = dfg_graph(obj)
    po = detect_rule_based_po(rules, alphabet_of(obj), graph)
    if po is None:
        return None
    branches = po_to_parallel_sequence_branches(po, rules)
    if branches is None:
        return None
    branch_alphabets = [set().union(*branch) for branch in branches]
    projected = RuleAwareConcurrencyCutUVCL.project_rules(rules, branch_alphabets)
    models = []
    for groups, branch_rules in zip(branches, projected):
        model = mine_sequence_branch(obj, branch_rules, groups, mine, parameters)
        if model is None:
            return None
        models.append(model)
    if not models:
        return None
    if len(models) == 1:
        return models[0]
    return build_model(PartialOrderSpec(len(models)), models)
