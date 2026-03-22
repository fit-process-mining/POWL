from pm4py import PetriNet

from powl.conversion.to_powl.from_pn.utils.cut_detection import (
    mine_base_case,
    mine_choice_graph,
    mine_partial_order,
)

from powl.conversion.to_powl.from_pn.utils.preprocessing import (
    preprocess,
    validate_workflow_net, make_self_loop_explicit,
)
from powl.conversion.to_powl.from_pn.utils.subnet_creation import (
    apply_projection, clone_subnet,
)
from powl.conversion.to_powl.from_pn.utils.weak_reachability import (
    get_simplified_reachability_graph,
)
from powl.objects.tagged_powl.activity import Activity
from powl.objects.tagged_powl.base import TaggedPOWL
from powl.objects.tagged_powl.choice_graph import ChoiceGraph
from powl.objects.tagged_powl.partial_order import PartialOrder


def convert_workflow_net_to_powl(net: PetriNet) -> TaggedPOWL:
    """
    Convert a Petri net to a POWL model.

    Parameters:
    - net: PetriNet

    Returns:
    - POWL model
    """
    start_place, end_place = validate_workflow_net(net)
    net = preprocess(net)
    res = __translate_petri_to_powl(net, start_place, end_place)
    return res.normalize()


def __translate_petri_to_powl(
    net: PetriNet, start_place: PetriNet.Place, end_place: PetriNet.Place
) -> TaggedPOWL:

    net, start_place, end_place = make_self_loop_explicit(net, start_place, end_place)

    base_case = mine_base_case(net)
    if base_case:
        return base_case

    reachability_map = get_simplified_reachability_graph(net)

    partition = mine_partial_order(net, start_place, end_place, reachability_map)
    if len(partition) > 1:
        return __translate_partial_order(net, partition, start_place, end_place)

    partition = mine_choice_graph(net)
    if len(partition) > 1:
        return __translate_choice_graph(net, partition, start_place, end_place)

    raise Exception(
        f"Failed to detected a POWL structure over the following transitions: {net.transitions}"
    )

def __translate_to_relation(
    net, transition_groups, i_place: PetriNet.Place, f_place: PetriNet.Place, enforce_unique_connection_points
):

    groups = [tuple(g) for g in transition_groups]
    transition_to_group_map = {transition: g for g in groups for transition in g}

    group_start_places = {g: set() for g in groups}
    group_end_places = {g: set() for g in groups}

    connection_edges = set()
    start_groups = set()
    end_groups = set()

    complex_places = set()

    for p in net.places:
        source_groups = {transition_to_group_map[arc.source] for arc in p.in_arcs}
        target_groups = {transition_to_group_map[arc.target] for arc in p.out_arcs}

        is_complex = (
                enforce_unique_connection_points
                and len(source_groups) > 1
                and len(target_groups) > 1
        )

        if is_complex:
            complex_places.add(p)

        # if p is start place and (p -> t), then p should be a start place in the subnet that contains t
        if p == i_place:
            if is_complex:
                start_groups.add(p)
            for group in target_groups:
                group_start_places[group].add(p)
                if not is_complex:
                    start_groups.add(group)
        # if p is end place and (t -> p), then p should be end place in the subnet that contains t
        if p == f_place:
            if is_complex:
                end_groups.add(p)
            for group in source_groups:
                group_end_places[group].add(p)
                if not is_complex:
                    end_groups.add(group)

        # if (t1 -> p -> t2) and t1 and t2 are in different subsets, then add an edge in the partial order
        # and set p as end place in g1 and as start place in g2
        for group_1 in source_groups:
            for group_2 in target_groups:
                if group_1 != group_2:
                    if is_complex:
                        connection_edges.add((group_1, p))
                        connection_edges.add((p, group_2))
                    else:
                        connection_edges.add((group_1, group_2))

                    group_end_places[group_1].add(p)
                    group_start_places[group_2].add(p)

    group_to_powl_map = {}
    children = []
    for group in groups:

        subnet, subnet_start_place, subnet_end_place = apply_projection(
            net, set(group), group_start_places[group], group_end_places[group], enforce_unique_connection_points
        )
        child = __translate_petri_to_powl(subnet, subnet_start_place, subnet_end_place)

        group_to_powl_map[group] = child
        children.append(child)

    for group in complex_places:
        child = Activity(label=None)
        group_to_powl_map[group] = child
        children.append(child)

    child_edges = [
        (group_to_powl_map[g1], group_to_powl_map[g2]) for (g1, g2) in connection_edges
    ]

    start_nodes = [group_to_powl_map[g] for g in start_groups]
    end_nodes = [group_to_powl_map[g] for g in end_groups]

    return children, child_edges, start_nodes, end_nodes


def __translate_partial_order(
    net, transition_groups, i_place: PetriNet.Place, f_place: PetriNet.Place
):

    children, edges, _, _ = __translate_to_relation(net,
                                                    transition_groups,
                                                    i_place,
                                                    f_place,
                                                    enforce_unique_connection_points=False)
    po = PartialOrder(nodes=children, edges=edges)
    po.validate_and_remove_transitive_edges()
    return po


def __translate_choice_graph(
    net, transition_groups, i_place: PetriNet.Place, f_place: PetriNet.Place
):

    children, edges, start_nodes, end_nodes = __translate_to_relation(net,
                                                                      transition_groups,
                                                                      i_place,
                                                                      f_place,
                                                                      enforce_unique_connection_points=True)
    cg = ChoiceGraph(
        nodes=children,
        edges=edges,
        start_nodes=start_nodes,
        end_nodes=end_nodes,
    )
    cg.validate_connectivity()
    return cg


def __create_sub_powl_model(
    net,
    branch: set[PetriNet.Transition],
    start_place: PetriNet.Place,
    end_place: PetriNet.Place,
):
    subnet, subnet_start_place, subnet_end_place = clone_subnet(
        net, branch, start_place, end_place
    )
    powl_model = __translate_petri_to_powl(subnet, subnet_start_place, subnet_end_place)
    return powl_model
