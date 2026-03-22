from pm4py.objects.petri_net.obj import Marking, PetriNet

from powl.conversion.utils.pn_reduction import add_arc_from_to
from powl.objects.tagged_powl.activity import Activity
from powl.objects.tagged_powl.builders import loop, silent_activity, xor
from powl.objects.tagged_powl.choice_graph import ChoiceGraph
from powl.objects.tagged_powl.partial_order import PartialOrder
from powl.objects.oc_powl import ComplexModel, LeafNode, ObjectCentricPOWL


DIV_IF_ALL_CHILDREN_DIV = True


def generate_xor(children):
    if len(children) == 1:
        return children[0]
    return xor(children)


def generate_flower_model(children):
    xor = generate_xor(children)
    return loop(silent_activity(), xor)


def clone_workflow_net(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    name_suffix: str = "_copy",
    label_delimiter: str = "<|>",
):
    """
    Create a deep copy of a Petri net and its initial/final markings.
    - Appends `name_suffix` to place/transition names.
    - If a transition label contains `label_delimiter`, keeps only the part before it.
    """
    mapping = {}

    new_net = PetriNet(f"{net.name}{name_suffix}")

    for place in net.places:
        new_place = PetriNet.Place(f"{place.name}{name_suffix}")
        new_net.places.add(new_place)
        mapping[place] = new_place

    for t in net.transitions:
        label = t.label
        if label and label_delimiter in label:
            label = label.split(label_delimiter, 1)[0].strip()
        new_t = PetriNet.Transition(name=f"{t.name}{name_suffix}", label=label)
        new_net.transitions.add(new_t)
        mapping[t] = new_t

    for arc in net.arcs:
        add_arc_from_to(mapping[arc.source], mapping[arc.target], new_net)

    new_im = Marking({mapping[p]: im[p] for p in im})
    new_fm = Marking({mapping[p]: fm[p] for p in fm})

    return new_net, new_im, new_fm


def project_oc_powl(oc_powl: ObjectCentricPOWL, object_type, div_edges):

    if isinstance(oc_powl, LeafNode):
        if oc_powl.activity == "" or object_type not in oc_powl.related:
            return Activity(label=None)
        activity = oc_powl.activity
        if object_type in oc_powl.get_type_information()[(activity, "div")]:
            return generate_flower_model([Activity(label=activity)])
        return Activity(label=activity)

    assert isinstance(oc_powl, ComplexModel)
    related_activities = set(
        [
            a
            for a in oc_powl.get_activities()
            if object_type in oc_powl.get_type_information()[(a, "rel")] and a != ""
        ]
    )

    if not related_activities:
        return Activity(label=None)

    if all(
        object_type in oc_powl.get_type_information()[(a, "div")]
        for a in related_activities
    ):
        return generate_flower_model([Activity(label=a) for a in related_activities])

    else:
        parallel = (
            isinstance(oc_powl.flat_model, PartialOrder)
            and len(oc_powl.flat_model.get_edges()) == 0
        )
        if parallel:
            return PartialOrder(
                nodes=[
                    project_oc_powl(sub, object_type, div_edges)
                    for sub in oc_powl.oc_children
                ],
            )

        diverging = [
            i
            for i in range(len(oc_powl.oc_children))
            if oc_powl.oc_children[i].get_activities() & related_activities
            and all(
                object_type in oc_powl.get_type_information()[(a, "div")]
                for a in oc_powl.oc_children[i].get_activities() & related_activities
            )
        ]
        non_diverging = [
            i
            for i in range(len(oc_powl.oc_children))
            if oc_powl.oc_children[i].get_activities() & related_activities
            and i not in diverging
        ]

        if isinstance(oc_powl.flat_model, PartialOrder):
            div_activities = set(
                sum(
                    [
                        list(
                            oc_powl.oc_children[i].get_activities() & related_activities
                        )
                        for i in diverging
                    ],
                    [],
                )
            )
            div_activities = {a for a in div_activities if a != ""}

            if div_activities:
                div_subtree = generate_flower_model(
                    [Activity(label=a) for a in div_activities]
                )
                if len(non_diverging) > 0:
                    mapping = {
                        oc_powl.flat_model.children[i]: (
                            project_oc_powl(
                                oc_powl.oc_children[i], object_type, div_edges
                            )
                            if i in non_diverging
                            else Activity(label=None)
                        )
                        for i in range(len(oc_powl.oc_children))
                    }
                    non_div_subtree = oc_powl.flat_model.map_nodes(mapping)
                    return PartialOrder([non_div_subtree, div_subtree])
                else:
                    return div_subtree
            else:
                mapping = {
                    oc_powl.flat_model.children[i]: project_oc_powl(
                        oc_powl.oc_children[i], object_type, div_edges
                    )
                    for i in range(len(oc_powl.oc_children))
                }
                return oc_powl.flat_model.map_nodes(mapping)

        elif isinstance(oc_powl.flat_model, ChoiceGraph):

            if DIV_IF_ALL_CHILDREN_DIV:
                parts = _partition_children(
                    oc_powl, diverging, related_activities, div_edges
                )

                mapping = {}
                processed_ids = set()
                for group in parts:
                    div_children = [
                        Activity(a)
                        for i in group
                        for a in oc_powl.oc_children[i].get_activities()
                        & related_activities
                    ]
                    flower = generate_flower_model(div_children)
                    for i in group:
                        processed_ids.add(i)
                        mapping[oc_powl.flat_model.children[i]] = flower

                for i in range(len(oc_powl.oc_children)):
                    if i in processed_ids:
                        continue
                    mapping[oc_powl.flat_model.children[i]] = project_oc_powl(
                        oc_powl.oc_children[i], object_type, div_edges
                    )

                return oc_powl.flat_model.map_nodes(mapping)

            else:
                parts = _partition_children(
                    oc_powl, diverging, related_activities, div_edges
                )

                mapping = {}

                for group in parts:
                    if len(group) == 1:
                        i = list(group)[0]
                        mapping[oc_powl.flat_model.children[i]] = project_oc_powl(
                            oc_powl.oc_children[i], object_type, div_edges
                        )
                    else:
                        div_children = [
                            project_oc_powl(
                                oc_powl.oc_children[i], object_type, div_edges
                            )
                            for i in group
                        ]
                        flower = generate_flower_model(div_children)
                        for i in group:
                            mapping[oc_powl.flat_model.children[i]] = flower

                return oc_powl.flat_model.map_nodes(mapping)

        else:
            raise NotImplementedError


def _partition_children(oc_powl, diverging, related_activities, div_edges):
    edges = [tuple(edge) for edge in div_edges]

    if DIV_IF_ALL_CHILDREN_DIV:
        parts = [{i} for i in diverging]
    else:
        parts = [{i} for i in range(len(oc_powl.oc_children))]

    def find_group(a):
        for g in parts:
            for i in g:
                if a in oc_powl.oc_children[i].get_activities():
                    return g
        return None

    for u, v in edges:
        if u in related_activities and v in related_activities:
            gu = find_group(u)
            gv = find_group(v)
            if gu is None or gv is None:
                continue
            if gu is gv:
                continue
            gu |= gv
            parts.remove(gv)

    return parts


def handle_deficiency(oc_powl: ObjectCentricPOWL):

    if isinstance(oc_powl, LeafNode):
        if oc_powl.activity == "":
            return oc_powl, []
        else:
            from itertools import chain, combinations

            stable_types = oc_powl.related - oc_powl.deficient
            variable_types = oc_powl.related & oc_powl.deficient
            if variable_types:
                ot_sets = [
                    stable_types | {c for c in comb}
                    for comb in chain.from_iterable(
                        combinations(variable_types, n)
                        for n in range(len(variable_types) + 1)
                    )
                ]

                mapping = {}
                for ots in ot_sets:
                    transition = Activity(
                        oc_powl.activity + "<|>" + str(sorted(list(ots)))
                    )
                    mapping[transition] = LeafNode(
                        transition=transition,
                        related=ots,
                        convergent=oc_powl.convergent & ots,
                        deficient=set(),
                        divergent=oc_powl.divergent & ots,
                    )
                flat_model = generate_xor(children=list(mapping.keys()))
                return ComplexModel(flat_model=flat_model, mapping=mapping), [
                    oc_powl.activity
                ]
            else:
                return oc_powl, []

    assert isinstance(oc_powl, ComplexModel)
    sub_results = [handle_deficiency(sub) for sub in oc_powl.oc_children]
    flat_children = [child[0].flat_model for child in sub_results]
    flat_to_oc_mapping = {
        flat_children[i]: sub_results[i][0] for i in range(len(sub_results))
    }
    old_flat_to_new_flat_mapping = {
        oc_powl.flat_model.children[i]: flat_children[i]
        for i in range(len(oc_powl.flat_model.children))
    }

    flat_model = oc_powl.flat_model

    if isinstance(flat_model, PartialOrder) or isinstance(flat_model, ChoiceGraph):
        new_flat_model = flat_model.map_nodes(old_flat_to_new_flat_mapping)
    else:
        raise NotImplementedError

    return ComplexModel(new_flat_model, flat_to_oc_mapping), sum(
        [sub[1] for sub in sub_results], []
    )


def convert_ocpowl_to_ocpn(oc_powl: ObjectCentricPOWL, divergence_matrices):

    assert isinstance(oc_powl, ObjectCentricPOWL)
    nets = {}
    nets_duplicates = {}

    convergent_activities = {}
    activities = set()
    oc_powl, special_activities = handle_deficiency(oc_powl)

    for ot in oc_powl.get_object_types():
        powl_model = project_oc_powl(oc_powl, ot, divergence_matrices[ot])
        powl_model = powl_model.normalize()
        from powl.conversion.converter import apply as to_pn

        net, im, fm = to_pn(powl_model)
        nets[ot] = net, im, fm
        nets_duplicates[ot] = clone_workflow_net(net, im, fm, label_delimiter="<|>")
        activities.update(
            {
                a
                for a in oc_powl.get_activities()
                if ot in oc_powl.get_type_information()[(a, "rel")]
            }
        )
        convergent_activities[ot] = {
            a: ot in oc_powl.get_type_information()[(a, "con")]
            for a in oc_powl.get_activities()
        }

    ocpn = {
        "activities": activities,
        "object_types": nets.keys(),
        "petri_nets": nets,
        "petri_nets_with_duplicates": nets_duplicates,
        "double_arcs_on_activity": convergent_activities,
        "tbr_results": {},
    }

    return ocpn
