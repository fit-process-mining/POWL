import copy


def filter_ocpn_by_object_types(ocpn: dict, object_types_to_keep: list[str]) -> dict:
    """
    Filters an OCPN (Object-Centric Petri Net) model object based on a
    provided list of object types to keep.

    This function creates a new, deep-copied OCPN object containing only
    the data related to the specified object types.

    Aggregated views (like 'activities' and 'activities_indep') are
    re-calculated from the filtered object-type-specific views.

    Args:
        ocpn: The original OCPN dictionary structure.
        object_types_to_keep: A list of object type strings to preserve.

    Returns:
        A new, filtered OCPN dictionary.
    """

    # Use a set for efficient O(1) lookups
    object_types_set = set(object_types_to_keep)

    # Initialize the new, filtered ocpn object
    new_ocpn = {}

    # 1. Set the new complete set of object types
    new_ocpn['object_types'] = object_types_set

    # 2. Filter all dictionaries that are keyed by object_type
    #    We use deepcopy to ensure the new object is fully independent

    if 'activities_ot' in ocpn:
        new_ocpn['activities_ot'] = {
            ot: copy.deepcopy(data)
            for ot, data in ocpn['activities_ot'].items()
            if ot in object_types_set
        }

    if 'start_activities' in ocpn:
        new_ocpn['start_activities'] = {
            ot: copy.deepcopy(data)
            for ot, data in ocpn['start_activities'].items()
            if ot in object_types_set
        }

    if 'end_activities' in ocpn:
        new_ocpn['end_activities'] = {
            ot: copy.deepcopy(data)
            for ot, data in ocpn['end_activities'].items()
            if ot in object_types_set
        }

    if 'edges' in ocpn:
        new_ocpn['edges'] = {
            ot: copy.deepcopy(data)
            for ot, data in ocpn['edges'].items()
            if ot in object_types_set
        }

    if 'petri_nets' in ocpn:
        new_ocpn['petri_nets'] = {
            ot: copy.deepcopy(data)
            for ot, data in ocpn['petri_nets'].items()
            if ot in object_types_set
        }

    if 'double_arcs_on_activity' in ocpn:
        new_ocpn['double_arcs_on_activity'] = {
            ot: copy.deepcopy(data)
            for ot, data in ocpn['double_arcs_on_activity'].items()
            if ot in object_types_set
        }

    if 'tbr_results' in ocpn:
        new_ocpn['tbr_results'] = {
            ot: copy.deepcopy(data)
            for ot, data in ocpn['tbr_results'].items()
            if ot in object_types_set
        }

    # 4. Re-calculate 'activities_indep' by aggregating the filtered 'activities_ot'
    new_activities_indep = {}
    if 'activities_ot' in new_ocpn:
        for ot_data in new_ocpn['activities_ot'].values():
            for act, act_data in ot_data.items():
                if act not in new_activities_indep:
                    # Initialize the entry for this activity
                    new_activities_indep[act] = {
                        'events': set(),
                        'unique_objects': set(),
                        'total_objects': set()
                    }

                # Aggregate the sets from the kept object types
                new_activities_indep[act]['events'].update(act_data.get('events', set()))
                new_activities_indep[act]['unique_objects'].update(act_data.get('unique_objects', set()))
                new_activities_indep[act]['total_objects'].update(act_data.get('total_objects', set()))

    new_ocpn['activities_indep'] = new_activities_indep

    # 3. Re-calculate the 'activities' set from the filtered 'activities_indep'
    # Also include start and end activities as they might not be in activities_ot
    activities_set = set(new_activities_indep.keys())

    if 'start_activities' in new_ocpn:
        for ot_data in new_ocpn['start_activities'].values():
            activities_set.update(ot_data.keys())

    if 'end_activities' in new_ocpn:
        for ot_data in new_ocpn['end_activities'].values():
            activities_set.update(ot_data.keys())

    if 'petri_nets' in new_ocpn:
        for (net, im, fm) in new_ocpn['petri_nets'].values():
            for trans in net.transitions:
                if trans.label:
                    activities_set.add(trans.label)

    new_ocpn['activities'] = activities_set

    return new_ocpn
