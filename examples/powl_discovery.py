import os

import pm4py
import powl


def execute_script():

    # Read event log (csv, xes, or xes.gz)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "running-example.csv")
    log = powl.import_event_log(path)

    # Discover POWL 2.0 model
    model = powl.discover(
        log,
        dfg_frequency_filtering_threshold=0.0,
        variant=powl.POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC,
    )

    # View the discovered model
    powl.view(model)

    # Export visualization
    image_path = os.path.join(current_dir, "powl_vis.svg")
    powl.save_visualization(model, file_path=image_path)

    # Convert into a PM4Py Petri net
    petri_net, initial_marking, final_marking = powl.convert_to_petri_net(model)
    pm4py.view_petri_net(petri_net, initial_marking, final_marking, format="SVG")

    # Convert into a PM4Py BPMN model
    bpmn = powl.convert_to_bpmn(model)
    pm4py.view_bpmn(bpmn, format="SVG")


if __name__ == "__main__":
    execute_script()
