import os

import powl


def example_1():
    log = powl.import_event_log(r"./examples/running-example.csv")
    model = powl.discover(log, dfg_frequency_filtering_threshold=0.0)
    # Export POWL directly as BPMN, including diagram layout.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bpmn_path = os.path.join(current_dir, "powl_bpmn_laid_out.bpmn")
    powl.export(model, bpmn_path)


if __name__ == "__main__":
    example_1()
