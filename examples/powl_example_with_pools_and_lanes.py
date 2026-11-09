import os

import powl
from powl.objects.tagged_powl import (
    Activity,
    ChoiceGraph,
    PartialOrder

)
from powl.objects.tagged_powl.builders import (
    xor, sequence
)

def generate_process_1():
    order_coffee = Activity("Order Coffee", "Customer", "Customer")
    pay = Activity("Pay", "Customer", "Customer")
    prepare_coffee = Activity("Prepare Coffee", "Cafe", "Customer")
    serve_coffee = Activity("Serve Coffee", "Cafe", "Barista")
    none = Activity()
    serve_coffee = xor(children=[serve_coffee, none])
    # Embed the sequence as one choice-graph child.
    seq = sequence(children=[pay, prepare_coffee])

    dg = ChoiceGraph(
        nodes=[order_coffee, seq, serve_coffee],
        edges=[(order_coffee, seq), (seq, serve_coffee)],
        start_nodes=[order_coffee],
        end_nodes=[serve_coffee],
    )
    # Visualize it
    powl.view(dg)
    powl.export(dg, "coffee_shop_process_no_pools_lanes.bpmn")

    assignments = {
        "Order Coffee": ("Customer Pool", "Customer Lane"),
        "Pay": ("Customer Pool", "Customer Lane"),
        "Prepare Coffee": ("Cafe Pool", "Cafe Lane"),
        "Serve Coffee": ("Cafe Pool", "Barista Lane"),
    }
    powl.export(dg, "coffee_shop_process.bpmn", activity_to_pool_lane=assignments)


def generate_process_2():

    log = powl.import_event_log(r"./examples/running-example.csv")
    model = powl.discover(log, dfg_frequency_filtering_threshold=0.0)

    activity_to_pool_lane = {
        "register request": ("P1", "Lane1"),
        "reinitiate request": ("P2", "Lane1"),
        "pay compensation": ("P1", "Lane1"),
        "reject request": ("P1", "Lane1"),
        "examine casually": ("P2", "Lane2"),
        "examine thoroughly": ("P1", "Lane2"),
        "check ticket": ("P2", "Lane3"),
        "decide": ("P1", "Lane3"),
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    bpmn_path = os.path.join(current_dir, "powl_bpmn.bpmn")
    powl.export(model, bpmn_path, activity_to_pool_lane=activity_to_pool_lane)


if __name__ == "__main__":
    generate_process_2()
