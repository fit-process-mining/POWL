# POWL Miner

**Process mining with the Partially Ordered Workflow Language (POWL 2.0).**

POWL is a Python library for discovering process models from event data. You can view the models, build your own, and export them as POWL JSON or BPMN diagrams, including pools and lanes.

## Installation

Requires **Python 3.11+**. Install the library with:

```bash
pip install powl
```

For SVG visualizations and BPMN export, install the **Graphviz `dot` executable** and ensure it is on your `PATH` (for example, `apt install graphviz` or `brew install graphviz`). The Python dependency alone does not include this executable.

## Quick start

```python
import powl

log = powl.import_event_log("events.csv")  # Also supports .xes and .xes.gz
model = powl.discover(log)

powl.view(model)                       # View the POWL model
powl.save_visualization(model, "model.svg")  # Save an image
powl.export(model, "model.powl")       # POWL JSON
powl.export(model, "model.bpmn")       # BPMN 2.0
```

The log needs a case ID, an activity name, and a timestamp for each event. By default, these columns are named `case:concept:name`, `concept:name`, and `time:timestamp`. You can also pass a pandas DataFrame directly to `powl.discover()`.

## Use your own column names

For a CSV with columns named `case_id`, `activity`, and `timestamp`:

```python
custom_log = powl.import_event_log("events.csv", timestamp_key="timestamp")
custom_model = powl.discover(
    custom_log,
    case_id_key="case_id",
    activity_key="activity",
    timestamp_key="timestamp",
)
```

To reduce the influence of infrequent behavior, try filtering during discovery:

```python
filtered_model = powl.discover(log, dfg_frequency_filtering_threshold=0.1)
```

The threshold ranges from `0` to less than `1`; higher values apply stronger filtering.

## Discover parallel activities

If activities can happen at the same time, use their timestamps to discover a model that preserves this parallel behavior. When start and completion events are available, the library also uses overlapping activity durations.

```python
parallel_model = powl.discover_from_partially_ordered_log(log)
powl.view(parallel_model)
```

## Work with object-centric event logs

For event data that connects several kinds of objects, such as orders and items, import an OCEL file and discover an object-centric Petri net:

```python
ocel = powl.import_ocel("events.sqlite")
object_centric_model = powl.discover_petri_net_from_ocel(ocel)
powl.view_ocpn(object_centric_model)
```

## Rule-aware discovery

Include known process rules when discovering a model. For example, CT and MRI must not occur in the same case, and treatment must be followed immediately by discharge:

```python
from powl.discovery.rule_based import ChainResponseRule, NotCoExistenceRule

rules = {NotCoExistenceRule("CT", "MRI"), ChainResponseRule("Treat", "Discharge")}
rule_model = powl.discover_with_rules(log, rules)
powl.export(rule_model, "rule-aware.powl")
```

## Construct and exchange models

POWL models can also be constructed directly, without an event log:

```python
model = powl.sequence([powl.Activity("Receive"), powl.Activity("Check")])
powl.export(model, "handmade.powl")
powl.export(model, "handmade.bpmn")
```

Reopen a saved model with `powl.read_powl_json("handmade.powl")`.

To use a POWL model with other process-mining tools, convert it to a Petri net:

```python
net, initial_marking, final_marking = powl.convert_to_petri_net(model)
```

To add pools and lanes to the BPMN export, assign each activity a pool and lane:

```python
assignments = {
    "Receive": ("Customer", "Requester"),
    "Check": ("Team", "Reviewer"),
}
powl.export(model, "with-resources.bpmn", activity_to_pool_lane=assignments)
```

The file extension selects the export format: `.powl` for POWL JSON, or `.bpmn` for BPMN.

## Streamlit app

Try the [hosted app](https://powl-miner.streamlit.app/), or run it from a repository checkout:

```bash
pip install -e ".[app]"
streamlit run app.py
```

## Reference

Kourani, H., Park, G. & van der Aalst, W.M.P. A discovery technique for expressive yet sound process models. Process Sci 3, 14 (2026). https://doi.org/10.1007/s44311-026-00046-8
