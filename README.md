# POWL Miner

**Process mining with the Partially Ordered Workflow Language (POWL 2.0).**

POWL Miner is a Python library for discovering, constructing, visualizing, and exporting process models. Models can be saved as POWL JSON or as BPMN 2.0 with diagram layout, including pools and lanes.

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
powl.export(model, "model.powl")       # POWL JSON
powl.export(model, "model.bpmn")       # BPMN 2.0
```

By default, discovery expects the `case:concept:name`, `concept:name`, and `time:timestamp` columns. Pass the corresponding options to `powl.discover()` for custom columns and filtering thresholds.

## Construct and exchange models

POWL models can also be constructed directly, without an event log:

```python
model = powl.sequence([powl.Activity("Receive"), powl.Activity("Check")])
powl.export(model, "handmade.powl")
powl.export(model, "handmade.bpmn")
powl.save_visualization(model, "handmade.svg")
```

Import a previously exported POWL JSON file with `powl.read_powl_json("model.powl")`.

To export a BPMN collaboration with pools and lanes, pass the activity assignments:

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
