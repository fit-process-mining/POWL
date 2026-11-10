from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pm4py

import powl
from powl.discovery.rule_based import (
    ChainResponseRule,
    NotCoExistenceRule,
    PrecedenceRule,
    ResponseRule,
)


def execute_script(output: Path | None = None):
    log = Counter(
        {
            ("Register", "CT", "Diagnose", "Treat", "Discharge"): 6,
            ("Register", "MRI", "Diagnose", "Treat", "Discharge"): 4,
            ("Register", "CT", "MRI", "Diagnose", "Treat", "Discharge"): 2,
        }
    )
    rows = []
    case_id = 0

    for trace, frequency in log.items():
        for _ in range(frequency):
            case_id += 1

            for event_index, activity in enumerate(trace):
                rows.append({
                    "case:concept:name": str(case_id),
                    "concept:name": activity,
                    "time:timestamp": pd.Timestamp("2026-01-01")
                                      + pd.Timedelta(minutes=event_index),
                })

    df = pd.DataFrame(rows)

    rules = {
        NotCoExistenceRule("CT", "MRI"),
        ResponseRule("MRI", "Diagnose"),
        PrecedenceRule("Diagnose", "Treat"),
        ChainResponseRule("Treat", "Discharge"),
    }

    print("Supplied rules on the original log:")
    for result in powl.validate_rules(log, rules):
        print(
            f"  {str(result.rule):42s} "
            f"support={result.support:.3f}  "
            f"confidence={result.confidence:.3f}  "
            f"activated={result.activated_traces:2d}  "
            f"violated={result.violated_traces:2d}/{result.total_traces}"
        )
    print(
        "Support counts every trace, including vacuous satisfaction; confidence "
        "counts activated traces. Always read confidence with its activation count."
    )

    # model without rules
    model = powl.discover(
        df
    )

    powl.view(model)

    net, initial_marking, final_marking = powl.convert_to_petri_net(model)
    pm4py.view_petri_net(net, initial_marking, final_marking, format="svg")

    # model with rules
    model_2 = powl.discover_with_rules(
        df,
        rules,
        repair_mode="event_level",
        noise_threshold=0.0,
        verify_rules=True,
    )

    powl.view(model_2)

    net_2, initial_marking_2, final_marking_2 = powl.convert_to_petri_net(model_2)
    pm4py.view_petri_net(net_2, initial_marking_2, final_marking_2, format="svg")


if __name__ == "__main__":
    execute_script()
