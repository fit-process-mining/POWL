"""Discover POWL from observed behavior and user-supplied declarative rules.

From a repository checkout with ``pip install -e .``:

    python examples/rule_based_discovery.py
    python examples/rule_based_discovery.py --output rule-aware.powl

The default run prints validation and model information. It does not open a
viewer or write files. ``--output`` explicitly exports POWL JSON or BPMN.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import powl
from powl.discovery.rule_based import (
    ChainResponseRule,
    NotCoExistenceRule,
    PrecedenceRule,
    ResponseRule,
)


def execute_script(output: Path | None = None):
    # Frequencies represent 12 cases without expanding them into 12 traces.
    # Two observed cases contain both tests, contrary to the supplied rule.
    log = Counter(
        {
            ("Register", "CT", "Diagnose", "Treat", "Discharge"): 6,
            ("Register", "MRI", "Diagnose", "Treat", "Discharge"): 4,
            ("Register", "CT", "MRI", "Diagnose", "Treat", "Discharge"): 2,
        }
    )
    rules = {
        NotCoExistenceRule("CT", "MRI"),
        ResponseRule("MRI", "Diagnose"),
        PrecedenceRule("Diagnose", "Treat"),
        ChainResponseRule("Treat", "Discharge"),
    }

    # This step is optional and evaluates the original observations. It neither
    # mines constraints nor changes or filters rules based on their scores.
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

    # Rules can reject an ordinary cut. Event-level repair follows the reference
    # rule-specific deletion/truncation policies, then discovery retries. The
    # supplied Counter and immutable rule objects remain unchanged.
    model = powl.discover_with_rules(
        log,
        rules,
        repair_mode="event_level",
        noise_threshold=0.0,
        verify_rules=True,
    )
    print(f"\nDiscovered {type(model).__name__}; original rules verified.")
    print(model)

    # Conversion needs no GUI. The result is usable by ordinary PM4Py tooling.
    net, initial_marking, final_marking = powl.convert_to_petri_net(model)
    print(f"Petri net: {len(net.places)} places, {len(net.transitions)} transitions.")

    if output is not None:
        if output.suffix.lower() not in {".powl", ".bpmn"}:
            raise ValueError("Choose an output filename ending in .powl or .bpmn.")
        powl.export(model, str(output))
        print(f"Exported {output}.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional .powl or .bpmn export path; no files are written by default.",
    )
    arguments = parser.parse_args()
    execute_script(arguments.output)
