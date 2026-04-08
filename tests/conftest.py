import io
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import powl
from powl.discovery.total_order_based.inductive.variants.powl_discovery_varaints import (
    POWLDiscoveryVariant,
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def examples_dir(repo_root: Path) -> Path:
    return repo_root / "examples"


@pytest.fixture(scope="session")
def running_example_path(examples_dir: Path) -> Path:
    return examples_dir / "running-example.csv"


@pytest.fixture(scope="session")
def hospital_example_path(examples_dir: Path) -> Path:
    return examples_dir / "hospital.csv"


@pytest.fixture(scope="session")
def workflow_net_path(examples_dir: Path) -> Path:
    return examples_dir / "hospital_wf_net.pnml"


@pytest.fixture(scope="session")
def running_example_log(running_example_path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return powl.import_event_log(str(running_example_path))


@pytest.fixture(scope="session")
def hospital_example_log(hospital_example_path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return powl.import_event_log(str(hospital_example_path))


@pytest.fixture(scope="session")
def running_example_model(running_example_log):
    return powl.discover(
        running_example_log,
        dfg_frequency_filtering_threshold=0.0,
        variant=POWLDiscoveryVariant.DECISION_GRAPH_CYCLIC,
    )


@pytest.fixture(scope="session")
def hospital_example_model(hospital_example_log):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(io.StringIO()):
            return powl.discover_from_partially_ordered_log(hospital_example_log)
