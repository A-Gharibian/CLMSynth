"""
Shared setup for the whole suite.
"""

import gzip
import io
import urllib.error
from email.message import Message

import matplotlib
import numpy as np
import pytest

# Before any test imports pyplot. matplotlib's default backend needs a GUI main
# thread; Agg writes files and needs nothing, which is what CI has. Set once
# here rather than in each module.
matplotlib.use("Agg")

# Must follow the backend selection above.
import clmsynth._extras.visualization
from clmsynth._input_sources import clustbench_source, dataset_cache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """This test's dataset cache folder, named by CLMSYNTH_CACHE_DIR."""
    folder = tmp_path / "cache"
    monkeypatch.setenv(dataset_cache.ENV_VAR, str(folder))
    return folder


class Network(list):
    """Every clustbench URL requested, in order. Clearing `up` takes the network down;
    a URL ending in one of `flaky` times out once. Each dataset has `rows` points."""

    up = True
    rows = 40

    def __init__(self) -> None:
        super().__init__()
        self.flaky: set[str] = set()


def _gz(lines) -> io.BytesIO:
    return io.BytesIO(gzip.compress("\n".join(lines).encode()))


@pytest.fixture
def network(monkeypatch):
    """clustbench's downloads, stubbed: each dataset and each base_url has data of its
    own, two labelings, three clusters."""
    calls = Network()
    n = calls.rows

    def urlopen(url, timeout=None):
        calls.append(url)
        if not calls.up:
            raise urllib.error.URLError("no network")
        for ending in [e for e in calls.flaky if url.endswith(e)]:
            calls.flaky.discard(ending)
            raise urllib.error.URLError("timed out")
        shift = sum(map(ord, url.rsplit(".", 2)[0])) % 97
        if url.endswith(".data.gz"):
            return _gz(f"{(i * 0.37 + shift) % 11:.4f} {np.sin(i + shift):.6f}" for i in range(n))
        if url.endswith(".labels0.gz"):
            return _gz(str(i % 3) for i in range(n))
        if url.endswith(".labels1.gz"):
            return _gz(str(i % 2) for i in range(n))
        raise urllib.error.HTTPError(url, 404, "not found", Message(), None)

    monkeypatch.setattr(clustbench_source.urllib.request, "urlopen", urlopen)
    return calls


@pytest.fixture(autouse=True, scope="session")
def _a_dataset_cache_of_its_own(tmp_path_factory):
    """No test reads or writes the user's dataset cache: a run a test starts, a
    batch's runs included, finds it in a folder of the session's own."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CLMSYNTH_CACHE_DIR", str(tmp_path_factory.mktemp("dataset_cache")))
        yield


def pytest_report_collectionfinish(config):
    """Which clmsynth the suite imports: the source tree or an installed copy.
    Unlike `pytest_report_header`, this also prints under `-q`."""
    return f"clmsynth {clmsynth.__version__}: {clmsynth.__file__}"


@pytest.fixture
def no_plots(monkeypatch):
    """Stop `run_pipeline` from rendering anything: the patched plotter returns True.

    Patched on `clmsynth._extras.visualization`: `main` imports it per call.
    """
    monkeypatch.setattr(clmsynth._extras.visualization,"plot_feature_scatter", lambda *a, **k: True)


@pytest.fixture
def yaml():
    """PyYAML, which is optional (the `yaml` extra). A test that reads or writes
    YAML takes this fixture, and skips where PyYAML is not installed."""
    return pytest.importorskip("yaml")
