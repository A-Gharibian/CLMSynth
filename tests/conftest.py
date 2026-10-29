"""
Shared setup for the whole suite.
"""

import matplotlib
import pytest

# Before any test imports pyplot. matplotlib's default backend needs a GUI main
# thread; Agg writes files and needs nothing, which is what CI has. Set once
# here rather than in each module.
matplotlib.use("Agg")

# Must follow the backend selection above.
import clmsynth._visualization


def pytest_report_collectionfinish(config):
    """Which clmsynth the suite imports: the source tree or an installed copy.
    Unlike `pytest_report_header`, this also prints under `-q`."""
    return f"clmsynth {clmsynth.__version__}: {clmsynth.__file__}"


@pytest.fixture
def no_plots(monkeypatch):
    """Stop `run_pipeline` from rendering anything, returns True.

    Patched on `clmsynth._visualization`: `main` imports it per call.
    """
    monkeypatch.setattr(clmsynth._visualization, "plot_feature_scatter", lambda *a, **k: True)


@pytest.fixture
def yaml():
    """PyYAML, which is optional (the `yaml` extra). A test that reads or writes
    YAML takes this fixture, and skips where PyYAML is not installed."""
    return pytest.importorskip("yaml")
