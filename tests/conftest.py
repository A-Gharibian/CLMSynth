"""Shared setup for the whole suite.
"""

import matplotlib
import pytest

# Before any test imports pyplot. matplotlib's default backend needs a GUI main
# thread; Agg writes files and needs nothing, which is what CI has. Set once
# here rather than in each module, since the first import wins and a module
# that forgot would silently depend on another module's setting.
matplotlib.use("Agg")

# Must follow the backend selection above, not be hoisted to the import block.
import clmsynth.main


@pytest.fixture
def no_plots(monkeypatch):
    """Stop `run_pipeline` from rendering anything,
    Returns True, which is what `plot_feature_scatter` returns on success, so
    `run_pipeline` takes its success path and logs no plot-failure warning.
    """
    monkeypatch.setattr(clmsynth.main, "plot_feature_scatter", lambda *a, **k: True)
