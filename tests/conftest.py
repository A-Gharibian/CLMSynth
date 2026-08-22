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
import clmsynth.main  # noqa: I001

@pytest.fixture
def no_plots(monkeypatch):
    """Stop `run_pipeline` from rendering anything, returns True.
    """
    monkeypatch.setattr(clmsynth.main, "plot_feature_scatter", lambda *a, **k: True)
