"""Shared setup for the whole suite.

Two things live here because three modules each had their own copy, and a
divergence between copies would be invisible until one of them started
rendering real plots on a headless runner.

Deliberately NOT here: an autouse plot patch. `04_failure_modes` exercises
plot *failure* on purpose, so a suite-wide patch would quietly defeat the tests
that matter most about plotting. Modules that want plots suppressed opt in with
their own one-line autouse fixture delegating to `no_plots` below, which keeps
that choice visible in the module that makes it.
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
