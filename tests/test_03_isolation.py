"""Category 3: State isolation.

Components must keep their state to themselves. Every test here is a
single-threaded, deterministic assertion about *ownership of state*.

What is checked, in order:

  RNG ownership        `_fabricated_generator` reproduces a seed exactly after
                       an interleaved call with a different seed, and genuinely
                       differs between seeds (so the first is not vacuous).
                       Direct use of the global np.random stream is ruff
                       NPY002's job (pyproject.toml).
  config ownership     `run_pipeline` does not mutate the config dict it is
                       handed, or a second call with the same object resolves
                       zero datasets. mypy guards the suite block itself
                       (typed Mapping); this guards the lists inside it.
  run-folder ownership `build_run_dir` creates the name it returns, and never
                       returns the same name twice. Finding N2 (check-then-act)
                       was closed in 0.6.3; both halves are now guarantees
                       rather than one guarantee and one characterization.
  module-level state   Every registry (`CODES`, `FETCHERS`, `SOURCE_METADATA`,
                       `_METRIC_FUNCS`, ...) is byte-identical after a real run.
  logging ownership    Importing the package configures nothing. The console
                       scripts configure logging because they own the process;
                       a library caller keeps their own handlers and format.
                       What the filter they install *does* is not this module's
                       subject -- that is `05_config_safety`.
  viz isolation        `main` imports no plotting stack; --no-viz renders nothing.
  optional PyYAML      `main` imports no yaml. With yaml unimportable, a .json
                       config runs and its resolved config repeats the run, and
                       any other config file and the config tools name the extra.
  batch isolation      Several configs run one process each, with their own run
                       folder, log and exit status; a batch where some
                       configs wrote their datasets and others did not
                       exits 3.

"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import clmsynth._visualization
import clmsynth.main
from clmsynth import _fabricated_generator
from clmsynth._cli_logging import SingleLineFilter, configure_cli_logging
from clmsynth.main import build_run_dir, run_pipeline

# The folder this session imported clmsynth from. -I also ignores PYTHONPATH, which
# is how an uninstalled checkout (src/ as an IDE source root) reaches the package,
# so each child adds this folder itself. Added after interpreter start, it brings no
# sitecustomize.py with it, and the child tests the same copy as this session.
IMPORT_ROOT = str(Path(clmsynth.__file__).resolve().parents[1])


def run_isolated(code: str) -> subprocess.CompletedProcess:
    prelude = f"import sys\nsys.path.insert(0, {IMPORT_ROOT!r})\n"
    return subprocess.run([sys.executable, "-I", "-c", prelude + code],
                          capture_output=True, text=True, check=False)


def pipeline_config(output_dir):
    """The plainest run that processes exactly one dataset.

    A factory rather than a constant because `output_dir` differs per test, and
    because a shared mutable dict is precisely what the two tests below exist to
    catch `run_pipeline` doing.

    Local to this module on purpose. A fixture shared across modules couples
    their assertions: `run_pipeline(...) == 1` is only meaningful while this
    config names exactly one dataset, and a change made for another module's
    needs would silently redefine what these tests check.
    """
    return {
        "global_settings": {"data_source": "fabricated_data", "output_dir": str(output_dir)},
        "fabricated_data_suite": {
            "batteries": ["fabricated"],
            "datasets": ["baseline_4class"],
            "seed": 42,
        },
        "label_generation": {
            "n_labels": 1,
            "source_labeling": "labels0",
            "noise": 0.1,
            "seed": 42,
            "clm_label": {"num_classes": 4, "matching_mode": "perfect"},
        },
    }


@pytest.fixture(autouse=True)
def _no_plots(no_plots):
    """Plotting is irrelevant to state ownership and costs a second per call.
    Implementation is in conftest; opted into per module so that
    `04_failure_modes`, which tests plot failure deliberately, is unaffected.
    """


# ---------------------------------------------------------------------------
# The RNG must not travel through global interpreter state
# ---------------------------------------------------------------------------


def _fabricate(seed):
    return _fabricated_generator.generate_synthetic_data(n_samples=200, output_file=None, seed=seed)


def test_interleaved_seeds_do_not_contaminate_each_other():
    """A, B, A: the third call must reproduce the first exactly.
    Catches state carried *between* calls rather than in from outside. a
    generator instance reused across calls, or a module-level stream advanced by
    whatever ran last.

    Also the guard for global-RNG draws that ruff NPY002 cannot see, such as
    pandas `.sample()` or an sklearn call without `random_state`: each moves the
    global stream between the calls.
    """
    first = _fabricate(42)
    _fabricate(7)
    third = _fabricate(42)
    assert first["Feature_1"].equals(third["Feature_1"])


def test_different_seeds_actually_differ():
    """Guards the test above from being vacuous.

    If the generator ignored its seed entirely and returned constant data, the
    interleaved test would pass perfectly. This is what makes it mean
    something.
    """
    assert not _fabricate(42)["Feature_1"].equals(_fabricate(7)["Feature_1"])


# ---------------------------------------------------------------------------
# run_pipeline must not mutate the config it is handed
# ---------------------------------------------------------------------------


def test_run_pipeline_does_not_mutate_the_callers_config(tmp_path):
    """
    A second call with the same config, a caller looping over sources, or any
    library user reusing a parsed config must see it unchanged, or it would
    silently resolve zero datasets.

    `run_pipeline` reads the suite with `.get()`, and its `Mapping` type makes a
    `.pop()` or assignment on the suite a mypy error. The values inside are
    `Any`, though: an in-place change to the caller's `batteries` or `datasets`
    list (`.clear()`, `.sort()`) passes mypy, and this test is what catches it.
    """
    # Asserted behaviourally. This was previously checked by reading the function
    # source with `inspect.getsource` and searching for the literal text.
    config = pipeline_config(tmp_path)
    before = {
        k: list(v) if isinstance(v, list) else v for k, v in config["fabricated_data_suite"].items()
    }

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()
    assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir) == 1

    assert config["fabricated_data_suite"] == before, "run_pipeline mutated the caller's config"


# ---------------------------------------------------------------------------
# build_run_dir hands out a reservation, not just a name
# ---------------------------------------------------------------------------


def test_build_run_dir_never_hands_out_the_same_name_twice(tmp_path):
    """Repeated calls in the same wall-clock second must diverge.
    All three share a `YYYYMMDD_HHMMSS_Source` stem, so this is the collision path,
    not the happy one. No caller creates anything here: that is the point
    reservation is now the function's own job.
        Since 0.6.3 `build_run_dir` creates each folder as it hands it out, so the
    caller no longer creates them here, doing so would now raise
    FileExistsError against the folder the previous call just made. The suffix
    behavior under test is unchanged; only who does the `mkdir` moved.
    """
    paths = [build_run_dir(tmp_path, "IsolationTest") for _ in range(3)]
    assert len(set(paths)) == 3, f"a name was reused: {paths}"
    assert all(p.is_dir() for p in paths), "build_run_dir returned a path it did not create"
    assert len(list(tmp_path.iterdir())) == 3, "a folder was created that was never returned"


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


def test_module_level_registries_survive_a_run_unchanged(tmp_path):
    """Nothing accumulates in module-level state across a pipeline run.

    The package keeps a dozen module-level containers, `CODES`, `FETCHERS`,
    `SOURCE_METADATA`, `SOURCE_DATASETS`, `_METRIC_FUNCS`. All are
    lookup tables built once at import and typed read-only; mypy does not see a
    write below an `Any`, such as inside a `FABRICATED_CONFIGS` preset.

    If any of them were written during a run, they would be shared mutable
    state: the second run in a batch would see the first run's leftovers, and
    concurrent workers in one process would see each other's. Snapshotted
    around a real run rather than asserted by reading the source, so a new
    write added anywhere is caught regardless of how it is spelled.
    """
    import copy as _copy

    from clmsynth import clm_errors, clm_label_engine, dataset_sources
    from clmsynth import main as main_mod

    watched = {
        "clm_errors.CODES": clm_errors.CODES,
        "clm_label_engine._METRIC_FUNCS": clm_label_engine._METRIC_FUNCS,
        "dataset_sources.SOURCE_METADATA": dataset_sources.SOURCE_METADATA,
        "dataset_sources.SOURCE_DATASETS": dataset_sources.SOURCE_DATASETS,
        "dataset_sources.CLUSTBENCH_DATASETS": dataset_sources.CLUSTBENCH_DATASETS,
        "dataset_sources.FABRICATED_CONFIGS": dataset_sources.FABRICATED_CONFIGS,
        "dataset_sources.HEAVY_BATTERIES": dataset_sources.HEAVY_BATTERIES,
        "main.FETCHERS": main_mod.FETCHERS,
        "main.SOURCE_DISPLAY": main_mod.SOURCE_DISPLAY,
    }
    before = {name: _copy.deepcopy(obj) for name, obj in watched.items()}

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()
    assert (
        run_pipeline("fabricated_data", pipeline_config(tmp_path), csv_dir, png_dir, txt_dir) == 1
    )

    _fabricate(42)
    from clmsynth.clm_label_engine import generate_clm_labels

    clusters = np.concatenate([np.full(50, k) for k in range(4)])
    generate_clm_labels(
        clusters,
        np.random.default_rng(0).normal(size=(200, 2)),
        {"num_classes": 4, "matching_mode": "perfect"},
        seed=1,
    )

    changed = [name for name, obj in watched.items() if obj != before[name]]
    assert not changed, f"module-level state mutated during a run: {changed}"


# ---------------------------------------------------------------------------
# Logging configuration belongs to the process owner, not to the package
# ---------------------------------------------------------------------------


def test_importing_the_package_configures_no_logging():
    """A library must not reconfigure the logging of the process it is imported into.

    Every module logs to `clmsynth.<module>`, so `clmsynth` is the logger a
    package-level handler or filter would be attached to. It must stay bare:
    a caller embedding this in their own program keeps their handlers, their
    format and their levels, and gets our records through ordinary propagation.

    Robust to test ordering on purpose. `configure_cli_logging` attaches to the
    ROOT handlers, so another test calling it cannot make this one pass or fail
    by accident.
    """
    import clmsynth  # noqa: F401  (the import is the thing under test)

    package_logger = logging.getLogger("clmsynth")
    assert package_logger.handlers == [], (
        f"importing clmsynth installed handlers: {package_logger.handlers}"
    )
    assert package_logger.filters == [], (
        f"importing clmsynth installed filters: {package_logger.filters}"
    )


def test_configure_cli_logging_is_idempotent(monkeypatch):
    """Two calls must not stack two filters.

    A second SingleLineFilter would escape the backslash the first one wrote,
    turning a neutralized newline into a doubled escape, and growing on every
    further call. The console scripts each call this once today, but nothing
    stops a caller driving two of them in one process.

    The root logger's handler list is swapped for a private one first. Calling
    the real thing against the session's own handlers would attach the filter to
    pytest's capture handler and leave it there for every test that ran
    afterward.
    """
    probe = logging.NullHandler()
    monkeypatch.setattr(logging.getLogger(), "handlers", [probe])

    configure_cli_logging()
    configure_cli_logging()

    scrubbers = [f for f in probe.filters if isinstance(f, SingleLineFilter)]
    assert len(scrubbers) == 1, f"expected exactly one filter, got {len(scrubbers)}"


# ---------------------------------------------------------------------------
# The plotting stack stays out of the import graph until a plot is asked for
# ---------------------------------------------------------------------------


def test_importing_main_loads_no_plotting_stack():
    """Importing the pipeline must not load matplotlib."""
    # Subprocess: conftest imports matplotlib session-wide. -I (isolated) ignores
    # PYTHONPATH: an IDE's plot support puts a sitecustomize.py on it that imports
    # matplotlib at interpreter start (PyCharm's "Show plots in tool window"), which
    # the child would otherwise inherit and blame on clmsynth.main.
    probe = (
        "import json, sys\n"
        "def plotting():\n"
        "    return sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib')\n"
        "before = plotting()\n"
        "import clmsynth.main\n"
        "print(json.dumps([before, [m for m in plotting() if m not in before]]))\n"
    )
    result = run_isolated(probe)

    assert result.returncode == 0, result.stderr
    before, loaded = json.loads(result.stdout)
    # Otherwise the import below proves nothing: modules already present cannot be
    # loaded again, so the check would pass whatever clmsynth.main imports.
    assert not before, (
        f"the interpreter loaded the plotting stack before clmsynth was imported: {before}. "
        "A sitecustomize.py or .pth file in the environment imports it."
    )
    assert not loaded, f"importing clmsynth.main loaded the plotting stack: {loaded}"


def test_a_no_viz_run_never_reaches_the_plotter(tmp_path, monkeypatch):
    """`render_plots=False` reaches no plotting call."""
    calls = []

    def record(*args, **kwargs):
        calls.append(1)
        return True

    monkeypatch.setattr(clmsynth._visualization, "plot_feature_scatter", record)

    def run(name, render_plots):
        dirs = [tmp_path / name / d for d in ("csv", "png", "txt")]
        for d in dirs:
            d.mkdir(parents=True)
        # mypy cannot see that `dirs` holds exactly three paths.
        return run_pipeline(  # type: ignore[misc]
            "fabricated_data", pipeline_config(tmp_path / name), *dirs, render_plots=render_plots
        )

    assert run("on", True) == 1
    assert calls, "the default run rendered nothing, which would make the check below vacuous"

    calls.clear()
    assert run("off", False) == 1
    assert calls == [], f"a --no-viz run still called the plotter {len(calls)} time(s)"
    assert not list((tmp_path / "off" / "png").iterdir()), "a PNG was written anyway"


@pytest.mark.parametrize(
    "flags,renders", [([], True), (["--no-viz"], False)], ids=["default", "no-viz"]
)
def test_the_no_viz_flag_reaches_the_pipeline(tmp_path, monkeypatch, flags, renders):
    """The CLI flag reaches `run_pipeline`: driven through `main()`, not the argument."""
    calls = []

    def record(*args, **kwargs):
        calls.append(1)
        return True

    monkeypatch.setattr(clmsynth._visualization, "plot_feature_scatter", record)
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps(pipeline_config(tmp_path / "out")), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["clmsynth", str(config_path), *flags])

    clmsynth.main.main()

    assert bool(calls) is renders, (
        f"flags {flags}: expected rendering={renders}, plotter called {len(calls)} time(s)"
    )


# ---------------------------------------------------------------------------
# PyYAML is optional
#
# Each child runs with -I, for the reason given in the plotting probe above, and
# with yaml unimportable, which is what an environment without PyYAML gives it.
# ---------------------------------------------------------------------------

NO_YAML = "sys.modules['yaml'] = None\n"


def without_yaml(code: str) -> subprocess.CompletedProcess:
    return run_isolated(NO_YAML + code)


def run_cli_without_yaml(config_path) -> subprocess.CompletedProcess:
    return without_yaml(f"sys.argv = ['clmsynth', {str(config_path)!r}, '--no-viz']\n"
                        "from clmsynth.main import main\nmain()\n")


def test_importing_main_loads_no_yaml():
    probe = "import clmsynth.main\nprint('yaml' in sys.modules)\n"
    result = run_isolated(probe)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "importing clmsynth.main loaded yaml"


def test_a_json_config_runs_without_pyyaml(tmp_path):
    """The whole CLI path, and its resolved config run again, with no PyYAML."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(pipeline_config(tmp_path / "first")), encoding="utf-8")

    first = run_cli_without_yaml(config_path)

    assert first.returncode == 0, first.stderr
    (run_dir,) = (tmp_path / "first").iterdir()
    resolved = json.loads((run_dir / "info" / f"{run_dir.name}.json").read_text(encoding="utf-8"))
    resolved["global_settings"]["output_dir"] = str(tmp_path / "again")
    resolved_path = tmp_path / "resolved.json"
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")

    again = run_cli_without_yaml(resolved_path)

    assert again.returncode == 0, again.stderr
    (rerun_dir,) = (tmp_path / "again").iterdir()
    name = "fabricated_data-fabricated-baseline_4class.csv"
    assert (rerun_dir / "csv" / name).read_bytes() == (run_dir / "csv" / name).read_bytes()


def test_any_other_config_file_without_pyyaml_names_the_extra(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("global_settings: {}\n", encoding="utf-8")

    result = run_cli_without_yaml(config_path)

    assert result.returncode == 1, result.stderr
    assert 'pip install "clmsynth[yaml]"' in result.stderr, result.stderr


@pytest.mark.parametrize("module", ["generate_config", "_config_wizard"])
def test_the_config_tools_without_pyyaml_name_the_extra(module):
    """Before any payload is read or any question is asked."""
    result = without_yaml(f"from clmsynth.{module} import main\nmain()\n")

    assert result.returncode == 1, result.stdout + result.stderr
    assert 'pip install "clmsynth[yaml]"' in result.stdout + result.stderr


def test_a_batch_runs_each_config_in_its_own_process(tmp_path, monkeypatch, caplog):
    """Two configs, one refused, two at a time. Each run keeps its own folder and
    log, the summary names each config's run, and the batch exits 3, as a run
    does that wrote some of its datasets."""
    good, bad = pipeline_config(tmp_path / "out"), pipeline_config(tmp_path / "out")
    bad["label_generation"]["clm_label"]["matching_mode"] = "bogus"
    paths = []
    for name, config in (("good.json", good), ("bad.json", bad)):
        (tmp_path / name).write_text(json.dumps(config), encoding="utf-8")
        paths.append(str(tmp_path / name))
    monkeypatch.setattr(sys, "argv",
                        ["clmsynth", *paths, "--jobs", "2", "--no-viz", "--keep-failed"])
    caplog.set_level(logging.INFO)

    with pytest.raises(SystemExit) as exit_signal:
        clmsynth.main.main()

    assert exit_signal.value.code == 3
    logs = {run.name: (run / "info" / f"{run.name}.log").read_text(encoding="utf-8")
            for run in (tmp_path / "out").iterdir()}
    assert len(logs) == 2, sorted(logs)
    good_run = next(name for name, text in logs.items() if "good.json" in text)
    bad_run = next(name for name, text in logs.items() if "bad.json" in text)
    assert "[CLM-101]" in logs[bad_run] and "[CLM-101]" not in logs[good_run]
    assert f"good.json: exit 0, run {good_run}" in caplog.text
    assert f"bad.json: exit 2, run {bad_run}" in caplog.text
