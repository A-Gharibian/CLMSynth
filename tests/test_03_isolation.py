"""Category 3: State isolation.

Components must keep their state to themselves. Every test here is a
single-threaded, deterministic assertion about *ownership of state*.

What is checked, in order:

  RNG ownership        `fabricated_generator` reproduces a seed exactly even
                       after `np.random.seed()`/`rand()` have moved the global
                       stream, reproduces it again after an interleaved call
                       with a different seed, and genuinely differs between
                       seeds (so the first two are not vacuous).
  config ownership     `run_pipeline` does not mutate the config dict it is
                       handed, it pops keys off the per-source suite block,
                       so without a defensive copy a second call with the same
                       object resolves zero datasets. Asserted both directly
                       and through that consequence.
  run-folder ownership `build_run_dir` creates the name it returns, and never
                       returns the same name twice. Finding N2 (check-then-act)
                       was closed in 0.6.3; both halves are now guarantees
                       rather than one guarantee and one characterization.
  module-level state   Every registry (`CODES`, `FETCHERS`, `SOURCE_METADATA`,
                       `_METRIC_FUNCS`, ...) is byte-identical after a real run.
  no async             The package contains no `async def`, `await` or
                       `asyncio`, which every claim above assumes.

"""

from pathlib import Path

import numpy as np
import pytest

from clmsynth import fabricated_generator
from clmsynth.main import build_run_dir, run_pipeline


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
    return fabricated_generator.generate_synthetic_data(n_samples=200, output_file=None, seed=seed)


def test_generator_output_survives_global_rng_mutation():
    """Finding F5: a fixed seed must reproduce even if the global RNG moved.

    `np.random.seed()` / `np.random.rand()` mutate process-global state. A
    generator built on `np.random.*` rather than its own `default_rng(seed)`
    would silently produce different data depending on what else in the process
    had drawn from numpy first, and the caller would have no way to tell,
    because the seed they passed did not change.
    """
    before = _fabricate(42)

    np.random.seed(999)
    np.random.rand(10_000)

    after = _fabricate(42)
    assert before["Feature_1"].equals(after["Feature_1"])
    assert before["Feature_6"].equals(after["Feature_6"])


def test_interleaved_seeds_do_not_contaminate_each_other():
    """A, B, A: the third call must reproduce the first exactly.
    Catches state carried *between* calls rather than in from outside. a
    generator instance reused across calls, or a module-level stream advanced by
    whatever ran last.
    """
    first = _fabricate(42)
    _fabricate(7)
    third = _fabricate(42)
    assert first["Feature_1"].equals(third["Feature_1"])


def test_different_seeds_actually_differ():
    """Guards the two tests above from being vacuous.

    If the generator ignored its seed entirely and returned constant data, both
    isolation tests would pass perfectly. This is what makes them mean
    something.
    """
    assert not _fabricate(42)["Feature_1"].equals(_fabricate(7)["Feature_1"])


# ---------------------------------------------------------------------------
# run_pipeline must not mutate the config it is handed
# ---------------------------------------------------------------------------

def test_run_pipeline_does_not_mutate_the_callers_config(tmp_path):
    """The suite block is consumed by popping keys, so it must be copied first.
    `run_pipeline` pops `batteries`, `datasets` off the per-source
    suite block as it resolves them. Without the defensive copy those keys are
    gone from the caller's dict afterward, so a second call with the same
    config, a caller looping over sources, or any library user reusing a
    parsed config, would silently resolve zero datasets.
    """
    # Asserted behaviourally. This was previously checked by reading the function
    # source with `inspect.getsource` and searching for the literal text
    # `_suite", {}).copy()`, which would survive the defense being removed and
    # reimplemented differently, and would break on a reformatting that changed
    # the quote style.
    config = {
        "global_settings": {"data_source": "fabricated_data", "output_dir": str(tmp_path)},
        "fabricated_data_suite": {
            "batteries": ["fabricated"], "datasets": ["baseline_4class"], "seed": 42,
        },
        "label_generation": {
            "n_labels": 1, "source_labeling": "labels0", "noise": 0.1, "seed": 42,
            "clm_label": {"num_classes": 4, "matching_mode": "perfect"},
        },
    }
    before = {k: list(v) if isinstance(v, list) else v
              for k, v in config["fabricated_data_suite"].items()}

    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir()
    assert run_pipeline("fabricated_data", config, csv_dir, png_dir, txt_dir) == 1

    assert config["fabricated_data_suite"] == before, "run_pipeline mutated the caller's config"


def test_run_pipeline_is_repeatable_with_the_same_config_object(tmp_path):
    """The consequence of the above, stated directly: two calls, same result.
    This is the failure a caller would actually see, the second run quietly
    processing nothing.
    """
    config = {
        "global_settings": {"data_source": "fabricated_data", "output_dir": str(tmp_path)},
        "fabricated_data_suite": {
            "batteries": ["fabricated"], "datasets": ["baseline_4class"], "seed": 42,
        },
        "label_generation": {
            "n_labels": 1, "source_labeling": "labels0", "noise": 0.1, "seed": 42,
            "clm_label": {"num_classes": 4, "matching_mode": "perfect"},
        },
    }
    counts = []
    for run in ("first", "second"):
        out = tmp_path / run
        dirs = [out / "csv", out / "png", out / "txt"]
        for d in dirs:
            d.mkdir(parents=True)
        counts.append(run_pipeline("fabricated_data", config, *dirs))
    assert counts == [1, 1], f"second run with the same config object processed {counts[1]}"


# ---------------------------------------------------------------------------
# build_run_dir hands out a reservation, not just a name
# ---------------------------------------------------------------------------

def test_build_run_dir_reserves_what_it_returns(tmp_path):
    """Finding N2, closed in 0.6.3: the name and the claim are now one step.

    `build_run_dir` used to pick a free name with `while unique.exists()` while
    the `mkdir` happened in the caller a full call later. Between those two
    moments the name was unclaimed, so a second caller checking in that window
    was told the same name was free, two runs starting in the same second
    against one `output_dir` interleaved their csv/png/txt and config copy into
    one folder.
    """
    path = build_run_dir(tmp_path, "IsolationTest")
    assert path.exists(), "build_run_dir returned a name it had not claimed"
    assert path.is_dir()


def test_build_run_dir_never_hands_out_the_same_name_twice(tmp_path):
    """Repeated calls in the same wall-clock second must diverge.
    All three share a `DDMMYY_Source_HHMMSS` stem, so this is the collision path,
    not the happy one. No caller creates anything here: that is the point --
    reservation is now the function's own job.
    """
    paths = [build_run_dir(tmp_path, "IsolationTest") for _ in range(3)]

    assert len(set(paths)) == 3, f"a name was reused: {paths}"
    assert all(p.is_dir() for p in paths)
    assert len(list(tmp_path.iterdir())) == 3, "a folder was created that was never returned"


# ---------------------------------------------------------------------------
# Structural: the package is synchronous
# ---------------------------------------------------------------------------

def test_module_level_registries_survive_a_run_unchanged():
    """Nothing accumulates in module-level state across a pipeline run.

    The package keeps a dozen module-level containers, `CODES`, `FETCHERS`,
    `SOURCE_METADATA`, `SOURCE_DATASETS`, `_METRIC_FUNCS` and friends. All are
    lookup tables built once at import; two `SOURCE_METADATA` writes exist but
    both run at module level during import, not inside any function.

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

    _fabricate(42)
    from clmsynth.clm_label_engine import generate_clm_labels
    clusters = np.concatenate([np.full(50, k) for k in range(4)])
    generate_clm_labels(clusters, np.random.default_rng(0).normal(size=(200, 2)),
                        {"num_classes": 4, "matching_mode": "perfect"}, seed=1)

    changed = [name for name, obj in watched.items() if obj != before[name]]
    assert not changed, f"module-level state mutated during a run: {changed}"


def test_package_contains_no_async_constructs():
    """The async section was a prose declaration; this makes it checkable.

    Every claim in this module rests on the pipeline being synchronous, one
    thread, one process, a sequential loop. If `async def`, `await` or
    `asyncio` ever appear, that assumption needs revisiting and so does
    everything above.
    """
    import clmsynth

    offenders = []
    for path in sorted(Path(clmsynth.__path__[0]).glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in ("async def", "await ", "import asyncio"):
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, f"package is no longer synchronous: {offenders}"
