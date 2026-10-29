"""Category 6: Coded diagnostics.

`00_contract` covers right input produces right output.
 This covers wrong input produces the right diagnostic.

**This module authors no cases.** Every configuration it runs already exists
in `docs/troubleshooting_catalog/`, written and verified by `CLM_catalog_gen.py`.
The only thing asserted here is that the expected outcome still appears. A new
catalog entry is picked up automatically; nothing needs editing here when a
code is added.

Coverage of the `[CLM-###]` registry is a **union** property across the whole
suite, not this module's exclusive job. A code asserted in `01_logic` or
`02_edge_cases` is covered; it does not need repeating. This module exists so
that no code can fall through the cracks, it is where a diagnostic with no
home elsewhere gets one.

Four bands, two mechanisms, decided by what the pipeline actually does:

  ValueError_1xx           run_pipeline re-raises: a coded config error is
                           equally wrong for every dataset, so it aborts the
                           run rather than being swallowed per dataset.
                           Asserted on the exception and its `.code`.
                           EXCEPTIONS, logged per dataset while the batch
                           continues. See MECHANISM_OVERRIDES below.
                           The codes in main.PER_DATASET_CODES judge the
                           DATASET's K, cluster ids or features rather than
                           the config, so that dataset is skipped.
                           [CLM-120] (no target_metric level fits) is judged
                           against the dataset's cluster sizes, so another
                           dataset may fit the same config. It skips no
                           dataset: only that dataset's label is left out and
                           the dataset is still written, as under 15x.
  InfeasibleAllocation_15x per-dataset skip (another dataset's cluster sizes
                           may satisfy the same rules), logged with its code.
  Warnings_3xx             run succeeds, warning logged with its code.
  KeyError_2xx             run_pipeline re-raises, like 1xx. A missing required
                           key has no default, so no dataset can be labelled.
                           Coded KeyErrors, asserted on the code.

Warnings matter more than exceptions. An exception that stops
firing announces itself: the run crashes differently or produces nothing. A
warning that stops firing is silent, the run completes and writes a CSV that
looks correct while being not what was configured.
Those are the cases where a regression is invisible without this module.
"""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

# The Agg backend is selected once in conftest, before any test module is
# imported, so no GUI toolkit is initialized even if something reaches
# matplotlib despite the plot suppression below.
from clmsynth.byoc_source import fetch_byoc_data
from clmsynth.clm_label_engine import generate_clm_labels
from clmsynth.main import PER_DATASET_CODES, run_pipeline

# How each catalog band surfaces its diagnostic.
RAISES, LOGGED = "raises", "logged"
BANDS = {
    "ValueError_1xx": RAISES,
    "InfeasibleAllocation_15x": LOGGED,
    "Warnings_3xx": LOGGED,
    "KeyError_2xx": RAISES,
}


MECHANISM_OVERRIDES = {**dict.fromkeys(PER_DATASET_CODES, LOGGED), 120: LOGGED}


def _catalog_dir() -> Path:
    """Locate docs/troubleshooting_catalog by walking up from this file.

    Resolved by walking up rather than by a relative path, so the module keeps
    working from any depth under the repository root and pytest can be invoked
    from anywhere.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "docs" / "troubleshooting_catalog"
        if candidate.is_dir():
            return candidate
    raise RuntimeError("docs/troubleshooting_catalog not found above " + __file__)


CATALOG = _catalog_dir()


@pytest.fixture(autouse=True)
def _no_plots(no_plots):
    """Never render a plot in this module.
    """


def _fixtures():
    for band in sorted(BANDS):
        band_dir = CATALOG / band
        if not band_dir.is_dir():
            continue
        for cfg in sorted(band_dir.glob("CLM-*.yaml")):
            code = int(cfg.stem.split("-")[1])
            yield pytest.param(cfg, MECHANISM_OVERRIDES.get(code, BANDS[band]), id=cfg.stem)


FIXTURES = list(_fixtures())


def test_catalog_is_present():
    """Guard the parametrization itself.
    If the catalog folder were renamed or emptied, every parametrized case
    would silently vanish and the suite would report all-green over nothing.
    """
    assert FIXTURES, "no catalog fixtures collected from " + str(CATALOG)
    bands = {cast(Path, p.values[0]).parent.name for p in FIXTURES}
    assert bands == set(BANDS), "a whole catalog band is missing: " + str(bands)


@pytest.mark.parametrize("config_path,mechanism", FIXTURES)
def test_catalog_diagnostic_fires(config_path, mechanism, tmp_path, caplog, yaml):
    """Each catalog config must still produce the diagnostic it documents."""
    code = config_path.stem.split("-")[1]  # "CLM-101" -> "101"
    tag = f"[CLM-{code}]"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = str(config["global_settings"]["data_source"]).lower()

    # Catalog paths are relative to the catalog root, which is what makes a
    # fixture reproducible by hand (`cd docs/troubleshooting_catalog && python -m
    # clmsynth.main ValueError_1xx/CLM-127.yaml`). pytest runs from wherever it
    # was invoked, so both ends are resolved here rather than depending on cwd:
    # output into tmp_path, input against the catalog's own _data/.
    if source == "byoc":
        suite = config.get("byoc_suite", {})
        input_dir = Path(suite.get("input_dir", ""))
        if not input_dir.is_absolute():
            input_dir = CATALOG / input_dir
        assert input_dir.is_dir(), (
            f"catalog byoc fixture points at a missing input folder: {input_dir}. "
            "CLM_catalog_gen.py generates it next to the configs; it must ship with them."
        )
        suite["input_dir"] = str(input_dir)

    config["global_settings"]["output_dir"] = str(tmp_path)
    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir(parents=True, exist_ok=True)

    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        if mechanism == RAISES:
            with pytest.raises((ValueError, KeyError)) as excinfo:
                run_pipeline(source, config, csv_dir, png_dir, txt_dir)
            exc = excinfo.value
            # The code is attached twice by design (message prefix and .code
            # attribute); accept either so neither can drift unnoticed.
            assert tag in str(exc) or str(getattr(exc, "code", "")) == code, (
                f"expected {tag} , got: {exc}"
            )
            return

        run_pipeline(source, config, csv_dir, png_dir, txt_dir)

    assert tag in caplog.text, f"{tag} was not logged. Captured:\n{caplog.text}"


CLUSTERS = np.concatenate([np.full(400, 0), np.full(300, 1), np.full(200, 2), np.full(100, 3)])
COORDS = np.random.default_rng(0).normal(size=(1000, 2))
N = 1000


@pytest.mark.parametrize("cluster_column", [["cluster"], 123], ids=["list", "int"])
def test_byoc_cluster_column_wrong_type_is_rejected(cluster_column, tmp_path):
    """A non-str cluster_column must be rejected, never silently coerced."""
    pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6], "cluster": [0, 0, 1]}).to_csv(
        tmp_path / "tc.csv", index=False
    )
    assert (
        fetch_byoc_data(dataset_name="tc", input_dir=str(tmp_path), cluster_column=cluster_column)
        is None
    )


def test_assignment_matrix_string_cluster_does_not_match_int_id():
    """clusters: ["1"] must not silently match int cluster 1, no type juggling.
    Cluster ids are integers, so it is a config error ([CLM-141]); it was the id
    not found in the dataset ([CLM-105])."""
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(
            CLUSTERS,
            COORDS,
            {
                "num_classes": 2,
                "balance": "balanced",
                "matching_mode": "custom",
                "assignment_matrix": [{"label": 0, "clusters": ["1"], "recall_target": 0.5}],
                "split_rule": "equal",
                "spillover_rule": "proportional_to_marginal",
            },
            seed=1,
        )
    assert "[CLM-141]" in str(excinfo.value)


def test_competing_noise_float_cluster_is_rejected():
    """competing_noise.cluster as 1.5 is no int id, and must be caught as such:
    [CLM-141], a config error; it was the id not found in the dataset ([CLM-119])."""
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(
            CLUSTERS,
            COORDS,
            {
                "num_classes": 4,
                "balance": "unbalanced",
                "proportions": [0.4, 0.3, 0.2, 0.1],
                "matching_mode": "custom",
                "assignment_matrix": [
                    {"label": i, "clusters": [i], "recall_target": 0.5} for i in range(4)
                ],
                "split_rule": "proportional_to_size",
                "spillover_rule": "proportional_to_marginal",
                "competing_noise": [{"cluster": 1.5, "label": 0, "share": 0.5}],
            },
            seed=1,
        )
    assert "[CLM-141]" in str(excinfo.value)


def test_cardinality_guard():
    """[CLM-126]/[CLM-127]: the 64 cap on num_classes and on the dataset's own K.
    A coarse backstop against runaway configs, not a statistical-significance
    boundary.

    Overlaps the catalog's byoc CLM-127 fixture, deliberately. That fixture was
    unrunnable from a source distribution until 0.6.5 shipped the `_data/` folder
    it reads, so this asserted the code directly instead. The catalog is
    self-contained now and `test_catalog_diagnostic_fires` covers it too; keeping
    both means the guard stays asserted without a byoc input file.

    A noninteger is refused under the same code: a quoted "4" or a float used
    to crash uncoded, and `yes`/`true` (bool is an int subclass) passed as 1.
    """
    for m in (0, 65, 20_000, "4", 4.0, True):
        with pytest.raises(ValueError) as excinfo:
            generate_clm_labels(
                CLUSTERS,
                COORDS,
                {
                    "num_classes": m,
                    "balance": "balanced",
                    "matching_mode": "random",
                },
                seed=1,
            )
        assert "[CLM-126]" in str(excinfo.value), f"num_classes={m!r}"

    # 64 is the inclusive boundary and must still be accepted.
    generate_clm_labels(
        CLUSTERS,
        COORDS,
        {
            "num_classes": 64,
            "balance": "balanced",
            "matching_mode": "random",
        },
        seed=1,
    )

    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(
            np.arange(100),
            np.zeros((100, 1)),
            {
                "num_classes": 2,
                "balance": "balanced",
                "matching_mode": "random",
            },
            seed=1,
        )
    assert "[CLM-127]" in str(excinfo.value)


# The base payload the render-time cases below mutate.
#
# `test_smoke.py` carries a MINIMAL_PAYLOAD with the same contents: the gate must
# import nothing from another test module, and diagnostics needs a payload it is
# free to mutate per case.
MINIMAL_PAYLOAD = {
    "data_source": "fabricated_data",
    "batteries": ["fabricated"],
    "datasets": ["baseline_4class"],
    "source_seed": 42,
    "n_labels": 1,
    "source_labeling": "labels0",
    "label_seed": 42,
    "num_classes": 4,
    "proportions": [0.25, 0.25, 0.25, 0.25],
    "balance": "unbalanced",
    "skew_rule": "geometric",
    "matching_mode": "custom",
    "assignment_matrix": [{"clusters": [i], "label": i, "recall_target": 0.8} for i in range(4)],
    "split_rule": "proportional_to_size",
    "spillover_rule": "proportional_to_marginal",
    "centroid_enabled": True,
    "centroid_profile": "linear",
    "centroid_favors": "core",
}


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"matching_mode": "random", "target_metric": {"type": "mcc", "value": 0.5}},
         "[CLM-114]"),
        ({"centroid_enabled": "false"}, "[CLM-141]"),
    ],
    ids=["random+target", "quoted-false"],
)
@pytest.mark.usefixtures("yaml")
def test_generate_config_writes_nothing_the_engine_refuses(overrides, expected, tmp_path):
    """The rendered clm_label block is checked with the engine's own checks before
    it is written, and a refusal is raised with the engine's code. Values are
    written as given, so a quoted "false" reaches the check as a string."""
    from clmsynth.generate_config import generate_base_config

    out = tmp_path / "out.yaml"
    with pytest.raises(ValueError) as excinfo:
        generate_base_config({**MINIMAL_PAYLOAD, **overrides}, output_path=str(out))
    assert expected in str(excinfo.value)
    assert not out.exists()


@pytest.mark.usefixtures("yaml")
def test_generate_config_warns_about_an_unknown_data_source(tmp_path, caplog):
    """data_source is not a clm_label key, so the engine never sees it; the run
    refuses an unknown one, and the renderer says so."""
    from clmsynth.generate_config import generate_base_config

    with caplog.at_level(logging.WARNING):
        generate_base_config({**MINIMAL_PAYLOAD, "data_source": "bogus_source"},
                             output_path=str(tmp_path / "out.yaml"))

    assert "data_source" in caplog.text, f"no render-time warning. Captured:\n{caplog.text}"


@pytest.mark.parametrize(
    "overrides,drop",
    [
        ({"balance": "Balanced"}, ()),
        ({"balance": "balanced"}, ()),
        ({"skew_rule": "bogus_rule"}, ("proportions",)),
    ],
    ids=["unknown-balance", "balanced+proportions", "unknown-skew"],
)
@pytest.mark.usefixtures("yaml")
def test_generate_config_ignores_sizing_keys_under_perfect(overrides, drop, tmp_path, caplog):
    """'perfect' neither reads nor validates balance, proportions or skew_rule, so
    clmsynth-config does not write them: a value the engine refuses under another
    mode passes, and the config does not warn [CLM-302]."""
    from clmsynth.generate_config import generate_base_config

    payload = {k: v for k, v in MINIMAL_PAYLOAD.items() if k not in drop}
    payload.update(overrides, matching_mode="perfect")

    with caplog.at_level(logging.WARNING):
        generate_base_config(payload, output_path=str(tmp_path / "out.yaml"))

    assert not caplog.text.strip(), f"perfect payload produced warnings:\n{caplog.text}"


@pytest.mark.usefixtures("yaml")
def test_generate_config_stays_silent_on_a_valid_payload(tmp_path, caplog):
    """The warnings must be worth reading,
    Without this, every case above would still pass if the renderer warned
    unconditionally, and a config that warns on every run trains the user to
    ignore the output.
    """
    from clmsynth.generate_config import generate_base_config

    with caplog.at_level(logging.WARNING):
        generate_base_config(dict(MINIMAL_PAYLOAD), output_path=str(tmp_path / "out.yaml"))

    assert not caplog.text.strip(), f"valid payload produced warnings:\n{caplog.text}"


def test_byoc_render_round_trips_a_windows_path(tmp_path, yaml):
    """Rendered byoc config parses back unchanged."""
    from clmsynth.generate_config import generate_base_config

    payload = dict(MINIMAL_PAYLOAD)
    payload.update(
        {
            "data_source": "byoc",
            "batteries": ["clinical"],
            "datasets": ["cohort_a"],
            "input_dir": r"C:\data\in",
            "cluster_column": "cluster",
            "standardize": True,
        }
    )
    out = tmp_path / "byoc.yaml"
    generate_base_config(payload, output_path=str(out))

    suite = yaml.safe_load(out.read_text(encoding="utf-8"))["byoc_suite"]
    assert suite["input_dir"] == r"C:\data\in"
    assert suite["cluster_column"] == "cluster"
    assert suite["batteries"] == ["clinical"]
    assert suite["datasets"] == ["cohort_a"]


@pytest.mark.parametrize(
    "key,expected",
    [
        ("proportions", []),
        ("batteries", "all"),
        ("datasets", "all"),
    ],
    ids=["proportions", "batteries", "datasets"],
)
def test_valueless_payload_key_renders_its_default(key, expected, tmp_path, yaml):
    """Null key renders its documented default."""
    from clmsynth.generate_config import generate_base_config

    payload = dict(MINIMAL_PAYLOAD)
    payload[key] = None
    out = tmp_path / "out.yaml"
    generate_base_config(payload, output_path=str(out))

    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    got = (
        loaded["fabricated_data_suite"][key]
        if key != "proportions"
        else loaded["label_generation"]["clm_label"][key]
    )
    assert got == expected


# ---------------------------------------------------------------------------
# A key missing inside a block is coded and names where it is missing.
# ---------------------------------------------------------------------------


def test_assignment_matrix_missing_key_is_coded():
    """[CLM-207] names the row the rule came from."""
    with pytest.raises(KeyError) as excinfo:
        generate_clm_labels(
            CLUSTERS,
            COORDS,
            {
                "num_classes": 2,
                "balance": "balanced",
                "matching_mode": "custom",
                "assignment_matrix": [{"label": 0}],
                "split_rule": "equal",
                "spillover_rule": "proportional_to_marginal",
            },
            seed=1,
        )
    assert "[CLM-207] assignment_matrix row 0" in str(excinfo.value)
