"""Category 6: Coded diagnostics.

`00_contract` covers right input produces right output.
 This covers wrong input produces the right diagnostic.

**This module authors no cases.** Every configuration it runs already exists
in `docs/troubleshooting_catalog/`, written and verified by `make_catalog.py`.
The only thing asserted here is that the expected outcome still appears. A new
catalog entry is picked up automatically; nothing needs editing here when a
code is added.

Coverage of the `[CLM-###]` registry is a **union** property across the whole
suite, not this module's exclusive job. A code asserted in `01_logic` or
`02_edge_cases` is covered; it does not need repeating. This module exists so
that no code can fall through the cracks, it is where a diagnostic with no
home elsewhere gets one.

Four bands, three mechanisms, decided by what the pipeline actually does:

  ValueError_1xx           run_pipeline re-raises: a coded config error is
                           equally wrong for every dataset, so it aborts the
                           run rather than being swallowed per dataset.
                           Asserted on the exception and its `.code`.
                           TWO EXCEPTIONS, [CLM-104] and [CLM-105]: they judge
                           the DATASET's ids rather than the config, so since
                           0.6.3 they are logged per dataset and the batch
                           continues. See MECHANISM_OVERRIDES below.
  InfeasibleAllocation_15x per-dataset skip (another dataset's cluster sizes
                           may satisfy the same rules), logged with its code.
  Warnings_3xx             run succeeds, warning logged with its code.
  KeyError_2xx             documentation-only band. Missing required config
                           keys are deliberately left as raw KeyErrors with no
                           [CLM-###] code, so these are asserted on the
                           documented behavior instead, label generation is
                           skipped and the run continues.

Warnings matter more than exceptions. An exception that stops
firing announces itself: the run crashes differently or produces nothing. A
warning that stops firing is silent, the run completes and writes a CSV that
looks correct while being not what was configured ([CLM-304] and
[CLM-308] are exactly that shape).
Those are the cases where a regression is invisible without this module.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# The Agg backend is selected once in conftest, before any test module is
# imported, so no GUI toolkit is initialized even if something reaches
# matplotlib despite the plot suppression below.
import clmsynth.main
from clmsynth.byoc_source import fetch_byoc_data
from clmsynth.clm_errors import CODES
from clmsynth.clm_label_engine import generate_clm_labels, resolve_label_counts
from clmsynth.main import run_pipeline

# How each catalog band surfaces its diagnostic.
RAISES, LOGGED, UNCODED_SKIP = "raises", "logged", "uncoded_skip"
BANDS = {
    "ValueError_1xx": RAISES,
    "InfeasibleAllocation_15x": LOGGED,
    "Warnings_3xx": LOGGED,
    "KeyError_2xx": UNCODED_SKIP,
}


MECHANISM_OVERRIDES = {104: LOGGED, 105: LOGGED}


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
DOCS = CATALOG.parent

# Shipped .tex documents carry a machine-readable banner in their first few
# lines, in the style of the `% !TeX program = ...` magic comments editors
# already understand:
#
#     % !CLMSynth-doc = configuration-troubleshooting
#     % !CLMSynth-version = 0.6.3
#
# Locating a document by that marker rather than by filename means the file can
# be renamed without breaking anything here.
DOC_BANNER = re.compile(r"^%\s*!CLMSynth-(doc|version)\s*=\s*(\S+)", re.M)


def find_document(kind: str):
    """Return (path, declared_version) for the one .tex declaring `kind`.

    Fails on zero matches and on more than one. Ambiguity matters here:
    keeping an older copy of a document beside the current one is a natural
    thing to do, and silently picking whichever sorted first would be a worse
    failure than the filename coupling this replaces.
    """
    hits = []
    for path in sorted(DOCS.rglob("*.tex")):
        fields = dict(DOC_BANNER.findall(
            path.read_text(encoding="utf-8", errors="replace")[:2000]))
        if fields.get("doc") == kind:
            hits.append((path, fields.get("version")))

    assert hits, (
        f"no .tex under {DOCS} declares '% !CLMSynth-doc = {kind}'. Every shipped "
        "document needs the banner; see any of them for the format."
    )
    assert len(hits) == 1, (
        f"{len(hits)} documents claim to be '{kind}': "
        f"{[p.name for p, _ in hits]}. Exactly one may carry a given marker"
        "reference or archived copies belong outside this tree, or must carry no banner."
    )
    return hits[0]

# Registry codes with no catalog fixture, and why. Deliberately a mapping rather
# than a bare set: an entry without a stated reason is how a genuine gap gets
# waved through. Empty is the goal; every entry is a debt.
NO_CATALOG_FIXTURE: dict = {}


@pytest.fixture(autouse=True)
def _no_plots(no_plots):
    """Never render a plot in this module.

    No [CLM-###] diagnostic depends on plotting: every code is decided during
    label generation, before run_pipeline reaches its plot calls. Rendering
    would only add matplotlib's cost to every case that runs to completion, and
    leave PNGs behind for nothing. Implementation is in conftest.
    """


def _catalog_codes() -> set:
    return {int(p.stem.split("-")[1]) for p in CATALOG.rglob("CLM-*.yaml")}


def test_every_registry_code_has_a_catalog_fixture():
    """Coverage is asserted against the registry, not against a stored checksum.

    This replaces the CATALOG.sha256 fingerprint that used to guard this file.
    The fingerprint answered "have the bytes changed", which git already answers,
    and it had to be updated by hand on every deliberate catalog edit. It also
    guarded less than it appeared to: a config edited so that it stops triggering
    its code already fails `test_catalog_diagnostic_fires`, because the expected
    code is parsed from the FILENAME and cannot drift from it.

    What the fingerprint really caught was a *deleted* fixture silently shrinking
    the parametrization. This catches that and more: a code added to the registry
    with no fixture written for it fails here immediately, which is exactly how
    [CLM-310] and [CLM-131] went missing unnoticed.
    """
    missing = set(CODES) - _catalog_codes() - set(NO_CATALOG_FIXTURE)
    assert not missing, (
        f"registry codes with no catalog fixture: {sorted(missing)}. "
        "Add one with make_catalog.py, or record it in NO_CATALOG_FIXTURE with "
        "the reason it cannot have one."
    )


def test_exemptions_are_real_codes_and_still_needed():
    """An exemption must name a code that exists and genuinely lacks a fixture.

    Without this, an exemption outlives the problem it documents: the fixture
    gets written, nobody removes the entry, and the next genuinely missing code
    slips through under a stale reason.
    """
    for code, reason in NO_CATALOG_FIXTURE.items():
        assert code in CODES, f"exemption for {code}, which is not a registry code"
        assert reason and isinstance(reason, str), f"exemption for {code} has no reason"
        assert code not in _catalog_codes(), (
            f"[CLM-{code}] now HAS a catalog fixture; remove it from "
            "NO_CATALOG_FIXTURE rather than leaving a stale exemption."
        )


def test_registry_reference_and_catalog_agree():
    """The three artifacts a user meets must describe the same set of codes.

    The registry is what the engine raises, the troubleshooting reference is what
    the manual documents, and the catalog is what a user can run. A code present
    in one and absent from another is a code somebody will hit and find no help
    for. Asserted one-directionally.
    every registry code appears in the other
    two, because the reference also writes band ranges in prose that are not
    codes at all.
    """
    reference, _ = find_document("configuration-troubleshooting")
    documented = {int(m) for m in re.findall(r"\\[a-z]*code\{(\d+)\}",
                                             reference.read_text(encoding="utf-8"))}
    undocumented = set(CODES) - documented
    assert not undocumented, (
        f"registry codes missing from {reference.name}: {sorted(undocumented)}"
    )


@pytest.mark.parametrize("kind", ["manual", "configuration-troubleshooting"])
def test_shipped_documents_declare_the_current_version(kind):
    """Documentation cannot silently fall behind a release.
    """
    path, declared = find_document(kind)
    assert declared == clmsynth.__version__, (
        f"{path.name} declares version {declared}, package is "
        f"{clmsynth.__version__}. Update the '% !CLMSynth-version' banner when the "
        "document is revised for a release, or revise the document, if it still "
        "describes older behaviour."
    )


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
    bands = {p.values[1] for p in FIXTURES}
    assert bands == {RAISES, LOGGED, UNCODED_SKIP}, "a whole catalog band is missing: " + str(bands)


@pytest.mark.parametrize("config_path,mechanism", FIXTURES)
def test_catalog_diagnostic_fires(config_path, mechanism, tmp_path, caplog):
    """Each catalog config must still produce the diagnostic it documents."""
    code = config_path.stem.split("-")[1]          # "CLM-101" -> "101"
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
            "make_catalog.py generates it next to the configs; it must ship with them."
        )
        suite["input_dir"] = str(input_dir)

    config["global_settings"]["output_dir"] = str(tmp_path)
    csv_dir, png_dir, txt_dir = tmp_path / "csv", tmp_path / "png", tmp_path / "txt"
    for d in (csv_dir, png_dir, txt_dir):
        d.mkdir(parents=True, exist_ok=True)

    with caplog.at_level(logging.DEBUG, logger="clmsynth"):
        if mechanism == RAISES:
            with pytest.raises(ValueError) as excinfo:
                run_pipeline(source, config, csv_dir, png_dir, txt_dir)
            exc = excinfo.value
            # The code is attached twice by design (message prefix and .code
            # attribute); accept either so neither can drift unnoticed.
            assert tag in str(exc) or str(getattr(exc, "code", "")) == code, \
                f"expected {tag} , got: {exc}"
            return

        run_pipeline(source, config, csv_dir, png_dir, txt_dir)

    if mechanism == LOGGED:
        assert tag in caplog.text, f"{tag} was not logged. Captured:\n{caplog.text}"
    else:  # UNCODED_SKIP, the 2xx band carries no code, by design
        assert "Skipping label generation" in caplog.text, \
            f"expected a skipped labeling for the {code} KeyError band. Captured:\n{caplog.text}"


CLUSTERS = np.concatenate([np.full(400, 0), np.full(300, 1), np.full(200, 2), np.full(100, 3)])
COORDS = np.random.default_rng(0).normal(size=(1000, 2))
N = 1000


@pytest.mark.parametrize("cluster_column", [["cluster"], 123], ids=["list", "int"])
def test_byoc_cluster_column_wrong_type_is_rejected(cluster_column, tmp_path):
    """A non-str cluster_column must be rejected, never silently coerced."""
    pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6], "cluster": ["A", "A", "B"]}) \
        .to_csv(tmp_path / "tc.csv", index=False)
    assert fetch_byoc_data(dataset_name="tc", input_dir=str(tmp_path),
                           cluster_column=cluster_column) is None


def test_single_match_cluster_as_list_is_unknown_id():
    """A list where a scalar cluster id belongs is an unknown id, not a set."""
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, {
            "num_classes": 2, "balance": "balanced", "matching_mode": "single",
            "single_match": {"cluster": [0, 1], "label": 0},
        }, seed=1)
    assert "[CLM-105]" in str(excinfo.value)


def test_assignment_matrix_string_cluster_does_not_match_int_id():
    """clusters: ["1"] must not silently match int cluster 1, no type juggling."""
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, {
            "num_classes": 2, "balance": "balanced", "matching_mode": "custom",
            "assignment_matrix": [{"label": 0, "clusters": ["1"], "recall_target": 0.5}],
            "split_rule": "equal", "spillover_rule": "proportional_to_marginal",
        }, seed=1)
    assert "[CLM-105]" in str(excinfo.value)


def test_competing_noise_float_cluster_is_rejected():
    """competing_noise.cluster as 1.5 is no int id, and must be caught as such."""
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(CLUSTERS, COORDS, {
            "num_classes": 4, "balance": "unbalanced", "proportions": [0.4, 0.3, 0.2, 0.1],
            "matching_mode": "custom",
            "assignment_matrix": [{"label": i, "clusters": [i], "recall_target": 0.5} for i in range(4)],
            "split_rule": "proportional_to_size", "spillover_rule": "proportional_to_marginal",
            "competing_noise": [{"cluster": 1.5, "label": 0, "share": 0.5}],
        }, seed=1)
    assert "[CLM-119]" in str(excinfo.value)


def test_cardinality_guard():
    """[CLM-126]/[CLM-127]: the 64 cap on num_classes and on the dataset's own K.
    A coarse backstop against runaway configs, not a statistical-significance
    boundary.

    Overlaps the catalog's byoc CLM-127 fixture, deliberately. That fixture was
    unrunnable from a source distribution until 0.6.5 shipped the `_data/` folder
    it reads, so this asserted the code directly instead. The catalog is
    self-contained now and `test_catalog_diagnostic_fires` covers it too; keeping
    both means the guard stays asserted without a byoc input file.
    """
    for m in (0, 65, 20_000):
        with pytest.raises(ValueError) as excinfo:
            generate_clm_labels(CLUSTERS, COORDS, {
                "num_classes": m, "balance": "balanced", "matching_mode": "random",
            }, seed=1)
        assert "[CLM-126]" in str(excinfo.value), f"num_classes={m}"

    # 64 is the inclusive boundary and must still be accepted.
    generate_clm_labels(CLUSTERS, COORDS, {
        "num_classes": 64, "balance": "balanced", "matching_mode": "random",
    }, seed=1)

    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(np.arange(100), np.zeros((100, 1)), {
            "num_classes": 2, "balance": "balanced", "matching_mode": "random",
        }, seed=1)
    assert "[CLM-127]" in str(excinfo.value)




# The base payload the render-time cases below mutate via `overrides`/`drop`.
#
# `test_smoke.py` carries a MINIMAL_PAYLOAD with the same contents. The two are
# deliberately independent copies, not one shared fixture: the smoke gate must
# import nothing from another test module, and this one needs a payload it is
# free to mutate per case. Same name on both sides so the relationship is
# visible; a change to the payload schema fails both, so neither can drift into
# testing less than it claims.
MINIMAL_PAYLOAD = {
    "data_source": "fabricated_data",
    "batteries": ["fabricated"], "datasets": ["baseline_4class"], "source_seed": 42,
    "n_labels": 1, "source_labeling": "labels0", "label_seed": 42,
    "num_classes": 4, "proportions": [0.25, 0.25, 0.25, 0.25],
    "balance": "unbalanced", "skew_rule": "geometric",
    "matching_mode": "custom",
    "assignment_matrix": [{"clusters": [i], "label": i, "recall_target": 0.8}
                          for i in range(4)],
    "split_rule": "proportional_to_size",
    "spillover_rule": "proportional_to_marginal",
    "centroid_enabled": True, "centroid_profile": "linear", "centroid_favors": "core",
}

@pytest.mark.parametrize("overrides,drop,expected", [
    ({"matching_mode": "perfect", "target_metric": {"type": "mcc", "value": 0.5}},
     (), "[CLM-111]"),
    ({"matching_mode": "random", "target_metric": {"type": "mcc", "value": 0.5}},
     (), "[CLM-114]"),
    ({"matching_mode": "random",
      "competing_noise": [{"cluster": 0, "label": 1, "share": 0.5}]}, (), "[CLM-115]"),
    ({"matching_mode": "custom",
      "target_metric": {"type": "mcc", "value": 0.5, "scope": "pair"}}, (), "[CLM-124]"),
    # skew_rule is only consulted when it is actually live, so the warning is
    # correctly silent unless proportions are absent.
    ({"skew_rule": "bogus_rule"}, ("proportions",), "skew_rule"),
    ({"balance": "balanced"}, (), "ignores explicit proportions"),
    ({"data_source": "bogus_source"}, (), "data_source"),
], ids=["perfect+target", "random+target", "random+competing", "pair+custom",
        "unknown-skew", "balanced+proportions", "unknown-source"])
def test_generate_config_warns_about_configs_the_engine_will_reject(
        overrides, drop, expected, tmp_path, caplog):
    """A bad payload should be caught at render time, not at run time.

    The renderer cannot refuse , its job is to produce the YAML the user asked
    for, but it can say the engine will reject it, and it names the code so
    the message and the troubleshooting reference line up.

    `drop` removes keys from the base payload, because several of these warnings
    fire only when the setting they concern is actually in use.
    """
    from clmsynth.generate_config import generate_base_config

    payload = {k: v for k, v in MINIMAL_PAYLOAD.items() if k not in drop}
    payload.update(overrides)

    with caplog.at_level(logging.WARNING):
        generate_base_config(payload, output_path=str(tmp_path / "out.yaml"))

    assert expected in caplog.text, f"no render-time warning. Captured:\n{caplog.text}"


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


# ---------------------------------------------------------------------------
# Characterisation: currently uncoded, and on the roadmap to gain a code
# (ROADMAP item 2, "uncoded errors to give [CLM-###] diagnostics").
# ---------------------------------------------------------------------------

def test_num_classes_as_string_is_still_uncoded():
    """`num_classes: "4"` a plausible YAML quoting slip has no config check."""
    with pytest.raises(TypeError):
        generate_clm_labels(CLUSTERS, COORDS, {
            "num_classes": "4", "balance": "balanced", "matching_mode": "random",
        }, seed=1)


def test_proportions_as_dict_is_still_uncoded():
    """sum() over a dict iterates its KEYS, so a dict fails on string addition."""
    with pytest.raises(TypeError):
        resolve_label_counts({"num_classes": 2, "balance": "unbalanced",
                              "proportions": {"a": 0.5, "b": 0.5}}, N, np.random.default_rng(0))


def test_assignment_matrix_missing_key_is_a_bare_error():
    """The 2xx band: missing required keys stay raw KeyErrors, uncoded by design."""
    with pytest.raises(KeyError):
        generate_clm_labels(CLUSTERS, COORDS, {
            "num_classes": 2, "balance": "balanced", "matching_mode": "custom",
            "assignment_matrix": [{"label": 0}],
            "split_rule": "equal", "spillover_rule": "proportional_to_marginal",
        }, seed=1)
