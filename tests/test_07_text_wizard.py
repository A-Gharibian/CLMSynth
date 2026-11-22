"""The text (CLI) wizard and its question schema.

The wizard is a deletable convenience: a config creator, nothing the rest of the
program depends on. Four properties make it safe to keep and safe to throw
away. This module pins the first two; ruff enforces the other two
(TID253 and TID251 in pyproject.toml):

* **Ranges agree with the run.** Each bound of a question is the run's own, read
  from its settings record, except the sides where the wizard is narrower on
  purpose. A wizard bound *wider* than the run's is a config the wizard builds
  happily and the engine rejects, which is the whole reason the wizard carries
  ranges at all.
* **The negative-target guard is wizard-only.** MCC/ARI targets floor at 0.0 in
  the wizard while the engine is left exactly as it is (it takes -1 to 1), because
  a negative target is meaningful but not something the program promises to solve.
* **The wizard's import graph stays light.** Importing it (or the schema) does not
  pull matplotlib, so a future help command built on the schema pays for nothing
  it does not use.
* **Deletability.** No core module imports the wizard.

    python -m pytest tests/test_07_text_wizard.py
    python -m pytest tests/test_07_text_wizard.py -v
"""

import re
from pathlib import Path

import numpy as np
import pytest

import clmsynth._config_gen.config_wizard as wizard
from clmsynth import load_config
from clmsynth._config_gen.questions import SCHEMA
from clmsynth.clm_label_engine import generate_clm_labels
from clmsynth.config_tree import SETTINGS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def geometry(sizes, seed=0):
    """K gaussian blobs of the given sizes -> (cluster_labels, coords)."""
    rng = np.random.default_rng(seed)
    labels, coords = [], []
    for k, n in enumerate(sizes):
        centre = np.zeros(2)
        centre[0] = k * 10.0
        coords.append(rng.normal(centre, 1.0, size=(n, 2)))
        labels.append(np.full(n, k))
    return np.concatenate(labels), np.concatenate(coords)


@pytest.fixture
def stdin(monkeypatch):
    """Queue of predetermined answers for the wizard's `input()` calls.
    Populate it before calling the builders. An empty queue raises `EOFError`."""
    queue: list[str] = []

    def fake_input(prompt=""):
        if not queue:
            raise EOFError(f"wizard asked for input past the canned queue (prompt {prompt!r})")
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return queue


# ---------------------------------------------------------------------------
# Schema integrity
# ---------------------------------------------------------------------------


def test_schema_keys_and_prompts_agree():
    assert SCHEMA, "SCHEMA is empty"
    src = Path(wizard.__file__).read_text(encoding="utf-8")
    asked = set(re.findall(r'ask_from\(\s*["\']([^"\']+)["\']', src))
    # A family of questions asked in one loop over its SCHEMA prefix.
    prefixes = tuple(re.findall(r'key\.startswith\(\s*["\']([^"\']+)["\']', src))
    asked |= {key for key in SCHEMA if key.startswith(prefixes)}
    assert not asked - set(SCHEMA), f"prompts with no schema entry: {sorted(asked - set(SCHEMA))}"
    assert not set(SCHEMA) - asked, f"schema entries no prompt asks: {sorted(set(SCHEMA) - asked)}"


# ---------------------------------------------------------------------------
# The negative-target guard is wizard-only
# ---------------------------------------------------------------------------


def test_target_value_bound_is_the_wizard_only_floor():
    q = SCHEMA["clm_label.target_metric.value"]
    assert (q.lo, q.hi) == (0.0, 1.0), "the MCC/ARI target must floor at 0.0 in the wizard"
    assert SETTINGS["clm_label.target_metric.value"].lo == -1, (
        "the target floor is deliberately wizard-only; the engine is left unchanged"
    )


def test_wizard_rejects_a_negative_target_and_re_asks(stdin):
    """A negative MCC/ARI target is re-asked until it is in [0, 1]."""
    stdin.extend(["-0.5", "1.5", "0.6"])
    assert wizard.ask_from("clm_label.target_metric.value") == pytest.approx(0.6)
    assert not stdin, "expected exactly the two rejects then the accepted value"


def test_wizard_rejects_alpha_of_zero(stdin):
    """dirichlet alpha must be > 0 (exactly 0 divides by zero in the engine); the
    wizard's strict lower bound refuses 0.0 as the engine does."""
    stdin.extend(["0", "-1", "0.5"])
    assert wizard.ask_from("clm_label.skew_params.alpha") == pytest.approx(0.5)
    assert not stdin


def test_wizard_rejects_num_classes_above_the_engine_cap(stdin):
    """M above 64 is re-asked, matching [CLM-126]."""
    stdin.extend(["65", "4"])
    assert wizard.ask_from("clm_label.num_classes") == 4
    assert not stdin


@pytest.mark.parametrize(("key", "bad", "valid"), [
    pytest.param("label_generation.seed", "-1", 0, id="negative-label-seed"),
    pytest.param("registry_suite.seed", "-1", 0, id="negative-source-seed"),
    pytest.param("label_generation.n_labels", "10001", 10000, id="too-many-labels"),
])
def test_an_answer_the_run_refuses_is_asked_again(stdin, key, bad, valid):
    """A seed below 0 and `n_labels` above 10 000 went into a config the command
    then refused. The answer at the bound itself is taken."""
    stdin.extend([bad, str(valid)])
    assert wizard.ask_from(key) == valid
    assert not stdin, "the replacement answer must have been read"


def test_several_ids_where_one_cluster_is_asked_are_asked_again(stdin):
    """`1, 2` kept 1 without a word."""
    stdin.extend(["1, 2", "2"])
    assert wizard.ask_from("clm_label.single_match.cluster") == 2
    assert not stdin, "the replacement answer must have been read"


# The sides where the wizard asks within a narrower range than the run takes, on purpose.
NARROWER_ON_PURPOSE = {
    "label_generation.n_labels": "lo",
    "clm_label.num_classes": "lo",
    "clm_label.target_metric.value": "lo",
    "clm_label.target_metric.tolerance": "hi",
    "clm_label.skew_params.ratio": "hi",
}


def test_a_question_asks_within_the_run_s_own_bounds():
    """Each bound of a question is the run's, from its settings record, but for the
    sides named above, where the wizard is narrower and never wider."""
    assert set(NARROWER_ON_PURPOSE) <= set(SCHEMA)
    for key, q in SCHEMA.items():
        record = SETTINGS.get(key)
        if record is None or (record.lo is None and record.hi is None):
            continue
        narrower = NARROWER_ON_PURPOSE.get(key)
        if narrower == "lo":
            assert q.lo is not None and (record.lo is None or q.lo > record.lo), key
        else:
            assert (q.lo, q.lo_strict) == (record.lo, record.lo_strict), key
        if narrower == "hi":
            assert q.hi is not None and (record.hi is None or q.hi < record.hi), key
        else:
            assert q.hi == record.hi, key


def test_proportions_prompt_re_asks_on_empty_input(stdin):
    """An empty list answer must re-ask."""
    stdin.extend(["", "  ", "0.25, 0.25, 0.25, 0.25"])
    assert wizard.ask_from("clm_label.proportions") == pytest.approx([0.25] * 4)
    assert not stdin


# ---------------------------------------------------------------------------
# End to end: canned answers build a config the engine accepts
# ---------------------------------------------------------------------------


def test_canned_answers_build_an_engine_valid_config(stdin, yaml):
    """Drive the builders with canned answers exactly as main() does, then run the
    assembled clm_label against matching geometry and round-trip it through YAML.
    A regression pin on the refactor: the schema-driven wizard produces the same
    kind of config the inline one did."""
    stdin.extend(
        [
            # build_source
            "byoc",  # data source
            "",  # output_dir -> OUTPUT
            # _byoc_suite
            "",  # input_dir -> INPUT
            "my_clusters",  # CSV file name(s)
            "",  # cluster_column -> cluster
            "",  # carry any column through? -> no
            "",  # standardize -> no
            # build_label_generation
            "",  # n_labels -> 1
            "",  # label seed -> 42
            # build_clm
            "",  # num_classes -> 3
            "",  # matching_mode -> custom
            "",  # balance -> balanced (returns before skew)
            "",  # aim for a target? -> no
            "",  # how many rules -> 1
            "",  # rule 1 label -> 0
            "0",  # rule 1 clusters
            "",  # recall_target -> 0.8
            "",  # split_rule -> proportional_to_size
            "",  # spillover_rule -> proportional_to_marginal
            "",  # competing noise? -> no
            "",  # centroid dependence? -> no
            "",  # follow a feature? -> no
        ]
    )

    source, gs, suite, known_k = wizard.build_source()
    lg = wizard.build_label_generation(source)
    lg["clm_label"] = wizard.build_clm(known_k)
    config = {"global_settings": gs, f"{source}_suite": suite, "label_generation": lg}

    assert not stdin, "the canned queue and the wizard's questions must line up exactly"
    assert source == "byoc"
    clm = config["label_generation"]["clm_label"]
    assert clm["num_classes"] == 3 and clm["matching_mode"] == "custom"

    c, X = geometry([200, 200, 200])
    out = np.asarray(generate_clm_labels(c, X, clm, rng=42))
    assert len(out) == len(c)
    assert set(np.unique(out)).issubset({0, 1, 2})

    # survives serialization: the wizard writes YAML, so what it built must reload.
    reloaded = yaml.safe_load(yaml.dump(config, sort_keys=False))["label_generation"]["clm_label"]
    out2 = np.asarray(generate_clm_labels(c, X, reloaded, rng=42))
    assert np.array_equal(out, out2)


# ---------------------------------------------------------------------------
# Saving a completed interview
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_save_creates_a_missing_parent_directory(tmp_path, yaml, suffix):
    """Answering "configs/run1.yaml" must not fail. The file is written in the
    format its suffix names, as clmsynth reads it."""
    target = tmp_path / "configs" / f"run1.{suffix}"

    wizard._save_config({"global_settings": {"data_source": "byoc"}}, str(target))

    assert load_config(str(target)) == {"global_settings": {"data_source": "byoc"}}


@pytest.mark.parametrize("answer,fmt,saved", [
    ("run1", "json", "run1.json"),
    ("run1.yaml", "json", "run1.json"),
    ("run1.json", "yaml", "run1.yaml"),
    ("run1.yml", "yaml", "run1.yml"),
    ("run1.v2", "yaml", "run1.v2.yaml"),
])
def test_the_chosen_format_sets_the_suffix(answer, fmt, saved):
    """One format per file: never run1.yaml.json."""
    assert wizard._config_path(answer, fmt) == saved


@pytest.mark.usefixtures("yaml")
def test_a_failed_save_reports_and_prints_the_answers(tmp_path, capsys):
    """A save failure must not discard the interview."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    config = {"label_generation": {"clm_label": {"num_classes": 3}}}

    with pytest.raises(SystemExit):
        wizard._save_config(config, str(blocker / "nested" / "run.yaml"))

    printed = capsys.readouterr().out
    assert "Could not save" in printed
    assert "num_classes: 3" in printed, "the answers must survive the failure"


def test_a_failing_resolve_does_not_traceback(monkeypatch):
    """resolve() must not decide the interview's fate."""
    assert Path(wizard._resolved_or_raw("OUTPUT")).is_absolute()

    def boom(self):
        raise OSError("name too long")

    monkeypatch.setattr(Path, "resolve", boom)
    assert wizard._resolved_or_raw("configs/run1.yaml") == "configs/run1.yaml"


def test_the_wizard_asks_again_for_nan_or_inf(stdin):
    """float() accepts 'nan' and 'inf', and NaN fails no bound check, so the
    wizard used to write them into a [0, 1] question itself."""
    stdin.extend(["nan", "inf", "0.5"])
    assert wizard.ask_float("share", lo=0.0, hi=1.0) == 0.5
    stdin.extend(["0.5, nan", "0.4, 0.6"])
    assert wizard.ask_floats("proportions") == [0.4, 0.6]
    assert not stdin


@pytest.mark.parametrize(("ask", "then"), [
    (lambda: wizard.ask_choice("Pick", ("a", "b")), []),
    (lambda: wizard.ask_selection("Pick", ["a", "b"]), []),
    (lambda: wizard._registry_suite("fabricated_data"), ["", ""]),
], ids=["a-choice", "a-selection", "a-group"])
def test_a_character_isdigit_accepts_is_asked_again(stdin, ask, then):
    """A superscript 2 passes isdigit and int() refuses it, so one keystroke ended
    the wizard with a traceback."""
    stdin.extend(["²", "1", *then])
    ask()
    assert not stdin


def test_byoc_answers_every_run_refuses_are_asked_again(stdin):
    """A stem with a folder part and the cluster column as a tag: each question
    says it must not be, and the answer stood. No seed is asked: byoc reads a file."""
    stdin.extend([
        "",  # input_dir -> INPUT
        "sub/data",  # CSV file name(s), asked again
        "data",
        "",  # cluster_column -> cluster
        "yes",  # carry a column through
        "cluster",  # the cluster column, asked again
        "site",
        "",  # standardize -> no
    ])
    suite = wizard._byoc_suite()
    assert not stdin
    assert suite["datasets"] == ["data"]
    assert suite["tag_columns"] == ["site"]
    assert "seed" not in suite


def test_a_seed_is_asked_only_of_a_source_that_reads_it(stdin):
    """clustbench downloads fixed data, and its seed question promised a result
    that nothing computed."""
    stdin.extend(["", ""])  # the group, its datasets
    assert "seed" not in wizard._registry_suite("clustbench")
    stdin.extend(["", "", ""])  # the group, its datasets, the seed
    assert wizard._registry_suite("fabricated_data")["seed"] == 42
    assert not stdin
