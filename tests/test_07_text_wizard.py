"""The text (CLI) wizard and its question schema.

The wizard is a deletable convenience: a config creator, nothing the rest of the
program depends on. This module pins the four properties that make it safe to
keep and safe to throw away:

* **Ranges agree with the engine.** Every bound the schema marks as engine-owned
  is probed just outside; the engine must refuse it. A wizard bound *wider* than
  the engine's is a config the wizard builds happily and the engine rejects,
  which is the whole reason the wizard carries ranges at all.
* **The negative-target guard is wizard-only.** MCC/ARI targets floor at 0.0 in
  the wizard while the engine is left exactly as it is (no engine bound), because
  a negative target is meaningful but not something the program promises to solve.
* **The wizard's import graph stays light.** Importing it (or the schema) pulls
  neither matplotlib nor seaborn, so a future help command built on the schema
  pays for nothing it does not use.
* **Deletability.** No core module imports the wizard.

    python -m pytest tests/test_07_text_wizard.py
    python -m pytest tests/test_07_text_wizard.py -v
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import clmsynth
import clmsynth.config_wizard as wizard
from clmsynth.clm_label_engine import generate_clm_labels
from clmsynth.questions import SCHEMA

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


def live_clm(skew_rule, skew_params):
    """A clm_label config in which skew_params are actually consumed: unbalanced,
    no explicit proportions, and the matching valid custom bijection so
    generate_clm_labels reaches _validate_skew_cfg."""
    return {
        "num_classes": 4, "balance": "unbalanced", "matching_mode": "custom",
        "assignment_matrix": [{"clusters": [i], "label": i, "recall_target": 0.2}
                              for i in range(4)],
        "split_rule": "proportional_to_size",
        "spillover_rule": "proportional_to_marginal",
        "skew_rule": skew_rule, "skew_params": skew_params,
        "centroid_dependence": {"enabled": False},
    }


# How each bounded question goes live.
ENGINE_BOUNDED = {
    "clm_label.skew_params.ratio": ("geometric", "[CLM-131]"),
    "clm_label.skew_params.dominant_index": ("dominant_minority", "[CLM-131]"),
    "clm_label.skew_params.dominant_share": ("dominant_minority", "[CLM-131]"),
    "clm_label.skew_params.alpha": ("dirichlet", "[CLM-131]"),
    "clm_label.num_classes": (None, "[CLM-126]"),
    "clm_label.concentrated_labels": (None, "[CLM-128]"),
}
# Extra clm keys a bounded value needs before the engine consults it.
LIVE_EXTRAS = {
    "clm_label.concentrated_labels": {"spillover_rule": "concentrated"},
}
VALID_PARAMS = {
    "geometric": {"ratio": 0.5},
    "dominant_minority": {"dominant_index": 0, "dominant_share": 0.6},
    "dirichlet": {"alpha": 1.0},
}


@pytest.fixture
def stdin(monkeypatch):
    """Queue of canned answers for the wizard's input() calls. Append to it, then
    call the builders; an empty queue mid-run fails loudly rather than blocking."""
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
        assert not asked - set(SCHEMA), f"prompts with no schema entry: {sorted(asked - set(SCHEMA))}"
        assert not set(SCHEMA) - asked, f"schema entries no prompt asks: {sorted(set(SCHEMA) - asked)}"

# ---------------------------------------------------------------------------
# The agreement property: a value outside an engine bound is refused by the engine
# ---------------------------------------------------------------------------

def _bad_values(q):
    """Values just outside q's engine bound(s), which the engine must reject."""
    integral = q.kind in ("int", "int_list")
    vals = []
    if q.engine_min is not None:
        vals.append(int(q.engine_min) - 1 if integral else q.engine_min - 0.5)
        if q.engine_min_strict:
            vals.append(float(q.engine_min))          # the strict boundary itself
    if q.engine_max is not None:
        vals.append(int(q.engine_max) + 1 if integral else q.engine_max + 0.5)
    return [[v] for v in vals] if q.kind == "int_list" else vals


_CASES = [(q.key, bad)
          for q in SCHEMA.values()
          if q.engine_min is not None or q.engine_max is not None
          for bad in _bad_values(q)]


def test_every_engine_bounded_question_is_handled():
    """New engine bounds need deliberate handling."""
    bounded = {q.key for q in SCHEMA.values()
               if q.engine_min is not None or q.engine_max is not None}
    assert bounded == set(ENGINE_BOUNDED), \
        f"unhandled engine-bounded questions: {bounded - set(ENGINE_BOUNDED)}"


@pytest.mark.parametrize("key,bad", _CASES, ids=[f"{k.split('.')[-1]}={b}" for k, b in _CASES])
def test_a_value_outside_a_declared_engine_bound_is_refused(key, bad):
    """This fails precisely when a wizard bound is WIDER than the engine's, the
    outcome the wizard exists to prevent. Each bound names its owning code.
    """
    q = SCHEMA[key]
    rule, code = ENGINE_BOUNDED[key]
    leaf = key.split(".")[-1]
    if rule is None:
        clm = live_clm("geometric", dict(VALID_PARAMS["geometric"]))
        clm[leaf] = bad
    else:
        params = dict(VALID_PARAMS[rule])
        params[leaf] = bad
        clm = live_clm(rule, params)
    clm.update(LIVE_EXTRAS.get(key, {}))
    assert q.visible_when(clm), "the value must be live for the engine to consult it"

    c, X = geometry([200, 200, 200, 200])
    with pytest.raises(ValueError) as excinfo:
        generate_clm_labels(c, X, clm, seed=1)
    assert code in str(excinfo.value), f"got a different error: {excinfo.value}"


# ---------------------------------------------------------------------------
# The negative-target guard is wizard-only
# ---------------------------------------------------------------------------

def test_target_value_bound_is_the_wizard_only_floor():
    q = SCHEMA["clm_label.target_metric.value"]
    assert (q.lo, q.hi) == (0.0, 1.0), "the MCC/ARI target must floor at 0.0 in the wizard"
    assert q.engine_min is None and q.engine_max is None, \
        "the target floor is deliberately wizard-only; the engine is left unchanged"


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


def test_proportions_prompt_re_asks_on_empty_input(stdin):
    """An empty list answer must re-ask."""
    stdin.extend(["", "  ", "0.25, 0.25, 0.25, 0.25"])
    assert wizard.ask_from("clm_label.proportions") == pytest.approx([0.25] * 4)
    assert not stdin


# ---------------------------------------------------------------------------
# Import graph and deletability
# ---------------------------------------------------------------------------

def test_wizard_import_graph_excludes_plotting():
    """A fresh interpreter, because the suite's conftest imports matplotlib for
    the whole process; an in-process check could never fail."""
    # Path injected because -E drops PYTHONPATH.
    root = str(Path(clmsynth.__file__).resolve().parents[1])
    code = (f"import sys; sys.path.insert(0, {root!r})\n"
            "import clmsynth.config_wizard, clmsynth.questions\n"
            "bad = [m for m in ('matplotlib', 'seaborn') if m in sys.modules]\n"
            "assert not bad, bad\n")
    result = subprocess.run(
        [sys.executable, "-E", "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_no_core_module_imports_the_wizard():
    """Deleting config_wizard.py must leave a working package: nothing in core may
    import it. A plain mention in a help string or docstring is fine, an import is
    not, so this looks only at import statements."""
    pkg = Path(clmsynth.__file__).parent
    offenders = []
    for path in pkg.glob("*.py"):
        if path.name == "config_wizard.py":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if (stripped.startswith(("import ", "from "))) and "config_wizard" in stripped:
                offenders.append(f"{path.name}: {stripped}")
    assert not offenders, "core imports the wizard:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# End to end: canned answers build a config the engine accepts
# ---------------------------------------------------------------------------

def test_canned_answers_build_an_engine_valid_config(stdin):
    """Drive the builders with canned answers exactly as main() does, then run the
    assembled clm_label against matching geometry and round-trip it through YAML.
    A regression pin on the refactor: the schema-driven wizard produces the same
    kind of config the inline one did."""
    stdin.extend([
        # build_source
        "byoc",           # data source
        "",               # output_dir -> OUTPUT
        # _byoc_suite
        "",               # input_dir -> INPUT
        "my_clusters",    # CSV file name(s)
        "",               # cluster_column -> cluster
        "",               # carry any column through? -> no
        "",               # standardize -> no
        "",               # seed -> 42
        # build_label_generation
        "",               # n_labels -> 1
        "",               # label seed -> 42
        # build_clm
        "",               # num_classes -> 3
        "",               # matching_mode -> custom
        "",               # balance -> balanced (returns before skew)
        "",               # aim for a target? -> no
        "",               # how many rules -> 1
        "",               # rule 1 label -> 0
        "0",              # rule 1 clusters
        "",               # recall_target -> 0.8
        "",               # split_rule -> proportional_to_size
        "",               # spillover_rule -> proportional_to_marginal
        "",               # competing noise? -> no
        "",               # centroid dependence? -> no
    ])

    source, gs, suite, known_k = wizard.build_source()
    lg = wizard.build_label_generation(source)
    lg["clm_label"] = wizard.build_clm(known_k)
    config = {"global_settings": gs, f"{source}_suite": suite, "label_generation": lg}

    assert not stdin, "the canned queue and the wizard's questions must line up exactly"
    assert source == "byoc"
    clm = config["label_generation"]["clm_label"]
    assert clm["num_classes"] == 3 and clm["matching_mode"] == "custom"

    c, X = geometry([200, 200, 200])
    out = np.asarray(generate_clm_labels(c, X, clm, seed=42))
    assert len(out) == len(c)
    assert set(np.unique(out)).issubset({0, 1, 2})

    # survives serialization: the wizard writes YAML, so what it built must reload.
    reloaded = yaml.safe_load(yaml.dump(config, sort_keys=False))["label_generation"]["clm_label"]
    out2 = np.asarray(generate_clm_labels(c, X, reloaded, seed=42))
    assert np.array_equal(out, out2)


# ---------------------------------------------------------------------------
# Saving a completed interview
# ---------------------------------------------------------------------------

def test_save_creates_a_missing_parent_directory(tmp_path):
    """Answering "configs/run1.yaml" must not fail."""
    target = tmp_path / "configs" / "run1.yaml"

    wizard._save_config({"global_settings": {"data_source": "byoc"}}, str(target))

    assert yaml.safe_load(target.read_text(encoding="utf-8")) == {
        "global_settings": {"data_source": "byoc"}}


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
