"""Category 5: Configuration safety.

**What the program does to protect the machine running a configuration it did
not author.** The premise is ordinary and scientific rather than adversarial: a
config here is a shareable artifact, and reproducing someone's published results
means running a YAML they wrote. These tests assert the measures that
make that safe to do.
Two measures now exist, so the category does:

  log forgery      `SingleLineFilter` keeps one log call to one line, so a
                   configuration value carrying a newline cannot split a record
                   into what reads as two and forge the second.
  destination      `resolved_for_report` states the absolute output and input
                   directories before any work begins, so a configuration cannot
                   quietly direct a run somewhere its operator did not intend.

**Explicitly out of scope**, and these are the sections whose absence retired the
original category. Re-read this list before adding anything here:

  authorization    There is no auth boundary. CLMSynth is a local single-user
                   CLI and library, with no accounts, sessions or roles, so
                   there is nothing to bypass.
  ReDoS            The package contains no regular expressions.
  recursion        No recursive descent over user input, so no depth to exhaust.
  timing           Wall-clock characterizations are not tests, here or anywhere
                   in this suite. They assert against the clock rather than
                   against behavior.

Related, and deliberately not moved here: `04_failure_modes` asserts that
path-shaped battery and dataset names are refused. It is the third config-borne
guard, and it lives there because the behavior it pins is that the batch skips
the offender and continues, which is `04`'s subject. Overlap between categories
is fine; moving a test away from the behavior it asserts is not.

See SECURITY.md for the threat model these measures serve, and for the standing
verdicts on the scanner findings that are accepted rather than fixed.
"""

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

from clmsynth.cli_logging import SingleLineFilter
from clmsynth.main import resolved_for_report

# ---------------------------------------------------------------------------
# Log forgery: one log call is one line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("plain message", "plain message"),
    ("forged\nsecond line", "forged\\nsecond line"),
    ("carriage\rreturn", "carriage\\rreturn"),
    ("windows\r\nline", "windows\\r\\nline"),
], ids=["untouched", "newline", "carriage-return", "crlf"])
def test_a_newline_in_a_message_cannot_forge_a_second_record(raw, expected):
    """Whatever the configuration value contained, the record stays one line.

    Config values reach messages by design: a warning naming an unrecognized
    `skew_rule` has to quote it. A value containing a newline would otherwise
    split one record into what reads as two, and the second can be shaped to
    look like a line the program never emitted.

    Escaped rather than stripped, and asserted as such: `\\n` in the output says
    a newline was present and was neutralized, where deleting it would leave a
    plausible single line and hide that anything was attempted.

    Asserted on the filter directly rather than through `caplog`. caplog installs
    its own handler, and the filter is attached to the handlers configured for
    the CLI, so a caplog-based test would never see it -- passing or failing for
    reasons unrelated to the code under test.
    """
    record = logging.LogRecord(
        name="clmsynth.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=raw, args=(), exc_info=None,
    )
    assert SingleLineFilter().filter(record) is True, "the filter must not drop records"
    assert record.getMessage() == expected
    assert "\n" not in record.getMessage() and "\r" not in record.getMessage()


def test_the_filter_finishes_lazy_percent_formatting_before_scrubbing():
    """`log.warning("bad rule: %s", value)` is scrubbed too, not only f-strings.

    Both call styles appear in the package, and a filter that only handled
    finished strings would leave the lazy ones injectable while appearing to
    work. `getMessage()` applies `args` to `msg`; the result then has to be
    stored and `args` cleared, or surviving placeholders would be formatted a
    second time against arguments that are no longer there.
    """
    record = logging.LogRecord(
        name="clmsynth.test", level=logging.WARNING, pathname=__file__, lineno=1,
        msg="skew_rule %s is unknown", args=("geometric\nFORGED",), exc_info=None,
    )
    SingleLineFilter().filter(record)

    assert record.getMessage() == "skew_rule geometric\\nFORGED is unknown"
    assert record.args in ((), None), "args must be cleared once applied"
    # Formatting must be settled, not merely correct once.
    assert record.getMessage() == record.getMessage()


# ---------------------------------------------------------------------------
# Destination: a run states where it writes, before it writes
# ---------------------------------------------------------------------------


def test_a_configured_path_is_reported_as_an_absolute_path(tmp_path, monkeypatch):
    """`output_dir: OUTPUT` means different folders in different working directories.
    Deliberately a report and not a restriction. `output_dir` is *supposed* to be
    a path, so there is no category error to refuse, and every candidate
    restriction rejects something legitimate: scratch space on a cluster, an
    output volume, `../results`. See SECURITY.md.
    """
    monkeypatch.chdir(tmp_path)
    reported = resolved_for_report("OUTPUT")

    assert Path(reported).is_absolute(), f"not resolved to an absolute path: {reported}"
    assert Path(reported) == (tmp_path / "OUTPUT").resolve()


# ---------------------------------------------------------------------------
# Destructive tooling
# ---------------------------------------------------------------------------

def _catalog_gen(monkeypatch, out_dir):
    """Import the tool with OUT_DIR as its argv."""
    monkeypatch.setattr(sys, "argv", ["CLM_catalog_gen.py", str(out_dir)])
    path = Path(__file__).resolve().parents[1] / "tools" / "CLM_catalog_gen.py"
    spec = importlib.util.spec_from_file_location("clm_catalog_gen_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_generator_refuses_a_non_catalog_target(tmp_path, monkeypatch):
    """`CLM_catalog_gen.py .` must not erase a working tree.

    The tool's own docstring invites a custom OUT_DIR, and its first act is a
    recursive delete of it.
    """
    victim = tmp_path / "worktree"
    victim.mkdir()
    (victim / "precious.txt").write_text("keep me", encoding="utf-8")
    gen = _catalog_gen(monkeypatch, victim)

    with pytest.raises(SystemExit) as excinfo:
        gen._checked_rmtree(gen.OUT, ignore_errors=True)

    assert "refusing to delete" in str(excinfo.value)
    assert (victim / "precious.txt").read_text(encoding="utf-8") == "keep me"


def test_catalog_generator_accepts_an_empty_or_absent_target(tmp_path, monkeypatch):
    """A new candidate catalog is still allowed."""
    empty = tmp_path / "candidate"
    empty.mkdir()
    gen = _catalog_gen(monkeypatch, empty)

    gen._checked_rmtree(gen.OUT, ignore_errors=True)
    gen._checked_rmtree(tmp_path / "never_created")

    assert not empty.exists()


def test_catalog_generator_replaces_a_real_catalog(tmp_path, monkeypatch):
    """A folder holding a table folder is a catalog."""
    catalog = tmp_path / "troubleshooting_catalog"
    (catalog / "ValueError_1xx").mkdir(parents=True)
    (catalog / "stale.log").write_text("old", encoding="utf-8")
    gen = _catalog_gen(monkeypatch, catalog)

    gen._checked_rmtree(gen.OUT, ignore_errors=True)

    assert not catalog.exists()


def test_catalog_generator_refuses_to_delete_outside_its_target(tmp_path, monkeypatch):
    """Scratch deletes stay inside OUT_DIR."""
    catalog = tmp_path / "troubleshooting_catalog"
    (catalog / "ValueError_1xx").mkdir(parents=True)
    outsider = tmp_path / "elsewhere"
    outsider.mkdir()
    gen = _catalog_gen(monkeypatch, catalog)

    with pytest.raises(SystemExit) as excinfo:
        gen._checked_rmtree(outsider)

    assert "outside" in str(excinfo.value)
    assert outsider.exists()
