"""Category 5: Configuration safety.

**What the program does to protect the machine running a configuration it did
not author.** The premise is ordinary and scientific rather than adversarial: a
config here is a shareable artifact, and reproducing someone's published results
means running a YAML you did not write. These tests assert the measures that
make that safe to do.

This slot was empty until 0.6.6, and deliberately so. The category was retired
because it had no subject, the program implemented no protective measure worth
asserting, and what remained under the name "security" was input validation
producing coded `[CLM-###]` errors, which is diagnostics and belongs in `06`.
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
  timing           Wall-clock characterisations are not tests, here or anywhere
                   in this suite. They assert against the clock rather than
                   against behaviour.

Nothing here uses a pipeline configuration or touches the filesystem: these are
unit assertions on a log record and a path string. That is deliberate. A measure
whose test needs a full run is a measure that cannot be reasoned about.

Related, and deliberately not moved here: `04_failure_modes` asserts that
path-shaped battery and dataset names are refused. It is the third config-borne
guard, and it lives there because the behaviour it pins is that the batch skips
the offender and continues, which is `04`'s subject. Overlap between categories
is fine; moving a test away from the behaviour it asserts is not.

See SECURITY.md for the threat model these measures serve, and for the standing
verdicts on the scanner findings that are accepted rather than fixed.
"""

import logging
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

    Config values reach messages by design: a warning naming an unrecognised
    `skew_rule` has to quote it. A value containing a newline would otherwise
    split one record into what reads as two, and the second can be shaped to
    look like a line the program never emitted.

    Escaped rather than stripped, and asserted as such: `\\n` in the output says
    a newline was present and was neutralised, where deleting it would leave a
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

    Reporting the resolved form is what makes a captured stdout from a cluster
    job or a pipeline step reconstructable afterwards, and what stops a borrowed
    configuration from writing somewhere unremarked.

    Deliberately a report and not a restriction. `output_dir` is *supposed* to be
    a path, so there is no category error to refuse, and every candidate
    restriction rejects something legitimate: scratch space on a cluster, an
    output volume, `../results`. See SECURITY.md.
    """
    monkeypatch.chdir(tmp_path)
    reported = resolved_for_report("OUTPUT")

    assert Path(reported).is_absolute(), f"not resolved to an absolute path: {reported}"
    assert Path(reported) == (tmp_path / "OUTPUT").resolve()
