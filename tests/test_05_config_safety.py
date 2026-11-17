"""Category 5: Configuration safety.

**What the program does to protect the machine running a configuration it did
not author.** The premise is ordinary and scientific rather than adversarial: a
config here is a shareable artifact, and reproducing someone's published results
means running a YAML they wrote. These tests assert the measures that
make that safe to do.
Three measures now exist, so the category does:

  log forgery      `SingleLineFilter` keeps one log call to one line, so a
                   configuration value carrying a newline cannot split a record
                   into what reads as two and forge the second.
  destination      `resolved_for_report` states the absolute output and input
                   directories before any work begins, so a configuration cannot
                   quietly direct a run somewhere its operator did not intend.
  dataset cache    An entry is read without unpickling anything, a link planted
                   under an entry's name is replaced and never written through,
                   and the folder clmsynth creates is the user's own: whoever can
                   write the cache folder can change the data a run reads, never
                   run code or write elsewhere.

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

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from clmsynth import dataset_sources
from clmsynth._input_sources import dataset_cache
from clmsynth._utils.cli_logging import SingleLineFilter
from clmsynth._utils.write_files import resolved_for_report

# ---------------------------------------------------------------------------
# Log forgery: one log call is one line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("plain message", "plain message"),
        ("forged\nsecond line", "forged\\nsecond line"),
        ("carriage\rreturn", "carriage\\rreturn"),
        ("windows\r\nline", "windows\\r\\nline"),
    ],
    ids=["untouched", "newline", "carriage-return", "crlf"],
)
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
        name="clmsynth.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=raw,
        args=(),
        exc_info=None,
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
        name="clmsynth.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="skew_rule %s is unknown",
        args=("geometric\nFORGED",),
        exc_info=None,
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
# Dataset cache: whoever can write its folder cannot run code through it
#
# The network is stubbed (`network` in conftest.py), and each test keeps its
# cache in a folder of its own (`cache_dir`).
# ---------------------------------------------------------------------------


def cached() -> pd.DataFrame:
    """clustbench's wut/x1, fetched with the cache on."""
    frame = dataset_sources.fetch("clustbench", "wut", "x1", cache=True)
    assert frame is not None
    return frame


SPRUNG: list[str] = []


def _spring() -> None:
    SPRUNG.append("unpickled")


class _Trap:
    def __reduce__(self) -> tuple[Any, ...]:
        return _spring, ()


def test_an_entry_is_read_without_unpickling_anything(cache_dir, network, caplog):
    """Whoever can write the folder can change the data a run reads, never run code."""
    first = cached()
    [entry] = cache_dir.iterdir()
    with open(entry, "wb") as file:
        np.savez(file, meta=np.array([_Trap()], dtype=object))
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        assert cached().equals(first)
    assert SPRUNG == []
    assert "Cannot read the cached 'wut/x1'" in caplog.text


def test_a_link_in_the_cache_folder_is_replaced_and_never_written_through(
        cache_dir, network, tmp_path, caplog):
    """In a shared folder someone else can plant a link under an entry's name: the run
    replaces the link and leaves the file it points at alone."""
    cached()
    [entry] = cache_dir.iterdir()
    entry.unlink()
    victim = tmp_path / "victim.txt"
    victim.write_text("mine")
    try:
        entry.symlink_to(victim)
    except OSError:
        pytest.skip("this system does not let the user make a link")
    assert not dataset_sources.is_cached("clustbench", "wut", "x1")
    with caplog.at_level(logging.WARNING, logger="clmsynth"):
        cached()
    assert victim.read_text() == "mine"
    assert not entry.is_symlink() and dataset_sources.is_cached("clustbench", "wut", "x1")
    assert "is a link, not a cache entry" in caplog.text


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_the_folder_clmsynth_creates_is_the_users_own(tmp_path, network, monkeypatch):
    monkeypatch.delenv(dataset_cache.ENV_VAR, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    old = os.umask(0o022)
    try:
        cached()
    finally:
        os.umask(old)
    folder = dataset_cache.cache_folder()
    assert folder.is_dir() and folder.stat().st_mode & 0o777 == 0o700
