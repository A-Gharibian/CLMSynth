# Security Policy

[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

## Supported Versions

Security fixes land on the latest release of the current series. There is no
long-term-support branch.

| Version   | Supported          |
|-----------|--------------------|
| `0.6.6`   | :white_check_mark: |
| < `0.6.6` | :x:                |


## Reporting a vulnerability

Report privately through GitHub's
[private vulnerability reporting](https://github.com/A-Gharibian/CLMSynth/security/advisories/new)
rather than in a public issue. If you would rather use email, the maintainer
address is the one published in `CITATION.cff` and `pyproject.toml`
(arootin.gharibian@vsb.cz).

## Threat model

This document exists because CLMSynth's static analysis raises findings that are
*real data flows* but not vulnerabilities here, and the reasoning that
distinguishes the two should be written down once rather than rediscovered every
time a scanner runs.

**CLMSynth is a local, single-user command-line program and library.** It has no
authentication, no authorization, no network service, no multi-tenancy, and no
persistent state shared between users. It reads files the invoking user can
already read, and writes files that user can already write, with that user's own
privileges.

The consequence: **there is no privilege boundary between the person supplying
input and the person running the program.** Someone who can pass a path on the
command line can already open that path with any other tool on the machine.

Where that reasoning stops is the interesting part. **A configuration file is a
shareable artifact.** Reproducing published results means running a YAML you did
not write, so a config is the one input that can plausibly come from someone
other than the person at the keyboard. Guards therefore apply to configuration
values, not to command-line arguments or interactive prompts.

## Standing findings and why they are accepted

### `py/path-injection` — paths from argv and interactive prompts

CodeQL reports "uncontrolled data used in path expression" where a path typed by
the invoking user reaches `open()`:

- `main.py` — `sys.argv[1]`, the config path (`python -m clmsynth.main my_config.yaml`)
- `generate_config.py` — `sys.argv[1]` and `sys.argv[2]`, the payload and output paths
- `config_wizard.py` — the save path the wizard asks for at the prompt

These are the program's interface. Naming the file you want to read or write is
the entire point of a command-line tool, and refusing a path because the user
typed it would remove documented functionality while hardening nothing: the same
user, at the same privilege level, can reach that path with `cat` or any editor.

**Accepted, with severity judged not applicable.** The finding's "High" rating
reflects the query's usual context, a server acting on remote input, and does
not transfer to a local CLI. These are dismissed in code scanning as *Won't fix*
rather than *False positive*, because the data flow genuinely exists.

### `py/path-injection` — paths from the configuration file

One sink is different in kind: `output_dir` in the configuration reaches the
run-folder and summary writes in `main.py`. Under the shareable-config case
above, that value can come from someone else.

Impact is nonetheless limited, and limited by construction rather than by luck:

- `build_run_dir` creates a fresh `DDMMYY_Source_HHMMSS` folder with
  `mkdir(exist_ok=False)` and retries the suffix on collision, so a hostile
  `output_dir` can create a directory tree somewhere unintended but **cannot
  overwrite a chosen existing file**.
- `byoc_suite.input_dir` is read-only, and what it can read is constrained by the
  BYOC import requirements: at least two clusters, at least three points each,
  numeric feature columns, no reserved column names. Anything it does read is
  written to the user's own output folder and sent nowhere.

**Names are guarded; directories are deliberately not.** Battery and dataset
names from the configuration are refused when path-shaped — `_is_plain_name` and
`_drop_path_shaped_names` in `main.py` reject any name containing `/`, `\`, `:`,
or equal to `.` or `..`, tested against traversal, separator and drive-relative
forms. A name is not supposed to be a path, so a separator in one is a category
error and can be refused without ambiguity.

`output_dir` and `byoc_suite.input_dir` **are** paths, so there is no category
error to detect, and every candidate restriction refuses something legitimate:
rejecting absolute paths breaks scratch space on a cluster and output on another
volume; requiring containment under the working directory breaks the same cases;
rejecting `..` breaks `../results`. The asymmetry is the correct outcome, not a
gap.

Since 0.6.6 both directories are instead **resolved and logged before any work
begins**, so a configuration cannot quietly direct a run somewhere unexpected,
the absolute destination is the first thing the run states. That addresses the
residual risk, which is surprise rather than damage: the run folder is created
fresh with `mkdir(exist_ok=False)`, so no existing file can be overwritten.

This alert therefore stays **dismissed** rather than becoming closeable.

### `B310` — `urlopen` with a configurable base URL

bandit flags `dataset_sources.py`'s `urlopen`, because `base_url` comes from the
configuration and a `file://` scheme would read a local file rather than fetch
over HTTP.

**Accepted under the model above**, on the same reasoning: the result is written to their own
output folder. Restricting the accepted URL schemes is noted as optional
hardening. If it is ever implemented, the exemption in `[tool.bandit]` comes out
and a test replaces it.

### `B101`, `B404`, `B603`

Recorded with their individual reasons in `[tool.bandit]` in `pyproject.toml`:
two `assert`s guarding internal allocation invariants that no configuration can
reach, and the wizard's `subprocess.run`, whose argument vector is built from
`sys.executable` and a path the wizard itself just wrote, with no shell and no
user string reaching `argv`.

## What would change this document

The model above rests on CLMSynth being run locally by the person who supplied
its input. It stops holding if the package is ever driven by a service that
accepts configurations from untrusted submitters.
