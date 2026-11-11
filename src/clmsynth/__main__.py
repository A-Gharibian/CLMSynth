# SPDX-License-Identifier: MIT
# __main__.py
"""The ``clmsynth`` command, and ``python -m clmsynth``: reads the command line and
hands the configs to a single run or a batch. The only place that exits the process."""

import argparse
import logging
import sys
from pathlib import Path

from ._single_run import run_single
from ._utils.cli_logging import configure_cli_logging
from ._utils.compat import is_installed, not_installed_message

log: logging.Logger = logging.getLogger(__name__)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Config paths, plus --jobs, --viz or --no-viz, and --keep-failed."""
    parser = argparse.ArgumentParser(
        prog="clmsynth",
        description="Generate CLM-labeled datasets from configuration files.")
    parser.add_argument("configs", nargs="*", default=["test_data_config.yaml"],
                        metavar="config",
                        help="Configuration file: .json, or YAML with PyYAML installed "
                             "(default: test_data_config.yaml), or a folder of them. "
                             "Several configs, or a folder, run as a batch, each config "
                             "in its own process; a config named twice runs once.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="How many configs of a batch run at once (default: 1).")
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--viz", action="store_true",
                       help="Draw the plots, or exit 1 if matplotlib is not installed "
                            "(pip install \"clmsynth[viz]\"). A single run draws them "
                            "whenever matplotlib is installed, a batch only with --viz.")
    plots.add_argument("--no-viz", action="store_true",
                       help="Skip plot rendering; the plotting stack is never imported.")
    parser.add_argument("--keep-failed", action="store_true",
                        help="Keep a failed run's folder, with its config and the log of "
                             "why it failed.")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be 1 or more")
    if "" in args.configs:
        # An empty path reads as the working folder: every config in it would run.
        parser.error("a config path is empty (an unset shell variable?)")
    return args


def _names_one_config(paths: list[str]) -> bool:
    """One path that is not a folder: it runs on its own, without the batch code."""
    return len(paths) == 1 and not Path(paths[0]).is_dir()


def main() -> None:
    """``clmsynth [config|folder ...] [--jobs N] [--viz | --no-viz] [--keep-failed]``."""
    # Parsed first: --help and a usage error print before any log line.
    args = _parse_args(sys.argv[1:])
    configure_cli_logging()

    if args.viz and not is_installed("matplotlib"):
        # Before any config starts and before any run folder exists.
        log.critical(f"--viz: {not_installed_message('matplotlib', 'Drawing the plots')}.")
        sys.exit(1)
    if _names_one_config(args.configs):
        config_path = args.configs[0]
    else:
        # Deferred so a single run loads no batch code.
        from ._extras.batch_run import batch_configs, run_batch

        try:
            configs, is_batch = batch_configs(args.configs)
        except ValueError as e:
            log.critical(str(e))
            sys.exit(2)
        if is_batch:
            sys.exit(run_batch(configs, args.jobs, render_plots=args.viz,
                               keep_failed=args.keep_failed))
        config_path = configs[0]

    status = run_single(config_path, render_plots=not args.no_viz,
                        keep_failed=args.keep_failed)
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()
