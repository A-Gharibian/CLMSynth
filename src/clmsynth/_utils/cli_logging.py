# SPDX-License-Identifier: MIT
# cli_logging.py
"""Logging configuration for the console scripts.
"""

import logging
import re

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

_RUN_LINE = re.compile(r" - INFO - Run (\S+) from config ")


def run_line(run: str, config: object) -> str:
    """The log line that ties a run folder, named by time and source only, to its config."""
    return f"Run {run} from config '{config}'."


def run_named_in(output: str) -> str | None:
    """The run folder named by `run_line` in a run's captured output, if it made one."""
    found = _RUN_LINE.search(output)
    return found.group(1) if found else None


class SingleLineFilter(logging.Filter):
    """Keeps one log call to one line, so a config value cannot forge a record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "\n" in message or "\r" in message:
            # getMessage() has already applied `args` to `msg`. Store the result
            # and clear the args, or the % formatting would be attempted a
            # second time against a message that no longer carries placeholders.
            record.msg = message.replace("\r", "\\r").replace("\n", "\\n")
            record.args = ()
        return True


def configure_cli_logging(level: int = logging.INFO) -> None:
    """Configure logging for a console script. Safe to call more than once.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT)

    root = logging.getLogger()
    for handler in root.handlers:
        # Idempotent: a second call must not stack a second copy, which would
        # escape an already-escaped backslash.
        if not any(isinstance(f, SingleLineFilter) for f in handler.filters):
            handler.addFilter(SingleLineFilter())
