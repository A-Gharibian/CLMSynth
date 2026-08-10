# cli_logging.py
"""Logging configuration for the console scripts.

**Nothing here runs on import, and that is the point.** `clmsynth` is a library
as well as a CLI, and a library that calls `basicConfig` silently reconfigures
the logging of whatever process imported it. Configuration belongs to whoever
owns the process: the console scripts call `configure_cli_logging()` because
they *are* the process, and a library caller gets nothing and keeps their own
handlers, format and levels.

It also gives the package one place where a log record is finished, which is
what `SingleLineFilter` needs to exist at all.
"""

import logging

# One format for every console script. `generate_config` previously took
# basicConfig's default while `main` set its own, so the same run could produce
# two different line shapes depending on which command emitted them.
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


class SingleLineFilter(logging.Filter):
    """Keeps one log call to one line, so a config value cannot forge a record.

    Configuration values reach log messages by design: a warning naming the
    `skew_rule` it does not recognise has to quote it. A value containing a
    newline therefore splits one record into what reads as two, and the second
    can be made to look like a line the program never emitted. A configuration
    is a shareable artifact here (see SECURITY.md), so the value need not have
    been written by the person reading the output.

    Escaped rather than stripped: `\\n` in the output says a newline was present
    and was neutralised. Deleting it would leave a plausible-looking single line
    and hide that anything was attempted.
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

    The filter goes on the **handlers**, not on the `clmsynth` logger, and the
    difference is not cosmetic. A filter attached to a logger only sees records
    logged directly to it; records from child loggers reach ancestors through
    `callHandlers`, which consults ancestor *handlers* and never re-applies
    ancestor filters. Every module here logs to `clmsynth.<module>`, so a filter
    on `clmsynth` would see nothing at all while appearing to work.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT)

    root = logging.getLogger()
    for handler in root.handlers:
        # Idempotent: a second call must not stack a second copy, which would
        # escape an already-escaped backslash.
        if not any(isinstance(f, SingleLineFilter) for f in handler.filters):
            handler.addFilter(SingleLineFilter())
