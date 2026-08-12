# cli_logging.py
"""Logging configuration for the console scripts.
"""

import logging

# One format for every console script. `generate_config` previously took
# basicConfig's default while `main` set its own, so the same run could produce
# two different line shapes depending on which command emitted them.
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


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
