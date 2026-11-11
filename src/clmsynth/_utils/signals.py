# SPDX-License-Identifier: MIT
# signals.py
"""SIGTERM as an exception, so a run and a batch clean up when a scheduler
cancels the job."""

import signal
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["Terminated", "sigterm_raises"]


class Terminated(BaseException):
    """SIGTERM, which a scheduler sends to cancel a job or at its time limit. A
    BaseException, like KeyboardInterrupt, so no `except Exception` catches it."""


def _raise_terminated(signum: int, frame: object) -> None:
    raise Terminated


@contextmanager
def sigterm_raises() -> Iterator[None]:
    """A SIGTERM inside the block raises Terminated; the handler before it returns after."""
    previous = signal.signal(signal.SIGTERM, _raise_terminated)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
