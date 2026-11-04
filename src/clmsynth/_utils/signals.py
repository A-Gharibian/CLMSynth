# SPDX-License-Identifier: MIT
# signals.py
"""SIGTERM as an exception, so a run and a batch clean up when a scheduler
cancels the job."""

__all__ = ["Terminated", "raise_terminated"]


class Terminated(BaseException):
    """SIGTERM, which a scheduler sends to cancel a job or at its time limit. A
    BaseException, like KeyboardInterrupt, so no `except Exception` catches it."""


def raise_terminated(signum: int, frame: object) -> None:
    raise Terminated
