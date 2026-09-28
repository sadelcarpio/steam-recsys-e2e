"""Graceful stop of the scraping tasks on SIGTERM (Fargate Spot interruption: 2 min warning).

The handler only sets a flag. The scrapers poll it between games and the Steam client's sleeps
(pacing, retry backoff, throttle cooldown) raise `ShutdownRequested` as soon as it is set, so a
stop never waits out a 60 s cooldown. The scraper then flushes what it has, commits the state of
the games that are fully written and exits with `EXIT_CODE`: a non-zero exit, so Step Functions
retries the partition, which resumes from the committed state.
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable
from types import FrameType

logger = logging.getLogger(__name__)

EXIT_CODE = 128 + signal.SIGTERM  # 143, what a process killed by SIGTERM reports


class ShutdownRequested(Exception):  # noqa: N818 (a request, not an error)
    """Raised from a sleep or a poll once SIGTERM was received."""


class Shutdown:
    def __init__(
        self,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._requested = False
        self._sleep = sleep
        self._clock = clock

    @property
    def requested(self) -> bool:
        return self._requested

    def request(self, signum: int | None = None, frame: FrameType | None = None) -> None:
        if not self._requested:
            logger.warning("stop requested (signal %s): finishing up", signum)
        self._requested = True

    def install(self) -> None:
        """Handle SIGTERM (as PID 1 in the container, Python would otherwise ignore it)."""
        signal.signal(signal.SIGTERM, self.request)

    def check(self) -> None:
        if self._requested:
            raise ShutdownRequested

    def sleep(self, seconds: float) -> None:
        """`time.sleep` in short steps (a signal only ends the current step), raising
        `ShutdownRequested` once a stop is requested."""
        end = self._clock() + seconds
        while True:
            self.check()
            remaining = end - self._clock()
            if remaining <= 0:
                return
            self._sleep(min(1.0, remaining))
