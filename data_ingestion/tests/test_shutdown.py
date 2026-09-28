from __future__ import annotations

import signal

import pytest

from steam_ingestion.shutdown import EXIT_CODE, Shutdown, ShutdownRequested


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.steps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.steps.append(seconds)
        self.now += seconds


def test_sleep_runs_in_short_steps_until_done() -> None:
    clock = _Clock()
    Shutdown(sleep=clock.sleep, clock=lambda: clock.now).sleep(2.5)
    assert clock.steps == [1.0, 1.0, 0.5]


def test_sleep_raises_as_soon_as_a_stop_is_requested() -> None:
    clock = _Clock()
    shutdown = Shutdown(sleep=clock.sleep, clock=lambda: clock.now)

    def step(seconds: float) -> None:
        clock.sleep(seconds)
        if len(clock.steps) == 2:
            shutdown.request(signal.SIGTERM)  # the signal arrives during the second step

    shutdown._sleep = step
    with pytest.raises(ShutdownRequested):
        shutdown.sleep(60)  # e.g. a throttle cooldown
    assert clock.steps == [1.0, 1.0]


def test_install_handles_sigterm() -> None:
    previous = signal.getsignal(signal.SIGTERM)
    shutdown = Shutdown()
    try:
        shutdown.install()
        signal.raise_signal(signal.SIGTERM)
        assert shutdown.requested
        with pytest.raises(ShutdownRequested):
            shutdown.check()
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert EXIT_CODE == 143
