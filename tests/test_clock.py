from datetime import datetime

import pytest
from astropy.time import Time

from src.clock import SimulationClock


def test_clock_accepts_armenia_iso_string():
    clock = SimulationClock.from_iso("2026-06-21T13:00:00+04:00")

    assert clock.current_time.isoformat() == "2026-06-21T13:00:00+04:00"


def test_clock_converts_utc_to_armenia_time_if_given():
    clock = SimulationClock.from_iso("2026-06-21T09:00:00Z")

    assert clock.current_time.isoformat() == "2026-06-21T13:00:00+04:00"


def test_clock_returns_astropy_time():
    clock = SimulationClock.from_iso("2026-06-21T13:00:00+04:00")
    astropy_time = clock.to_astropy_time()

    assert isinstance(astropy_time, Time)


def test_clock_rejects_naive_datetime():
    naive_dt = datetime(2026, 6, 21, 13, 0, 0)

    with pytest.raises(ValueError):
        SimulationClock(naive_dt)


def test_clock_creates_time_range():
    time_range = SimulationClock.make_range(
        start_iso="2026-06-21T00:00:00+04:00",
        end_iso="2026-06-21T01:00:00+04:00",
        step_minutes=15,
    )

    assert len(time_range) == 4
    assert all(isinstance(t, Time) for t in time_range)


def test_clock_rejects_negative_step():
    with pytest.raises(ValueError):
        SimulationClock.make_range(
            start_iso="2026-06-21T00:00:00+04:00",
            end_iso="2026-06-21T01:00:00+04:00",
            step_minutes=-15,
        )


def test_clock_rejects_wrong_time_order():
    with pytest.raises(ValueError):
        SimulationClock.make_range(
            start_iso="2026-06-21T01:00:00+04:00",
            end_iso="2026-06-21T00:00:00+04:00",
            step_minutes=15,
        )