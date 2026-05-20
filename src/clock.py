from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
from zoneinfo import ZoneInfo

from astropy.time import Time


PROJECT_TIMEZONE_NAME = "Asia/Yerevan"
PROJECT_TIMEZONE = ZoneInfo(PROJECT_TIMEZONE_NAME)


@dataclass(frozen=True)
class SimulationClock:
    """
    Single Armenia-time-based simulation clock for the ROT-54/2.6 digital twin.

    Rules:
    - Project timezone: Asia/Yerevan.
    - Naive datetime is forbidden.
    - ISO 8601 strings must contain timezone information.
    """

    current_time: datetime

    def __post_init__(self) -> None:
        validated = self._validate_datetime(self.current_time)
        object.__setattr__(self, "current_time", validated)

    @staticmethod
    def from_iso(iso_string: str) -> "SimulationClock":
        """
        Create SimulationClock from ISO 8601 string.

        Correct example:
        2026-06-21T13:00:00+04:00
        """

        if iso_string.endswith("Z"):
            iso_string = iso_string.replace("Z", "+00:00")

        dt = datetime.fromisoformat(iso_string)

        return SimulationClock(dt)

    @staticmethod
    def _validate_datetime(dt: datetime) -> datetime:
        """
        Validate datetime and convert it to Armenia project time.
        """

        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError(
                "Naive datetime is forbidden. Use timezone-aware Armenia time."
            )

        return dt.astimezone(PROJECT_TIMEZONE)

    def to_astropy_time(self) -> Time:
        """
        Convert current simulation time to astropy.time.Time.
        """

        return Time(self.current_time)

    @staticmethod
    def make_range(
        start_iso: str,
        end_iso: str,
        step_minutes: int,
    ) -> List[Time]:
        """
        Create Armenia-time range as a list of astropy.time.Time objects.

        End time is not included.
        """

        if step_minutes <= 0:
            raise ValueError("step_minutes must be positive.")

        start_clock = SimulationClock.from_iso(start_iso)
        end_clock = SimulationClock.from_iso(end_iso)

        start = start_clock.current_time
        end = end_clock.current_time

        if end <= start:
            raise ValueError("end time must be later than start time.")

        result: List[Time] = []
        current = start

        while current < end:
            result.append(Time(current))
            current += timedelta(minutes=step_minutes)

        return result