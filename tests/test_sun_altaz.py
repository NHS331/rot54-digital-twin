import numpy as np
import pytest

from src.astronomy_engine import AstronomyEngine, SunPosition
from src.clock import SimulationClock
from src.observer import ObserverSite


def test_sun_position_returns_valid_object():
    observer = ObserverSite.from_yaml("configs/site.yaml")
    engine = AstronomyEngine(observer)

    clock = SimulationClock.from_iso("2026-06-21T13:00:00+04:00")
    sun = engine.sun_position(clock)

    assert isinstance(sun, SunPosition)


def test_sun_altitude_and_azimuth_are_in_valid_ranges():
    observer = ObserverSite.from_yaml("configs/site.yaml")
    engine = AstronomyEngine(observer)

    clock = SimulationClock.from_iso("2026-06-21T13:00:00+04:00")
    sun = engine.sun_position(clock)

    assert -90.0 <= sun.altitude_deg <= 90.0
    assert 0.0 <= sun.azimuth_deg < 360.0


def test_sun_enu_vector_is_unit_vector():
    observer = ObserverSite.from_yaml("configs/site.yaml")
    engine = AstronomyEngine(observer)

    clock = SimulationClock.from_iso("2026-06-21T13:00:00+04:00")
    sun = engine.sun_position(clock)

    assert np.linalg.norm(sun.enu_vector) == pytest.approx(1.0)


def test_summer_noon_sun_is_high_above_horizon_for_orgov():
    """
    Around summer solstice and near local solar noon,
    the Sun must be high above the horizon at latitude ~40 deg N.
    """

    observer = ObserverSite.from_yaml("configs/site.yaml")
    engine = AstronomyEngine(observer)

    clock = SimulationClock.from_iso("2026-06-21T13:00:00+04:00")
    sun = engine.sun_position(clock)

    assert 65.0 <= sun.altitude_deg <= 80.0


def test_summer_noon_sun_is_roughly_south_for_orgov():
    """
    Near local solar noon in the northern hemisphere,
    the Sun should be roughly in the southern sky.
    """

    observer = ObserverSite.from_yaml("configs/site.yaml")
    engine = AstronomyEngine(observer)

    clock = SimulationClock.from_iso("2026-06-21T13:00:00+04:00")
    sun = engine.sun_position(clock)

    assert 130.0 <= sun.azimuth_deg <= 230.0


def test_altaz_to_enu_known_directions():
    north = AstronomyEngine.altaz_to_enu(
        altitude_deg=0.0,
        azimuth_deg=0.0,
    )
    assert north == pytest.approx(np.array([0.0, 1.0, 0.0]))

    east = AstronomyEngine.altaz_to_enu(
        altitude_deg=0.0,
        azimuth_deg=90.0,
    )
    assert east == pytest.approx(np.array([1.0, 0.0, 0.0]), abs=1e-12)

    up = AstronomyEngine.altaz_to_enu(
        altitude_deg=90.0,
        azimuth_deg=0.0,
    )
    assert up == pytest.approx(np.array([0.0, 0.0, 1.0]), abs=1e-12)