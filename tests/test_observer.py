import numpy as np
import pytest
from astropy.coordinates import EarthLocation

from src.observer import ObserverSite


def test_observer_loads_from_yaml():
    observer = ObserverSite.from_yaml("configs/site.yaml")

    assert observer.name == "ROT-54/2.6 Orgov"
    assert observer.latitude_deg == pytest.approx(40.3508609)
    assert observer.longitude_deg == pytest.approx(44.2417924)
    assert observer.altitude_m == pytest.approx(1711.0)
    assert observer.timezone == "Asia/Yerevan"


def test_observer_creates_earth_location():
    observer = ObserverSite.from_yaml("configs/site.yaml")
    location = observer.earth_location()

    assert isinstance(location, EarthLocation)


def test_enu_basis_is_orthonormal():
    observer = ObserverSite.from_yaml("configs/site.yaml")
    basis = observer.enu_basis_ecef()

    east = basis["east"]
    north = basis["north"]
    up = basis["up"]

    assert np.linalg.norm(east) == pytest.approx(1.0)
    assert np.linalg.norm(north) == pytest.approx(1.0)
    assert np.linalg.norm(up) == pytest.approx(1.0)

    assert np.dot(east, north) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(east, up) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(north, up) == pytest.approx(0.0, abs=1e-12)


def test_antenna_axis_is_tilted_south():
    observer = ObserverSite.from_yaml("configs/site.yaml")
    axis = observer.antenna_axis_enu()

    expected_tilt_rad = np.deg2rad(15.0)

    assert np.linalg.norm(axis) == pytest.approx(1.0)

    assert axis[0] == pytest.approx(0.0)

    assert axis[1] < 0.0
    assert axis[1] == pytest.approx(-np.sin(expected_tilt_rad))

    assert axis[2] == pytest.approx(np.cos(expected_tilt_rad))


def test_observer_rejects_wrong_timezone():
    with pytest.raises(ValueError):
        ObserverSite(
            name="Bad Site",
            latitude_deg=40.0,
            longitude_deg=44.0,
            altitude_m=1000.0,
            timezone="UTC",
            coordinate_status="test",
            antenna_axis_tilt_deg=15.0,
            antenna_axis_tilt_direction="south",
        )


def test_observer_rejects_wrong_tilt_direction():
    with pytest.raises(ValueError):
        ObserverSite(
            name="Bad Site",
            latitude_deg=40.0,
            longitude_deg=44.0,
            altitude_m=1000.0,
            timezone="Asia/Yerevan",
            coordinate_status="test",
            antenna_axis_tilt_deg=15.0,
            antenna_axis_tilt_direction="north",
        )