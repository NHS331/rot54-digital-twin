from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, get_sun

from src.clock import SimulationClock
from src.observer import ObserverSite


@dataclass(frozen=True)
class SunPosition:
    """
    Solar position in local horizontal coordinates.

    altitude_deg:
        Sun altitude above the local horizon.

    azimuth_deg:
        Sun azimuth measured eastward from north:
        north = 0 deg, east = 90 deg, south = 180 deg, west = 270 deg.

    enu_vector:
        Unit vector in local ENU coordinates:
        x = East
        y = North
        z = Up
    """

    altitude_deg: float
    azimuth_deg: float
    enu_vector: np.ndarray


class AstronomyEngine:
    """
    Astronomy engine for the ROT-54/2.6 digital twin.

    Stage 3 responsibility:
    - compute Sun altitude;
    - compute Sun azimuth;
    - compute Sun vector in local ENU coordinates.

    Important:
    - pressure = 0 means atmospheric refraction is disabled.
    - This module is the only source of solar position data.
    """

    def __init__(self, observer: ObserverSite) -> None:
        self.observer = observer
        self.location = observer.earth_location()

    def sun_position(self, clock: SimulationClock) -> SunPosition:
        """
        Compute Sun altitude, azimuth, and ENU vector for one simulation time.
        """

        time = clock.to_astropy_time()

        altaz_frame = AltAz(
            obstime=time,
            location=self.location,
            pressure=0 * u.hPa,
        )

        sun_altaz = get_sun(time).transform_to(altaz_frame)

        altitude_deg = float(sun_altaz.alt.to_value(u.deg))
        azimuth_deg = float(sun_altaz.az.to_value(u.deg))

        enu_vector = self.altaz_to_enu(
            altitude_deg=altitude_deg,
            azimuth_deg=azimuth_deg,
        )

        return SunPosition(
            altitude_deg=altitude_deg,
            azimuth_deg=azimuth_deg,
            enu_vector=enu_vector,
        )

    @staticmethod
    def altaz_to_enu(altitude_deg: float, azimuth_deg: float) -> np.ndarray:
        """
        Convert AltAz coordinates to local ENU unit vector.

        Convention:
        x = East
        y = North
        z = Up

        Formula:
        x = cos(alt) * sin(az)
        y = cos(alt) * cos(az)
        z = sin(alt)
        """

        alt_rad = np.deg2rad(altitude_deg)
        az_rad = np.deg2rad(azimuth_deg)

        x_east = np.cos(alt_rad) * np.sin(az_rad)
        y_north = np.cos(alt_rad) * np.cos(az_rad)
        z_up = np.sin(alt_rad)

        vector = np.array(
            [
                x_east,
                y_north,
                z_up,
            ],
            dtype=float,
        )

        return vector / np.linalg.norm(vector)