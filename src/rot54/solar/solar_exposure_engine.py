from __future__ import annotations

import math


class SolarExposureEngine:
    """
    Calculates front-side solar exposure for the ROT-54/2.6 main reflector.

    Coordinate convention:
        x axis: East
        y axis: North
        z axis: Up / zenith

    Solar azimuth convention:
        azimuth = 0 deg   -> North
        azimuth = 90 deg  -> East
        azimuth = 180 deg -> South
        azimuth = 270 deg -> West

    ROT-54/2.6 reflector axis:
        tilted by 15 deg toward the South from zenith.

    Therefore the reflector normal vector is:
        n = (0, -sin(15 deg), cos(15 deg))
    """

    def __init__(
        self,
        reflector_tilt_south_deg: float = 15.0,
    ) -> None:
        self.reflector_tilt_south_deg = reflector_tilt_south_deg
        self.reflector_normal = self._calculate_reflector_normal()

    def _calculate_reflector_normal(self) -> tuple[float, float, float]:
        tilt_rad = math.radians(self.reflector_tilt_south_deg)

        nx = 0.0
        ny = -math.sin(tilt_rad)
        nz = math.cos(tilt_rad)

        return nx, ny, nz

    @staticmethod
    def solar_vector_from_alt_az(
        altitude_deg: float,
        azimuth_deg: float,
    ) -> tuple[float, float, float]:
        """
        Converts solar altitude and azimuth into a unit vector.

        Input:
            altitude_deg: solar altitude above horizon
            azimuth_deg: solar azimuth, 0 deg North, 90 deg East

        Output:
            solar vector in East-North-Up coordinates:
                sx: East component
                sy: North component
                sz: Up component
        """

        altitude_rad = math.radians(altitude_deg)
        azimuth_rad = math.radians(azimuth_deg)

        sx = math.cos(altitude_rad) * math.sin(azimuth_rad)
        sy = math.cos(altitude_rad) * math.cos(azimuth_rad)
        sz = math.sin(altitude_rad)

        return sx, sy, sz

    @staticmethod
    def dot_product(
        a: tuple[float, float, float],
        b: tuple[float, float, float],
    ) -> float:
        return (
            a[0] * b[0]
            + a[1] * b[1]
            + a[2] * b[2]
        )

    def calculate_front_side_exposure(
        self,
        altitude_deg: float,
        azimuth_deg: float,
    ) -> dict:
        """
        Calculates the front-side solar exposure coefficient.

        If the Sun is below the horizon, exposure is zero.

        Output:
            raw_dot: n · s
            mu_front: max(0, n · s), with horizon filtering
            is_sun_above_horizon: altitude_deg > 0
            is_front_side_illuminated: mu_front > 0
        """

        if altitude_deg <= 0.0:
            return {
                "raw_dot": 0.0,
                "mu_front": 0.0,
                "is_sun_above_horizon": False,
                "is_front_side_illuminated": False,
            }

        solar_vector = self.solar_vector_from_alt_az(
            altitude_deg=altitude_deg,
            azimuth_deg=azimuth_deg,
        )

        raw_dot = self.dot_product(
            self.reflector_normal,
            solar_vector,
        )

        mu_front = max(0.0, raw_dot)

        return {
            "raw_dot": raw_dot,
            "mu_front": mu_front,
            "is_sun_above_horizon": True,
            "is_front_side_illuminated": mu_front > 0.0,
        }
