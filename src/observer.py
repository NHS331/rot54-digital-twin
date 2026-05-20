from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import astropy.units as u
from astropy.coordinates import EarthLocation


@dataclass(frozen=True)
class ObserverSite:
    """
    Observer/site model for the ROT-54/2.6 digital twin.

    Responsibilities:
    - store geodetic coordinates;
    - create Astropy EarthLocation;
    - define local ENU basis;
    - define the antenna axis direction.
    """

    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    timezone: str
    coordinate_status: str
    antenna_axis_tilt_deg: float
    antenna_axis_tilt_direction: str

    def __post_init__(self) -> None:
        self._validate()

    @staticmethod
    def from_yaml(path: str | Path) -> "ObserverSite":
        """
        Load observer site parameters from YAML config.
        """

        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            data: dict[str, Any] = yaml.safe_load(file)

        site = data["site"]
        antenna_axis = data["antenna_axis"]

        return ObserverSite(
            name=site["name"],
            latitude_deg=float(site["latitude_deg"]),
            longitude_deg=float(site["longitude_deg"]),
            altitude_m=float(site["altitude_m"]),
            timezone=site["timezone"],
            coordinate_status=site["coordinate_status"],
            antenna_axis_tilt_deg=float(antenna_axis["tilt_deg"]),
            antenna_axis_tilt_direction=antenna_axis["tilt_direction"],
        )

    def _validate(self) -> None:
        """
        Validate site parameters.
        """

        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("Latitude must be between -90 and 90 degrees.")

        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("Longitude must be between -180 and 180 degrees.")

        if self.timezone != "Asia/Yerevan":
            raise ValueError("Project timezone must be Asia/Yerevan.")

        if self.antenna_axis_tilt_deg < 0.0:
            raise ValueError("Antenna axis tilt must be non-negative.")

        if self.antenna_axis_tilt_direction.lower() != "south":
            raise ValueError("Currently only south antenna-axis tilt is supported.")

    def earth_location(self) -> EarthLocation:
        """
        Return Astropy EarthLocation for the observer site.
        """

        return EarthLocation.from_geodetic(
            lon=self.longitude_deg * u.deg,
            lat=self.latitude_deg * u.deg,
            height=self.altitude_m * u.m,
        )

    def enu_basis_ecef(self) -> dict[str, np.ndarray]:
        """
        Return local East-North-Up basis vectors expressed in ECEF coordinates.

        The basis is orthonormal:
        - east
        - north
        - up
        """

        lat = np.deg2rad(self.latitude_deg)
        lon = np.deg2rad(self.longitude_deg)

        east = np.array(
            [
                -np.sin(lon),
                np.cos(lon),
                0.0,
            ],
            dtype=float,
        )

        north = np.array(
            [
                -np.sin(lat) * np.cos(lon),
                -np.sin(lat) * np.sin(lon),
                np.cos(lat),
            ],
            dtype=float,
        )

        up = np.array(
            [
                np.cos(lat) * np.cos(lon),
                np.cos(lat) * np.sin(lon),
                np.sin(lat),
            ],
            dtype=float,
        )

        return {
            "east": east,
            "north": north,
            "up": up,
        }

    def antenna_axis_enu(self) -> np.ndarray:
        """
        Return antenna axis vector in local ENU coordinates.

        ENU convention:
        x = East
        y = North
        z = Up

        ROT-54/2.6 working assumption:
        main axis is tilted by 15 degrees toward south.
        """

        tilt_rad = np.deg2rad(self.antenna_axis_tilt_deg)

        axis = np.array(
            [
                0.0,
                -np.sin(tilt_rad),
                np.cos(tilt_rad),
            ],
            dtype=float,
        )

        return axis / np.linalg.norm(axis)