from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time


ARMENIA_TIMEZONE = ZoneInfo("Asia/Yerevan")


class AstronomyEngine:
    """
    Solar geometry engine for the ROT-54/2.6 site.

    Input:
        timezone-aware datetime

    Output:
        local Armenia time,
        solar altitude,
        solar azimuth,
        solar zenith angle.
    """

    def __init__(
        self,
        latitude_deg: float,
        longitude_deg: float,
        altitude_m: float,
        timezone=ARMENIA_TIMEZONE,
    ) -> None:
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.altitude_m = altitude_m
        self.timezone = timezone

        self.location = EarthLocation(
            lat=latitude_deg * u.deg,
            lon=longitude_deg * u.deg,
            height=altitude_m * u.m,
        )

    def sun_alt_az(self, dt: datetime) -> dict:
        """
        Calculate Sun altitude, azimuth and zenith angle.

        The datetime must not be naive.
        """

        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError(
                "Naive datetime is forbidden. Use timezone-aware datetime with Asia/Yerevan."
            )

        dt_local = dt.astimezone(self.timezone)

        astropy_time = Time(dt_local)

        altaz_frame = AltAz(
            obstime=astropy_time,
            location=self.location,
            pressure=0 * u.hPa,
        )

        sun_altaz = get_sun(astropy_time).transform_to(altaz_frame)

        altitude_deg = float(sun_altaz.alt.deg)
        azimuth_deg = float(sun_altaz.az.deg)
        zenith_deg = 90.0 - altitude_deg

        return {
            "time_local": dt_local,
            "altitude_deg": altitude_deg,
            "azimuth_deg": azimuth_deg,
            "zenith_deg": zenith_deg,
        }
