from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SiteParameters:
    """
    Geographic and time-zone parameters of the observing site.

    Coordinate convention:
        latitude_deg  > 0 north
        longitude_deg > 0 east
    """

    latitude_deg: float
    longitude_deg: float
    timezone: str


@dataclass(frozen=True)
class MirrorAxisParameters:
    """
    Main reflector axis orientation.

    axis_tilt_south_deg:
        Tilt of the central front-facing normal from local zenith toward south.

    ENU convention:
        E = east
        N = north
        U = up

    If the axis is tilted southward by psi:
        axis_ENU = [0, -sin(psi), cos(psi)]
    """

    axis_tilt_south_deg: float


def day_of_year(local_time: datetime) -> int:
    """
    Return day number in the year, starting from 1.
    """
    return int(local_time.timetuple().tm_yday)


def equation_of_time_minutes(day_number: int) -> float:
    """
    Engineering approximation of the equation of time.

    Output:
        minutes

    Formula:
        B = 360 deg * (N - 81) / 364
        EoT = 9.87 sin(2B) - 7.53 cos(B) - 1.5 sin(B)
    """
    b_rad = np.deg2rad(360.0 * (day_number - 81.0) / 364.0)
    return float(
        9.87 * np.sin(2.0 * b_rad)
        - 7.53 * np.cos(b_rad)
        - 1.5 * np.sin(b_rad)
    )


def solar_declination_deg(day_number: int) -> float:
    """
    Standard engineering approximation for solar declination.

    Output:
        degrees
    """
    return float(23.45 * np.sin(np.deg2rad(360.0 * (284.0 + day_number) / 365.0)))


def local_clock_hours(local_time: datetime) -> float:
    """
    Convert local clock time to decimal hours.
    """
    return (
        float(local_time.hour)
        + float(local_time.minute) / 60.0
        + float(local_time.second) / 3600.0
    )


def local_standard_meridian_deg(local_time: datetime) -> float:
    """
    Return the local standard meridian from UTC offset.

    For Armenia, UTC+4 gives:
        LSTM = 15 deg/hour * 4 = 60 deg east
    """
    utc_offset = local_time.utcoffset()

    if utc_offset is None:
        raise ValueError("Timezone-aware datetime is required.")

    offset_hours = utc_offset.total_seconds() / 3600.0
    return float(15.0 * offset_hours)


def solar_time_hours(local_time: datetime, longitude_deg: float) -> float:
    """
    Convert local clock time into local solar time.

    Time correction:
        TC = EoT + 4 * (longitude - LSTM)

    where:
        EoT is in minutes,
        longitude and LSTM are in degrees,
        4 min/deg converts longitude difference into time.
    """
    n = day_of_year(local_time)
    eot_min = equation_of_time_minutes(n)
    lstm_deg = local_standard_meridian_deg(local_time)

    correction_min = eot_min + 4.0 * (longitude_deg - lstm_deg)

    return local_clock_hours(local_time) + correction_min / 60.0


def hour_angle_deg(local_time: datetime, longitude_deg: float) -> float:
    """
    Solar hour angle in degrees.

    At local solar noon:
        omega = 0 deg

    Morning:
        omega < 0

    Afternoon:
        omega > 0
    """
    st_h = solar_time_hours(local_time, longitude_deg)
    return float(15.0 * (st_h - 12.0))


def solar_vector_enu(
    latitude_deg: float,
    declination_deg: float,
    hour_angle_deg_value: float,
) -> tuple[float, float, float]:
    """
    Return the unit vector from the observing site toward the Sun.

    ENU convention:
        east, north, up

    Formula:
        E = -cos(delta) sin(omega)
        N =  cos(phi) sin(delta) - sin(phi) cos(delta) cos(omega)
        U =  sin(phi) sin(delta) + cos(phi) cos(delta) cos(omega)
    """
    phi = np.deg2rad(latitude_deg)
    delta = np.deg2rad(declination_deg)
    omega = np.deg2rad(hour_angle_deg_value)

    east = -np.cos(delta) * np.sin(omega)
    north = (
        np.cos(phi) * np.sin(delta)
        - np.sin(phi) * np.cos(delta) * np.cos(omega)
    )
    up = (
        np.sin(phi) * np.sin(delta)
        + np.cos(phi) * np.cos(delta) * np.cos(omega)
    )

    vector = np.array([east, north, up], dtype=float)
    norm = np.linalg.norm(vector)

    if norm <= 0.0:
        raise ValueError("Invalid solar vector norm.")

    vector = vector / norm

    return float(vector[0]), float(vector[1]), float(vector[2])


def solar_altitude_azimuth_deg(
    solar_east: float,
    solar_north: float,
    solar_up: float,
) -> tuple[float, float]:
    """
    Convert the solar ENU vector into altitude and azimuth.

    Altitude:
        degrees above mathematical horizon.

    Azimuth:
        degrees clockwise from north:
            0   = north
            90  = east
            180 = south
            270 = west
    """
    up_clipped = float(np.clip(solar_up, -1.0, 1.0))
    altitude = float(np.rad2deg(np.arcsin(up_clipped)))

    azimuth = float(np.rad2deg(np.arctan2(solar_east, solar_north)))
    azimuth = (azimuth + 360.0) % 360.0

    return altitude, azimuth


def mirror_axis_enu(axis_tilt_south_deg: float) -> tuple[float, float, float]:
    """
    Return central front-facing normal of the main reflector in ENU coordinates.
    """
    psi = np.deg2rad(axis_tilt_south_deg)

    east = 0.0
    north = -np.sin(psi)
    up = np.cos(psi)

    return float(east), float(north), float(up)


def compute_solar_position_at_time(
    local_time: datetime,
    site: SiteParameters,
    mirror_axis: MirrorAxisParameters,
    apparent_horizon_altitude_deg: float,
) -> dict[str, object]:
    """
    Compute solar position and reflector front-side condition for one local time.
    """
    if local_time.tzinfo is None:
        raise ValueError("local_time must be timezone-aware.")

    n = day_of_year(local_time)
    eot_min = equation_of_time_minutes(n)
    decl_deg = solar_declination_deg(n)
    solar_time_h = solar_time_hours(local_time, site.longitude_deg)
    omega_deg = hour_angle_deg(local_time, site.longitude_deg)

    solar_e, solar_n, solar_u = solar_vector_enu(
        latitude_deg=site.latitude_deg,
        declination_deg=decl_deg,
        hour_angle_deg_value=omega_deg,
    )

    altitude_deg, azimuth_deg = solar_altitude_azimuth_deg(
        solar_east=solar_e,
        solar_north=solar_n,
        solar_up=solar_u,
    )

    axis_e, axis_n, axis_u = mirror_axis_enu(mirror_axis.axis_tilt_south_deg)

    axis_dot_sun = (
        axis_e * solar_e
        + axis_n * solar_n
        + axis_u * solar_u
    )

    above_apparent_horizon = altitude_deg >= apparent_horizon_altitude_deg
    in_front_of_reflector_axis = axis_dot_sun > 0.0

    front_side_illumination = bool(
        above_apparent_horizon and in_front_of_reflector_axis
    )

    return {
        "local_time": local_time.isoformat(),
        "date": local_time.date().isoformat(),
        "clock_time": local_time.strftime("%H:%M:%S"),
        "day_of_year": n,
        "equation_of_time_min": eot_min,
        "solar_declination_deg": decl_deg,
        "solar_time_h": solar_time_h,
        "hour_angle_deg": omega_deg,
        "solar_altitude_deg": altitude_deg,
        "solar_azimuth_deg": azimuth_deg,
        "solar_east": solar_e,
        "solar_north": solar_n,
        "solar_up": solar_u,
        "mirror_axis_east": axis_e,
        "mirror_axis_north": axis_n,
        "mirror_axis_up": axis_u,
        "axis_dot_sun": axis_dot_sun,
        "above_apparent_horizon": bool(above_apparent_horizon),
        "in_front_of_reflector_axis": bool(in_front_of_reflector_axis),
        "front_side_illumination": front_side_illumination,
    }


def make_local_day_times(
    date_iso: str,
    timezone_name: str,
    step_minutes: int,
) -> list[datetime]:
    """
    Generate timezone-aware local times for one full local day.

    The end time is exclusive:
        00:00, 00:01, ..., 23:59 for step_minutes = 1
    """
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive.")

    tz = ZoneInfo(timezone_name)
    start = datetime.fromisoformat(date_iso).replace(tzinfo=tz)

    times: list[datetime] = []

    total_minutes = 24 * 60
    for minute in range(0, total_minutes, step_minutes):
        times.append(start + timedelta(minutes=minute))

    return times


def compute_solar_day_table(
    case_key: str,
    case_label: str,
    date_iso: str,
    site: SiteParameters,
    mirror_axis: MirrorAxisParameters,
    step_minutes: int,
    apparent_horizon_altitude_deg: float,
) -> pd.DataFrame:
    """
    Compute solar geometry table for a seasonal case.
    """
    local_times = make_local_day_times(
        date_iso=date_iso,
        timezone_name=site.timezone,
        step_minutes=step_minutes,
    )

    rows = [
        compute_solar_position_at_time(
            local_time=local_time,
            site=site,
            mirror_axis=mirror_axis,
            apparent_horizon_altitude_deg=apparent_horizon_altitude_deg,
        )
        for local_time in local_times
    ]

    df = pd.DataFrame(rows)
    df.insert(0, "case_key", case_key)
    df.insert(1, "case_label", case_label)

    return df


def summarize_solar_day(
    solar_day: pd.DataFrame,
    step_minutes: int,
    expected_front_duration_h: float | None,
) -> dict[str, object]:
    """
    Create a compact summary for one seasonal solar day.
    """
    if solar_day.empty:
        raise ValueError("solar_day is empty.")

    case_key = str(solar_day["case_key"].iloc[0])
    case_label = str(solar_day["case_label"].iloc[0])
    date_iso = str(solar_day["date"].iloc[0])

    step_h = step_minutes / 60.0

    daylight_mask = solar_day["above_apparent_horizon"].astype(bool)
    front_mask = solar_day["front_side_illumination"].astype(bool)

    daylight_duration_h = float(daylight_mask.sum() * step_h)
    front_duration_h = float(front_mask.sum() * step_h)

    daylight_rows = solar_day[daylight_mask]
    front_rows = solar_day[front_mask]

    max_alt_idx = solar_day["solar_altitude_deg"].idxmax()
    max_axis_idx = solar_day["axis_dot_sun"].idxmax()

    if not daylight_rows.empty:
        first_daylight_time = str(daylight_rows["clock_time"].iloc[0])
        last_daylight_time = str(daylight_rows["clock_time"].iloc[-1])
    else:
        first_daylight_time = ""
        last_daylight_time = ""

    if not front_rows.empty:
        first_front_time = str(front_rows["clock_time"].iloc[0])
        last_front_time = str(front_rows["clock_time"].iloc[-1])
    else:
        first_front_time = ""
        last_front_time = ""

    if expected_front_duration_h is None:
        front_duration_error_h = np.nan
    else:
        front_duration_error_h = front_duration_h - expected_front_duration_h

    return {
        "case_key": case_key,
        "case_label": case_label,
        "date": date_iso,
        "step_minutes": step_minutes,
        "daylight_duration_h": daylight_duration_h,
        "front_duration_h": front_duration_h,
        "expected_front_duration_h": expected_front_duration_h,
        "front_duration_error_h": front_duration_error_h,
        "first_daylight_time": first_daylight_time,
        "last_daylight_time": last_daylight_time,
        "first_front_time": first_front_time,
        "last_front_time": last_front_time,
        "max_altitude_time": str(solar_day.loc[max_alt_idx, "clock_time"]),
        "max_altitude_deg": float(solar_day.loc[max_alt_idx, "solar_altitude_deg"]),
        "max_axis_dot_time": str(solar_day.loc[max_axis_idx, "clock_time"]),
        "max_axis_dot": float(solar_day.loc[max_axis_idx, "axis_dot_sun"]),
        "solar_declination_deg_at_noon_like": float(
            solar_day.loc[max_alt_idx, "solar_declination_deg"]
        ),
    }
