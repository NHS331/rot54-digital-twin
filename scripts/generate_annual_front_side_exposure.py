from __future__ import annotations

import csv
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import AltAz, get_sun
from astropy.time import Time
from astropy.utils import iers


# Avoid external IERS auto-download during local calculations.
iers.conf.auto_download = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from rot54.astronomy import AstronomyEngine
from rot54.solar import SolarExposureEngine


ARMENIA_TZ = ZoneInfo("Asia/Yerevan")


# Working coordinates for the ROT-54/2.6 site near Orgov / Aragats Scientific Center.
ORGOV_LATITUDE_DEG = 40.35
ORGOV_LONGITUDE_DEG = 44.25
ORGOV_ALTITUDE_M = 1711.0


OUTPUT_DIR = PROJECT_ROOT / "outputs"
CSV_OUTPUT_PATH = OUTPUT_DIR / "front_side_exposure_2026.csv"
FIGURE_OUTPUT_PATH = OUTPUT_DIR / "annual_front_side_exposure_2026.png"


def generate_dates_for_year(year: int) -> list[date]:
    start = date(year, 1, 1)
    end = date(year, 12, 31)

    dates = []
    current = start

    while current <= end:
        dates.append(current)
        current += timedelta(days=1)

    return dates


def generate_local_day_times(
    day: date,
    step_seconds: int,
) -> list[datetime]:
    start = datetime(
        day.year,
        day.month,
        day.day,
        0,
        0,
        0,
        tzinfo=ARMENIA_TZ,
    )

    end = datetime(
        day.year,
        day.month,
        day.day,
        23,
        59,
        59,
        tzinfo=ARMENIA_TZ,
    )

    times = []
    current = start

    while current <= end:
        times.append(current)
        current += timedelta(seconds=step_seconds)

    return times


def calculate_day_exposure_rows(
    astronomy_engine: AstronomyEngine,
    exposure_engine: SolarExposureEngine,
    day: date,
    step_seconds: int,
) -> list[dict]:
    """
    Vectorized solar geometry calculation for one day.

    This is more suitable for annual calculations than calling Astropy
    separately for every single timestamp.
    """

    times = generate_local_day_times(
        day=day,
        step_seconds=step_seconds,
    )

    astropy_times = Time(times)

    altaz_frame = AltAz(
        obstime=astropy_times,
        location=astronomy_engine.location,
        pressure=0 * u.hPa,
    )

    sun_altaz = get_sun(astropy_times).transform_to(altaz_frame)

    altitudes_deg = sun_altaz.alt.deg
    azimuths_deg = sun_altaz.az.deg

    rows = []

    for index, dt_local in enumerate(times):
        altitude_deg = float(altitudes_deg[index])
        azimuth_deg = float(azimuths_deg[index])
        zenith_deg = 90.0 - altitude_deg

        exposure = exposure_engine.calculate_front_side_exposure(
            altitude_deg=altitude_deg,
            azimuth_deg=azimuth_deg,
        )

        row = {
            "time_local": dt_local,
            "altitude_deg": altitude_deg,
            "azimuth_deg": azimuth_deg,
            "zenith_deg": zenith_deg,
            "raw_dot": exposure["raw_dot"],
            "mu_front": exposure["mu_front"],
            "is_sun_above_horizon": exposure["is_sun_above_horizon"],
            "is_front_side_illuminated": exposure["is_front_side_illuminated"],
        }

        rows.append(row)

    return rows


def calculate_trapezoidal_integral_mu_h(
    rows: list[dict],
) -> float:
    """
    Trapezoidal integration of mu_front over one day.

    Unit:
        mu * hour
    """

    if len(rows) < 2:
        return 0.0

    integral_h = 0.0

    for i in range(len(rows) - 1):
        row_0 = rows[i]
        row_1 = rows[i + 1]

        dt_hours = (
            row_1["time_local"] - row_0["time_local"]
        ).total_seconds() / 3600.0

        mu_0 = row_0["mu_front"]
        mu_1 = row_1["mu_front"]

        integral_h += 0.5 * (mu_0 + mu_1) * dt_hours

    return integral_h


def interpolate_threshold_crossing_time(
    row_0: dict,
    row_1: dict,
    value_key: str,
    threshold: float = 0.0,
) -> datetime:
    """
    Linear interpolation of a threshold crossing between two time samples.
    """

    t_0 = row_0["time_local"]
    t_1 = row_1["time_local"]

    y_0 = row_0[value_key]
    y_1 = row_1[value_key]

    denominator = y_1 - y_0

    if abs(denominator) < 1e-15:
        return t_0

    fraction = (threshold - y_0) / denominator
    fraction = max(0.0, min(1.0, fraction))

    delta_seconds = (t_1 - t_0).total_seconds() * fraction

    return t_0 + timedelta(seconds=delta_seconds)


def refine_local_maximum_parabolic(
    rows: list[dict],
    value_key: str,
    index: int,
) -> tuple[datetime, float]:
    """
    Refine local maximum using a three-point parabolic interpolation.

    This avoids reporting only the nearest sampled minute.
    """

    current_row = rows[index]
    current_time = current_row["time_local"]
    current_value = current_row[value_key]

    if index <= 0 or index >= len(rows) - 1:
        return current_time, current_value

    previous_row = rows[index - 1]
    next_row = rows[index + 1]

    y_left = previous_row[value_key]
    y_mid = current_value
    y_right = next_row[value_key]

    a = 0.5 * (y_left + y_right) - y_mid
    b = 0.5 * (y_right - y_left)
    c = y_mid

    if a >= 0.0 or abs(a) < 1e-15:
        return current_time, current_value

    offset = -b / (2.0 * a)

    if offset < -1.0 or offset > 1.0:
        return current_time, current_value

    refined_value = a * offset * offset + b * offset + c

    step_seconds = (
        next_row["time_local"] - current_row["time_local"]
    ).total_seconds()

    refined_time = current_time + timedelta(seconds=offset * step_seconds)

    return refined_time, refined_value


def format_time(dt: datetime | None) -> str:
    if dt is None:
        return ""

    return dt.strftime("%H:%M:%S")


def format_float(value: float) -> str:
    """
    Keep enough precision for engineering review without fake integer-like output.
    """

    return f"{value:.9f}"


def summarize_day(
    day: date,
    rows: list[dict],
) -> dict:
    positive_indices = [
        index for index, row in enumerate(rows)
        if row["mu_front"] > 0.0
    ]

    visible_indices = [
        index for index, row in enumerate(rows)
        if row["altitude_deg"] > 0.0
    ]

    daily_integral_mu_h = calculate_trapezoidal_integral_mu_h(rows)

    if positive_indices:
        first_positive_index = positive_indices[0]
        last_positive_index = positive_indices[-1]

        if first_positive_index > 0:
            front_start_time = interpolate_threshold_crossing_time(
                row_0=rows[first_positive_index - 1],
                row_1=rows[first_positive_index],
                value_key="mu_front",
                threshold=0.0,
            )
        else:
            front_start_time = rows[first_positive_index]["time_local"]

        if last_positive_index < len(rows) - 1:
            front_end_time = interpolate_threshold_crossing_time(
                row_0=rows[last_positive_index],
                row_1=rows[last_positive_index + 1],
                value_key="mu_front",
                threshold=0.0,
            )
        else:
            front_end_time = rows[last_positive_index]["time_local"]

        front_illumination_duration_h = (
            front_end_time - front_start_time
        ).total_seconds() / 3600.0

        max_mu_index = max(
            positive_indices,
            key=lambda index: rows[index]["mu_front"],
        )

        time_of_max_mu, max_mu = refine_local_maximum_parabolic(
            rows=rows,
            value_key="mu_front",
            index=max_mu_index,
        )

    else:
        front_start_time = None
        front_end_time = None
        front_illumination_duration_h = 0.0
        max_mu = 0.0
        time_of_max_mu = None

    if visible_indices:
        max_altitude_index = max(
            visible_indices,
            key=lambda index: rows[index]["altitude_deg"],
        )

        time_of_max_solar_altitude, max_solar_altitude_deg = refine_local_maximum_parabolic(
            rows=rows,
            value_key="altitude_deg",
            index=max_altitude_index,
        )

    else:
        max_solar_altitude_deg = 0.0
        time_of_max_solar_altitude = None

    return {
        "date": day.isoformat(),
        "front_illumination_start": format_time(front_start_time),
        "front_illumination_end": format_time(front_end_time),
        "front_illumination_duration_h": format_float(front_illumination_duration_h),
        "max_mu": format_float(max_mu),
        "time_of_max_mu": format_time(time_of_max_mu),
        "daily_integral_mu_h": format_float(daily_integral_mu_h),
        "max_solar_altitude_deg": format_float(max_solar_altitude_deg),
        "time_of_max_solar_altitude": format_time(time_of_max_solar_altitude),
    }


def write_csv(
    rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "date",
        "front_illumination_start",
        "front_illumination_end",
        "front_illumination_duration_h",
        "max_mu",
        "time_of_max_mu",
        "daily_integral_mu_h",
        "max_solar_altitude_deg",
        "time_of_max_solar_altitude",
    ]

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


def plot_annual_front_side_exposure(
    annual_rows: list[dict],
    output_path: Path,
) -> None:
    days_of_year = list(range(1, len(annual_rows) + 1))

    durations_h = [
        float(row["front_illumination_duration_h"])
        for row in annual_rows
    ]

    integrals_mu_h = [
        float(row["daily_integral_mu_h"])
        for row in annual_rows
    ]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(
        days_of_year,
        durations_h,
        linewidth=2.0,
        label="Interpolated front-side illumination duration",
    )

    ax1.set_xlabel("Day of year, 2026")
    ax1.set_ylabel("Front-side illumination duration, h")
    ax1.set_xlim(1, 365)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        days_of_year,
        integrals_mu_h,
        linewidth=2.0,
        linestyle="--",
        label="Trapezoidal daily integral of μ",
    )

    ax2.set_ylabel("Daily integral of μ, h")

    ax1.set_title(
        "Annual front-side solar exposure for ROT-54/2.6 — 2026"
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved figure: {output_path}")


def print_key_days(
    annual_rows: list[dict],
) -> None:
    max_duration_row = max(
        annual_rows,
        key=lambda row: float(row["front_illumination_duration_h"]),
    )

    max_integral_row = max(
        annual_rows,
        key=lambda row: float(row["daily_integral_mu_h"]),
    )

    max_mu_row = max(
        annual_rows,
        key=lambda row: float(row["max_mu"]),
    )

    print("")
    print("Annual verification summary")
    print("---------------------------")
    print(f"Maximum duration date:  {max_duration_row['date']}")
    print(f"Maximum duration:       {max_duration_row['front_illumination_duration_h']} h")
    print("")
    print(f"Maximum integral date:  {max_integral_row['date']}")
    print(f"Maximum integral mu*h:  {max_integral_row['daily_integral_mu_h']}")
    print("")
    print(f"Maximum mu date:        {max_mu_row['date']}")
    print(f"Maximum mu:             {max_mu_row['max_mu']}")
    print(f"Time of maximum mu:     {max_mu_row['time_of_max_mu']}")


def main() -> None:
    year = 2026

    # Final engineering mode:
    # 60 s gives non-coarse annual values without making the run unnecessarily heavy.
    step_seconds = 60

    astronomy_engine = AstronomyEngine(
        latitude_deg=ORGOV_LATITUDE_DEG,
        longitude_deg=ORGOV_LONGITUDE_DEG,
        altitude_m=ORGOV_ALTITUDE_M,
        timezone=ARMENIA_TZ,
    )

    exposure_engine = SolarExposureEngine(
        reflector_tilt_south_deg=15.0,
    )

    annual_rows = []

    all_dates = generate_dates_for_year(year)

    for index, day in enumerate(all_dates, start=1):
        rows = calculate_day_exposure_rows(
            astronomy_engine=astronomy_engine,
            exposure_engine=exposure_engine,
            day=day,
            step_seconds=step_seconds,
        )

        summary_row = summarize_day(
            day=day,
            rows=rows,
        )

        annual_rows.append(summary_row)

        if index % 25 == 0 or index == len(all_dates):
            print(f"Processed {index}/{len(all_dates)} days")

    write_csv(
        rows=annual_rows,
        output_path=CSV_OUTPUT_PATH,
    )

    plot_annual_front_side_exposure(
        annual_rows=annual_rows,
        output_path=FIGURE_OUTPUT_PATH,
    )

    print_key_days(
        annual_rows=annual_rows,
    )


if __name__ == "__main__":
    main()
