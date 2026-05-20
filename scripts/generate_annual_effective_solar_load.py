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
from rot54.solar import SolarExposureEngine, ClearSkyIrradianceModel
from rot54.weather import WeatherCoefficientModel


ARMENIA_TZ = ZoneInfo("Asia/Yerevan")


# Working coordinates for the ROT-54/2.6 site near Orgov / Aragats Scientific Center.
ORGOV_LATITUDE_DEG = 40.35
ORGOV_LONGITUDE_DEG = 44.25
ORGOV_ALTITUDE_M = 1711.0


OUTPUT_DIR = PROJECT_ROOT / "outputs"

CSV_OUTPUT_PATH = OUTPUT_DIR / "effective_solar_load_2026.csv"
FIGURE_OUTPUT_PATH = OUTPUT_DIR / "annual_effective_solar_load_2026.png"


def generate_dates_for_year(
    year: int,
) -> list[date]:
    start = date(year, 1, 1)
    end = date(year, 12, 31)

    days = []
    current = start

    while current <= end:
        days.append(current)
        current += timedelta(days=1)

    return days


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


def calculate_day_rows(
    astronomy_engine: AstronomyEngine,
    exposure_engine: SolarExposureEngine,
    irradiance_model: ClearSkyIrradianceModel,
    weather_model: WeatherCoefficientModel,
    day: date,
    step_seconds: int,
) -> list[dict]:
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

    day_of_year = day.timetuple().tm_yday

    k_weather = weather_model.coefficient_for_date(day)

    rows = []

    for index, dt_local in enumerate(times):
        altitude_deg = float(altitudes_deg[index])
        azimuth_deg = float(azimuths_deg[index])
        zenith_deg = 90.0 - altitude_deg

        exposure = exposure_engine.calculate_front_side_exposure(
            altitude_deg=altitude_deg,
            azimuth_deg=azimuth_deg,
        )

        mu_front = exposure["mu_front"]

        dni_clear_w_m2 = irradiance_model.direct_normal_irradiance_clear_sky(
            altitude_deg=altitude_deg,
            day_of_year=day_of_year,
        )

        clear_projected_load_w_m2 = dni_clear_w_m2 * mu_front

        effective_load_w_m2 = clear_projected_load_w_m2 * k_weather

        row = {
            "time_local": dt_local,
            "altitude_deg": altitude_deg,
            "azimuth_deg": azimuth_deg,
            "zenith_deg": zenith_deg,
            "mu_front": mu_front,
            "dni_clear_w_m2": dni_clear_w_m2,
            "clear_projected_load_w_m2": clear_projected_load_w_m2,
            "k_weather": k_weather,
            "effective_load_w_m2": effective_load_w_m2,
            "is_sun_above_horizon": exposure["is_sun_above_horizon"],
            "is_front_side_illuminated": exposure["is_front_side_illuminated"],
        }

        rows.append(row)

    return rows


def trapezoidal_integral_h(
    rows: list[dict],
    value_key: str,
) -> float:
    """
    Integrate a time series over the day.

    If value_key is W/m^2, output is Wh/m^2.
    If value_key is dimensionless, output is h.
    """

    if len(rows) < 2:
        return 0.0

    integral = 0.0

    for index in range(len(rows) - 1):
        row_0 = rows[index]
        row_1 = rows[index + 1]

        dt_hours = (
            row_1["time_local"] - row_0["time_local"]
        ).total_seconds() / 3600.0

        y_0 = row_0[value_key]
        y_1 = row_1[value_key]

        integral += 0.5 * (y_0 + y_1) * dt_hours

    return integral


def interpolate_threshold_crossing_time(
    row_0: dict,
    row_1: dict,
    value_key: str,
    threshold: float,
) -> datetime:
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


def format_time(
    dt: datetime | None,
) -> str:
    if dt is None:
        return ""

    return dt.strftime("%H:%M:%S")


def format_float(
    value: float,
) -> str:
    return f"{value:.9f}"


def summarize_day(
    day: date,
    rows: list[dict],
) -> dict:
    positive_indices = [
        index for index, row in enumerate(rows)
        if row["effective_load_w_m2"] > 0.0
    ]

    visible_indices = [
        index for index, row in enumerate(rows)
        if row["altitude_deg"] > 0.0
    ]

    daily_mu_integral_h = trapezoidal_integral_h(
        rows=rows,
        value_key="mu_front",
    )

    daily_clear_projected_energy_wh_m2 = trapezoidal_integral_h(
        rows=rows,
        value_key="clear_projected_load_w_m2",
    )

    daily_effective_energy_wh_m2 = trapezoidal_integral_h(
        rows=rows,
        value_key="effective_load_w_m2",
    )

    k_weather = rows[0]["k_weather"] if rows else 0.0

    if positive_indices:
        first_positive_index = positive_indices[0]
        last_positive_index = positive_indices[-1]

        if first_positive_index > 0:
            load_start_time = interpolate_threshold_crossing_time(
                row_0=rows[first_positive_index - 1],
                row_1=rows[first_positive_index],
                value_key="effective_load_w_m2",
                threshold=0.0,
            )
        else:
            load_start_time = rows[first_positive_index]["time_local"]

        if last_positive_index < len(rows) - 1:
            load_end_time = interpolate_threshold_crossing_time(
                row_0=rows[last_positive_index],
                row_1=rows[last_positive_index + 1],
                value_key="effective_load_w_m2",
                threshold=0.0,
            )
        else:
            load_end_time = rows[last_positive_index]["time_local"]

        load_duration_h = (
            load_end_time - load_start_time
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

        max_clear_projected_index = max(
            positive_indices,
            key=lambda index: rows[index]["clear_projected_load_w_m2"],
        )

        time_of_max_clear_projected, max_clear_projected_load_w_m2 = refine_local_maximum_parabolic(
            rows=rows,
            value_key="clear_projected_load_w_m2",
            index=max_clear_projected_index,
        )

        max_effective_index = max(
            positive_indices,
            key=lambda index: rows[index]["effective_load_w_m2"],
        )

        time_of_max_effective, max_effective_load_w_m2 = refine_local_maximum_parabolic(
            rows=rows,
            value_key="effective_load_w_m2",
            index=max_effective_index,
        )

    else:
        load_start_time = None
        load_end_time = None
        load_duration_h = 0.0

        max_mu = 0.0
        time_of_max_mu = None

        max_clear_projected_load_w_m2 = 0.0
        time_of_max_clear_projected = None

        max_effective_load_w_m2 = 0.0
        time_of_max_effective = None

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
        "k_weather": format_float(k_weather),
        "load_start": format_time(load_start_time),
        "load_end": format_time(load_end_time),
        "load_duration_h": format_float(load_duration_h),
        "max_mu": format_float(max_mu),
        "time_of_max_mu": format_time(time_of_max_mu),
        "daily_mu_integral_h": format_float(daily_mu_integral_h),
        "max_clear_projected_load_w_m2": format_float(max_clear_projected_load_w_m2),
        "time_of_max_clear_projected_load": format_time(time_of_max_clear_projected),
        "daily_clear_projected_energy_wh_m2": format_float(daily_clear_projected_energy_wh_m2),
        "max_effective_load_w_m2": format_float(max_effective_load_w_m2),
        "time_of_max_effective_load": format_time(time_of_max_effective),
        "daily_effective_energy_wh_m2": format_float(daily_effective_energy_wh_m2),
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
        "k_weather",
        "load_start",
        "load_end",
        "load_duration_h",
        "max_mu",
        "time_of_max_mu",
        "daily_mu_integral_h",
        "max_clear_projected_load_w_m2",
        "time_of_max_clear_projected_load",
        "daily_clear_projected_energy_wh_m2",
        "max_effective_load_w_m2",
        "time_of_max_effective_load",
        "daily_effective_energy_wh_m2",
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


def plot_annual_effective_solar_load(
    annual_rows: list[dict],
    output_path: Path,
) -> None:
    days_of_year = list(range(1, len(annual_rows) + 1))

    daily_effective_energy = [
        float(row["daily_effective_energy_wh_m2"])
        for row in annual_rows
    ]

    max_effective_load = [
        float(row["max_effective_load_w_m2"])
        for row in annual_rows
    ]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(
        days_of_year,
        daily_effective_energy,
        linewidth=2.0,
        label="Daily effective energy",
    )

    ax1.set_xlabel("Day of year, 2026")
    ax1.set_ylabel("Daily effective solar energy, Wh/m²")
    ax1.set_xlim(1, 365)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        days_of_year,
        max_effective_load,
        linewidth=2.0,
        linestyle="--",
        label="Maximum effective load",
    )

    ax2.set_ylabel("Maximum effective solar load, W/m²")

    ax1.set_title(
        "Annual effective front-side solar load for ROT-54/2.6 — 2026"
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
    max_energy_row = max(
        annual_rows,
        key=lambda row: float(row["daily_effective_energy_wh_m2"]),
    )

    max_load_row = max(
        annual_rows,
        key=lambda row: float(row["max_effective_load_w_m2"]),
    )

    max_clear_row = max(
        annual_rows,
        key=lambda row: float(row["daily_clear_projected_energy_wh_m2"]),
    )

    print("")
    print("Annual effective-load verification summary")
    print("------------------------------------------")
    print(f"Maximum effective energy date:       {max_energy_row['date']}")
    print(f"Daily effective energy:              {max_energy_row['daily_effective_energy_wh_m2']} Wh/m^2")
    print(f"K_weather:                           {max_energy_row['k_weather']}")
    print("")
    print(f"Maximum effective load date:         {max_load_row['date']}")
    print(f"Maximum effective load:              {max_load_row['max_effective_load_w_m2']} W/m^2")
    print(f"Time of maximum effective load:      {max_load_row['time_of_max_effective_load']}")
    print("")
    print(f"Maximum clear projected energy date: {max_clear_row['date']}")
    print(f"Clear projected energy:              {max_clear_row['daily_clear_projected_energy_wh_m2']} Wh/m^2")


def main() -> None:
    year = 2026
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

    irradiance_model = ClearSkyIrradianceModel(
        solar_constant_w_m2=1361.0,
        clear_sky_transmittance=0.72,
    )

    weather_model = WeatherCoefficientModel()

    annual_rows = []

    all_dates = generate_dates_for_year(year)

    for index, day in enumerate(all_dates, start=1):
        day_rows = calculate_day_rows(
            astronomy_engine=astronomy_engine,
            exposure_engine=exposure_engine,
            irradiance_model=irradiance_model,
            weather_model=weather_model,
            day=day,
            step_seconds=step_seconds,
        )

        summary_row = summarize_day(
            day=day,
            rows=day_rows,
        )

        annual_rows.append(summary_row)

        if index % 25 == 0 or index == len(all_dates):
            print(f"Processed {index}/{len(all_dates)} days")

    write_csv(
        rows=annual_rows,
        output_path=CSV_OUTPUT_PATH,
    )

    plot_annual_effective_solar_load(
        annual_rows=annual_rows,
        output_path=FIGURE_OUTPUT_PATH,
    )

    print_key_days(
        annual_rows=annual_rows,
    )


if __name__ == "__main__":
    main()
