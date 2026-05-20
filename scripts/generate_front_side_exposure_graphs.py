from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt

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


def generate_local_day_times(
    date_string: str,
    step_minutes: int = 10,
) -> list[datetime]:
    year, month, day = map(int, date_string.split("-"))

    start = datetime(
        year,
        month,
        day,
        0,
        0,
        0,
        tzinfo=ARMENIA_TZ,
    )

    end = datetime(
        year,
        month,
        day,
        23,
        59,
        0,
        tzinfo=ARMENIA_TZ,
    )

    times = []
    current = start

    while current <= end:
        times.append(current)
        current += timedelta(minutes=step_minutes)

    return times


def calculate_front_side_exposure_for_day(
    astronomy_engine: AstronomyEngine,
    exposure_engine: SolarExposureEngine,
    date_string: str,
    step_minutes: int = 10,
) -> list[dict]:
    times = generate_local_day_times(
        date_string=date_string,
        step_minutes=step_minutes,
    )

    results = []

    for dt in times:
        solar_geometry = astronomy_engine.sun_alt_az(dt)

        exposure = exposure_engine.calculate_front_side_exposure(
            altitude_deg=solar_geometry["altitude_deg"],
            azimuth_deg=solar_geometry["azimuth_deg"],
        )

        row = {
            "time_local": solar_geometry["time_local"],
            "altitude_deg": solar_geometry["altitude_deg"],
            "azimuth_deg": solar_geometry["azimuth_deg"],
            "zenith_deg": solar_geometry["zenith_deg"],
            "raw_dot": exposure["raw_dot"],
            "mu_front": exposure["mu_front"],
            "is_sun_above_horizon": exposure["is_sun_above_horizon"],
            "is_front_side_illuminated": exposure["is_front_side_illuminated"],
        }

        results.append(row)

    return results


def plot_front_side_exposure_graph(
    date_string: str,
    exposure_data: list[dict],
    output_dir: Path,
) -> None:
    visible_data = [
        row for row in exposure_data
        if row["is_sun_above_horizon"]
    ]

    if not visible_data:
        raise RuntimeError(f"No visible solar data for date {date_string}")

    local_hours = [
        row["time_local"].hour + row["time_local"].minute / 60.0
        for row in visible_data
    ]

    altitudes_deg = [
        row["altitude_deg"]
        for row in visible_data
    ]

    mu_front_values = [
        row["mu_front"]
        for row in visible_data
    ]

    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(
        local_hours,
        mu_front_values,
        linewidth=2.2,
        label="Front-side exposure coefficient μ",
    )

    ax1.set_xlabel("Local time in Armenia, h")
    ax1.set_ylabel("Front-side exposure coefficient μ")
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        local_hours,
        altitudes_deg,
        linewidth=1.8,
        linestyle="--",
        label="Solar altitude",
    )

    ax2.set_ylabel("Solar altitude, deg")

    ax1.set_title(
        f"Front-side solar exposure for ROT-54/2.6 — {date_string}"
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"front_side_exposure_{date_string}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def print_daily_summary(
    date_string: str,
    exposure_data: list[dict],
) -> None:
    illuminated_data = [
        row for row in exposure_data
        if row["is_front_side_illuminated"]
    ]

    if not illuminated_data:
        print(f"{date_string}: no front-side illumination")
        return

    first_time = illuminated_data[0]["time_local"]
    last_time = illuminated_data[-1]["time_local"]

    max_row = max(
        illuminated_data,
        key=lambda row: row["mu_front"],
    )

    duration_hours = len(illuminated_data) * 10 / 60.0

    print("")
    print(f"Date: {date_string}")
    print(f"Front-side illumination starts: {first_time.strftime('%H:%M')}")
    print(f"Front-side illumination ends:   {last_time.strftime('%H:%M')}")
    print(f"Approximate duration:           {duration_hours:.2f} h")
    print(f"Maximum μ:                      {max_row['mu_front']:.4f}")
    print(f"Time of maximum μ:              {max_row['time_local'].strftime('%H:%M')}")


def main() -> None:
    astronomy_engine = AstronomyEngine(
        latitude_deg=ORGOV_LATITUDE_DEG,
        longitude_deg=ORGOV_LONGITUDE_DEG,
        altitude_m=ORGOV_ALTITUDE_M,
        timezone=ARMENIA_TZ,
    )

    exposure_engine = SolarExposureEngine(
        reflector_tilt_south_deg=15.0,
    )

    dates = [
        "2026-03-20",
        "2026-06-21",
        "2026-12-21",
    ]

    for date_string in dates:
        exposure_data = calculate_front_side_exposure_for_day(
            astronomy_engine=astronomy_engine,
            exposure_engine=exposure_engine,
            date_string=date_string,
            step_minutes=10,
        )

        plot_front_side_exposure_graph(
            date_string=date_string,
            exposure_data=exposure_data,
            output_dir=OUTPUT_DIR,
        )

        print_daily_summary(
            date_string=date_string,
            exposure_data=exposure_data,
        )


if __name__ == "__main__":
    main()
