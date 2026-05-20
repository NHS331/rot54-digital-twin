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
    """
    Generate timezone-aware local Armenia times for one complete day.
    """

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


def calculate_solar_geometry_for_day(
    engine: AstronomyEngine,
    date_string: str,
    step_minutes: int = 10,
) -> list[dict]:
    """
    Calculate solar altitude, azimuth and zenith angle for one day.
    """

    times = generate_local_day_times(
        date_string=date_string,
        step_minutes=step_minutes,
    )

    results = []

    for dt in times:
        result = engine.sun_alt_az(dt)
        results.append(result)

    return results


def plot_solar_graph(
    date_string: str,
    solar_data: list[dict],
    output_dir: Path,
) -> None:
    """
    Save one solar geometry graph.

    The graph shows only the visible part of the day:
        solar altitude > 0 deg
    """

    visible_data = [
        row for row in solar_data
        if row["altitude_deg"] > 0.0
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

    azimuths_deg = [
        row["azimuth_deg"]
        for row in visible_data
    ]

    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.plot(
        local_hours,
        altitudes_deg,
        linewidth=2.0,
        label="Solar altitude",
    )

    ax1.set_xlabel("Local time in Armenia, h")
    ax1.set_ylabel("Solar altitude, deg")
    ax1.set_xlim(0, 24)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        local_hours,
        azimuths_deg,
        linewidth=2.0,
        linestyle="--",
        label="Solar azimuth",
    )

    ax2.set_ylabel("Solar azimuth, deg")

    ax1.set_title(
        f"Solar geometry for ROT-54/2.6 site — {date_string}"
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"solar_graph_{date_string}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main() -> None:
    engine = AstronomyEngine(
        latitude_deg=ORGOV_LATITUDE_DEG,
        longitude_deg=ORGOV_LONGITUDE_DEG,
        altitude_m=ORGOV_ALTITUDE_M,
        timezone=ARMENIA_TZ,
    )

    dates = [
        "2026-03-20",
        "2026-06-21",
        "2026-12-21",
    ]

    for date_string in dates:
        solar_data = calculate_solar_geometry_for_day(
            engine=engine,
            date_string=date_string,
            step_minutes=10,
        )

        plot_solar_graph(
            date_string=date_string,
            solar_data=solar_data,
            output_dir=OUTPUT_DIR,
        )


if __name__ == "__main__":
    main()
