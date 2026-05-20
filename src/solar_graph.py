from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.astronomy_engine import AstronomyEngine
from src.clock import (
    PROJECT_TIMEZONE,
    PROJECT_TIMEZONE_NAME,
    SimulationClock,
)
from src.observer import ObserverSite


@dataclass(frozen=True)
class SolarGraphConfig:
    """
    Configuration for one-day solar graph generation.
    """

    site_config_path: str = "configs/site.yaml"
    start_iso: str = "2026-06-21T00:00:00+04:00"
    end_iso: str = "2026-06-22T00:00:00+04:00"
    step_minutes: int = 10
    project_timezone: str = PROJECT_TIMEZONE_NAME
    output_table_path: str = "output/tables/sun_altaz_day.csv"
    output_altitude_figure_path: str = "output/figures/sun_altitude_day.png"
    output_azimuth_figure_path: str = "output/figures/sun_azimuth_day.png"


def build_sun_altaz_table(config: SolarGraphConfig) -> pd.DataFrame:
    """
    Build a table of Sun altitude, azimuth, and ENU vector over Armenia time.
    """

    observer = ObserverSite.from_yaml(config.site_config_path)
    engine = AstronomyEngine(observer)
    project_timezone = ZoneInfo(config.project_timezone)

    time_range = SimulationClock.make_range(
        start_iso=config.start_iso,
        end_iso=config.end_iso,
        step_minutes=config.step_minutes,
    )

    rows: list[dict[str, float | str | bool]] = []

    for astropy_time in time_range:
        dt_project = astropy_time.to_datetime(timezone=project_timezone)
        clock = SimulationClock(dt_project)

        sun = engine.sun_position(clock)

        rows.append(
            {
                "time_armenia": clock.current_time.isoformat(),
                "project_timezone": config.project_timezone,
                "sun_altitude_deg": sun.altitude_deg,
                "sun_azimuth_deg": sun.azimuth_deg,
                "sun_enu_x_east": float(sun.enu_vector[0]),
                "sun_enu_y_north": float(sun.enu_vector[1]),
                "sun_enu_z_up": float(sun.enu_vector[2]),
                "sun_above_horizon": sun.altitude_deg > 0.0,
            }
        )

    return pd.DataFrame(rows)


def save_sun_altaz_table(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save Sun AltAz table as CSV.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False, encoding="utf-8")


def _armenia_time_for_plot(df: pd.DataFrame) -> pd.Series:
    """
    Convert stored Armenia-time strings to timezone-free Armenia local values for plotting.
    """

    return (
        pd.to_datetime(df["time_armenia"], utc=True)
        .dt.tz_convert(PROJECT_TIMEZONE)
        .dt.tz_localize(None)
    )


def plot_sun_altitude(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Plot Sun altitude versus Armenia local time.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    time = _armenia_time_for_plot(df)

    plt.figure(figsize=(10, 5))
    plt.plot(time, df["sun_altitude_deg"])
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Armenia time")
    plt.ylabel("Sun altitude, deg")
    plt.title("Sun Altitude for ROT-54/2.6 Orgov Site, Armenia Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_sun_azimuth(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Plot Sun azimuth versus Armenia local time.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    time = _armenia_time_for_plot(df)

    plt.figure(figsize=(10, 5))
    plt.plot(time, df["sun_azimuth_deg"])
    plt.xlabel("Armenia time")
    plt.ylabel("Sun azimuth, deg")
    plt.title("Sun Azimuth for ROT-54/2.6 Orgov Site, Armenia Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def generate_solar_graph_outputs(config: SolarGraphConfig) -> pd.DataFrame:
    """
    Generate CSV table and PNG figures for Sun altitude and azimuth.
    """

    df = build_sun_altaz_table(config)

    save_sun_altaz_table(df, config.output_table_path)
    plot_sun_altitude(df, config.output_altitude_figure_path)
    plot_sun_azimuth(df, config.output_azimuth_figure_path)

    return df


def main() -> None:
    """
    Command-line entry point.
    """

    config = SolarGraphConfig()
    df = generate_solar_graph_outputs(config)

    print("Solar graph outputs generated.")
    print(f"Rows: {len(df)}")
    print(f"Project timezone: {config.project_timezone}")
    print(f"Table: {config.output_table_path}")
    print(f"Altitude figure: {config.output_altitude_figure_path}")
    print(f"Azimuth figure: {config.output_azimuth_figure_path}")


if __name__ == "__main__":
    main()