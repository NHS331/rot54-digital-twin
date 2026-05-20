from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rot54.config import load_config
from rot54.solar_geometry import (
    MirrorAxisParameters,
    SiteParameters,
    compute_solar_day_table,
    summarize_solar_day,
)


def plot_solar_quantity(
    table: pd.DataFrame,
    output_path: Path,
    y_column: str,
    y_label: str,
    title: str,
) -> None:
    """
    Plot one solar quantity for all seasonal cases.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    for case_key, case_data in table.groupby("case_key"):
        case_label = str(case_data["case_label"].iloc[0])

        x = pd.to_datetime(case_data["local_time"]).dt.strftime("%H:%M")
        y = case_data[y_column]

        # Plot using numeric index to avoid overcrowded time-axis labels.
        ax.plot(range(len(case_data)), y, label=case_label)

    step = max(1, len(table[table["case_key"] == table["case_key"].iloc[0]]) // 12)
    first_case = table[table["case_key"] == table["case_key"].iloc[0]].copy()
    tick_positions = list(range(0, len(first_case), step))
    tick_labels = pd.to_datetime(first_case["local_time"]).dt.strftime("%H:%M").iloc[tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_title(title)
    ax.set_xlabel("Local time, Asia/Yerevan")
    ax.set_ylabel(y_label)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "rot54_config.yaml"
    config = load_config(config_path)

    site_config = config["site"]
    mirror_config = config["main_reflector"]
    solar_config = config["solar_geometry"]
    output_config = config["solar_outputs"]

    site = SiteParameters(
        latitude_deg=float(site_config["latitude_deg"]),
        longitude_deg=float(site_config["longitude_deg"]),
        timezone=str(site_config["timezone"]),
    )

    mirror_axis = MirrorAxisParameters(
        axis_tilt_south_deg=float(mirror_config["axis_tilt_south_deg"]),
    )

    step_minutes = int(solar_config["time_step_minutes"])
    apparent_horizon_altitude_deg = float(
        solar_config["apparent_horizon_altitude_deg"]
    )

    all_tables: list[pd.DataFrame] = []
    all_summaries: list[dict[str, object]] = []

    for case_key, case_data in solar_config["seasonal_cases"].items():
        case_label = str(case_data["label"])
        date_iso = str(case_data["date"])
        expected_front_duration_h = float(case_data["expected_front_duration_h"])

        solar_day = compute_solar_day_table(
            case_key=case_key,
            case_label=case_label,
            date_iso=date_iso,
            site=site,
            mirror_axis=mirror_axis,
            step_minutes=step_minutes,
            apparent_horizon_altitude_deg=apparent_horizon_altitude_deg,
        )

        summary = summarize_solar_day(
            solar_day=solar_day,
            step_minutes=step_minutes,
            expected_front_duration_h=expected_front_duration_h,
        )

        all_tables.append(solar_day)
        all_summaries.append(summary)

    seasonal_table = pd.concat(all_tables, ignore_index=True)
    seasonal_summary = pd.DataFrame(all_summaries)

    positions_csv = PROJECT_ROOT / output_config["seasonal_positions_csv"]
    summary_csv = PROJECT_ROOT / output_config["seasonal_summary_csv"]

    positions_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    seasonal_table.to_csv(positions_csv, index=False, encoding="utf-8")
    seasonal_summary.to_csv(summary_csv, index=False, encoding="utf-8")

    plot_solar_quantity(
        table=seasonal_table,
        output_path=PROJECT_ROOT / output_config["altitude_figure_png"],
        y_column="solar_altitude_deg",
        y_label="Solar altitude, deg",
        title="ROT-54/2.6 Seasonal Solar Altitude, Local Armenia Time",
    )

    plot_solar_quantity(
        table=seasonal_table,
        output_path=PROJECT_ROOT / output_config["azimuth_figure_png"],
        y_column="solar_azimuth_deg",
        y_label="Solar azimuth, deg clockwise from north",
        title="ROT-54/2.6 Seasonal Solar Azimuth, Local Armenia Time",
    )

    plot_solar_quantity(
        table=seasonal_table,
        output_path=PROJECT_ROOT / output_config["axis_dot_figure_png"],
        y_column="axis_dot_sun",
        y_label="dot(axis, sun)",
        title="Alignment Between Sun Vector and Tilted Main Reflector Axis",
    )

    print("Seasonal solar geometry generated successfully.")
    print(f"Solar table: {positions_csv}")
    print(f"Solar summary: {summary_csv}")
    print("")
    print("Front-side illumination duration check:")
    for _, row in seasonal_summary.iterrows():
        print(
            f"- {row['case_label']}: "
            f"computed = {row['front_duration_h']:.3f} h, "
            f"expected = {row['expected_front_duration_h']:.3f} h, "
            f"error = {row['front_duration_error_h']:.3f} h"
        )


if __name__ == "__main__":
    main()
