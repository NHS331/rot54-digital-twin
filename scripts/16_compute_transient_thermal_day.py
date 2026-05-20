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
from rot54.thermal_steady import WeatherCase
from rot54.thermal_transient import (
    build_transient_thermal_parameters,
    run_transient_day,
    save_transient_outputs,
)


def wind_code(wind_speed_m_s: float) -> str:
    if abs(wind_speed_m_s - int(wind_speed_m_s)) < 1e-12:
        return f"v{int(wind_speed_m_s):02d}"

    return "v" + f"{wind_speed_m_s:.2f}".replace(".", "p")


def get_weather_case(
    case_key: str,
    wind_speed_m_s: float,
    thermal_steady_config: dict,
) -> WeatherCase:
    weather_map = thermal_steady_config["thermal_steady"]["seasonal_weather"]

    if case_key not in weather_map:
        raise KeyError(
            f"No weather scenario found for {case_key} in thermal_steady_config.yaml"
        )

    raw = weather_map[case_key]

    return WeatherCase(
        air_temperature_C=float(raw["air_temperature_C"]),
        sky_temperature_C=float(raw["sky_temperature_C"]),
        wind_speed_m_s=float(wind_speed_m_s),
    )


def plot_timeseries(
    timeseries: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))

    x = pd.to_datetime(timeseries["local_time"])
    x_labels = x.dt.strftime("%H:%M")

    ax.plot(
        range(len(timeseries)),
        timeseries["deltaT_air_max_C"],
        marker="o",
        markersize=2,
        label="max Delta T relative to air",
    )

    ax.plot(
        range(len(timeseries)),
        timeseries["deltaT_air_mean_C"],
        marker="o",
        markersize=2,
        label="mean Delta T relative to air",
    )

    step = max(1, len(timeseries) // 12)
    tick_positions = list(range(0, len(timeseries), step))
    tick_labels = x_labels.iloc[tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_title(title)
    ax.set_xlabel("Local time, Asia/Yerevan")
    ax.set_ylabel("Temperature difference, °C")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_snapshot_temperature(
    snapshot: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        snapshot["x_m"],
        snapshot["y_m"],
        c=snapshot["surface_temperature_C"],
        s=2,
        linewidths=0,
    )

    ax.set_title(title)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Surface temperature, °C")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_snapshot_deltaT_mean(
    snapshot: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        snapshot["x_m"],
        snapshot["y_m"],
        c=snapshot["deltaT_aperture_mean_C"],
        s=2,
        linewidths=0,
    )

    ax.set_title(title)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Delta T relative to aperture mean, °C")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    main_config = load_config(PROJECT_ROOT / "configs" / "rot54_config.yaml")
    transient_config = load_config(PROJECT_ROOT / "configs" / "thermal_transient_config.yaml")
    thermal_steady_config = load_config(PROJECT_ROOT / "configs" / "thermal_steady_config.yaml")

    params = build_transient_thermal_parameters(
        main_config=main_config,
        transient_config=transient_config,
    )

    geometry_csv = PROJECT_ROOT / main_config["geometry_grid"]["output_csv"]
    solar_csv = PROJECT_ROOT / main_config["solar_outputs"]["seasonal_positions_csv"]

    if not geometry_csv.exists():
        raise FileNotFoundError(
            f"Geometry grid not found: {geometry_csv}. "
            "Run scripts/03_generate_mirror_grid.py first."
        )

    if not solar_csv.exists():
        raise FileNotFoundError(
            f"Solar positions not found: {solar_csv}. "
            "Run scripts/05_generate_solar_tables.py first."
        )

    if not (PROJECT_ROOT / "configs" / "shadow_v2_config.yaml").exists():
        raise FileNotFoundError(
            "configs/shadow_v2_config.yaml not found. "
            "Run Step 4.3b first."
        )

    mirror_grid = pd.read_csv(geometry_csv)
    solar_table = pd.read_csv(solar_csv)

    raw = transient_config["thermal_transient"]
    output_dir = PROJECT_ROOT / raw["output_dir"]
    figures_dir = PROJECT_ROOT / raw["figures_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cases = [str(v) for v in raw["cases"]]
    wind_speeds = [float(v) for v in raw["wind_speeds_m_s"]]

    all_final_rows: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []

    for case_key in cases:
        case_solar = solar_table[solar_table["case_key"] == case_key].copy()

        if case_solar.empty:
            raise ValueError(f"No solar rows found for case: {case_key}")

        case_label = str(case_solar["case_label"].iloc[0])

        for wind_speed in wind_speeds:
            wcode = wind_code(wind_speed)

            weather = get_weather_case(
                case_key=case_key,
                wind_speed_m_s=wind_speed,
                thermal_steady_config=thermal_steady_config,
            )

            print("")
            print(f"Running transient model: {case_label}, wind={wind_speed:.1f} m/s")

            timeseries, snapshots, validation = run_transient_day(
                case_key=case_key,
                case_label=case_label,
                solar_case_full=case_solar,
                mirror_grid=mirror_grid,
                weather=weather,
                main_config=main_config,
                transient_config=transient_config,
                params=params,
            )

            save_transient_outputs(
                timeseries=timeseries,
                snapshots=snapshots,
                output_dir=output_dir,
                case_key=case_key,
                wind_code=wcode,
            )

            validations.append(validation)

            peak_row = timeseries.loc[timeseries["surface_temperature_max_C"].idxmax()].to_dict()
            peak_row["wind_code"] = wcode
            peak_row["daily_peak_local_time"] = peak_row["local_time"]
            all_final_rows.append(peak_row)

            timeseries_png = figures_dir / f"transient_deltaT_air_max_{case_key}_{wcode}.png"

            plot_timeseries(
                timeseries=timeseries,
                output_path=timeseries_png,
                title=(
                    f"ROT-54/2.6 Transient Thermal Response\n"
                    f"{case_label}, wind={wind_speed:.1f} m/s"
                ),
            )

            for snapshot_code, snapshot in snapshots.items():
                local_time = str(snapshot["selected_local_time"].iloc[0])

                temp_png = figures_dir / (
                    f"transient_surface_temperature_{case_key}_{snapshot_code}_{wcode}.png"
                )

                delta_png = figures_dir / (
                    f"transient_deltaT_aperture_mean_{case_key}_{snapshot_code}_{wcode}.png"
                )

                title_base = (
                    f"ROT-54/2.6 Transient Thermal Snapshot\n"
                    f"{case_label}, {snapshot_code}, {local_time}\n"
                    f"wind={wind_speed:.1f} m/s"
                )

                plot_snapshot_temperature(
                    snapshot=snapshot,
                    output_path=temp_png,
                    title=title_base,
                )

                plot_snapshot_deltaT_mean(
                    snapshot=snapshot,
                    output_path=delta_png,
                    title=title_base + "\nAnomaly relative to aperture mean",
                )

            print(f"  Timeseries saved for {case_key} / {wcode}")
            print(f"  Peak local time: {peak_row['daily_peak_local_time']}")
            print(f"  Peak max surface temperature: {peak_row['surface_temperature_max_C']:.3f} °C")
            print(f"  Peak max Delta T relative to air: {peak_row['deltaT_air_max_C']:.3f} °C")

    summary_df = pd.DataFrame(all_final_rows)
    validation_df = pd.DataFrame(validations)

    summary_csv = output_dir / "transient_summary.csv"
    validation_csv = output_dir / "transient_validation.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    validation_df.to_csv(validation_csv, index=False, encoding="utf-8")

    print("")
    print("Transient thermal model completed.")
    print(f"Summary: {summary_csv}")
    print(f"Validation: {validation_csv}")
    print("")
    print("Validation:")
    print(validation_df.to_string(index=False))

    if not validation_df["all_transient_checks_ok"].all():
        raise ValueError(
            "At least one transient validation check failed. "
            "Inspect outputs/thermal_transient/transient_validation.csv"
        )


if __name__ == "__main__":
    main()
