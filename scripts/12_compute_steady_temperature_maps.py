from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rot54.config import load_config
from rot54.thermal_steady import (
    build_steady_thermal_parameters,
    compute_steady_temperature_map,
    get_weather_case_for_flux,
    save_steady_temperature_map,
    summarize_steady_temperature,
    validate_steady_temperature,
)


def wind_code(wind_speed_m_s: float) -> str:
    """
    Convert wind speed into a filename-safe code.
    """
    if abs(wind_speed_m_s - int(wind_speed_m_s)) < 1e-12:
        return f"v{int(wind_speed_m_s):02d}"

    text = f"{wind_speed_m_s:.2f}".replace(".", "p")
    return f"v{text}"


def parse_flux_filename(path: Path) -> tuple[str, str]:
    """
    Parse:
        absorbed_flux_summer_solstice_morning.csv

    into:
        case_key = summer_solstice
        time_code = morning
    """
    name = path.stem

    prefix = "absorbed_flux_"

    if not name.startswith(prefix):
        raise ValueError(f"Unexpected absorbed flux filename: {path.name}")

    rest = name[len(prefix):]

    match = re.match(r"(.+)_(morning|axis|evening)$", rest)

    if not match:
        raise ValueError(
            f"Could not parse case_key and time_code from filename: {path.name}"
        )

    case_key = match.group(1)
    time_code = match.group(2)

    return case_key, time_code


def plot_surface_temperature(
    thermal: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot absolute surface temperature in Celsius.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        thermal["x_m"],
        thermal["y_m"],
        c=thermal["surface_temperature_C"],
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


def plot_deltaT_air(
    thermal: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot temperature difference relative to air temperature.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        thermal["x_m"],
        thermal["y_m"],
        c=thermal["deltaT_air_C"],
        s=2,
        linewidths=0,
    )

    ax.set_title(title)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Delta T relative to air, °C")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_deltaT_aperture_mean(
    thermal: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot temperature anomaly relative to full-aperture mean temperature.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        thermal["x_m"],
        thermal["y_m"],
        c=thermal["deltaT_aperture_mean_C"],
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
    thermal_config = load_config(PROJECT_ROOT / "configs" / "thermal_steady_config.yaml")

    params = build_steady_thermal_parameters(
        main_config=main_config,
        thermal_config=thermal_config,
    )

    raw = thermal_config["thermal_steady"]

    irradiation_dir = PROJECT_ROOT / "outputs" / "irradiation"
    output_dir = PROJECT_ROOT / raw["output_dir"]
    figures_dir = PROJECT_ROOT / raw["figures_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    wind_speeds = [float(v) for v in raw["wind_speeds_m_s"]]

    flux_files = sorted(irradiation_dir.glob("absorbed_flux_*_*.csv"))

    flux_files = [
        path for path in flux_files
        if path.name not in [
            "absorbed_flux_summary.csv",
            "absorbed_flux_validation.csv",
        ]
    ]

    if not flux_files:
        raise FileNotFoundError(
            f"No absorbed flux maps found in {irradiation_dir}. "
            "Run scripts/10_compute_absorbed_flux_maps.py first."
        )

    summaries: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []

    for flux_path in flux_files:
        case_key, time_code = parse_flux_filename(flux_path)

        flux = pd.read_csv(flux_path)

        case_label = str(flux["case_label"].iloc[0]) if "case_label" in flux.columns else case_key
        selected_local_time = (
            str(flux["selected_local_time"].iloc[0])
            if "selected_local_time" in flux.columns
            else ""
        )

        for wind_speed in wind_speeds:
            weather = get_weather_case_for_flux(
                case_key=case_key,
                wind_speed_m_s=wind_speed,
                thermal_config=thermal_config,
            )

            thermal = compute_steady_temperature_map(
                flux=flux,
                weather=weather,
                params=params,
            )

            wcode = wind_code(wind_speed)

            output_csv = output_dir / (
                f"steady_temperature_{case_key}_{time_code}_{wcode}.csv"
            )

            surface_png = figures_dir / (
                f"surface_temperature_{case_key}_{time_code}_{wcode}.png"
            )

            delta_air_png = figures_dir / (
                f"deltaT_air_{case_key}_{time_code}_{wcode}.png"
            )

            delta_mean_png = figures_dir / (
                f"deltaT_aperture_mean_{case_key}_{time_code}_{wcode}.png"
            )

            save_steady_temperature_map(
                thermal=thermal,
                output_path=output_csv,
            )

            title_base = (
                f"ROT-54/2.6 Steady Thermal Map\n"
                f"{case_label}, {time_code}, {selected_local_time}\n"
                f"v={wind_speed:.1f} m/s, "
                f"T_air={weather.air_temperature_C:.1f} °C, "
                f"T_sky={weather.sky_temperature_C:.1f} °C"
            )

            plot_surface_temperature(
                thermal=thermal,
                output_path=surface_png,
                title=title_base,
            )

            plot_deltaT_air(
                thermal=thermal,
                output_path=delta_air_png,
                title=title_base + "\nSurface temperature relative to air",
            )

            plot_deltaT_aperture_mean(
                thermal=thermal,
                output_path=delta_mean_png,
                title=title_base + "\nSurface temperature anomaly relative to aperture mean",
            )

            summaries.append(
                summarize_steady_temperature(
                    case_key=case_key,
                    case_label=case_label,
                    time_code=time_code,
                    selected_local_time=selected_local_time,
                    thermal=thermal,
                )
            )

            validations.append(
                validate_steady_temperature(
                    case_key=case_key,
                    time_code=time_code,
                    wind_speed_m_s=wind_speed,
                    thermal=thermal,
                    params=params,
                )
            )

            print(f"{case_label} / {time_code} / {wcode}: steady temperature calculated.")
            print(f"  Time: {selected_local_time}")
            print(f"  CSV: {output_csv}")
            print(f"  Surface temperature PNG: {surface_png}")
            print(f"  DeltaT air PNG: {delta_air_png}")
            print(f"  DeltaT aperture mean PNG: {delta_mean_png}")

    summary_df = pd.DataFrame(summaries)
    validation_df = pd.DataFrame(validations)

    summary_csv = output_dir / "steady_temperature_summary.csv"
    validation_csv = output_dir / "steady_temperature_validation.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    validation_df.to_csv(validation_csv, index=False, encoding="utf-8")

    print("")
    print("Steady thermal calculation completed.")
    print(f"Summary: {summary_csv}")
    print(f"Validation: {validation_csv}")
    print("")

    failed = validation_df[
        validation_df["all_steady_temperature_checks_ok"] != True
    ]

    for _, row in validation_df.iterrows():
        print(
            f"- {row['case_key']} / {row['time_code']} / "
            f"v={row['wind_speed_m_s']}: "
            f"ok={row['all_steady_temperature_checks_ok']}, "
            f"max_residual={row['max_abs_thermal_balance_residual_W_m2']:.3e} W/m^2"
        )

    if not failed.empty:
        raise ValueError(
            "At least one steady thermal validation check failed. "
            "Inspect outputs/thermal_steady/steady_temperature_validation.csv"
        )


if __name__ == "__main__":
    main()
