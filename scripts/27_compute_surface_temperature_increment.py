from __future__ import annotations

import csv
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "surface_temperature_model.yaml"

STAGE_10_SCRIPT = PROJECT_ROOT / "scripts" / "26_compute_absorbed_solar_flux.py"

CONTROL_POINT_FLUX_CSV = PROJECT_ROOT / "outputs" / "solar_flux" / "control_point_absorbed_flux.csv"
PANEL_GRID_CSV = PROJECT_ROOT / "outputs" / "geometry" / "panel_grid_3738.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "surface_temperature"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "surface_temperature"

CONTROL_POINT_TEMPERATURE_CSV = OUTPUT_DIR / "control_point_surface_temperature.csv"
PANEL_TEMPERATURE_SUMMARY_CSV = OUTPUT_DIR / "panel_temperature_summary.csv"
SCENARIO_TEMPERATURE_SUMMARY_CSV = OUTPUT_DIR / "scenario_temperature_summary.csv"


@dataclass(frozen=True)
class FluxInputRow:
    scenario_key: str
    scenario_label: str
    date_iso: str
    day_of_year: int
    time_key: str
    hour_angle_deg: float
    solar_declination_deg: float
    panel_id: int
    control_point_id: str
    x: float
    y: float
    z: float
    mu_front: float
    chi: int
    dni_w_m2: float
    alpha_s: float
    q_incident_w_m2: float
    q_abs_w_m2: float
    shadow_source: str


@dataclass(frozen=True)
class TemperatureRow:
    scenario_key: str
    scenario_label: str
    date_iso: str
    day_of_year: int
    time_key: str
    hour_angle_deg: float
    solar_declination_deg: float
    panel_id: int
    control_point_id: str
    x: float
    y: float
    z: float
    mu_front: float
    chi: int
    dni_w_m2: float
    alpha_s: float
    q_abs_w_m2: float
    ambient_temperature_C: float
    wind_speed_m_s: float
    h_conv_w_m2_k: float
    h_rad_w_m2_k: float
    h_eff_w_m2_k: float
    delta_t_s_C: float
    surface_temperature_C: float
    shadow_source: str


@dataclass(frozen=True)
class PanelCentre:
    panel_id: int
    x: float
    y: float
    z: float
    radius: float


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    return data


def ensure_stage_10_exists() -> None:
    if CONTROL_POINT_FLUX_CSV.exists() and PANEL_GRID_CSV.exists():
        return

    if not STAGE_10_SCRIPT.exists():
        raise FileNotFoundError(
            "Stage 10 outputs are missing and scripts/26_compute_absorbed_solar_flux.py was not found."
        )

    subprocess.run(
        [sys.executable, str(STAGE_10_SCRIPT)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def read_flux_csv(path: Path) -> list[FluxInputRow]:
    rows: list[FluxInputRow] = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                FluxInputRow(
                    scenario_key=row["scenario_key"],
                    scenario_label=row["scenario_label"],
                    date_iso=row["date"],
                    day_of_year=int(row["day_of_year"]),
                    time_key=row["time_key"],
                    hour_angle_deg=float(row["hour_angle_deg"]),
                    solar_declination_deg=float(row["solar_declination_deg"]),
                    panel_id=int(row["panel_id"]),
                    control_point_id=row["control_point_id"],
                    x=float(row["x_m"]),
                    y=float(row["y_m"]),
                    z=float(row["z_m"]),
                    mu_front=float(row["mu_front"]),
                    chi=int(row["chi"]),
                    dni_w_m2=float(row["dni_w_m2"]),
                    alpha_s=float(row["alpha_s"]),
                    q_incident_w_m2=float(row["q_incident_w_m2"]),
                    q_abs_w_m2=float(row["q_abs_w_m2"]),
                    shadow_source=row["shadow_source"],
                )
            )

    return rows


def read_panel_centres(path: Path) -> dict[int, PanelCentre]:
    panels: dict[int, PanelCentre] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            panel_id = int(row["panel_id"])
            panels[panel_id] = PanelCentre(
                panel_id=panel_id,
                x=float(row["x_m"]),
                y=float(row["y_m"]),
                z=float(row["z_m"]),
                radius=float(row["projected_radius_m"]),
            )

    return panels


def get_environment(
    config: dict,
    scenario_key: str,
    time_key: str,
) -> tuple[float, float]:
    environment = config["surface_temperature_model"]["physical_model"]["environment"]

    if scenario_key not in environment:
        raise KeyError(f"Environment scenario was not found in config: {scenario_key}")

    if time_key not in environment[scenario_key]:
        raise KeyError(f"Environment time key was not found in config: {scenario_key}/{time_key}")

    item = environment[scenario_key][time_key]

    ambient_temperature_C = float(item["ambient_temperature_C"])
    wind_speed_m_s = float(item["wind_speed_m_s"])

    return ambient_temperature_C, wind_speed_m_s


def convection_coefficient_w_m2_k(wind_speed_m_s: float) -> float:
    if wind_speed_m_s < 0.0:
        raise ValueError("wind_speed_m_s must be non-negative.")

    return 5.7 + 3.8 * wind_speed_m_s


def radiative_linearized_coefficient_w_m2_k(
    ambient_temperature_C: float,
    emissivity_lw: float,
    sigma: float,
) -> float:
    ambient_temperature_K = ambient_temperature_C + 273.15

    if ambient_temperature_K <= 0.0:
        raise ValueError("Ambient temperature in Kelvin must be positive.")

    return 4.0 * emissivity_lw * sigma * ambient_temperature_K**3


def compute_temperature_rows(
    flux_rows: list[FluxInputRow],
    config: dict,
) -> list[TemperatureRow]:
    model = config["surface_temperature_model"]
    physical = model["physical_model"]
    numerical = model["numerical"]

    emissivity_lw = float(physical["surface_emissivity_epsilon_lw"])
    sigma = float(physical["stefan_boltzmann_w_m2_k4"])
    min_h_eff = float(numerical["min_h_eff_w_m2_k"])

    if not (0.0 < emissivity_lw <= 1.0):
        raise ValueError("surface_emissivity_epsilon_lw must be within (0, 1].")

    if sigma <= 0.0:
        raise ValueError("stefan_boltzmann_w_m2_k4 must be positive.")

    temperature_rows: list[TemperatureRow] = []

    for row in flux_rows:
        ambient_temperature_C, wind_speed_m_s = get_environment(
            config=config,
            scenario_key=row.scenario_key,
            time_key=row.time_key,
        )

        h_conv = convection_coefficient_w_m2_k(wind_speed_m_s)
        h_rad = radiative_linearized_coefficient_w_m2_k(
            ambient_temperature_C=ambient_temperature_C,
            emissivity_lw=emissivity_lw,
            sigma=sigma,
        )
        h_eff = h_conv + h_rad

        if h_eff < min_h_eff:
            raise ValueError(
                f"h_eff is below configured minimum: {h_eff:.6f} W/m²K"
            )

        q_abs = max(0.0, row.q_abs_w_m2)
        delta_t_s = q_abs / h_eff
        surface_temperature_C = ambient_temperature_C + delta_t_s

        temperature_rows.append(
            TemperatureRow(
                scenario_key=row.scenario_key,
                scenario_label=row.scenario_label,
                date_iso=row.date_iso,
                day_of_year=row.day_of_year,
                time_key=row.time_key,
                hour_angle_deg=row.hour_angle_deg,
                solar_declination_deg=row.solar_declination_deg,
                panel_id=row.panel_id,
                control_point_id=row.control_point_id,
                x=row.x,
                y=row.y,
                z=row.z,
                mu_front=row.mu_front,
                chi=row.chi,
                dni_w_m2=row.dni_w_m2,
                alpha_s=row.alpha_s,
                q_abs_w_m2=q_abs,
                ambient_temperature_C=ambient_temperature_C,
                wind_speed_m_s=wind_speed_m_s,
                h_conv_w_m2_k=h_conv,
                h_rad_w_m2_k=h_rad,
                h_eff_w_m2_k=h_eff,
                delta_t_s_C=delta_t_s,
                surface_temperature_C=surface_temperature_C,
                shadow_source=row.shadow_source,
            )
        )

    return temperature_rows


def write_control_point_temperature_csv(rows: Iterable[TemperatureRow]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario_key",
        "scenario_label",
        "date",
        "day_of_year",
        "time_key",
        "hour_angle_deg",
        "solar_declination_deg",
        "panel_id",
        "control_point_id",
        "x_m",
        "y_m",
        "z_m",
        "mu_front",
        "chi",
        "dni_w_m2",
        "alpha_s",
        "q_abs_w_m2",
        "ambient_temperature_C",
        "wind_speed_m_s",
        "h_conv_w_m2_k",
        "h_rad_w_m2_k",
        "h_eff_w_m2_k",
        "delta_t_s_C",
        "surface_temperature_C",
        "shadow_source",
    ]

    with CONTROL_POINT_TEMPERATURE_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "scenario_key": row.scenario_key,
                    "scenario_label": row.scenario_label,
                    "date": row.date_iso,
                    "day_of_year": row.day_of_year,
                    "time_key": row.time_key,
                    "hour_angle_deg": f"{row.hour_angle_deg:.9f}",
                    "solar_declination_deg": f"{row.solar_declination_deg:.9f}",
                    "panel_id": row.panel_id,
                    "control_point_id": row.control_point_id,
                    "x_m": f"{row.x:.9f}",
                    "y_m": f"{row.y:.9f}",
                    "z_m": f"{row.z:.9f}",
                    "mu_front": f"{row.mu_front:.12f}",
                    "chi": row.chi,
                    "dni_w_m2": f"{row.dni_w_m2:.6f}",
                    "alpha_s": f"{row.alpha_s:.6f}",
                    "q_abs_w_m2": f"{row.q_abs_w_m2:.9f}",
                    "ambient_temperature_C": f"{row.ambient_temperature_C:.6f}",
                    "wind_speed_m_s": f"{row.wind_speed_m_s:.6f}",
                    "h_conv_w_m2_k": f"{row.h_conv_w_m2_k:.9f}",
                    "h_rad_w_m2_k": f"{row.h_rad_w_m2_k:.9f}",
                    "h_eff_w_m2_k": f"{row.h_eff_w_m2_k:.9f}",
                    "delta_t_s_C": f"{row.delta_t_s_C:.9f}",
                    "surface_temperature_C": f"{row.surface_temperature_C:.9f}",
                    "shadow_source": row.shadow_source,
                }
            )


def build_panel_temperature_summary(rows: list[TemperatureRow]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, int], list[TemperatureRow]] = defaultdict(list)

    for row in rows:
        grouped[(row.scenario_key, row.time_key, row.panel_id)].append(row)

    summary_rows: list[dict[str, str]] = []

    for (scenario_key, time_key, panel_id), items in sorted(grouped.items()):
        total = len(items)

        delta_values = [item.delta_t_s_C for item in items]
        surface_values = [item.surface_temperature_C for item in items]
        q_values = [item.q_abs_w_m2 for item in items]

        mean_delta = sum(delta_values) / total
        mean_surface = sum(surface_values) / total
        mean_q = sum(q_values) / total

        summary_rows.append(
            {
                "scenario_key": scenario_key,
                "scenario_label": items[0].scenario_label,
                "date": items[0].date_iso,
                "day_of_year": str(items[0].day_of_year),
                "time_key": time_key,
                "panel_id": str(panel_id),
                "control_point_count": str(total),
                "ambient_temperature_C": f"{items[0].ambient_temperature_C:.6f}",
                "wind_speed_m_s": f"{items[0].wind_speed_m_s:.6f}",
                "h_conv_w_m2_k": f"{items[0].h_conv_w_m2_k:.9f}",
                "h_rad_w_m2_k": f"{items[0].h_rad_w_m2_k:.9f}",
                "h_eff_w_m2_k": f"{items[0].h_eff_w_m2_k:.9f}",
                "mean_q_abs_w_m2": f"{mean_q:.9f}",
                "mean_delta_t_s_C": f"{mean_delta:.9f}",
                "min_delta_t_s_C": f"{min(delta_values):.9f}",
                "max_delta_t_s_C": f"{max(delta_values):.9f}",
                "mean_surface_temperature_C": f"{mean_surface:.9f}",
                "min_surface_temperature_C": f"{min(surface_values):.9f}",
                "max_surface_temperature_C": f"{max(surface_values):.9f}",
            }
        )

    return summary_rows


def write_panel_temperature_summary_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "scenario_key",
        "scenario_label",
        "date",
        "day_of_year",
        "time_key",
        "panel_id",
        "control_point_count",
        "ambient_temperature_C",
        "wind_speed_m_s",
        "h_conv_w_m2_k",
        "h_rad_w_m2_k",
        "h_eff_w_m2_k",
        "mean_q_abs_w_m2",
        "mean_delta_t_s_C",
        "min_delta_t_s_C",
        "max_delta_t_s_C",
        "mean_surface_temperature_C",
        "min_surface_temperature_C",
        "max_surface_temperature_C",
    ]

    with PANEL_TEMPERATURE_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_scenario_temperature_summary(rows: list[TemperatureRow]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[TemperatureRow]] = defaultdict(list)

    for row in rows:
        grouped[(row.scenario_key, row.time_key)].append(row)

    summary_rows: list[dict[str, str]] = []

    for (scenario_key, time_key), items in sorted(grouped.items()):
        total = len(items)

        delta_values = [item.delta_t_s_C for item in items]
        surface_values = [item.surface_temperature_C for item in items]
        q_values = [item.q_abs_w_m2 for item in items]

        nonzero_delta_values = [value for value in delta_values if value > 0.0]

        if nonzero_delta_values:
            mean_nonzero_delta = sum(nonzero_delta_values) / len(nonzero_delta_values)
        else:
            mean_nonzero_delta = 0.0

        summary_rows.append(
            {
                "scenario_key": scenario_key,
                "scenario_label": items[0].scenario_label,
                "date": items[0].date_iso,
                "day_of_year": str(items[0].day_of_year),
                "time_key": time_key,
                "control_point_count": str(total),
                "ambient_temperature_C": f"{items[0].ambient_temperature_C:.6f}",
                "wind_speed_m_s": f"{items[0].wind_speed_m_s:.6f}",
                "h_conv_w_m2_k": f"{items[0].h_conv_w_m2_k:.9f}",
                "h_rad_w_m2_k": f"{items[0].h_rad_w_m2_k:.9f}",
                "h_eff_w_m2_k": f"{items[0].h_eff_w_m2_k:.9f}",
                "mean_q_abs_w_m2": f"{sum(q_values) / total:.9f}",
                "mean_delta_t_s_C": f"{sum(delta_values) / total:.9f}",
                "mean_nonzero_delta_t_s_C": f"{mean_nonzero_delta:.9f}",
                "min_delta_t_s_C": f"{min(delta_values):.9f}",
                "max_delta_t_s_C": f"{max(delta_values):.9f}",
                "mean_surface_temperature_C": f"{sum(surface_values) / total:.9f}",
                "min_surface_temperature_C": f"{min(surface_values):.9f}",
                "max_surface_temperature_C": f"{max(surface_values):.9f}",
            }
        )

    return summary_rows


def write_scenario_temperature_summary_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "scenario_key",
        "scenario_label",
        "date",
        "day_of_year",
        "time_key",
        "control_point_count",
        "ambient_temperature_C",
        "wind_speed_m_s",
        "h_conv_w_m2_k",
        "h_rad_w_m2_k",
        "h_eff_w_m2_k",
        "mean_q_abs_w_m2",
        "mean_delta_t_s_C",
        "mean_nonzero_delta_t_s_C",
        "min_delta_t_s_C",
        "max_delta_t_s_C",
        "mean_surface_temperature_C",
        "min_surface_temperature_C",
        "max_surface_temperature_C",
    ]

    with SCENARIO_TEMPERATURE_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def panel_temperature_lookup(panel_summary_rows: list[dict[str, str]]) -> dict[tuple[str, str, int], float]:
    lookup = {}

    for row in panel_summary_rows:
        key = (
            row["scenario_key"],
            row["time_key"],
            int(row["panel_id"]),
        )

        lookup[key] = float(row["mean_delta_t_s_C"])

    return lookup


def plot_temperature_increment_maps(
    panels: dict[int, PanelCentre],
    panel_summary_rows: list[dict[str, str]],
    config: dict,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    lookup = panel_temperature_lookup(panel_summary_rows)

    scenario_keys = sorted({row["scenario_key"] for row in panel_summary_rows})
    time_keys = ["morning", "noon", "evening"]

    plotting = config["surface_temperature_model"]["plotting"]
    aperture_radius = float(plotting["aperture_radius_m"])
    marker_size = float(plotting["marker_size"])
    dpi = int(plotting["dpi"])

    global_max = max(float(row["mean_delta_t_s_C"]) for row in panel_summary_rows)

    for scenario_key in scenario_keys:
        for time_key in time_keys:
            x_values = []
            y_values = []
            delta_values = []

            for panel_id, panel in panels.items():
                key = (scenario_key, time_key, panel_id)

                if key not in lookup:
                    continue

                x_values.append(panel.x)
                y_values.append(panel.y)
                delta_values.append(lookup[key])

            fig, ax = plt.subplots(figsize=(8, 8))

            scatter = ax.scatter(
                x_values,
                y_values,
                c=delta_values,
                s=marker_size,
                vmin=0.0,
                vmax=global_max,
                cmap="inferno",
            )

            aperture = plt.Circle(
                (0.0, 0.0),
                aperture_radius,
                fill=False,
                linewidth=2.0,
            )
            ax.add_patch(aperture)

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-28.5, 28.5)
            ax.set_ylim(-28.5, 28.5)
            ax.set_xlabel("x, m")
            ax.set_ylabel("y, m")
            ax.set_title(f"Surface temperature increment — {scenario_key}, {time_key}")
            ax.grid(True, alpha=0.3)

            colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.set_label("Mean Delta_T_s, °C")

            output_path = FIGURE_DIR / f"surface_temperature_delta_{scenario_key}_{time_key}.png"

            plt.tight_layout()
            plt.savefig(output_path, dpi=dpi)
            plt.close(fig)


def print_summary(rows: list[dict[str, str]]) -> None:
    print("Stage 11 — First-order surface temperature increment")
    print("=" * 72)

    for row in rows:
        print(
            f"{row['scenario_key']:17s} {row['time_key']:8s} | "
            f"T_amb={float(row['ambient_temperature_C']):7.2f} °C | "
            f"wind={float(row['wind_speed_m_s']):5.2f} m/s | "
            f"h_eff={float(row['h_eff_w_m2_k']):7.3f} W/m²K | "
            f"mean ΔT={float(row['mean_delta_t_s_C']):7.3f} °C | "
            f"max ΔT={float(row['max_delta_t_s_C']):7.3f} °C"
        )

    print()
    print(f"Saved control-point temperature: {CONTROL_POINT_TEMPERATURE_CSV}")
    print(f"Saved panel summary:             {PANEL_TEMPERATURE_SUMMARY_CSV}")
    print(f"Saved scenario summary:          {SCENARIO_TEMPERATURE_SUMMARY_CSV}")
    print(f"Saved figures directory:         {FIGURE_DIR}")


def main() -> None:
    ensure_stage_10_exists()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    config = load_yaml(CONFIG_PATH)

    flux_rows = read_flux_csv(CONTROL_POINT_FLUX_CSV)
    panels = read_panel_centres(PANEL_GRID_CSV)

    temperature_rows = compute_temperature_rows(
        flux_rows=flux_rows,
        config=config,
    )

    write_control_point_temperature_csv(temperature_rows)

    panel_rows = build_panel_temperature_summary(temperature_rows)
    write_panel_temperature_summary_csv(panel_rows)

    scenario_rows = build_scenario_temperature_summary(temperature_rows)
    write_scenario_temperature_summary_csv(scenario_rows)

    plot_temperature_increment_maps(
        panels=panels,
        panel_summary_rows=panel_rows,
        config=config,
    )

    print_summary(scenario_rows)


if __name__ == "__main__":
    main()
