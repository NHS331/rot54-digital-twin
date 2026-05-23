from __future__ import annotations

import csv
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "solar_flux_model.yaml"

STAGE_9_SCRIPT = PROJECT_ROOT / "scripts" / "25_compute_ray_shadow_visibility.py"

VISIBILITY_CSV = PROJECT_ROOT / "outputs" / "shadow_ray" / "control_point_visibility.csv"
PANEL_GRID_CSV = PROJECT_ROOT / "outputs" / "geometry" / "panel_grid_3738.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "solar_flux"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "solar_flux"

CONTROL_POINT_FLUX_CSV = OUTPUT_DIR / "control_point_absorbed_flux.csv"
PANEL_FLUX_SUMMARY_CSV = OUTPUT_DIR / "panel_absorbed_flux_summary.csv"
SCENARIO_FLUX_SUMMARY_CSV = OUTPUT_DIR / "scenario_absorbed_flux_summary.csv"


@dataclass(frozen=True)
class VisibilityRow:
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
    shadow_source: str


@dataclass(frozen=True)
class FluxRow:
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


def ensure_stage_9_exists() -> None:
    if VISIBILITY_CSV.exists() and PANEL_GRID_CSV.exists():
        return

    if not STAGE_9_SCRIPT.exists():
        raise FileNotFoundError(
            "Stage 9 outputs are missing and scripts/25_compute_ray_shadow_visibility.py was not found."
        )

    subprocess.run(
        [sys.executable, str(STAGE_9_SCRIPT)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def read_visibility_csv(path: Path) -> list[VisibilityRow]:
    rows: list[VisibilityRow] = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                VisibilityRow(
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


def get_dni_w_m2(
    config: dict,
    scenario_key: str,
    time_key: str,
) -> float:
    dni_table = config["solar_flux_model"]["physical_model"]["scenario_dni_w_m2"]

    if scenario_key not in dni_table:
        raise KeyError(f"DNI scenario was not found in config: {scenario_key}")

    if time_key not in dni_table[scenario_key]:
        raise KeyError(f"DNI time key was not found in config: {scenario_key}/{time_key}")

    return float(dni_table[scenario_key][time_key])


def compute_flux_rows(
    visibility_rows: list[VisibilityRow],
    config: dict,
) -> list[FluxRow]:
    model = config["solar_flux_model"]
    physical = model["physical_model"]

    alpha_s = float(physical["solar_absorptivity_alpha_s"])

    if not (0.0 <= alpha_s <= 1.0):
        raise ValueError("solar_absorptivity_alpha_s must be within [0, 1].")

    flux_rows: list[FluxRow] = []

    for row in visibility_rows:
        if row.chi not in {0, 1}:
            raise ValueError(f"Invalid chi value: {row.chi}")

        mu_front = max(0.0, row.mu_front)
        dni = get_dni_w_m2(
            config=config,
            scenario_key=row.scenario_key,
            time_key=row.time_key,
        )

        q_incident = dni * mu_front * row.chi
        q_abs = alpha_s * q_incident

        flux_rows.append(
            FluxRow(
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
                mu_front=mu_front,
                chi=row.chi,
                dni_w_m2=dni,
                alpha_s=alpha_s,
                q_incident_w_m2=q_incident,
                q_abs_w_m2=q_abs,
                shadow_source=row.shadow_source,
            )
        )

    return flux_rows


def write_control_point_flux_csv(rows: Iterable[FluxRow]) -> None:
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
        "q_incident_w_m2",
        "q_abs_w_m2",
        "shadow_source",
    ]

    with CONTROL_POINT_FLUX_CSV.open("w", encoding="utf-8", newline="") as file:
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
                    "q_incident_w_m2": f"{row.q_incident_w_m2:.9f}",
                    "q_abs_w_m2": f"{row.q_abs_w_m2:.9f}",
                    "shadow_source": row.shadow_source,
                }
            )


def build_panel_flux_summary(rows: list[FluxRow]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, int], list[FluxRow]] = defaultdict(list)

    for row in rows:
        grouped[(row.scenario_key, row.time_key, row.panel_id)].append(row)

    summary_rows: list[dict[str, str]] = []

    for (scenario_key, time_key, panel_id), items in sorted(grouped.items()):
        total = len(items)
        illuminated_count = sum(1 for item in items if item.chi == 1)
        shadowed_count = total - illuminated_count

        mean_q_abs = sum(item.q_abs_w_m2 for item in items) / total
        max_q_abs = max(item.q_abs_w_m2 for item in items)
        min_q_abs = min(item.q_abs_w_m2 for item in items)

        mean_mu_front = sum(item.mu_front for item in items) / total
        mean_chi = sum(item.chi for item in items) / total

        summary_rows.append(
            {
                "scenario_key": scenario_key,
                "scenario_label": items[0].scenario_label,
                "date": items[0].date_iso,
                "day_of_year": str(items[0].day_of_year),
                "time_key": time_key,
                "panel_id": str(panel_id),
                "control_point_count": str(total),
                "illuminated_count": str(illuminated_count),
                "shadowed_count": str(shadowed_count),
                "illuminated_fraction": f"{mean_chi:.9f}",
                "mean_mu_front": f"{mean_mu_front:.12f}",
                "dni_w_m2": f"{items[0].dni_w_m2:.6f}",
                "alpha_s": f"{items[0].alpha_s:.6f}",
                "mean_q_abs_w_m2": f"{mean_q_abs:.9f}",
                "min_q_abs_w_m2": f"{min_q_abs:.9f}",
                "max_q_abs_w_m2": f"{max_q_abs:.9f}",
            }
        )

    return summary_rows


def write_panel_flux_summary_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "scenario_key",
        "scenario_label",
        "date",
        "day_of_year",
        "time_key",
        "panel_id",
        "control_point_count",
        "illuminated_count",
        "shadowed_count",
        "illuminated_fraction",
        "mean_mu_front",
        "dni_w_m2",
        "alpha_s",
        "mean_q_abs_w_m2",
        "min_q_abs_w_m2",
        "max_q_abs_w_m2",
    ]

    with PANEL_FLUX_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_scenario_flux_summary(rows: list[FluxRow]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[FluxRow]] = defaultdict(list)

    for row in rows:
        grouped[(row.scenario_key, row.time_key)].append(row)

    summary_rows: list[dict[str, str]] = []

    for (scenario_key, time_key), items in sorted(grouped.items()):
        total = len(items)
        illuminated_count = sum(1 for item in items if item.chi == 1)
        shadowed_count = total - illuminated_count

        mean_q_abs = sum(item.q_abs_w_m2 for item in items) / total
        max_q_abs = max(item.q_abs_w_m2 for item in items)
        min_q_abs = min(item.q_abs_w_m2 for item in items)

        nonzero_values = [item.q_abs_w_m2 for item in items if item.q_abs_w_m2 > 0.0]

        if nonzero_values:
            mean_nonzero_q_abs = sum(nonzero_values) / len(nonzero_values)
        else:
            mean_nonzero_q_abs = 0.0

        summary_rows.append(
            {
                "scenario_key": scenario_key,
                "scenario_label": items[0].scenario_label,
                "date": items[0].date_iso,
                "day_of_year": str(items[0].day_of_year),
                "time_key": time_key,
                "control_point_count": str(total),
                "illuminated_count": str(illuminated_count),
                "shadowed_count": str(shadowed_count),
                "illuminated_fraction": f"{illuminated_count / total:.9f}",
                "shadow_fraction": f"{shadowed_count / total:.9f}",
                "mean_mu_front": f"{sum(item.mu_front for item in items) / total:.12f}",
                "dni_w_m2": f"{items[0].dni_w_m2:.6f}",
                "alpha_s": f"{items[0].alpha_s:.6f}",
                "mean_q_abs_w_m2": f"{mean_q_abs:.9f}",
                "mean_nonzero_q_abs_w_m2": f"{mean_nonzero_q_abs:.9f}",
                "min_q_abs_w_m2": f"{min_q_abs:.9f}",
                "max_q_abs_w_m2": f"{max_q_abs:.9f}",
            }
        )

    return summary_rows


def write_scenario_flux_summary_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "scenario_key",
        "scenario_label",
        "date",
        "day_of_year",
        "time_key",
        "control_point_count",
        "illuminated_count",
        "shadowed_count",
        "illuminated_fraction",
        "shadow_fraction",
        "mean_mu_front",
        "dni_w_m2",
        "alpha_s",
        "mean_q_abs_w_m2",
        "mean_nonzero_q_abs_w_m2",
        "min_q_abs_w_m2",
        "max_q_abs_w_m2",
    ]

    with SCENARIO_FLUX_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def panel_flux_lookup(panel_summary_rows: list[dict[str, str]]) -> dict[tuple[str, str, int], float]:
    lookup = {}

    for row in panel_summary_rows:
        key = (
            row["scenario_key"],
            row["time_key"],
            int(row["panel_id"]),
        )

        lookup[key] = float(row["mean_q_abs_w_m2"])

    return lookup


def plot_flux_maps(
    panels: dict[int, PanelCentre],
    panel_summary_rows: list[dict[str, str]],
    config: dict,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    lookup = panel_flux_lookup(panel_summary_rows)

    scenario_keys = sorted({row["scenario_key"] for row in panel_summary_rows})
    time_keys = ["morning", "noon", "evening"]

    plotting = config["solar_flux_model"]["plotting"]
    aperture_radius = float(plotting["aperture_radius_m"])
    marker_size = float(plotting["marker_size"])
    dpi = int(plotting["dpi"])

    global_max = max(float(row["mean_q_abs_w_m2"]) for row in panel_summary_rows)

    for scenario_key in scenario_keys:
        for time_key in time_keys:
            x_values = []
            y_values = []
            q_values = []

            for panel_id, panel in panels.items():
                key = (scenario_key, time_key, panel_id)

                if key not in lookup:
                    continue

                x_values.append(panel.x)
                y_values.append(panel.y)
                q_values.append(lookup[key])

            fig, ax = plt.subplots(figsize=(8, 8))

            scatter = ax.scatter(
                x_values,
                y_values,
                c=q_values,
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
            ax.set_title(f"Absorbed solar flux — {scenario_key}, {time_key}")
            ax.grid(True, alpha=0.3)

            colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.set_label("Mean absorbed flux, W/m²")

            output_path = FIGURE_DIR / f"absorbed_flux_{scenario_key}_{time_key}.png"

            plt.tight_layout()
            plt.savefig(output_path, dpi=dpi)
            plt.close(fig)


def print_summary(rows: list[dict[str, str]]) -> None:
    print("Stage 10 — Absorbed solar flux")
    print("=" * 60)

    for row in rows:
        print(
            f"{row['scenario_key']:17s} {row['time_key']:8s} | "
            f"DNI={float(row['dni_w_m2']):7.1f} W/m² | "
            f"mean q_abs={float(row['mean_q_abs_w_m2']):8.3f} W/m² | "
            f"max q_abs={float(row['max_q_abs_w_m2']):8.3f} W/m² | "
            f"illum={float(row['illuminated_fraction']):.4f}"
        )

    print()
    print(f"Saved control-point flux: {CONTROL_POINT_FLUX_CSV}")
    print(f"Saved panel summary:      {PANEL_FLUX_SUMMARY_CSV}")
    print(f"Saved scenario summary:   {SCENARIO_FLUX_SUMMARY_CSV}")
    print(f"Saved figures directory:  {FIGURE_DIR}")


def main() -> None:
    ensure_stage_9_exists()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    config = load_yaml(CONFIG_PATH)

    visibility_rows = read_visibility_csv(VISIBILITY_CSV)
    panels = read_panel_centres(PANEL_GRID_CSV)

    flux_rows = compute_flux_rows(
        visibility_rows=visibility_rows,
        config=config,
    )

    write_control_point_flux_csv(flux_rows)

    panel_rows = build_panel_flux_summary(flux_rows)
    write_panel_flux_summary_csv(panel_rows)

    scenario_rows = build_scenario_flux_summary(flux_rows)
    write_scenario_flux_summary_csv(scenario_rows)

    plot_flux_maps(
        panels=panels,
        panel_summary_rows=panel_rows,
        config=config,
    )

    print_summary(scenario_rows)


if __name__ == "__main__":
    main()
