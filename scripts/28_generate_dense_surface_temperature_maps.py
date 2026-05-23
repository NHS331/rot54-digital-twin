from __future__ import annotations

import csv
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "dense_surface_temperature_map.yaml"

STAGE_11_SCRIPT = PROJECT_ROOT / "scripts" / "27_compute_surface_temperature_increment.py"

CONTROL_POINT_TEMPERATURE_CSV = (
    PROJECT_ROOT / "outputs" / "surface_temperature" / "control_point_surface_temperature.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "surface_temperature_dense"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "surface_temperature_dense"

SUMMARY_CSV = OUTPUT_DIR / "dense_temperature_grid_summary.csv"


@dataclass(frozen=True)
class TemperaturePoint:
    scenario_key: str
    scenario_label: str
    date_iso: str
    day_of_year: int
    time_key: str
    x: float
    y: float
    delta_t_s_C: float
    surface_temperature_C: float


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    return data


def ensure_stage_11_exists() -> None:
    if CONTROL_POINT_TEMPERATURE_CSV.exists():
        return

    if not STAGE_11_SCRIPT.exists():
        raise FileNotFoundError(
            "Stage 11 output is missing and scripts/27_compute_surface_temperature_increment.py was not found."
        )

    subprocess.run(
        [sys.executable, str(STAGE_11_SCRIPT)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def read_temperature_points(path: Path) -> list[TemperaturePoint]:
    points: list[TemperaturePoint] = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            points.append(
                TemperaturePoint(
                    scenario_key=row["scenario_key"],
                    scenario_label=row["scenario_label"],
                    date_iso=row["date"],
                    day_of_year=int(row["day_of_year"]),
                    time_key=row["time_key"],
                    x=float(row["x_m"]),
                    y=float(row["y_m"]),
                    delta_t_s_C=float(row["delta_t_s_C"]),
                    surface_temperature_C=float(row["surface_temperature_C"]),
                )
            )

    return points


def group_points(
    points: list[TemperaturePoint],
) -> dict[tuple[str, str], list[TemperaturePoint]]:
    grouped: dict[tuple[str, str], list[TemperaturePoint]] = defaultdict(list)

    for point in points:
        grouped[(point.scenario_key, point.time_key)].append(point)

    return grouped


def build_dense_grid(
    aperture_radius_m: float,
    dense_grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(
        -aperture_radius_m,
        aperture_radius_m,
        dense_grid_size,
        dtype=np.float64,
    )

    x_grid, y_grid = np.meshgrid(axis, axis)

    aperture_mask = (x_grid * x_grid + y_grid * y_grid) <= aperture_radius_m * aperture_radius_m

    return x_grid, y_grid, aperture_mask


def interpolate_case(
    case_points: list[TemperaturePoint],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    aperture_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_source = np.array([point.x for point in case_points], dtype=np.float64)
    y_source = np.array([point.y for point in case_points], dtype=np.float64)

    delta_source = np.array([point.delta_t_s_C for point in case_points], dtype=np.float64)
    surface_source = np.array([point.surface_temperature_C for point in case_points], dtype=np.float64)

    triangulation = mtri.Triangulation(x_source, y_source)

    delta_interpolator = mtri.LinearTriInterpolator(
        triangulation,
        delta_source,
    )

    surface_interpolator = mtri.LinearTriInterpolator(
        triangulation,
        surface_source,
    )

    delta_grid = delta_interpolator(x_grid, y_grid)
    surface_grid = surface_interpolator(x_grid, y_grid)

    delta_mask = np.ma.getmaskarray(delta_grid)
    surface_mask = np.ma.getmaskarray(surface_grid)

    invalid_mask = (~aperture_mask) | delta_mask | surface_mask

    delta_dense = np.ma.array(delta_grid, mask=invalid_mask).filled(np.nan)
    surface_dense = np.ma.array(surface_grid, mask=invalid_mask).filled(np.nan)

    return delta_dense.astype(np.float32), surface_dense.astype(np.float32)


def save_dense_npz(
    output_path: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    aperture_mask: np.ndarray,
    delta_dense: np.ndarray,
    surface_dense: np.ndarray,
) -> int:
    valid_mask = aperture_mask & np.isfinite(delta_dense) & np.isfinite(surface_dense)

    x_values = x_grid[valid_mask].astype(np.float32)
    y_values = y_grid[valid_mask].astype(np.float32)
    delta_values = delta_dense[valid_mask].astype(np.float32)
    surface_values = surface_dense[valid_mask].astype(np.float32)

    np.savez_compressed(
        output_path,
        x_m=x_values,
        y_m=y_values,
        delta_t_s_C=delta_values,
        surface_temperature_C=surface_values,
    )

    return int(x_values.size)


def plot_dense_delta_map(
    output_path: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    delta_dense: np.ndarray,
    aperture_radius_m: float,
    title: str,
    dpi: int,
    figure_size_inches: float,
    colormap: str,
    global_delta_max: float,
) -> None:
    fig, ax = plt.subplots(
        figsize=(figure_size_inches, figure_size_inches),
    )

    image = ax.imshow(
        delta_dense,
        extent=[
            float(np.nanmin(x_grid)),
            float(np.nanmax(x_grid)),
            float(np.nanmin(y_grid)),
            float(np.nanmax(y_grid)),
        ],
        origin="lower",
        interpolation="bilinear",
        cmap=colormap,
        vmin=0.0,
        vmax=global_delta_max,
    )

    aperture = plt.Circle(
        (0.0, 0.0),
        aperture_radius_m,
        fill=False,
        linewidth=2.0,
    )
    ax.add_patch(aperture)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-aperture_radius_m - 1.5, aperture_radius_m + 1.5)
    ax.set_ylim(-aperture_radius_m - 1.5, aperture_radius_m + 1.5)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_title(title)
    ax.grid(False)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Interpolated Delta_T_s, °C")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_summary(rows: list[dict[str, str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario_key",
        "scenario_label",
        "date",
        "day_of_year",
        "time_key",
        "source_point_count",
        "dense_coordinate_count",
        "density_multiplier",
        "grid_size",
        "aperture_radius_m",
        "mean_delta_t_s_C",
        "max_delta_t_s_C",
        "mean_surface_temperature_C",
        "max_surface_temperature_C",
        "npz_file",
        "figure_file",
    ]

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_stage_11_exists()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    config = load_yaml(CONFIG_PATH)
    model = config["dense_surface_temperature_map"]
    interpolation = model["interpolation"]
    plotting = model["plotting"]

    dense_grid_size = int(interpolation["dense_grid_size"])
    aperture_radius_m = float(interpolation["aperture_radius_m"])
    minimum_density_multiplier = float(interpolation["minimum_density_multiplier"])

    dpi = int(plotting["dpi"])
    figure_size_inches = float(plotting["figure_size_inches"])
    colormap = str(plotting["colormap"])

    points = read_temperature_points(CONTROL_POINT_TEMPERATURE_CSV)
    grouped = group_points(points)

    x_grid, y_grid, aperture_mask = build_dense_grid(
        aperture_radius_m=aperture_radius_m,
        dense_grid_size=dense_grid_size,
    )

    dense_results = []
    global_delta_max = 0.0

    for (scenario_key, time_key), case_points in sorted(grouped.items()):
        delta_dense, surface_dense = interpolate_case(
            case_points=case_points,
            x_grid=x_grid,
            y_grid=y_grid,
            aperture_mask=aperture_mask,
        )

        finite_delta = delta_dense[np.isfinite(delta_dense)]

        if finite_delta.size > 0:
            global_delta_max = max(global_delta_max, float(np.max(finite_delta)))

        dense_results.append(
            (
                scenario_key,
                time_key,
                case_points,
                delta_dense,
                surface_dense,
            )
        )

    if global_delta_max <= 0.0:
        global_delta_max = 1.0

    summary_rows: list[dict[str, str]] = []

    for scenario_key, time_key, case_points, delta_dense, surface_dense in dense_results:
        first = case_points[0]

        npz_path = OUTPUT_DIR / f"dense_temperature_{scenario_key}_{time_key}.npz"
        figure_path = FIGURE_DIR / f"dense_temperature_delta_{scenario_key}_{time_key}.png"

        dense_coordinate_count = save_dense_npz(
            output_path=npz_path,
            x_grid=x_grid,
            y_grid=y_grid,
            aperture_mask=aperture_mask,
            delta_dense=delta_dense,
            surface_dense=surface_dense,
        )

        source_point_count = len(case_points)
        density_multiplier = dense_coordinate_count / source_point_count

        if density_multiplier < minimum_density_multiplier:
            raise ValueError(
                "Dense grid is not dense enough: "
                f"{density_multiplier:.3f}x < {minimum_density_multiplier:.3f}x"
            )

        finite_delta = delta_dense[np.isfinite(delta_dense)]
        finite_surface = surface_dense[np.isfinite(surface_dense)]

        plot_dense_delta_map(
            output_path=figure_path,
            x_grid=x_grid,
            y_grid=y_grid,
            delta_dense=delta_dense,
            aperture_radius_m=aperture_radius_m,
            title=f"Dense interpolated surface temperature increment — {scenario_key}, {time_key}",
            dpi=dpi,
            figure_size_inches=figure_size_inches,
            colormap=colormap,
            global_delta_max=global_delta_max,
        )

        summary_rows.append(
            {
                "scenario_key": scenario_key,
                "scenario_label": first.scenario_label,
                "date": first.date_iso,
                "day_of_year": str(first.day_of_year),
                "time_key": time_key,
                "source_point_count": str(source_point_count),
                "dense_coordinate_count": str(dense_coordinate_count),
                "density_multiplier": f"{density_multiplier:.6f}",
                "grid_size": str(dense_grid_size),
                "aperture_radius_m": f"{aperture_radius_m:.6f}",
                "mean_delta_t_s_C": f"{float(np.mean(finite_delta)):.9f}",
                "max_delta_t_s_C": f"{float(np.max(finite_delta)):.9f}",
                "mean_surface_temperature_C": f"{float(np.mean(finite_surface)):.9f}",
                "max_surface_temperature_C": f"{float(np.max(finite_surface)):.9f}",
                "npz_file": str(npz_path.relative_to(PROJECT_ROOT)),
                "figure_file": str(figure_path.relative_to(PROJECT_ROOT)),
            }
        )

    write_summary(summary_rows)

    print("Stage 11b — Dense interpolated surface temperature maps")
    print("=" * 72)

    for row in summary_rows:
        print(
            f"{row['scenario_key']:17s} {row['time_key']:8s} | "
            f"source={int(row['source_point_count']):6d} | "
            f"dense={int(row['dense_coordinate_count']):8d} | "
            f"x{float(row['density_multiplier']):7.2f} | "
            f"max ΔT={float(row['max_delta_t_s_C']):7.3f} °C"
        )

    print()
    print(f"Saved dense summary: {SUMMARY_CSV}")
    print(f"Saved dense NPZ dir: {OUTPUT_DIR}")
    print(f"Saved figures dir:   {FIGURE_DIR}")


if __name__ == "__main__":
    main()
