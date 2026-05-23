from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_GEOMETRY_DIR = PROJECT_ROOT / "outputs" / "geometry"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "geometry"

PANEL_GRID_CSV = OUTPUT_GEOMETRY_DIR / "panel_grid_3738.csv"
CONTROL_POINTS_CSV = OUTPUT_GEOMETRY_DIR / "panel_control_points_3738.csv"
SUMMARY_CSV = OUTPUT_GEOMETRY_DIR / "panel_grid_summary.csv"
FIGURE_PATH = OUTPUT_FIGURE_DIR / "panel_grid_tripod_bases.png"


PANEL_COUNT = 3738
APERTURE_RADIUS_M = 27.0
SPHERE_RADIUS_M = 27.0
EFFECTIVE_APERTURE_M = 32.0

TRIPOD_BASE_RADIUS_M = 15.5
TRIPOD_BASE_AZIMUTHS_DEG = [90.0, 210.0, 330.0]

CONTROL_POINT_OFFSET_FACTOR = 0.45


@dataclass(frozen=True)
class PanelPoint:
    panel_id: int
    x_m: float
    y_m: float
    z_m: float
    projected_radius_m: float
    azimuth_deg: float
    normal_x: float
    normal_y: float
    normal_z: float
    projected_area_m2: float
    equivalent_cell_size_m: float


@dataclass(frozen=True)
class ControlPoint:
    panel_id: int
    control_point_id: str
    x_m: float
    y_m: float
    z_m: float
    projected_radius_m: float
    azimuth_deg: float
    normal_x: float
    normal_y: float
    normal_z: float
    clipped_to_aperture: bool


def projected_aperture_area_m2() -> float:
    return math.pi * APERTURE_RADIUS_M**2


def projected_cell_area_m2() -> float:
    return projected_aperture_area_m2() / PANEL_COUNT


def equivalent_cell_size_m() -> float:
    return math.sqrt(projected_cell_area_m2())


def spherical_bowl_z(x_m: float, y_m: float) -> float:
    radius_squared = x_m * x_m + y_m * y_m

    if radius_squared > SPHERE_RADIUS_M * SPHERE_RADIUS_M:
        raise ValueError("Point is outside the spherical aperture.")

    return -math.sqrt(max(0.0, SPHERE_RADIUS_M * SPHERE_RADIUS_M - radius_squared))


def azimuth_deg_from_xy(x_m: float, y_m: float) -> float:
    angle = math.degrees(math.atan2(y_m, x_m))

    if angle < 0.0:
        angle += 360.0

    return angle


def front_side_normal(x_m: float, y_m: float, z_m: float) -> tuple[float, float, float]:
    length = math.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)

    if length <= 0.0:
        raise ValueError("Zero-length radius vector is not valid.")

    # The spherical bowl is the lower hemisphere. The front/open-side normal is
    # opposite to the radial vector from the sphere centre.
    return (-x_m / length, -y_m / length, -z_m / length)


def clip_to_aperture(x_m: float, y_m: float) -> tuple[float, float, bool]:
    radius = math.hypot(x_m, y_m)

    if radius <= APERTURE_RADIUS_M:
        return x_m, y_m, False

    scale = (APERTURE_RADIUS_M - 1.0e-9) / radius

    return x_m * scale, y_m * scale, True


def generate_panel_centres() -> list[PanelPoint]:
    panels: list[PanelPoint] = []

    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    cell_area = projected_cell_area_m2()
    leq = equivalent_cell_size_m()

    for index in range(PANEL_COUNT):
        panel_id = index + 1

        radial_fraction = math.sqrt((index + 0.5) / PANEL_COUNT)
        radius_m = APERTURE_RADIUS_M * radial_fraction
        theta = index * golden_angle

        x_m = radius_m * math.cos(theta)
        y_m = radius_m * math.sin(theta)
        z_m = spherical_bowl_z(x_m, y_m)

        nx, ny, nz = front_side_normal(x_m, y_m, z_m)

        panels.append(
            PanelPoint(
                panel_id=panel_id,
                x_m=x_m,
                y_m=y_m,
                z_m=z_m,
                projected_radius_m=radius_m,
                azimuth_deg=azimuth_deg_from_xy(x_m, y_m),
                normal_x=nx,
                normal_y=ny,
                normal_z=nz,
                projected_area_m2=cell_area,
                equivalent_cell_size_m=leq,
            )
        )

    return panels


def make_control_point(
    panel_id: int,
    control_point_id: str,
    x_m: float,
    y_m: float,
) -> ControlPoint:
    x_clipped, y_clipped, clipped = clip_to_aperture(x_m, y_m)

    z_m = spherical_bowl_z(x_clipped, y_clipped)
    radius_m = math.hypot(x_clipped, y_clipped)
    nx, ny, nz = front_side_normal(x_clipped, y_clipped, z_m)

    return ControlPoint(
        panel_id=panel_id,
        control_point_id=control_point_id,
        x_m=x_clipped,
        y_m=y_clipped,
        z_m=z_m,
        projected_radius_m=radius_m,
        azimuth_deg=azimuth_deg_from_xy(x_clipped, y_clipped),
        normal_x=nx,
        normal_y=ny,
        normal_z=nz,
        clipped_to_aperture=clipped,
    )


def generate_control_points(panels: list[PanelPoint]) -> list[ControlPoint]:
    control_points: list[ControlPoint] = []

    offset = CONTROL_POINT_OFFSET_FACTOR * equivalent_cell_size_m()

    for panel in panels:
        candidates = [
            ("C", panel.x_m, panel.y_m),
            ("PX", panel.x_m + offset, panel.y_m),
            ("NX", panel.x_m - offset, panel.y_m),
            ("PY", panel.x_m, panel.y_m + offset),
            ("NY", panel.x_m, panel.y_m - offset),
        ]

        for control_point_id, x_m, y_m in candidates:
            control_points.append(
                make_control_point(
                    panel_id=panel.panel_id,
                    control_point_id=control_point_id,
                    x_m=x_m,
                    y_m=y_m,
                )
            )

    return control_points


def write_panel_grid_csv(panels: list[PanelPoint]) -> None:
    PANEL_GRID_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "panel_id",
        "x_m",
        "y_m",
        "z_m",
        "projected_radius_m",
        "azimuth_deg",
        "normal_x",
        "normal_y",
        "normal_z",
        "projected_area_m2",
        "equivalent_cell_size_m",
    ]

    with PANEL_GRID_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for panel in panels:
            writer.writerow(
                {
                    "panel_id": panel.panel_id,
                    "x_m": f"{panel.x_m:.9f}",
                    "y_m": f"{panel.y_m:.9f}",
                    "z_m": f"{panel.z_m:.9f}",
                    "projected_radius_m": f"{panel.projected_radius_m:.9f}",
                    "azimuth_deg": f"{panel.azimuth_deg:.9f}",
                    "normal_x": f"{panel.normal_x:.12f}",
                    "normal_y": f"{panel.normal_y:.12f}",
                    "normal_z": f"{panel.normal_z:.12f}",
                    "projected_area_m2": f"{panel.projected_area_m2:.12f}",
                    "equivalent_cell_size_m": f"{panel.equivalent_cell_size_m:.12f}",
                }
            )


def write_control_points_csv(control_points: list[ControlPoint]) -> None:
    CONTROL_POINTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "panel_id",
        "control_point_id",
        "x_m",
        "y_m",
        "z_m",
        "projected_radius_m",
        "azimuth_deg",
        "normal_x",
        "normal_y",
        "normal_z",
        "clipped_to_aperture",
    ]

    with CONTROL_POINTS_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for point in control_points:
            writer.writerow(
                {
                    "panel_id": point.panel_id,
                    "control_point_id": point.control_point_id,
                    "x_m": f"{point.x_m:.9f}",
                    "y_m": f"{point.y_m:.9f}",
                    "z_m": f"{point.z_m:.9f}",
                    "projected_radius_m": f"{point.projected_radius_m:.9f}",
                    "azimuth_deg": f"{point.azimuth_deg:.9f}",
                    "normal_x": f"{point.normal_x:.12f}",
                    "normal_y": f"{point.normal_y:.12f}",
                    "normal_z": f"{point.normal_z:.12f}",
                    "clipped_to_aperture": str(point.clipped_to_aperture).lower(),
                }
            )


def write_summary_csv(
    panels: list[PanelPoint],
    control_points: list[ControlPoint],
) -> None:
    clipped_count = sum(1 for point in control_points if point.clipped_to_aperture)

    rows = [
        ("panel_count", PANEL_COUNT),
        ("control_points_per_panel", 5),
        ("total_control_points", len(control_points)),
        ("aperture_radius_m", APERTURE_RADIUS_M),
        ("projected_aperture_area_m2", projected_aperture_area_m2()),
        ("projected_cell_area_m2", projected_cell_area_m2()),
        ("equivalent_cell_size_m", equivalent_cell_size_m()),
        ("control_point_offset_factor", CONTROL_POINT_OFFSET_FACTOR),
        ("control_point_offset_m", CONTROL_POINT_OFFSET_FACTOR * equivalent_cell_size_m()),
        ("clipped_control_point_count", clipped_count),
        ("tripod_base_radius_m", TRIPOD_BASE_RADIUS_M),
        ("tripod_base_count", len(TRIPOD_BASE_AZIMUTHS_DEG)),
    ]

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["quantity", "value"])

        for key, value in rows:
            writer.writerow([key, value])


def tripod_base_xy() -> list[tuple[float, float]]:
    points = []

    for angle_deg in TRIPOD_BASE_AZIMUTHS_DEG:
        angle_rad = math.radians(angle_deg)
        points.append(
            (
                TRIPOD_BASE_RADIUS_M * math.cos(angle_rad),
                TRIPOD_BASE_RADIUS_M * math.sin(angle_rad),
            )
        )

    return points


def plot_panel_grid(panels: list[PanelPoint]) -> None:
    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    x_values = [panel.x_m for panel in panels]
    y_values = [panel.y_m for panel in panels]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        x_values,
        y_values,
        s=2,
        label="Panel centres",
    )

    aperture = plt.Circle(
        (0.0, 0.0),
        APERTURE_RADIUS_M,
        fill=False,
        linewidth=2.0,
        label="54 m reflector rim",
    )

    tripod_radius = plt.Circle(
        (0.0, 0.0),
        TRIPOD_BASE_RADIUS_M,
        fill=False,
        linestyle="--",
        linewidth=2.0,
        label="Internal tripod-base radius",
    )

    effective_aperture = plt.Circle(
        (0.0, 0.0),
        EFFECTIVE_APERTURE_M / 2.0,
        fill=False,
        linestyle=":",
        linewidth=2.0,
        label="32 m effective aperture",
    )

    ax.add_patch(aperture)
    ax.add_patch(tripod_radius)
    ax.add_patch(effective_aperture)

    for index, (x_m, y_m) in enumerate(tripod_base_xy(), start=1):
        ax.scatter([x_m], [y_m], s=80, marker="x")
        ax.text(x_m, y_m, f" B{index}", fontsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-28.5, 28.5)
    ax.set_ylim(-28.5, 28.5)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_title("ROT-54/2.6 panel grid with corrected internal tripod bases")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300)
    plt.close(fig)


def print_summary(
    panels: list[PanelPoint],
    control_points: list[ControlPoint],
) -> None:
    clipped_count = sum(1 for point in control_points if point.clipped_to_aperture)

    print("Stage 8 — Panel grid and control-point generation")
    print("=" * 60)
    print(f"Panel count:                  {len(panels)}")
    print(f"Control points:               {len(control_points)}")
    print(f"Projected aperture area:       {projected_aperture_area_m2():.9f} m^2")
    print(f"Projected cell area:           {projected_cell_area_m2():.9f} m^2")
    print(f"Equivalent cell size:          {equivalent_cell_size_m():.9f} m")
    print(f"Control-point offset:          {CONTROL_POINT_OFFSET_FACTOR * equivalent_cell_size_m():.9f} m")
    print(f"Clipped edge control points:   {clipped_count}")
    print(f"Tripod base radius:            {TRIPOD_BASE_RADIUS_M:.3f} m")
    print()
    print(f"Saved panel grid:              {PANEL_GRID_CSV}")
    print(f"Saved control points:          {CONTROL_POINTS_CSV}")
    print(f"Saved summary:                 {SUMMARY_CSV}")
    print(f"Saved figure:                  {FIGURE_PATH}")


def main() -> None:
    OUTPUT_GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)

    panels = generate_panel_centres()
    control_points = generate_control_points(panels)

    write_panel_grid_csv(panels)
    write_control_points_csv(control_points)
    write_summary_csv(panels, control_points)
    plot_panel_grid(panels)
    print_summary(panels, control_points)


if __name__ == "__main__":
    main()
