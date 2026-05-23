from __future__ import annotations

import csv
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "ray_shadow_model.yaml"

CONTROL_POINTS_CSV = PROJECT_ROOT / "outputs" / "geometry" / "panel_control_points_3738.csv"
PANEL_GRID_CSV = PROJECT_ROOT / "outputs" / "geometry" / "panel_grid_3738.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shadow_ray"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "shadow_ray"

CONTROL_POINT_VISIBILITY_CSV = OUTPUT_DIR / "control_point_visibility.csv"
PANEL_SUMMARY_CSV = OUTPUT_DIR / "panel_shadow_summary.csv"
SCENARIO_SUMMARY_CSV = OUTPUT_DIR / "scenario_shadow_summary.csv"


@dataclass(frozen=True)
class ControlPoint:
    panel_id: int
    control_point_id: str
    x: float
    y: float
    z: float
    radius: float
    azimuth_deg: float
    normal_x: float
    normal_y: float
    normal_z: float


@dataclass(frozen=True)
class PanelCentre:
    panel_id: int
    x: float
    y: float
    z: float
    radius: float


@dataclass(frozen=True)
class SunCase:
    scenario_key: str
    scenario_label: str
    date_iso: str
    day_of_year: int
    time_key: str
    hour_angle_deg: float
    solar_declination_deg: float
    sun_x: float
    sun_y: float
    sun_z: float


@dataclass(frozen=True)
class VisibilityResult:
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


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    return data


def ensure_panel_grid_exists() -> None:
    if CONTROL_POINTS_CSV.exists() and PANEL_GRID_CSV.exists():
        return

    script_path = PROJECT_ROOT / "scripts" / "24_generate_panel_grid.py"

    if not script_path.exists():
        raise FileNotFoundError(
            "Stage 8 panel grid is missing and scripts/24_generate_panel_grid.py was not found."
        )

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def deg_to_rad(value_deg: float) -> float:
    return math.radians(value_deg)


def rad_to_deg(value_rad: float) -> float:
    return math.degrees(value_rad)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_acos_deg(value: float) -> float:
    return rad_to_deg(math.acos(clamp(value, -1.0, 1.0)))


def solar_declination_deg(day_of_year: int) -> float:
    argument_deg = 360.0 * (284.0 + day_of_year) / 365.0
    return 23.45 * math.sin(deg_to_rad(argument_deg))


def horizon_half_angle_deg(
    latitude_deg: float,
    declination_deg: float,
    horizon_altitude_deg: float = -0.833,
) -> float:
    phi = deg_to_rad(latitude_deg)
    delta = deg_to_rad(declination_deg)
    h0 = deg_to_rad(horizon_altitude_deg)

    denominator = math.cos(phi) * math.cos(delta)

    if abs(denominator) < 1.0e-15:
        return 0.0

    cos_h = (math.sin(h0) - math.sin(phi) * math.sin(delta)) / denominator

    return safe_acos_deg(cos_h)


def front_side_half_angle_deg(
    front_normal_declination_deg: float,
    declination_deg: float,
) -> float:
    delta_n = deg_to_rad(front_normal_declination_deg)
    delta = deg_to_rad(declination_deg)

    cos_h = -math.tan(delta_n) * math.tan(delta)

    return safe_acos_deg(cos_h)


def normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vector
    length = math.sqrt(x * x + y * y + z * z)

    if length <= 0.0:
        raise ValueError("Cannot normalize zero vector.")

    return x / length, y / length, z / length


def sun_vector_global_enu(
    latitude_deg: float,
    declination_deg: float,
    hour_angle_deg: float,
) -> tuple[float, float, float]:
    phi = deg_to_rad(latitude_deg)
    delta = deg_to_rad(declination_deg)
    h = deg_to_rad(hour_angle_deg)

    east = -math.cos(delta) * math.sin(h)
    north = math.cos(phi) * math.sin(delta) - math.sin(phi) * math.cos(delta) * math.cos(h)
    up = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(h)

    return normalize((east, north, up))


def rotate_global_to_reflector_local(
    vector_enu: tuple[float, float, float],
    southward_tilt_deg: float,
) -> tuple[float, float, float]:
    x, y, z = vector_enu

    psi = deg_to_rad(southward_tilt_deg)

    local_x = x
    local_y = math.cos(psi) * y + math.sin(psi) * z
    local_z = -math.sin(psi) * y + math.cos(psi) * z

    return normalize((local_x, local_y, local_z))


def read_control_points() -> list[ControlPoint]:
    points: list[ControlPoint] = []

    with CONTROL_POINTS_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            points.append(
                ControlPoint(
                    panel_id=int(row["panel_id"]),
                    control_point_id=row["control_point_id"],
                    x=float(row["x_m"]),
                    y=float(row["y_m"]),
                    z=float(row["z_m"]),
                    radius=float(row["projected_radius_m"]),
                    azimuth_deg=float(row["azimuth_deg"]),
                    normal_x=float(row["normal_x"]),
                    normal_y=float(row["normal_y"]),
                    normal_z=float(row["normal_z"]),
                )
            )

    return points


def read_panel_centres() -> dict[int, PanelCentre]:
    panels: dict[int, PanelCentre] = {}

    with PANEL_GRID_CSV.open("r", encoding="utf-8-sig", newline="") as file:
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


def point_segment_distance_2d(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx = bx - ax
    aby = by - ay

    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby

    if ab2 <= 1.0e-15:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab2
    t = clamp(t, 0.0, 1.0)

    closest_x = ax + t * abx
    closest_y = ay + t * aby

    return math.hypot(px - closest_x, py - closest_y)


def ray_intersects_vertical_cylinder(
    point: ControlPoint,
    sun_direction: tuple[float, float, float],
    obstacle: dict,
    epsilon: float,
) -> bool:
    """
    Robust ray-volume intersection with a finite vertical cylinder.

    The cylinder is defined by:
    - circular footprint in the x-y plane;
    - finite vertical interval z_min <= z <= z_max.

    This version correctly handles the near-axial solar-noon case:
    dx ~= 0, dy ~= 0, dz > 0.

    That case is essential for the secondary mirror shadow because
    central reflector points must be shadowed when the Sun direction
    is nearly aligned with the reflector-local z axis.
    """

    cx = float(obstacle["center_x_m"])
    cy = float(obstacle["center_y_m"])
    radius = float(obstacle["radius_m"])
    z_min = float(obstacle["z_min_m"])
    z_max = float(obstacle["z_max_m"])

    dx, dy, dz = sun_direction

    # The ray must pass through the obstacle's vertical interval.
    # If dz is zero, this simplified vertical-cylinder test cannot
    # enter a higher/lower finite z interval from the surface point.
    if abs(dz) < 1.0e-15:
        return False

    t_z0 = (z_min - point.z) / dz
    t_z1 = (z_max - point.z) / dz

    t_z_low = min(t_z0, t_z1)
    t_z_high = max(t_z0, t_z1)

    t_low = max(t_z_low, epsilon)
    t_high = t_z_high

    if t_high < t_low:
        return False

    ox = point.x - cx
    oy = point.y - cy

    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius

    # Near-vertical ray in x-y.
    # If the ray footprint is already inside the cylinder radius,
    # it intersects the finite cylinder volume across the z interval.
    if abs(a) < 1.0e-15:
        return c <= epsilon

    discriminant = b * b - 4.0 * a * c

    if discriminant < 0.0:
        # No side-wall radial crossing. If the ray starts inside the
        # cylinder footprint, it still intersects the finite volume.
        return c <= epsilon

    root = math.sqrt(discriminant)

    t_r0 = (-b - root) / (2.0 * a)
    t_r1 = (-b + root) / (2.0 * a)

    t_r_low = min(t_r0, t_r1)
    t_r_high = max(t_r0, t_r1)

    # Finite cylinder intersection exists if the radial-valid interval
    # overlaps the vertical-valid interval.
    overlap_low = max(t_low, t_r_low)
    overlap_high = min(t_high, t_r_high)

    return overlap_high >= overlap_low
def ray_intersects_vertical_capsule_prism(
    point: ControlPoint,
    sun_direction: tuple[float, float, float],
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    width_m: float,
    z_min_m: float,
    z_max_m: float,
    samples_per_ray: int,
    epsilon: float,
) -> bool:
    dx, dy, dz = sun_direction

    if abs(dz) < 1.0e-15:
        return False

    t0 = (z_min_m - point.z) / dz
    t1 = (z_max_m - point.z) / dz

    t_low = min(t0, t1)
    t_high = max(t0, t1)

    t_low = max(t_low, epsilon)

    if t_high <= t_low:
        return False

    half_width = 0.5 * width_m

    if samples_per_ray < 2:
        samples_per_ray = 2

    for index in range(samples_per_ray):
        fraction = index / (samples_per_ray - 1)
        t = t_low + fraction * (t_high - t_low)

        x = point.x + t * dx
        y = point.y + t * dy

        distance = point_segment_distance_2d(
            px=x,
            py=y,
            ax=start_x,
            ay=start_y,
            bx=end_x,
            by=end_y,
        )

        if distance <= half_width:
            return True

    return False


def tripod_segments(config: dict) -> list[tuple[float, float, float, float]]:
    tripod = config["ray_shadow_model"]["obstacles"]["tripod_legs"]

    angles = tripod["angles_deg"]
    start_radius = float(tripod["start_radius_m"])
    end_radius = float(tripod["end_radius_m"])

    segments = []

    for angle_deg in angles:
        angle_rad = math.radians(float(angle_deg))

        start_x = start_radius * math.cos(angle_rad)
        start_y = start_radius * math.sin(angle_rad)

        end_x = end_radius * math.cos(angle_rad)
        end_y = end_radius * math.sin(angle_rad)

        segments.append((start_x, start_y, end_x, end_y))

    return segments


def shadow_source_for_point(
    point: ControlPoint,
    sun_direction: tuple[float, float, float],
    config: dict,
) -> str:
    model = config["ray_shadow_model"]
    obstacles = model["obstacles"]
    numerical = model["numerical"]

    epsilon = float(numerical["epsilon"])
    samples_per_ray = int(numerical["capsule_samples_per_ray"])

    for key in ["secondary_mirror", "optical_reflector", "central_hub"]:
        obstacle = obstacles[key]

        if not bool(obstacle["enabled"]):
            continue

        if ray_intersects_vertical_cylinder(
            point=point,
            sun_direction=sun_direction,
            obstacle=obstacle,
            epsilon=epsilon,
        ):
            return str(obstacle["label"])

    tripod = obstacles["tripod_legs"]

    if bool(tripod["enabled"]):
        width = float(tripod["width_m"])
        z_min = float(tripod["z_min_m"])
        z_max = float(tripod["z_max_m"])

        for index, (start_x, start_y, end_x, end_y) in enumerate(
            tripod_segments(config),
            start=1,
        ):
            if ray_intersects_vertical_capsule_prism(
                point=point,
                sun_direction=sun_direction,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                width_m=width,
                z_min_m=z_min,
                z_max_m=z_max,
                samples_per_ray=samples_per_ray,
                epsilon=epsilon,
            ):
                return f"{tripod['label']}_{index}"

    return "none"


def build_sun_cases(config: dict) -> list[SunCase]:
    model = config["ray_shadow_model"]

    latitude = float(model["site"]["latitude_deg"])
    tilt = float(model["reflector"]["southward_axis_tilt_deg"])
    normal_declination = latitude - tilt

    cases: list[SunCase] = []

    for scenario_key, scenario in model["scenarios"].items():
        day_of_year = int(scenario["day_of_year"])
        declination = solar_declination_deg(day_of_year)

        horizon_half = horizon_half_angle_deg(
            latitude_deg=latitude,
            declination_deg=declination,
        )

        front_half = front_side_half_angle_deg(
            front_normal_declination_deg=normal_declination,
            declination_deg=declination,
        )

        active_half = min(horizon_half, front_half)

        for time_case in model["sun_model"]["selected_times"]:
            time_key = str(time_case["key"])
            fraction = float(time_case["active_half_angle_fraction"])
            hour_angle = fraction * active_half

            sun_global = sun_vector_global_enu(
                latitude_deg=latitude,
                declination_deg=declination,
                hour_angle_deg=hour_angle,
            )

            sun_local = rotate_global_to_reflector_local(
                vector_enu=sun_global,
                southward_tilt_deg=tilt,
            )

            cases.append(
                SunCase(
                    scenario_key=scenario_key,
                    scenario_label=str(scenario["label"]),
                    date_iso=str(scenario["date"]),
                    day_of_year=day_of_year,
                    time_key=time_key,
                    hour_angle_deg=hour_angle,
                    solar_declination_deg=declination,
                    sun_x=sun_local[0],
                    sun_y=sun_local[1],
                    sun_z=sun_local[2],
                )
            )

    return cases


def compute_visibility(
    control_points: list[ControlPoint],
    config: dict,
) -> list[VisibilityResult]:
    results: list[VisibilityResult] = []
    sun_cases = build_sun_cases(config)

    for sun_case in sun_cases:
        sun_direction = (sun_case.sun_x, sun_case.sun_y, sun_case.sun_z)

        for point in control_points:
            mu_front = (
                point.normal_x * sun_case.sun_x
                + point.normal_y * sun_case.sun_y
                + point.normal_z * sun_case.sun_z
            )

            if mu_front <= 0.0 or sun_case.sun_z <= 0.0:
                chi = 0
                shadow_source = "not_front_side"
            else:
                source = shadow_source_for_point(
                    point=point,
                    sun_direction=sun_direction,
                    config=config,
                )

                if source == "none":
                    chi = 1
                    shadow_source = "none"
                else:
                    chi = 0
                    shadow_source = source

            results.append(
                VisibilityResult(
                    scenario_key=sun_case.scenario_key,
                    scenario_label=sun_case.scenario_label,
                    date_iso=sun_case.date_iso,
                    day_of_year=sun_case.day_of_year,
                    time_key=sun_case.time_key,
                    hour_angle_deg=sun_case.hour_angle_deg,
                    solar_declination_deg=sun_case.solar_declination_deg,
                    panel_id=point.panel_id,
                    control_point_id=point.control_point_id,
                    x=point.x,
                    y=point.y,
                    z=point.z,
                    mu_front=max(0.0, mu_front),
                    chi=chi,
                    shadow_source=shadow_source,
                )
            )

    return results


def write_visibility_csv(results: Iterable[VisibilityResult]) -> None:
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
        "shadow_source",
    ]

    with CONTROL_POINT_VISIBILITY_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "scenario_key": result.scenario_key,
                    "scenario_label": result.scenario_label,
                    "date": result.date_iso,
                    "day_of_year": result.day_of_year,
                    "time_key": result.time_key,
                    "hour_angle_deg": f"{result.hour_angle_deg:.9f}",
                    "solar_declination_deg": f"{result.solar_declination_deg:.9f}",
                    "panel_id": result.panel_id,
                    "control_point_id": result.control_point_id,
                    "x_m": f"{result.x:.9f}",
                    "y_m": f"{result.y:.9f}",
                    "z_m": f"{result.z:.9f}",
                    "mu_front": f"{result.mu_front:.12f}",
                    "chi": result.chi,
                    "shadow_source": result.shadow_source,
                }
            )


def build_panel_summary(results: list[VisibilityResult]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, int], list[VisibilityResult]] = defaultdict(list)

    for result in results:
        grouped[(result.scenario_key, result.time_key, result.panel_id)].append(result)

    rows: list[dict[str, str]] = []

    for (scenario_key, time_key, panel_id), items in sorted(grouped.items()):
        scenario_label = items[0].scenario_label
        date_iso = items[0].date_iso
        day_of_year = items[0].day_of_year

        shadowed_count = sum(1 for item in items if item.chi == 0)
        illuminated_count = sum(1 for item in items if item.chi == 1)
        shadow_fraction = shadowed_count / len(items)
        mean_mu_front = sum(item.mu_front for item in items) / len(items)

        sources = [
            item.shadow_source
            for item in items
            if item.shadow_source not in {"none"}
        ]

        if sources:
            primary_source = Counter(sources).most_common(1)[0][0]
        else:
            primary_source = "none"

        rows.append(
            {
                "scenario_key": scenario_key,
                "scenario_label": scenario_label,
                "date": date_iso,
                "day_of_year": str(day_of_year),
                "time_key": time_key,
                "panel_id": str(panel_id),
                "control_point_count": str(len(items)),
                "illuminated_count": str(illuminated_count),
                "shadowed_count": str(shadowed_count),
                "shadow_fraction": f"{shadow_fraction:.9f}",
                "mean_mu_front": f"{mean_mu_front:.12f}",
                "primary_shadow_source": primary_source,
            }
        )

    return rows


def write_panel_summary_csv(rows: list[dict[str, str]]) -> None:
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
        "shadow_fraction",
        "mean_mu_front",
        "primary_shadow_source",
    ]

    with PANEL_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_scenario_summary(results: list[VisibilityResult]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[VisibilityResult]] = defaultdict(list)

    for result in results:
        grouped[(result.scenario_key, result.time_key)].append(result)

    rows: list[dict[str, str]] = []

    for (scenario_key, time_key), items in sorted(grouped.items()):
        total = len(items)
        illuminated = sum(1 for item in items if item.chi == 1)
        shadowed = total - illuminated

        source_counts = Counter(item.shadow_source for item in items)

        rows.append(
            {
                "scenario_key": scenario_key,
                "scenario_label": items[0].scenario_label,
                "date": items[0].date_iso,
                "day_of_year": str(items[0].day_of_year),
                "time_key": time_key,
                "control_point_count": str(total),
                "illuminated_count": str(illuminated),
                "shadowed_count": str(shadowed),
                "shadow_fraction": f"{shadowed / total:.9f}",
                "mean_mu_front": f"{sum(item.mu_front for item in items) / total:.12f}",
                "not_front_side_count": str(source_counts.get("not_front_side", 0)),
                "central_hub_count": str(source_counts.get("central_hub", 0)),
                "secondary_mirror_count": str(source_counts.get("secondary_mirror", 0)),
                "optical_reflector_count": str(source_counts.get("optical_reflector", 0)),
                "tripod_count": str(
                    sum(
                        count
                        for source, count in source_counts.items()
                        if source.startswith("tripod_legs_internal_bases")
                    )
                ),
            }
        )

    return rows


def write_scenario_summary_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "scenario_key",
        "scenario_label",
        "date",
        "day_of_year",
        "time_key",
        "control_point_count",
        "illuminated_count",
        "shadowed_count",
        "shadow_fraction",
        "mean_mu_front",
        "not_front_side_count",
        "central_hub_count",
        "secondary_mirror_count",
        "optical_reflector_count",
        "tripod_count",
    ]

    with SCENARIO_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def panel_shadow_lookup(panel_summary_rows: list[dict[str, str]]) -> dict[tuple[str, str, int], float]:
    lookup = {}

    for row in panel_summary_rows:
        key = (
            row["scenario_key"],
            row["time_key"],
            int(row["panel_id"]),
        )

        lookup[key] = float(row["shadow_fraction"])

    return lookup


def plot_shadow_maps(
    panels: dict[int, PanelCentre],
    panel_summary_rows: list[dict[str, str]],
    config: dict,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    lookup = panel_shadow_lookup(panel_summary_rows)

    scenario_keys = sorted({row["scenario_key"] for row in panel_summary_rows})
    time_keys = ["morning", "noon", "evening"]

    tripod = config["ray_shadow_model"]["obstacles"]["tripod_legs"]
    start_radius = float(tripod["start_radius_m"])
    end_radius = float(tripod["end_radius_m"])
    angles = [float(value) for value in tripod["angles_deg"]]

    for scenario_key in scenario_keys:
        for time_key in time_keys:
            x_values = []
            y_values = []
            shadow_values = []

            for panel_id, panel in panels.items():
                key = (scenario_key, time_key, panel_id)

                if key not in lookup:
                    continue

                x_values.append(panel.x)
                y_values.append(panel.y)
                shadow_values.append(lookup[key])

            fig, ax = plt.subplots(figsize=(8, 8))

            scatter = ax.scatter(
                x_values,
                y_values,
                c=shadow_values,
                s=3,
                vmin=0.0,
                vmax=1.0,
                cmap="viridis_r",
            )

            aperture = plt.Circle(
                (0.0, 0.0),
                27.0,
                fill=False,
                linewidth=2.0,
            )
            ax.add_patch(aperture)

            tripod_radius = plt.Circle(
                (0.0, 0.0),
                end_radius,
                fill=False,
                linestyle="--",
                linewidth=2.0,
            )
            ax.add_patch(tripod_radius)

            for angle_deg in angles:
                angle_rad = math.radians(angle_deg)
                x0 = start_radius * math.cos(angle_rad)
                y0 = start_radius * math.sin(angle_rad)
                x1 = end_radius * math.cos(angle_rad)
                y1 = end_radius * math.sin(angle_rad)

                ax.plot([x0, x1], [y0, y1], linewidth=2.0)

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-28.5, 28.5)
            ax.set_ylim(-28.5, 28.5)
            ax.set_xlabel("x, m")
            ax.set_ylabel("y, m")
            ax.set_title(f"Ray-based shadow fraction — {scenario_key}, {time_key}")
            ax.grid(True, alpha=0.3)

            colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.set_label("Panel shadow fraction")

            output_path = FIGURE_DIR / f"ray_shadow_{scenario_key}_{time_key}.png"

            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close(fig)


def print_summary(scenario_summary_rows: list[dict[str, str]]) -> None:
    print("Stage 9 — Numerical ray-based visibility function")
    print("=" * 60)

    for row in scenario_summary_rows:
        print(
            f"{row['scenario_key']:17s} {row['time_key']:8s} | "
            f"shadow_fraction={float(row['shadow_fraction']):.4f} | "
            f"tripod={row['tripod_count']} | "
            f"secondary={row['secondary_mirror_count']} | "
            f"hub={row['central_hub_count']}"
        )

    print()
    print(f"Saved control-point visibility: {CONTROL_POINT_VISIBILITY_CSV}")
    print(f"Saved panel summary:            {PANEL_SUMMARY_CSV}")
    print(f"Saved scenario summary:         {SCENARIO_SUMMARY_CSV}")
    print(f"Saved figures directory:        {FIGURE_DIR}")


def main() -> None:
    ensure_panel_grid_exists()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_yaml(CONFIG_PATH)
    control_points = read_control_points()
    panels = read_panel_centres()

    results = compute_visibility(
        control_points=control_points,
        config=config,
    )

    write_visibility_csv(results)

    panel_rows = build_panel_summary(results)
    write_panel_summary_csv(panel_rows)

    scenario_rows = build_scenario_summary(results)
    write_scenario_summary_csv(scenario_rows)

    plot_shadow_maps(
        panels=panels,
        panel_summary_rows=panel_rows,
        config=config,
    )

    print_summary(scenario_rows)


if __name__ == "__main__":
    main()

