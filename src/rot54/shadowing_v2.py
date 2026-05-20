from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CylinderBlocker:
    key: str
    label: str
    center_x_m: float
    center_y_m: float
    radius_m: float
    z_min_m: float
    z_max_m: float
    enabled: bool = True


@dataclass(frozen=True)
class SupportArmPrismModel:
    label: str
    angles_deg: tuple[float, ...]
    width_m: float
    start_radius_m: float
    end_radius_m: float
    z_min_m: float
    z_max_m: float
    enabled: bool = True


@dataclass(frozen=True)
class ShadowV2Parameters:
    axis_tilt_south_deg: float
    numerical_epsilon: float
    cylinders: tuple[CylinderBlocker, ...]
    support_arm_prisms: SupportArmPrismModel | None


def bool_series(series: pd.Series) -> pd.Series:
    """
    Robust conversion of CSV-loaded bool-like values.
    """
    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def solar_vector_global_enu_to_reflector_local(
    solar_east: float,
    solar_north: float,
    solar_up: float,
    axis_tilt_south_deg: float,
) -> tuple[float, float, float]:
    """
    Convert solar vector from global ENU into reflector-local coordinates.

    Global ENU:
        E = east
        N = north
        U = up

    Reflector-local:
        x = aperture-plane x
        y = aperture-plane y
        z = local central front-facing normal before global tilt

    Inverse of:
        E = x
        N = cos(psi) y - sin(psi) z
        U = sin(psi) y + cos(psi) z
    """
    psi = np.deg2rad(axis_tilt_south_deg)
    c = np.cos(psi)
    s = np.sin(psi)

    local_x = solar_east
    local_y = c * solar_north + s * solar_up
    local_z = -s * solar_north + c * solar_up

    vec = np.array([local_x, local_y, local_z], dtype=float)
    norm = np.linalg.norm(vec)

    if norm <= 0.0:
        raise ValueError("Solar vector norm became zero in local coordinates.")

    vec = vec / norm

    return float(vec[0]), float(vec[1]), float(vec[2])


def ray_interval_against_slab(
    p: np.ndarray,
    d: float,
    lower: float,
    upper: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ray-slab intersection for one coordinate.

    Ray:
        p(t) = p + t d, t >= 0

    Slab:
        lower <= p(t) <= upper

    Returns interval [t_min, t_max].
    Invalid rays receive an empty interval.
    """
    p = np.asarray(p, dtype=float)

    if lower > upper:
        lower, upper = upper, lower

    if abs(d) <= eps:
        inside = (p >= lower) & (p <= upper)
        t_min = np.full_like(p, -np.inf, dtype=float)
        t_max = np.full_like(p, np.inf, dtype=float)

        t_min[~inside] = np.inf
        t_max[~inside] = -np.inf

        return t_min, t_max

    t0 = (lower - p) / d
    t1 = (upper - p) / d

    t_min = np.minimum(t0, t1)
    t_max = np.maximum(t0, t1)

    return t_min, t_max


def ray_intersects_vertical_cylinder(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sx: float,
    sy: float,
    sz: float,
    cylinder: CylinderBlocker,
    eps: float,
) -> np.ndarray:
    """
    Vectorized ray intersection with a vertical finite cylinder.

    Cylinder:
        (x - cx)^2 + (y - cy)^2 <= radius^2
        z_min <= z <= z_max

    Ray:
        p(t) = p0 + t s, t >= 0
    """
    if not cylinder.enabled:
        return np.zeros_like(x, dtype=bool)

    if cylinder.radius_m <= 0.0:
        raise ValueError(f"Cylinder radius must be positive: {cylinder.key}")

    z_min, z_max = sorted([cylinder.z_min_m, cylinder.z_max_m])

    t_z_min, t_z_max = ray_interval_against_slab(
        p=z,
        d=sz,
        lower=z_min,
        upper=z_max,
        eps=eps,
    )

    dx = x - cylinder.center_x_m
    dy = y - cylinder.center_y_m

    a = sx * sx + sy * sy

    if a <= eps:
        inside_xy = (dx * dx + dy * dy) <= cylinder.radius_m**2

        t_xy_min = np.full_like(x, -np.inf, dtype=float)
        t_xy_max = np.full_like(x, np.inf, dtype=float)

        t_xy_min[~inside_xy] = np.inf
        t_xy_max[~inside_xy] = -np.inf

    else:
        b = 2.0 * (dx * sx + dy * sy)
        c = dx * dx + dy * dy - cylinder.radius_m**2

        disc = b * b - 4.0 * a * c

        valid_disc = disc >= 0.0

        sqrt_disc = np.zeros_like(disc, dtype=float)
        sqrt_disc[valid_disc] = np.sqrt(disc[valid_disc])

        root0 = (-b - sqrt_disc) / (2.0 * a)
        root1 = (-b + sqrt_disc) / (2.0 * a)

        t_xy_min = np.minimum(root0, root1)
        t_xy_max = np.maximum(root0, root1)

        t_xy_min[~valid_disc] = np.inf
        t_xy_max[~valid_disc] = -np.inf

    t_enter = np.maximum.reduce(
        [
            t_z_min,
            t_xy_min,
            np.zeros_like(x, dtype=float),
        ]
    )

    t_exit = np.minimum(t_z_max, t_xy_max)

    return t_exit >= t_enter


def ray_intersects_support_arm_prisms(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sx: float,
    sy: float,
    sz: float,
    support: SupportArmPrismModel,
    eps: float,
) -> np.ndarray:
    """
    Vectorized ray intersection with radial rectangular-prism support arms.

    Each arm is represented in its own local 2D basis:
        u = along arm axis
        v = perpendicular to arm axis

    Prism bounds:
        start_radius <= u <= end_radius
        -width/2 <= v <= width/2
        z_min <= z <= z_max
    """
    if support is None or not support.enabled:
        return np.zeros_like(x, dtype=bool)

    if support.width_m <= 0.0:
        raise ValueError("Support-arm width must be positive.")

    if support.end_radius_m <= support.start_radius_m:
        raise ValueError("Support-arm end radius must be greater than start radius.")

    z_min, z_max = sorted([support.z_min_m, support.z_max_m])
    half_width = 0.5 * support.width_m

    combined = np.zeros_like(x, dtype=bool)

    for angle_deg in support.angles_deg:
        angle_rad = np.deg2rad(angle_deg)

        ux = np.cos(angle_rad)
        uy = np.sin(angle_rad)

        # Coordinates in the local arm frame.
        p_u = x * ux + y * uy
        p_v = -x * uy + y * ux

        d_u = sx * ux + sy * uy
        d_v = -sx * uy + sy * ux

        t_u_min, t_u_max = ray_interval_against_slab(
            p=p_u,
            d=d_u,
            lower=support.start_radius_m,
            upper=support.end_radius_m,
            eps=eps,
        )

        t_v_min, t_v_max = ray_interval_against_slab(
            p=p_v,
            d=d_v,
            lower=-half_width,
            upper=half_width,
            eps=eps,
        )

        t_z_min, t_z_max = ray_interval_against_slab(
            p=z,
            d=sz,
            lower=z_min,
            upper=z_max,
            eps=eps,
        )

        t_enter = np.maximum.reduce(
            [
                t_u_min,
                t_v_min,
                t_z_min,
                np.zeros_like(x, dtype=float),
            ]
        )

        t_exit = np.minimum.reduce(
            [
                t_u_max,
                t_v_max,
                t_z_max,
            ]
        )

        hit = t_exit >= t_enter
        combined = combined | hit

    return combined


def build_shadow_v2_parameters(
    main_config: dict[str, Any],
    shadow_v2_config: dict[str, Any],
) -> ShadowV2Parameters:
    """
    Build V2 shadow model from main and V2 configs.
    """
    raw = shadow_v2_config["shadow_v2"]

    cylinders: list[CylinderBlocker] = []

    for key, item in raw["circular_cylinders"].items():
        cylinders.append(
            CylinderBlocker(
                key=str(key),
                label=str(item.get("label", key)),
                center_x_m=float(item["center_x_m"]),
                center_y_m=float(item["center_y_m"]),
                radius_m=float(item["radius_m"]),
                z_min_m=float(item["z_min_m"]),
                z_max_m=float(item["z_max_m"]),
                enabled=bool(item.get("enabled", True)),
            )
        )

    support_raw = raw.get("support_arm_prisms", None)

    support: SupportArmPrismModel | None

    if support_raw is None:
        support = None
    else:
        support = SupportArmPrismModel(
            label=str(support_raw.get("label", "support_arm_prisms")),
            angles_deg=tuple(float(v) for v in support_raw["angles_deg"]),
            width_m=float(support_raw["width_m"]),
            start_radius_m=float(support_raw["start_radius_m"]),
            end_radius_m=float(support_raw["end_radius_m"]),
            z_min_m=float(support_raw["z_min_m"]),
            z_max_m=float(support_raw["z_max_m"]),
            enabled=bool(support_raw.get("enabled", True)),
        )

    return ShadowV2Parameters(
        axis_tilt_south_deg=float(main_config["main_reflector"]["axis_tilt_south_deg"]),
        numerical_epsilon=float(raw["numerical_epsilon"]),
        cylinders=tuple(cylinders),
        support_arm_prisms=support,
    )


def apply_shadow_v2(
    incidence: pd.DataFrame,
    params: ShadowV2Parameters,
) -> pd.DataFrame:
    """
    Apply volumetric structural shadowing.

    Output:
        structure_shadow_v2
        visibility_chi_v2
        effective_solar_factor_v2

    Formula:
        effective_solar_factor_v2 = max(cos(theta_i), 0) * visibility_chi_v2
    """
    required = [
        "x_m",
        "y_m",
        "z_m",
        "cos_incidence",
        "potential_sunlit",
        "solar_east",
        "solar_north",
        "solar_up",
    ]

    missing = [col for col in required if col not in incidence.columns]
    if missing:
        raise ValueError(f"Incidence table is missing columns: {missing}")

    result = incidence.copy()

    x = result["x_m"].to_numpy(dtype=float)
    y = result["y_m"].to_numpy(dtype=float)
    z = result["z_m"].to_numpy(dtype=float)

    solar_e = float(result["solar_east"].iloc[0])
    solar_n = float(result["solar_north"].iloc[0])
    solar_u = float(result["solar_up"].iloc[0])

    sx, sy, sz = solar_vector_global_enu_to_reflector_local(
        solar_east=solar_e,
        solar_north=solar_n,
        solar_up=solar_u,
        axis_tilt_south_deg=params.axis_tilt_south_deg,
    )

    result["solar_local_x_v2"] = sx
    result["solar_local_y_v2"] = sy
    result["solar_local_z_v2"] = sz

    combined_shadow = np.zeros(len(result), dtype=bool)

    for cylinder in params.cylinders:
        hit = ray_intersects_vertical_cylinder(
            x=x,
            y=y,
            z=z,
            sx=sx,
            sy=sy,
            sz=sz,
            cylinder=cylinder,
            eps=params.numerical_epsilon,
        )

        result[f"shadow_v2_{cylinder.key}"] = hit
        combined_shadow = combined_shadow | hit

    if params.support_arm_prisms is not None:
        support_hit = ray_intersects_support_arm_prisms(
            x=x,
            y=y,
            z=z,
            sx=sx,
            sy=sy,
            sz=sz,
            support=params.support_arm_prisms,
            eps=params.numerical_epsilon,
        )
    else:
        support_hit = np.zeros(len(result), dtype=bool)

    result["shadow_v2_support_arm_prisms"] = support_hit
    combined_shadow = combined_shadow | support_hit

    potential = bool_series(result["potential_sunlit"]).to_numpy(dtype=bool)

    structure_shadow = combined_shadow & potential
    visible = potential & (~structure_shadow)

    chi = visible.astype(float)
    cos_positive = np.maximum(result["cos_incidence"].to_numpy(dtype=float), 0.0)

    result["structure_shadow_v2"] = structure_shadow
    result["visibility_chi_v2"] = chi
    result["effective_solar_factor_v2"] = cos_positive * chi

    return result


def summarize_shadow_v2(
    case_key: str,
    case_label: str,
    time_code: str,
    selected_local_time: str,
    solar_altitude_deg: float,
    solar_azimuth_deg: float,
    shadow: pd.DataFrame,
) -> dict[str, object]:
    potential = bool_series(shadow["potential_sunlit"])
    structure = bool_series(shadow["structure_shadow_v2"])
    visible = shadow["visibility_chi_v2"] > 0.5

    total = len(shadow)
    potential_count = int(potential.sum())
    structure_count = int(structure.sum())
    visible_count = int(visible.sum())

    if potential_count > 0:
        shadow_fraction = structure_count / potential_count
        visible_fraction_potential = visible_count / potential_count
    else:
        shadow_fraction = np.nan
        visible_fraction_potential = np.nan

    visible_rows = shadow[visible].copy()

    if visible_rows.empty:
        mean_factor = np.nan
        max_factor = np.nan
    else:
        mean_factor = float(visible_rows["effective_solar_factor_v2"].mean())
        max_factor = float(visible_rows["effective_solar_factor_v2"].max())

    return {
        "case_key": case_key,
        "case_label": case_label,
        "time_code": time_code,
        "selected_local_time": selected_local_time,
        "solar_altitude_deg": solar_altitude_deg,
        "solar_azimuth_deg": solar_azimuth_deg,
        "aperture_points": total,
        "potential_sunlit_points": potential_count,
        "structure_shadow_points_v2": structure_count,
        "visible_sunlit_points_v2": visible_count,
        "shadow_fraction_of_potential_sunlit_v2": shadow_fraction,
        "visible_fraction_of_potential_sunlit_v2": visible_fraction_potential,
        "visible_fraction_of_aperture_v2": visible_count / total if total else np.nan,
        "effective_solar_factor_mean_visible_v2": mean_factor,
        "effective_solar_factor_max_visible_v2": max_factor,
    }


def validate_shadow_v2(
    case_key: str,
    time_code: str,
    shadow: pd.DataFrame,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    chi = shadow["visibility_chi_v2"].to_numpy(dtype=float)
    factor = shadow["effective_solar_factor_v2"].to_numpy(dtype=float)
    cos_positive = np.maximum(shadow["cos_incidence"].to_numpy(dtype=float), 0.0)

    potential = bool_series(shadow["potential_sunlit"]).to_numpy(dtype=bool)
    structure = bool_series(shadow["structure_shadow_v2"]).to_numpy(dtype=bool)

    expected_factor = cos_positive * chi

    chi_binary_ok = bool(np.all((chi == 0.0) | (chi == 1.0)))
    shadow_only_inside_potential_ok = bool(np.all(structure <= potential))
    factor_error = float(np.max(np.abs(factor - expected_factor)))
    factor_equation_ok = bool(factor_error <= tolerance)
    factor_range_ok = bool(np.nanmin(factor) >= -tolerance and np.nanmax(factor) <= 1.0 + tolerance)

    all_ok = bool(
        chi_binary_ok
        and shadow_only_inside_potential_ok
        and factor_equation_ok
        and factor_range_ok
    )

    return {
        "case_key": case_key,
        "time_code": time_code,
        "chi_binary_ok": chi_binary_ok,
        "shadow_only_inside_potential_ok": shadow_only_inside_potential_ok,
        "factor_equation_ok": factor_equation_ok,
        "factor_range_ok": factor_range_ok,
        "max_factor_equation_error": factor_error,
        "all_shadow_v2_checks_ok": all_ok,
    }


def save_shadow_v2(
    shadow: pd.DataFrame,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shadow.to_csv(path, index=False, encoding="utf-8")
