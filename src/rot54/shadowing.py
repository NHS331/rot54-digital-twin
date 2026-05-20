from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CircularBlocker:
    """
    Circular obstruction footprint in a shadow reference plane.

    The shadow test is performed as follows:
        1. Take a surface point p = (x, y, z).
        2. Take the solar direction vector s in reflector-local coordinates.
        3. Trace from the surface point toward the Sun:
               p_shadow = p + t s
        4. Intersect with the obstruction plane:
               z_shadow = z_plane_m
               t = (z_plane_m - z) / s_z
        5. If t >= 0 and (x_shadow, y_shadow) is inside the circular footprint,
           then the original surface point is shadowed.

    This is not a full CAD ray-tracing model. It is a reduced, transparent,
    parameterized engineering model.
    """

    key: str
    label: str
    center_x_m: float
    center_y_m: float
    radius_m: float
    z_plane_m: float
    enabled: bool = True


@dataclass(frozen=True)
class SupportArmModel:
    """
    Rectangular radial support-arm obstruction model.

    The arms are represented as finite-width radial strips in a shadow plane.
    For each projected point q = (x_shadow, y_shadow), the model checks whether
    q lies within any support-arm strip.

    angles_deg:
        Arm-axis angles in the reflector-local aperture plane.
        0 deg points along +x.
        90 deg points along +y.
    """

    label: str
    angles_deg: tuple[float, ...]
    width_m: float
    start_radius_m: float
    end_radius_m: float
    z_plane_m: float
    enabled: bool = True


@dataclass(frozen=True)
class ShadowModelParameters:
    """
    Full first-stage structural shadowing model.
    """

    axis_tilt_south_deg: float
    sun_vector_local_z_epsilon: float
    circular_blockers: tuple[CircularBlocker, ...]
    support_arms: SupportArmModel | None


def solar_vector_global_enu_to_reflector_local(
    solar_east: float,
    solar_north: float,
    solar_up: float,
    axis_tilt_south_deg: float,
) -> tuple[float, float, float]:
    """
    Convert solar vector from global ENU coordinates into reflector-local coordinates.

    In previous incidence calculations, local vectors were rotated to global ENU as:

        E = x
        N = cos(psi) * y - sin(psi) * z
        U = sin(psi) * y + cos(psi) * z

    Therefore the inverse rotation is:

        x = E
        y = cos(psi) * N + sin(psi) * U
        z = -sin(psi) * N + cos(psi) * U

    where psi is the southward tilt of the main reflector axis.
    """
    psi = np.deg2rad(axis_tilt_south_deg)
    c = np.cos(psi)
    s = np.sin(psi)

    local_x = solar_east
    local_y = c * solar_north + s * solar_up
    local_z = -s * solar_north + c * solar_up

    vector = np.array([local_x, local_y, local_z], dtype=float)
    norm = np.linalg.norm(vector)

    if norm <= 0.0:
        raise ValueError("Solar vector has zero norm after local transformation.")

    vector = vector / norm

    return float(vector[0]), float(vector[1]), float(vector[2])


def project_points_to_shadow_plane(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    solar_local_x: float,
    solar_local_y: float,
    solar_local_z: float,
    z_plane_m: float,
    z_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project surface points toward the Sun until they intersect z = z_plane_m.

    Returns:
        x_shadow_m
        y_shadow_m
        valid_projection

    valid_projection is true only when:
        - solar_local_z is large enough,
        - the intersection is in front of the surface point along the Sun direction.
    """
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    z = np.asarray(z_m, dtype=float)

    x_shadow = np.full_like(x, np.nan, dtype=float)
    y_shadow = np.full_like(y, np.nan, dtype=float)
    valid = np.zeros_like(x, dtype=bool)

    if solar_local_z <= z_epsilon:
        return x_shadow, y_shadow, valid

    t = (float(z_plane_m) - z) / solar_local_z

    valid = t >= 0.0

    x_shadow[valid] = x[valid] + t[valid] * solar_local_x
    y_shadow[valid] = y[valid] + t[valid] * solar_local_y

    return x_shadow, y_shadow, valid


def circular_blocker_shadow_mask(
    incidence: pd.DataFrame,
    blocker: CircularBlocker,
    solar_local_x: float,
    solar_local_y: float,
    solar_local_z: float,
    z_epsilon: float,
) -> np.ndarray:
    """
    Compute shadow mask caused by one circular blocker.
    """
    n = len(incidence)
    mask = np.zeros(n, dtype=bool)

    if not blocker.enabled:
        return mask

    if blocker.radius_m <= 0.0:
        raise ValueError(f"Circular blocker radius must be positive: {blocker.key}")

    x_shadow, y_shadow, valid = project_points_to_shadow_plane(
        x_m=incidence["x_m"].to_numpy(dtype=float),
        y_m=incidence["y_m"].to_numpy(dtype=float),
        z_m=incidence["z_m"].to_numpy(dtype=float),
        solar_local_x=solar_local_x,
        solar_local_y=solar_local_y,
        solar_local_z=solar_local_z,
        z_plane_m=blocker.z_plane_m,
        z_epsilon=z_epsilon,
    )

    dx = x_shadow - blocker.center_x_m
    dy = y_shadow - blocker.center_y_m

    inside_circle = (dx**2 + dy**2) <= blocker.radius_m**2

    mask = valid & inside_circle

    return mask


def support_arms_shadow_mask(
    incidence: pd.DataFrame,
    support_model: SupportArmModel,
    solar_local_x: float,
    solar_local_y: float,
    solar_local_z: float,
    z_epsilon: float,
) -> np.ndarray:
    """
    Compute shadow mask caused by finite-width radial support arms.
    """
    n = len(incidence)
    combined_mask = np.zeros(n, dtype=bool)

    if not support_model.enabled:
        return combined_mask

    if support_model.width_m <= 0.0:
        raise ValueError("Support-arm width must be positive.")

    if support_model.end_radius_m <= support_model.start_radius_m:
        raise ValueError("Support-arm end radius must be larger than start radius.")

    x_shadow, y_shadow, valid = project_points_to_shadow_plane(
        x_m=incidence["x_m"].to_numpy(dtype=float),
        y_m=incidence["y_m"].to_numpy(dtype=float),
        z_m=incidence["z_m"].to_numpy(dtype=float),
        solar_local_x=solar_local_x,
        solar_local_y=solar_local_y,
        solar_local_z=solar_local_z,
        z_plane_m=support_model.z_plane_m,
        z_epsilon=z_epsilon,
    )

    half_width = 0.5 * support_model.width_m

    for angle_deg in support_model.angles_deg:
        angle_rad = np.deg2rad(angle_deg)

        ux = np.cos(angle_rad)
        uy = np.sin(angle_rad)

        longitudinal = x_shadow * ux + y_shadow * uy
        perpendicular = np.abs(-x_shadow * uy + y_shadow * ux)

        inside_length = (
            (longitudinal >= support_model.start_radius_m)
            & (longitudinal <= support_model.end_radius_m)
        )

        inside_width = perpendicular <= half_width

        arm_mask = valid & inside_length & inside_width

        combined_mask = combined_mask | arm_mask

    return combined_mask


def build_shadow_model_parameters(config: dict[str, Any]) -> ShadowModelParameters:
    """
    Build shadow model parameters from YAML configuration.
    """
    mirror_config = config["main_reflector"]
    shadow_config = config["shadow_model"]

    circular_blockers: list[CircularBlocker] = []

    for key, raw in shadow_config["circular_blockers"].items():
        circular_blockers.append(
            CircularBlocker(
                key=str(key),
                label=str(raw.get("label", key)),
                center_x_m=float(raw["center_x_m"]),
                center_y_m=float(raw["center_y_m"]),
                radius_m=float(raw["radius_m"]),
                z_plane_m=float(raw["z_plane_m"]),
                enabled=bool(raw.get("enabled", True)),
            )
        )

    support_raw = shadow_config.get("support_arms", None)

    support_arms: SupportArmModel | None

    if support_raw is None:
        support_arms = None
    else:
        support_arms = SupportArmModel(
            label=str(support_raw.get("label", "support_arms")),
            angles_deg=tuple(float(v) for v in support_raw["angles_deg"]),
            width_m=float(support_raw["width_m"]),
            start_radius_m=float(support_raw["start_radius_m"]),
            end_radius_m=float(support_raw["end_radius_m"]),
            z_plane_m=float(support_raw["z_plane_m"]),
            enabled=bool(support_raw.get("enabled", True)),
        )

    return ShadowModelParameters(
        axis_tilt_south_deg=float(mirror_config["axis_tilt_south_deg"]),
        sun_vector_local_z_epsilon=float(shadow_config["sun_vector_local_z_epsilon"]),
        circular_blockers=tuple(circular_blockers),
        support_arms=support_arms,
    )


def apply_shadow_model(
    incidence: pd.DataFrame,
    params: ShadowModelParameters,
) -> pd.DataFrame:
    """
    Apply structural shadowing model to one incidence map.

    Output quantities:
        potential_sunlit:
            From previous incidence step.

        structure_shadow:
            True only where the point is potentially sunlit and blocked by
            at least one structural element.

        visibility_chi:
            1.0 where the point receives direct solar radiation after shadowing,
            0.0 otherwise.

        effective_solar_factor:
            max(cos(theta_i), 0) * visibility_chi

    This is the exact multiplier needed before absorbed radiation:
        q_abs = alpha_s * I_sun * effective_solar_factor
    """
    required_columns = [
        "x_m",
        "y_m",
        "z_m",
        "cos_incidence",
        "potential_sunlit",
        "solar_east",
        "solar_north",
        "solar_up",
    ]

    missing = [column for column in required_columns if column not in incidence.columns]

    if missing:
        raise ValueError(f"Incidence table is missing required columns: {missing}")

    result = incidence.copy()

    solar_east = float(result["solar_east"].iloc[0])
    solar_north = float(result["solar_north"].iloc[0])
    solar_up = float(result["solar_up"].iloc[0])

    solar_local_x, solar_local_y, solar_local_z = solar_vector_global_enu_to_reflector_local(
        solar_east=solar_east,
        solar_north=solar_north,
        solar_up=solar_up,
        axis_tilt_south_deg=params.axis_tilt_south_deg,
    )

    result["solar_local_x"] = solar_local_x
    result["solar_local_y"] = solar_local_y
    result["solar_local_z"] = solar_local_z

    potential_sunlit = result["potential_sunlit"].astype(bool).to_numpy()

    combined_raw_shadow = np.zeros(len(result), dtype=bool)

    for blocker in params.circular_blockers:
        blocker_mask = circular_blocker_shadow_mask(
            incidence=result,
            blocker=blocker,
            solar_local_x=solar_local_x,
            solar_local_y=solar_local_y,
            solar_local_z=solar_local_z,
            z_epsilon=params.sun_vector_local_z_epsilon,
        )

        column_name = f"shadow_{blocker.key}"
        result[column_name] = blocker_mask

        combined_raw_shadow = combined_raw_shadow | blocker_mask

    if params.support_arms is not None:
        support_mask = support_arms_shadow_mask(
            incidence=result,
            support_model=params.support_arms,
            solar_local_x=solar_local_x,
            solar_local_y=solar_local_y,
            solar_local_z=solar_local_z,
            z_epsilon=params.sun_vector_local_z_epsilon,
        )

        result["shadow_support_arms"] = support_mask
        combined_raw_shadow = combined_raw_shadow | support_mask
    else:
        result["shadow_support_arms"] = False

    # Do not report structural shadow at points that are not geometrically sunlit.
    structure_shadow = combined_raw_shadow & potential_sunlit

    visibility_chi = potential_sunlit & (~structure_shadow)

    cos_positive = np.maximum(result["cos_incidence"].to_numpy(dtype=float), 0.0)

    result["structure_shadow"] = structure_shadow
    result["visibility_chi"] = visibility_chi.astype(float)
    result["effective_solar_factor"] = cos_positive * result["visibility_chi"]

    return result


def summarize_shadow_result(
    case_key: str,
    case_label: str,
    selected_time: str,
    shadow_result: pd.DataFrame,
) -> dict[str, object]:
    """
    Summarize one structural shadowing result.
    """
    total_points = len(shadow_result)

    potential = shadow_result["potential_sunlit"].astype(bool)
    shadow = shadow_result["structure_shadow"].astype(bool)
    visible = shadow_result["visibility_chi"] > 0.5

    potential_count = int(potential.sum())
    shadow_count = int(shadow.sum())
    visible_count = int(visible.sum())

    if potential_count > 0:
        shadow_fraction_of_potential = float(shadow_count / potential_count)
        visible_fraction_of_potential = float(visible_count / potential_count)
    else:
        shadow_fraction_of_potential = np.nan
        visible_fraction_of_potential = np.nan

    if total_points > 0:
        visible_fraction_of_aperture = float(visible_count / total_points)
    else:
        visible_fraction_of_aperture = np.nan

    visible_rows = shadow_result[visible].copy()

    if visible_rows.empty:
        mean_factor_visible = np.nan
        max_factor_visible = np.nan
    else:
        mean_factor_visible = float(visible_rows["effective_solar_factor"].mean())
        max_factor_visible = float(visible_rows["effective_solar_factor"].max())

    return {
        "case_key": case_key,
        "case_label": case_label,
        "selected_time": selected_time,
        "aperture_points": total_points,
        "potential_sunlit_points": potential_count,
        "structure_shadow_points": shadow_count,
        "visible_sunlit_points_after_shadowing": visible_count,
        "shadow_fraction_of_potential_sunlit": shadow_fraction_of_potential,
        "visible_fraction_of_potential_sunlit": visible_fraction_of_potential,
        "visible_fraction_of_aperture": visible_fraction_of_aperture,
        "effective_solar_factor_mean_visible": mean_factor_visible,
        "effective_solar_factor_max_visible": max_factor_visible,
    }


def validate_shadow_result(
    case_key: str,
    shadow_result: pd.DataFrame,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """
    Validate the algebra of the shadowing layer.

    Checks:
        1. visibility_chi is binary.
        2. structure_shadow does not occur outside potential_sunlit.
        3. effective_solar_factor = max(cos_incidence, 0) * visibility_chi.
        4. effective_solar_factor is in [0, 1].
    """
    chi = shadow_result["visibility_chi"].to_numpy(dtype=float)
    cos_positive = np.maximum(
        shadow_result["cos_incidence"].to_numpy(dtype=float),
        0.0,
    )
    factor = shadow_result["effective_solar_factor"].to_numpy(dtype=float)

    potential = shadow_result["potential_sunlit"].astype(bool).to_numpy()
    structure_shadow = shadow_result["structure_shadow"].astype(bool).to_numpy()

    chi_binary_ok = bool(np.all((chi == 0.0) | (chi == 1.0)))

    shadow_only_inside_potential_ok = bool(np.all(structure_shadow <= potential))

    expected_factor = cos_positive * chi
    max_factor_error = float(np.max(np.abs(factor - expected_factor)))

    factor_equation_ok = bool(max_factor_error <= tolerance)

    factor_range_ok = bool(
        np.nanmin(factor) >= -tolerance
        and np.nanmax(factor) <= 1.0 + tolerance
    )

    all_ok = bool(
        chi_binary_ok
        and shadow_only_inside_potential_ok
        and factor_equation_ok
        and factor_range_ok
    )

    return {
        "case_key": case_key,
        "chi_binary_ok": chi_binary_ok,
        "shadow_only_inside_potential_ok": shadow_only_inside_potential_ok,
        "factor_equation_ok": factor_equation_ok,
        "factor_range_ok": factor_range_ok,
        "max_factor_equation_error": max_factor_error,
        "all_shadow_checks_ok": all_ok,
    }


def save_shadow_result(
    shadow_result: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save one shadowing result table.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shadow_result.to_csv(path, index=False, encoding="utf-8")
