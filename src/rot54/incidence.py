from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IncidenceParameters:
    """
    Parameters required to compute solar incidence over the main reflector grid.
    """

    axis_tilt_south_deg: float


def rotate_local_vector_to_global_enu(
    vector_x: np.ndarray,
    vector_y: np.ndarray,
    vector_z: np.ndarray,
    axis_tilt_south_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate reflector-local vectors into global ENU coordinates.

    Local reflector frame:
        local z-axis = central front-facing normal before tilt

    Global ENU frame:
        E = east
        N = north
        U = up

    The main axis is tilted toward south by psi.
    This is represented as a rotation around the east axis:

        E_global = x_local
        N_global = cos(psi) * y_local - sin(psi) * z_local
        U_global = sin(psi) * y_local + cos(psi) * z_local

    Check:
        local central normal [0, 0, 1]
        becomes [0, -sin(psi), cos(psi)]
    """
    psi = np.deg2rad(axis_tilt_south_deg)

    global_east = vector_x
    global_north = np.cos(psi) * vector_y - np.sin(psi) * vector_z
    global_up = np.sin(psi) * vector_y + np.cos(psi) * vector_z

    return global_east, global_north, global_up


def compute_incidence_for_grid(
    mirror_grid: pd.DataFrame,
    solar_row: pd.Series,
    params: IncidenceParameters,
) -> pd.DataFrame:
    """
    Compute incidence angle and cosine factor for all aperture points.

    Input:
        mirror_grid:
            Output of create_main_reflector_grid()

        solar_row:
            One row from seasonal_solar_positions.csv

    Output:
        DataFrame with solar incidence quantities for every point.
    """
    inside = mirror_grid[mirror_grid["inside_aperture"] == True].copy()

    if inside.empty:
        raise ValueError("No aperture points found in mirror_grid.")

    normal_x = inside["normal_x"].to_numpy(dtype=float)
    normal_y = inside["normal_y"].to_numpy(dtype=float)
    normal_z = inside["normal_z"].to_numpy(dtype=float)

    normal_e, normal_n, normal_u = rotate_local_vector_to_global_enu(
        vector_x=normal_x,
        vector_y=normal_y,
        vector_z=normal_z,
        axis_tilt_south_deg=params.axis_tilt_south_deg,
    )

    # Numerical re-normalization after rotation.
    normal_norm = np.sqrt(normal_e**2 + normal_n**2 + normal_u**2)
    normal_e = normal_e / normal_norm
    normal_n = normal_n / normal_norm
    normal_u = normal_u / normal_norm

    solar_e = float(solar_row["solar_east"])
    solar_n = float(solar_row["solar_north"])
    solar_u = float(solar_row["solar_up"])

    cos_incidence = (
        normal_e * solar_e
        + normal_n * solar_n
        + normal_u * solar_u
    )

    cos_incidence_clipped = np.clip(cos_incidence, -1.0, 1.0)

    incidence_angle_deg = np.rad2deg(np.arccos(cos_incidence_clipped))

    solar_front_side = bool(solar_row["front_side_illumination"])

    potential_sunlit = (cos_incidence > 0.0) & solar_front_side

    result = pd.DataFrame(
        {
            "point_id": inside["point_id"].to_numpy(dtype=int),
            "x_m": inside["x_m"].to_numpy(dtype=float),
            "y_m": inside["y_m"].to_numpy(dtype=float),
            "z_m": inside["z_m"].to_numpy(dtype=float),
            "r_m": inside["r_m"].to_numpy(dtype=float),
            "radial_fraction": inside["radial_fraction"].to_numpy(dtype=float),
            "zone": inside["zone"].to_numpy(),
            "normal_local_x": normal_x,
            "normal_local_y": normal_y,
            "normal_local_z": normal_z,
            "normal_global_east": normal_e,
            "normal_global_north": normal_n,
            "normal_global_up": normal_u,
            "solar_east": solar_e,
            "solar_north": solar_n,
            "solar_up": solar_u,
            "solar_altitude_deg": float(solar_row["solar_altitude_deg"]),
            "solar_azimuth_deg": float(solar_row["solar_azimuth_deg"]),
            "axis_dot_sun": float(solar_row["axis_dot_sun"]),
            "cos_incidence": cos_incidence,
            "incidence_angle_deg": incidence_angle_deg,
            "potential_sunlit": potential_sunlit,
        }
    )

    return result


def summarize_incidence_map(
    case_key: str,
    case_label: str,
    selected_time: str,
    incidence: pd.DataFrame,
) -> dict[str, object]:
    """
    Create numerical checks for one incidence map.
    """
    sunlit = incidence[incidence["potential_sunlit"] == True].copy()

    if sunlit.empty:
        return {
            "case_key": case_key,
            "case_label": case_label,
            "selected_time": selected_time,
            "aperture_points": len(incidence),
            "sunlit_points": 0,
            "sunlit_fraction": 0.0,
            "cos_incidence_min_sunlit": np.nan,
            "cos_incidence_mean_sunlit": np.nan,
            "cos_incidence_max_sunlit": np.nan,
            "incidence_angle_min_deg_sunlit": np.nan,
            "incidence_angle_mean_deg_sunlit": np.nan,
            "incidence_angle_max_deg_sunlit": np.nan,
        }

    return {
        "case_key": case_key,
        "case_label": case_label,
        "selected_time": selected_time,
        "aperture_points": len(incidence),
        "sunlit_points": len(sunlit),
        "sunlit_fraction": float(len(sunlit) / len(incidence)),
        "cos_incidence_min_sunlit": float(sunlit["cos_incidence"].min()),
        "cos_incidence_mean_sunlit": float(sunlit["cos_incidence"].mean()),
        "cos_incidence_max_sunlit": float(sunlit["cos_incidence"].max()),
        "incidence_angle_min_deg_sunlit": float(sunlit["incidence_angle_deg"].min()),
        "incidence_angle_mean_deg_sunlit": float(sunlit["incidence_angle_deg"].mean()),
        "incidence_angle_max_deg_sunlit": float(sunlit["incidence_angle_deg"].max()),
    }


def save_incidence_map(
    incidence: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save incidence map as CSV.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    incidence.to_csv(path, index=False, encoding="utf-8")
