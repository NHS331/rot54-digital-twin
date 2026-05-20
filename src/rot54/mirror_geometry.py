from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MirrorGeometryParameters:
    """
    Main reflector geometric parameters.

    Coordinate convention in the reflector-local frame:
        x_m: horizontal coordinate in aperture projection
        y_m: horizontal coordinate in aperture projection
        z_m: spherical surface coordinate

    The generated surface is a concave hemispherical bowl:

        z = -sqrt(R^2 - x^2 - y^2)

    The local unit normal is taken as the inward/front-facing normal
    toward the sphere center:

        n = [-x, -y, -z] / R
    """

    diameter_m: float
    aperture_radius_m: float
    spherical_radius_m: float
    panel_count: int
    grid_size: int


def create_main_reflector_grid(params: MirrorGeometryParameters) -> pd.DataFrame:
    """
    Generate a dense geometric sampling grid for the ROT-54/2.6 main reflector.
    """
    if params.grid_size < 11:
        raise ValueError("grid_size must be at least 11.")

    if params.aperture_radius_m <= 0:
        raise ValueError("aperture_radius_m must be positive.")

    if params.spherical_radius_m < params.aperture_radius_m:
        raise ValueError(
            "spherical_radius_m must be greater than or equal to aperture_radius_m."
        )

    r_ap = float(params.aperture_radius_m)
    r_sph = float(params.spherical_radius_m)

    x_values = np.linspace(-r_ap, r_ap, params.grid_size)
    y_values = np.linspace(-r_ap, r_ap, params.grid_size)

    xx, yy = np.meshgrid(x_values, y_values)
    rr = np.sqrt(xx**2 + yy**2)

    inside = rr <= r_ap

    zz = np.full_like(xx, np.nan, dtype=float)
    zz[inside] = -np.sqrt(np.maximum(r_sph**2 - rr[inside] ** 2, 0.0))

    normal_x = np.full_like(xx, np.nan, dtype=float)
    normal_y = np.full_like(xx, np.nan, dtype=float)
    normal_z = np.full_like(xx, np.nan, dtype=float)

    normal_x[inside] = -xx[inside] / r_sph
    normal_y[inside] = -yy[inside] / r_sph
    normal_z[inside] = -zz[inside] / r_sph

    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)

    normal_x[inside] = normal_x[inside] / norm[inside]
    normal_y[inside] = normal_y[inside] / norm[inside]
    normal_z[inside] = normal_z[inside] / norm[inside]

    radial_fraction = np.full_like(xx, np.nan, dtype=float)
    radial_fraction[inside] = rr[inside] / r_ap

    zone = np.full(xx.shape, "outside", dtype=object)
    zone[(inside) & (radial_fraction <= 0.25)] = "central"
    zone[(inside) & (radial_fraction > 0.25) & (radial_fraction <= 0.50)] = "inner"
    zone[(inside) & (radial_fraction > 0.50) & (radial_fraction <= 0.75)] = "middle"
    zone[(inside) & (radial_fraction > 0.75)] = "outer"

    df = pd.DataFrame(
        {
            "point_id": np.arange(xx.size, dtype=int),
            "x_m": xx.ravel(),
            "y_m": yy.ravel(),
            "z_m": zz.ravel(),
            "r_m": rr.ravel(),
            "radial_fraction": radial_fraction.ravel(),
            "inside_aperture": inside.ravel(),
            "normal_x": normal_x.ravel(),
            "normal_y": normal_y.ravel(),
            "normal_z": normal_z.ravel(),
            "zone": zone.ravel(),
        }
    )

    return df


def summarize_main_reflector_grid(
    grid: pd.DataFrame,
    params: MirrorGeometryParameters,
) -> pd.DataFrame:
    """
    Create a compact numerical summary for checking the generated geometry.
    """
    inside = grid[grid["inside_aperture"]].copy()

    if inside.empty:
        raise ValueError("No points inside aperture. Geometry generation failed.")

    dx = (2.0 * params.aperture_radius_m) / (params.grid_size - 1)
    approximate_cell_area_m2 = dx * dx

    aperture_area_theoretical_m2 = np.pi * params.aperture_radius_m**2
    aperture_area_grid_m2 = len(inside) * approximate_cell_area_m2

    z_min = float(inside["z_m"].min())
    z_max = float(inside["z_m"].max())
    z_center_expected = -float(params.spherical_radius_m)

    normal_length = np.sqrt(
        inside["normal_x"] ** 2
        + inside["normal_y"] ** 2
        + inside["normal_z"] ** 2
    )

    summary = pd.DataFrame(
        [
            {
                "parameter": "diameter_m",
                "value": params.diameter_m,
                "unit": "m",
                "comment": "Main reflector diameter",
            },
            {
                "parameter": "aperture_radius_m",
                "value": params.aperture_radius_m,
                "unit": "m",
                "comment": "Projected aperture radius",
            },
            {
                "parameter": "spherical_radius_m",
                "value": params.spherical_radius_m,
                "unit": "m",
                "comment": "Spherical surface radius used in current model",
            },
            {
                "parameter": "panel_count",
                "value": params.panel_count,
                "unit": "count",
                "comment": "Nominal number of panel elements",
            },
            {
                "parameter": "grid_size",
                "value": params.grid_size,
                "unit": "points per axis",
                "comment": "Sampling resolution",
            },
            {
                "parameter": "inside_aperture_points",
                "value": len(inside),
                "unit": "points",
                "comment": "Number of generated points inside circular aperture",
            },
            {
                "parameter": "theoretical_projected_area_m2",
                "value": aperture_area_theoretical_m2,
                "unit": "m^2",
                "comment": "pi * R^2",
            },
            {
                "parameter": "grid_projected_area_m2",
                "value": aperture_area_grid_m2,
                "unit": "m^2",
                "comment": "Approximate area represented by the grid",
            },
            {
                "parameter": "grid_area_relative_error_percent",
                "value": 100.0
                * (aperture_area_grid_m2 - aperture_area_theoretical_m2)
                / aperture_area_theoretical_m2,
                "unit": "%",
                "comment": "Grid discretization area error",
            },
            {
                "parameter": "z_min_m",
                "value": z_min,
                "unit": "m",
                "comment": "Minimum z value of spherical bowl",
            },
            {
                "parameter": "z_max_m",
                "value": z_max,
                "unit": "m",
                "comment": "Maximum z value near aperture rim",
            },
            {
                "parameter": "z_center_expected_m",
                "value": z_center_expected,
                "unit": "m",
                "comment": "Expected center depth for hemispherical approximation",
            },
            {
                "parameter": "normal_length_min",
                "value": float(normal_length.min()),
                "unit": "-",
                "comment": "Minimum unit-normal length",
            },
            {
                "parameter": "normal_length_max",
                "value": float(normal_length.max()),
                "unit": "-",
                "comment": "Maximum unit-normal length",
            },
        ]
    )

    return summary


def save_geometry_outputs(
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    grid_output_path: str | Path,
    summary_output_path: str | Path,
) -> None:
    """
    Save grid and summary CSV files.
    """
    grid_path = Path(grid_output_path)
    summary_path = Path(summary_output_path)

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    grid.to_csv(grid_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
