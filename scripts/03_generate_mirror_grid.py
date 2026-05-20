from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rot54.config import load_config
from rot54.mirror_geometry import (
    MirrorGeometryParameters,
    create_main_reflector_grid,
    save_geometry_outputs,
    summarize_main_reflector_grid,
)


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "rot54_config.yaml"
    config = load_config(config_path)

    main_reflector = config["main_reflector"]
    grid_config = config["geometry_grid"]

    params = MirrorGeometryParameters(
        diameter_m=float(main_reflector["diameter_m"]),
        aperture_radius_m=float(main_reflector["aperture_radius_m"]),
        spherical_radius_m=float(main_reflector["spherical_radius_m"]),
        panel_count=int(main_reflector["panel_count"]),
        grid_size=int(grid_config["grid_size"]),
    )

    grid = create_main_reflector_grid(params)
    summary = summarize_main_reflector_grid(grid, params)

    grid_output_path = PROJECT_ROOT / grid_config["output_csv"]
    summary_output_path = PROJECT_ROOT / grid_config["output_summary_csv"]

    save_geometry_outputs(
        grid=grid,
        summary=summary,
        grid_output_path=grid_output_path,
        summary_output_path=summary_output_path,
    )

    inside_points = int(grid["inside_aperture"].sum())

    print("Main reflector geometry grid generated successfully.")
    print(f"Grid output: {grid_output_path}")
    print(f"Summary output: {summary_output_path}")
    print(f"Grid size: {params.grid_size} x {params.grid_size}")
    print(f"Points inside aperture: {inside_points}")
    print(f"Diameter: {params.diameter_m:.3f} m")
    print(f"Aperture radius: {params.aperture_radius_m:.3f} m")
    print(f"Spherical radius: {params.spherical_radius_m:.3f} m")


if __name__ == "__main__":
    main()
