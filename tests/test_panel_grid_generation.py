from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATH = ROOT / "scripts" / "24_generate_panel_grid.py"
PANEL_GRID_CSV = ROOT / "outputs" / "geometry" / "panel_grid_3738.csv"
CONTROL_POINTS_CSV = ROOT / "outputs" / "geometry" / "panel_control_points_3738.csv"
SUMMARY_CSV = ROOT / "outputs" / "geometry" / "panel_grid_summary.csv"
FIGURE_PATH = ROOT / "outputs" / "figures" / "geometry" / "panel_grid_tripod_bases.png"


def ensure_stage_8_outputs() -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=ROOT,
        check=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_stage_8_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_panel_grid_generation_outputs_exist() -> None:
    ensure_stage_8_outputs()

    assert PANEL_GRID_CSV.exists()
    assert CONTROL_POINTS_CSV.exists()
    assert SUMMARY_CSV.exists()
    assert FIGURE_PATH.exists()


def test_panel_count_is_3738() -> None:
    ensure_stage_8_outputs()

    panels = read_csv(PANEL_GRID_CSV)

    assert len(panels) == 3738


def test_control_points_are_five_per_panel() -> None:
    ensure_stage_8_outputs()

    control_points = read_csv(CONTROL_POINTS_CSV)

    assert len(control_points) == 3738 * 5

    first_panel = [
        row["control_point_id"]
        for row in control_points
        if int(row["panel_id"]) == 1
    ]

    assert sorted(first_panel) == ["C", "NX", "NY", "PX", "PY"]


def test_projected_cell_area_and_equivalent_size() -> None:
    ensure_stage_8_outputs()

    panels = read_csv(PANEL_GRID_CSV)

    expected_area = math.pi * 27.0**2 / 3738.0
    expected_leq = math.sqrt(expected_area)

    first = panels[0]

    assert abs(float(first["projected_area_m2"]) - expected_area) < 1.0e-9
    assert abs(float(first["equivalent_cell_size_m"]) - expected_leq) < 1.0e-9
    assert 0.612 < expected_area < 0.614
    assert 0.782 < expected_leq < 0.784


def test_all_panel_centres_are_inside_aperture() -> None:
    ensure_stage_8_outputs()

    panels = read_csv(PANEL_GRID_CSV)

    for row in panels:
        radius = float(row["projected_radius_m"])
        assert 0.0 <= radius <= 27.0


def test_all_control_points_are_inside_aperture_after_clipping() -> None:
    ensure_stage_8_outputs()

    control_points = read_csv(CONTROL_POINTS_CSV)

    for row in control_points:
        radius = float(row["projected_radius_m"])
        assert 0.0 <= radius <= 27.0000001


def test_front_side_normals_are_unit_vectors() -> None:
    ensure_stage_8_outputs()

    panels = read_csv(PANEL_GRID_CSV)

    for row in panels[0:200]:
        nx = float(row["normal_x"])
        ny = float(row["normal_y"])
        nz = float(row["normal_z"])

        length = math.sqrt(nx * nx + ny * ny + nz * nz)

        assert abs(length - 1.0) < 1.0e-9
        assert nz >= 0.0


def test_tripod_base_radius_is_internal() -> None:
    ensure_stage_8_outputs()

    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    summary = {row["quantity"]: row["value"] for row in rows}

    tripod_base_radius = float(summary["tripod_base_radius_m"])

    assert 15.0 <= tripod_base_radius <= 16.0
    assert tripod_base_radius < 27.0
