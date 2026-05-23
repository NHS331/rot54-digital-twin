from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "dense_surface_temperature_map.yaml"
SCRIPT_PATH = ROOT / "scripts" / "28_generate_dense_surface_temperature_maps.py"

SUMMARY_CSV = ROOT / "outputs" / "surface_temperature_dense" / "dense_temperature_grid_summary.csv"
DENSE_DIR = ROOT / "outputs" / "surface_temperature_dense"
FIGURE_DIR = ROOT / "outputs" / "figures" / "surface_temperature_dense"


def run_stage_11b() -> None:
    expected_figures = list(FIGURE_DIR.glob("dense_temperature_delta_*.png"))
    expected_npz = list(DENSE_DIR.glob("dense_temperature_*.npz"))

    if SUMMARY_CSV.exists() and len(expected_figures) == 9 and len(expected_npz) == 9:
        return

    subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=ROOT,
        check=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_stage_11b_script_and_config_exist() -> None:
    assert CONFIG_PATH.exists()
    assert SCRIPT_PATH.exists()


def test_dense_temperature_config_requests_at_least_100x_density() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    interpolation = config["dense_surface_temperature_map"]["interpolation"]

    assert int(interpolation["dense_grid_size"]) >= 1600
    assert float(interpolation["aperture_radius_m"]) == 27.0
    assert float(interpolation["minimum_density_multiplier"]) >= 100.0


def test_stage_11b_outputs_exist() -> None:
    run_stage_11b()

    assert SUMMARY_CSV.exists()
    assert DENSE_DIR.exists()
    assert FIGURE_DIR.exists()


def test_dense_summary_has_nine_cases() -> None:
    run_stage_11b()

    rows = read_csv(SUMMARY_CSV)

    assert len(rows) == 9


def test_dense_coordinate_count_is_at_least_100x_source_points() -> None:
    run_stage_11b()

    rows = read_csv(SUMMARY_CSV)

    for row in rows:
        source_count = int(row["source_point_count"])
        dense_count = int(row["dense_coordinate_count"])
        density_multiplier = float(row["density_multiplier"])

        assert source_count == 18690
        assert dense_count >= source_count * 100
        assert density_multiplier >= 100.0


def test_dense_npz_files_exist_and_contain_coordinate_arrays() -> None:
    run_stage_11b()

    rows = read_csv(SUMMARY_CSV)

    for row in rows:
        npz_path = ROOT / row["npz_file"]

        assert npz_path.exists()

    first_npz = ROOT / rows[0]["npz_file"]

    data = np.load(first_npz)

    assert "x_m" in data
    assert "y_m" in data
    assert "delta_t_s_C" in data
    assert "surface_temperature_C" in data

    assert data["x_m"].size >= 18690 * 100
    assert data["x_m"].size == data["y_m"].size
    assert data["x_m"].size == data["delta_t_s_C"].size
    assert data["x_m"].size == data["surface_temperature_C"].size


def test_dense_temperature_values_are_finite_and_nonnegative_for_delta() -> None:
    run_stage_11b()

    rows = read_csv(SUMMARY_CSV)
    first_npz = ROOT / rows[0]["npz_file"]

    data = np.load(first_npz)

    delta = data["delta_t_s_C"]
    surface = data["surface_temperature_C"]

    assert np.isfinite(delta).all()
    assert np.isfinite(surface).all()
    assert float(delta.min()) >= 0.0
    assert float(delta.max()) >= 0.0


def test_dense_figures_exist() -> None:
    run_stage_11b()

    figures = list(FIGURE_DIR.glob("dense_temperature_delta_*.png"))

    assert len(figures) == 9
