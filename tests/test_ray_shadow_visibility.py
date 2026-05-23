from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATH = ROOT / "scripts" / "25_compute_ray_shadow_visibility.py"
CONFIG_PATH = ROOT / "configs" / "ray_shadow_model.yaml"
VISIBILITY_CSV = ROOT / "outputs" / "shadow_ray" / "control_point_visibility.csv"
PANEL_SUMMARY_CSV = ROOT / "outputs" / "shadow_ray" / "panel_shadow_summary.csv"
SCENARIO_SUMMARY_CSV = ROOT / "outputs" / "shadow_ray" / "scenario_shadow_summary.csv"
FIGURE_DIR = ROOT / "outputs" / "figures" / "shadow_ray"


def run_stage_9() -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=ROOT,
        check=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_stage_9_script_and_config_exist() -> None:
    assert SCRIPT_PATH.exists()
    assert CONFIG_PATH.exists()


def test_tripod_geometry_is_internal_not_outer_rim() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    tripod = config["ray_shadow_model"]["obstacles"]["tripod_legs"]

    assert bool(tripod["enabled"]) is True
    assert float(tripod["start_radius_m"]) < float(tripod["end_radius_m"])
    assert 15.0 <= float(tripod["end_radius_m"]) <= 16.0
    assert float(tripod["outer_rim_radius_m"]) == 27.0
    assert float(tripod["end_radius_m"]) < float(tripod["outer_rim_radius_m"])
    assert bool(tripod["confirmed_not_outer_rim"]) is True


def test_stage_9_outputs_exist() -> None:
    run_stage_9()

    assert VISIBILITY_CSV.exists()
    assert PANEL_SUMMARY_CSV.exists()
    assert SCENARIO_SUMMARY_CSV.exists()
    assert FIGURE_DIR.exists()


def test_visibility_row_count() -> None:
    run_stage_9()

    rows = read_csv(VISIBILITY_CSV)

    expected = 3738 * 5 * 3 * 3

    assert len(rows) == expected


def test_panel_summary_row_count() -> None:
    run_stage_9()

    rows = read_csv(PANEL_SUMMARY_CSV)

    expected = 3738 * 3 * 3

    assert len(rows) == expected


def test_scenario_summary_row_count() -> None:
    run_stage_9()

    rows = read_csv(SCENARIO_SUMMARY_CSV)

    assert len(rows) == 9


def test_chi_values_are_binary_and_mu_nonnegative() -> None:
    run_stage_9()

    rows = read_csv(VISIBILITY_CSV)

    chi_values = {row["chi"] for row in rows}

    assert chi_values.issubset({"0", "1"})
    assert "0" in chi_values
    assert "1" in chi_values

    for row in rows[0:1000]:
        assert float(row["mu_front"]) >= 0.0


def test_shadow_sources_include_structural_sources() -> None:
    run_stage_9()

    rows = read_csv(VISIBILITY_CSV)
    sources = {row["shadow_source"] for row in rows}

    assert "none" in sources
    assert any(source.startswith("tripod_legs_internal_bases") for source in sources)
    assert (
        "secondary_mirror" in sources
        or "central_hub" in sources
        or "optical_reflector" in sources
    )


def test_generated_figures_exist() -> None:
    run_stage_9()

    figures = list(FIGURE_DIR.glob("ray_shadow_*.png"))

    assert len(figures) == 9


