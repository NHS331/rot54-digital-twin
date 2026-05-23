from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "solar_flux_model.yaml"
SCRIPT_PATH = ROOT / "scripts" / "26_compute_absorbed_solar_flux.py"

VISIBILITY_CSV = ROOT / "outputs" / "shadow_ray" / "control_point_visibility.csv"

CONTROL_POINT_FLUX_CSV = ROOT / "outputs" / "solar_flux" / "control_point_absorbed_flux.csv"
PANEL_FLUX_SUMMARY_CSV = ROOT / "outputs" / "solar_flux" / "panel_absorbed_flux_summary.csv"
SCENARIO_FLUX_SUMMARY_CSV = ROOT / "outputs" / "solar_flux" / "scenario_absorbed_flux_summary.csv"
FIGURE_DIR = ROOT / "outputs" / "figures" / "solar_flux"


def run_stage_10() -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=ROOT,
        check=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_stage_10_script_and_config_exist() -> None:
    assert CONFIG_PATH.exists()
    assert SCRIPT_PATH.exists()


def test_solar_flux_config_is_physically_bounded() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    physical = config["solar_flux_model"]["physical_model"]

    alpha_s = float(physical["solar_absorptivity_alpha_s"])

    assert 0.0 <= alpha_s <= 1.0

    dni_table = physical["scenario_dni_w_m2"]

    assert set(dni_table.keys()) == {
        "equinox",
        "summer_solstice",
        "winter_solstice",
    }

    for scenario_key, time_table in dni_table.items():
        assert set(time_table.keys()) == {"morning", "noon", "evening"}

        for time_key, dni in time_table.items():
            value = float(dni)
            assert 0.0 < value <= 1200.0


def test_stage_10_outputs_exist() -> None:
    run_stage_10()

    assert CONTROL_POINT_FLUX_CSV.exists()
    assert PANEL_FLUX_SUMMARY_CSV.exists()
    assert SCENARIO_FLUX_SUMMARY_CSV.exists()
    assert FIGURE_DIR.exists()


def test_control_point_flux_row_count_matches_stage_9_visibility() -> None:
    run_stage_10()

    visibility_rows = read_csv(VISIBILITY_CSV)
    flux_rows = read_csv(CONTROL_POINT_FLUX_CSV)

    assert len(flux_rows) == len(visibility_rows)

    expected = 3738 * 5 * 3 * 3

    assert len(flux_rows) == expected


def test_panel_flux_summary_row_count() -> None:
    run_stage_10()

    rows = read_csv(PANEL_FLUX_SUMMARY_CSV)

    expected = 3738 * 3 * 3

    assert len(rows) == expected


def test_scenario_flux_summary_row_count() -> None:
    run_stage_10()

    rows = read_csv(SCENARIO_FLUX_SUMMARY_CSV)

    assert len(rows) == 9


def test_absorbed_flux_is_nonnegative_and_not_above_alpha_dni() -> None:
    run_stage_10()

    rows = read_csv(CONTROL_POINT_FLUX_CSV)

    for row in rows:
        q_abs = float(row["q_abs_w_m2"])
        dni = float(row["dni_w_m2"])
        alpha_s = float(row["alpha_s"])

        assert q_abs >= 0.0
        assert q_abs <= alpha_s * dni + 1.0e-6


def test_shadowed_points_have_zero_absorbed_flux() -> None:
    run_stage_10()

    rows = read_csv(CONTROL_POINT_FLUX_CSV)

    checked = 0

    for row in rows:
        if int(row["chi"]) == 0:
            assert float(row["q_abs_w_m2"]) == 0.0
            checked += 1

        if checked >= 1000:
            break

    assert checked > 0


def test_illuminated_points_can_have_positive_absorbed_flux() -> None:
    run_stage_10()

    rows = read_csv(CONTROL_POINT_FLUX_CSV)

    positive_count = sum(1 for row in rows if float(row["q_abs_w_m2"]) > 0.0)

    assert positive_count > 0


def test_generated_flux_figures_exist() -> None:
    run_stage_10()

    figures = list(FIGURE_DIR.glob("absorbed_flux_*.png"))

    assert len(figures) == 9
