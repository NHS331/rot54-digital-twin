from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "surface_temperature_model.yaml"
SCRIPT_PATH = ROOT / "scripts" / "27_compute_surface_temperature_increment.py"

CONTROL_POINT_FLUX_CSV = ROOT / "outputs" / "solar_flux" / "control_point_absorbed_flux.csv"

CONTROL_POINT_TEMPERATURE_CSV = ROOT / "outputs" / "surface_temperature" / "control_point_surface_temperature.csv"
PANEL_TEMPERATURE_SUMMARY_CSV = ROOT / "outputs" / "surface_temperature" / "panel_temperature_summary.csv"
SCENARIO_TEMPERATURE_SUMMARY_CSV = ROOT / "outputs" / "surface_temperature" / "scenario_temperature_summary.csv"
FIGURE_DIR = ROOT / "outputs" / "figures" / "surface_temperature"


def run_stage_11() -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=ROOT,
        check=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_stage_11_script_and_config_exist() -> None:
    assert CONFIG_PATH.exists()
    assert SCRIPT_PATH.exists()


def test_surface_temperature_config_is_physically_bounded() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    physical = config["surface_temperature_model"]["physical_model"]
    numerical = config["surface_temperature_model"]["numerical"]

    emissivity = float(physical["surface_emissivity_epsilon_lw"])
    sigma = float(physical["stefan_boltzmann_w_m2_k4"])

    assert 0.0 < emissivity <= 1.0
    assert sigma > 0.0

    assert float(numerical["min_h_eff_w_m2_k"]) > 0.0
    assert float(numerical["max_reasonable_delta_t_C"]) > 0.0

    environment = physical["environment"]

    assert set(environment.keys()) == {
        "equinox",
        "summer_solstice",
        "winter_solstice",
    }

    for scenario_key, time_table in environment.items():
        assert set(time_table.keys()) == {"morning", "noon", "evening"}

        for time_key, item in time_table.items():
            wind_speed = float(item["wind_speed_m_s"])
            ambient = float(item["ambient_temperature_C"])

            assert wind_speed >= 0.0
            assert -60.0 <= ambient <= 60.0


def test_stage_11_outputs_exist() -> None:
    run_stage_11()

    assert CONTROL_POINT_TEMPERATURE_CSV.exists()
    assert PANEL_TEMPERATURE_SUMMARY_CSV.exists()
    assert SCENARIO_TEMPERATURE_SUMMARY_CSV.exists()
    assert FIGURE_DIR.exists()


def test_control_point_temperature_row_count_matches_flux_input() -> None:
    run_stage_11()

    flux_rows = read_csv(CONTROL_POINT_FLUX_CSV)
    temperature_rows = read_csv(CONTROL_POINT_TEMPERATURE_CSV)

    assert len(temperature_rows) == len(flux_rows)

    expected = 3738 * 5 * 3 * 3

    assert len(temperature_rows) == expected


def test_panel_temperature_summary_row_count() -> None:
    run_stage_11()

    rows = read_csv(PANEL_TEMPERATURE_SUMMARY_CSV)

    expected = 3738 * 3 * 3

    assert len(rows) == expected


def test_scenario_temperature_summary_row_count() -> None:
    run_stage_11()

    rows = read_csv(SCENARIO_TEMPERATURE_SUMMARY_CSV)

    assert len(rows) == 9


def test_heat_transfer_coefficients_are_positive() -> None:
    run_stage_11()

    rows = read_csv(CONTROL_POINT_TEMPERATURE_CSV)

    for row in rows[0:1000]:
        assert float(row["h_conv_w_m2_k"]) > 0.0
        assert float(row["h_rad_w_m2_k"]) > 0.0
        assert float(row["h_eff_w_m2_k"]) > 0.0


def test_temperature_increment_is_nonnegative() -> None:
    run_stage_11()

    rows = read_csv(CONTROL_POINT_TEMPERATURE_CSV)

    for row in rows:
        assert float(row["delta_t_s_C"]) >= 0.0
        assert float(row["surface_temperature_C"]) >= float(row["ambient_temperature_C"])


def test_zero_absorbed_flux_gives_zero_temperature_increment() -> None:
    run_stage_11()

    rows = read_csv(CONTROL_POINT_TEMPERATURE_CSV)

    checked = 0

    for row in rows:
        if float(row["q_abs_w_m2"]) == 0.0:
            assert float(row["delta_t_s_C"]) == 0.0
            checked += 1

        if checked >= 1000:
            break

    assert checked > 0


def test_positive_absorbed_flux_can_give_positive_temperature_increment() -> None:
    run_stage_11()

    rows = read_csv(CONTROL_POINT_TEMPERATURE_CSV)

    positive_count = sum(1 for row in rows if float(row["delta_t_s_C"]) > 0.0)

    assert positive_count > 0


def test_surface_temperature_identity() -> None:
    run_stage_11()

    rows = read_csv(CONTROL_POINT_TEMPERATURE_CSV)

    for row in rows[0:1000]:
        ambient = float(row["ambient_temperature_C"])
        delta = float(row["delta_t_s_C"])
        surface = float(row["surface_temperature_C"])

        assert abs(surface - (ambient + delta)) <= 1.0e-6


def test_temperature_increment_is_reasonable_under_current_engineering_inputs() -> None:
    run_stage_11()

    with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    max_reasonable = float(
        config["surface_temperature_model"]["numerical"]["max_reasonable_delta_t_C"]
    )

    rows = read_csv(CONTROL_POINT_TEMPERATURE_CSV)

    max_delta = max(float(row["delta_t_s_C"]) for row in rows)

    assert max_delta <= max_reasonable


def test_generated_temperature_figures_exist() -> None:
    run_stage_11()

    figures = list(FIGURE_DIR.glob("surface_temperature_delta_*.png"))

    assert len(figures) == 9
