from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "verify_front_side_exposure_2026.py"


def load_stage_6_module():
    spec = importlib.util.spec_from_file_location(
        "verify_front_side_exposure_2026",
        SCRIPT_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    # Required for dataclasses when a module is loaded manually through importlib.
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


def test_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_equivalent_front_side_normal_declination() -> None:
    module = load_stage_6_module()

    assert module.FRONT_SIDE_NORMAL_DECLINATION_DEG == 25.35


def test_summer_solstice_exposure_standard_refraction() -> None:
    module = load_stage_6_module()

    row = module.front_side_exposure_for_day(
        day_of_year=173,
        apparent_horizon_altitude_deg=module.STANDARD_REFRACTION_HORIZON_ALTITUDE_DEG,
    )

    assert abs(row.front_side_exposure_h - 13.58) < 0.02
    assert row.limiting_condition == "front_side"


def test_winter_solstice_exposure_standard_refraction() -> None:
    module = load_stage_6_module()

    row = module.front_side_exposure_for_day(
        day_of_year=356,
        apparent_horizon_altitude_deg=module.STANDARD_REFRACTION_HORIZON_ALTITUDE_DEG,
    )

    assert abs(row.front_side_exposure_h - 9.29) < 0.02
    assert row.limiting_condition == "horizon"


def test_standard_refraction_annual_exposure_is_diagnostic_value() -> None:
    module = load_stage_6_module()

    rows = module.generate_rows_for_year(
        apparent_horizon_altitude_deg=module.STANDARD_REFRACTION_HORIZON_ALTITUDE_DEG,
    )

    annual_exposure_h = sum(row.front_side_exposure_h for row in rows)

    assert 4260.0 < annual_exposure_h < 4263.0


def test_backfit_horizon_reproduces_manuscript_annual_target() -> None:
    module = load_stage_6_module()

    required_horizon_deg = module.find_horizon_for_annual_target(
        target_annual_exposure_h=module.MANUSCRIPT_ANNUAL_TARGET_H,
    )

    annual_exposure_h = module.annual_exposure_for_horizon(
        apparent_horizon_altitude_deg=required_horizon_deg,
    )

    assert -1.4 < required_horizon_deg < -1.2
    assert abs(annual_exposure_h - module.MANUSCRIPT_ANNUAL_TARGET_H) < 1e-6


def test_generated_rows_have_expected_length() -> None:
    module = load_stage_6_module()

    rows = module.generate_rows_for_year(
        apparent_horizon_altitude_deg=module.STANDARD_REFRACTION_HORIZON_ALTITUDE_DEG,
    )

    assert len(rows) == 365
    assert rows[0].date_iso == "2026-01-01"
    assert rows[-1].date_iso == "2026-12-31"
