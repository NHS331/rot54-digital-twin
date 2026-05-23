from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "configs" / "scientific_targets.yaml"
SCENARIOS_PATH = ROOT / "configs" / "scenarios_2026.yaml"


def load_targets() -> dict:
    with TARGETS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["paper_targets"]


def load_scenarios() -> dict:
    with SCENARIOS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_target_file_exists() -> None:
    assert TARGETS_PATH.exists()


def test_scenarios_file_exists() -> None:
    assert SCENARIOS_PATH.exists()


def test_fixed_geometry_targets() -> None:
    targets = load_targets()
    geometry = targets["geometry"]

    assert geometry["main_reflector_diameter_m"] == 54.0
    assert geometry["main_reflector_radius_m"] == 27.0
    assert geometry["effective_aperture_m"] == 32.0
    assert geometry["optical_channel_m"] == 2.6
    assert geometry["secondary_mirror_diameter_m"] == 5.0
    assert geometry["main_axis_tilt_deg"] == 15.0
    assert geometry["panel_count"] == 3738
    assert geometry["passport_surface_smoothness_mm"] == 0.070


def test_site_targets() -> None:
    targets = load_targets()
    site = targets["site"]

    assert site["latitude_deg"] == 40.35
    assert site["longitude_deg"] == 44.25
    assert site["altitude_m"] == 1711.0
    assert site["timezone_output"] == "Asia/Yerevan"
    assert site["internal_time_standard"] == "UTC"


def test_front_side_exposure_targets_are_defined() -> None:
    targets = load_targets()
    exposure = targets["front_side_exposure"]

    assert exposure["summer_solstice_hours"] == 13.58
    assert exposure["winter_solstice_hours"] == 9.29
    assert exposure["annual_hours"] == 4275.93
    assert exposure["tolerance_hours"] > 0


def test_panel_model_targets_are_defined() -> None:
    targets = load_targets()
    panel_model = targets["panel_model"]

    assert panel_model["structural_panel_area_m2"] == 1.21
    assert panel_model["projected_cell_area_m2"] == 0.613
    assert panel_model["equivalent_cell_size_m"] == 0.783
    assert panel_model["effective_panel_span_m"] == 0.95
    assert panel_model["panel_mass_kg"] == 55.0
    assert panel_model["aluminum_density_kg_m3"] == 2700.0
    assert panel_model["mass_equivalent_thickness_mm"] == 16.8


def test_mechanical_response_targets_are_defined() -> None:
    targets = load_targets()
    response = targets["mechanical_response"]

    assert response["alpha_T_1_per_K"] == 0.0000232
    assert response["ku_central_low_mm_per_C"] == 0.080
    assert response["ku_central_high_mm_per_C"] == 0.096
    assert response["ku_upper_mm_per_C"] == 0.160


def test_rms_upper_case_targets_are_defined() -> None:
    targets = load_targets()
    rms = targets["rms_upper_case"]

    assert rms["equinox_sigma_T_mm"] == 0.078
    assert rms["summer_sigma_T_mm"] == 0.098
    assert rms["winter_sigma_T_mm"] == 0.050
    assert rms["equinox_sigma_total_mm"] == 0.105
    assert rms["summer_sigma_total_mm"] == 0.120
    assert rms["winter_sigma_total_mm"] == 0.086


def test_ruze_upper_case_targets_are_defined() -> None:
    targets = load_targets()
    ruze = targets["ruze_upper_case"]

    assert ruze["summer_f10_GHz"] == 64.4
    assert ruze["summer_eta_100_GHz"] == 0.7750


def test_required_scenarios_are_defined() -> None:
    scenarios = load_scenarios()["scenarios"]

    assert "equinox" in scenarios
    assert "summer_solstice" in scenarios
    assert "winter_solstice" in scenarios

    assert scenarios["summer_solstice"]["date"] == "2026-06-22"
    assert scenarios["winter_solstice"]["date"] == "2026-12-22"
    assert scenarios["equinox"]["date"] == "2026-03-20"


def test_future_validation_cases_are_defined() -> None:
    scenarios = load_scenarios()
    validation = scenarios["future_validation_cases"]

    assert "sensitivity" in validation
    assert "convergence" in validation
    assert "ablation" in validation
    assert "five_points" in validation["convergence"]["panel_sampling_modes"]
    assert "full_model" in validation["ablation"]["cases"]
    assert "no_self_shadowing" in validation["ablation"]["cases"]
