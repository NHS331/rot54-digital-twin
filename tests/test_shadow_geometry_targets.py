from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
SHADOW_GEOMETRY_PATH = ROOT / "configs" / "shadow_geometry.yaml"


def load_shadow_geometry() -> dict:
    with SHADOW_GEOMETRY_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["shadow_geometry"]


def test_shadow_geometry_config_exists() -> None:
    assert SHADOW_GEOMETRY_PATH.exists()


def test_tripod_bases_are_internal_not_at_rim() -> None:
    geometry = load_shadow_geometry()

    reflector_radius = geometry["main_reflector"]["radius_m"]
    base_radius = geometry["tripod_support_bases"]["radial_distance_from_center_m"]

    assert reflector_radius == 27.0
    assert 15.0 <= base_radius <= 16.0
    assert base_radius < reflector_radius


def test_tripod_base_count_is_three() -> None:
    geometry = load_shadow_geometry()

    assert geometry["tripod_support_bases"]["count"] == 3
    assert len(geometry["tripod_support_bases"]["azimuths_deg_model_initial"]) == 3


def test_tripod_radius_status_is_recorded() -> None:
    geometry = load_shadow_geometry()

    status = geometry["tripod_support_bases"]["modelling_status"]

    assert status == "confirmed_radius_approximate_azimuth_pending"
    assert geometry["tripod_support_bases"]["future_ray_shadowing_use"] is True
