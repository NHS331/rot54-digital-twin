from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ROT54_CONFIG_PATH = ROOT / "configs" / "rot54_config.yaml"
SHADOW_V2_CONFIG_PATH = ROOT / "configs" / "shadow_v2_config.yaml"
SHADOW_GEOMETRY_PATH = ROOT / "configs" / "shadow_geometry.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)


def test_rot54_support_arms_stop_at_internal_base() -> None:
    config = load_yaml(ROT54_CONFIG_PATH)

    reflector_radius = float(config["main_reflector"]["aperture_radius_m"])
    support = config["shadow_model"]["support_arms"]

    start_radius = float(support["start_radius_m"])
    end_radius = float(support["end_radius_m"])

    assert reflector_radius == 27.0
    assert 1.5 <= start_radius <= 3.0
    assert 15.0 <= end_radius <= 16.0
    assert end_radius < reflector_radius
    assert end_radius != reflector_radius
    assert bool(support["confirmed_not_outer_rim"]) is True


def test_shadow_v2_support_prisms_stop_at_internal_base() -> None:
    config = load_yaml(SHADOW_V2_CONFIG_PATH)

    support = config["shadow_v2"]["support_arm_prisms"]

    start_radius = float(support["start_radius_m"])
    end_radius = float(support["end_radius_m"])
    outer_rim = float(support["outer_rim_radius_m"])

    assert outer_rim == 27.0
    assert 1.5 <= start_radius <= 3.0
    assert 15.0 <= end_radius <= 16.0
    assert end_radius < outer_rim
    assert end_radius != outer_rim
    assert bool(support["confirmed_not_outer_rim"]) is True


def test_shadow_geometry_contract_records_internal_tripod_bases() -> None:
    config = load_yaml(SHADOW_GEOMETRY_PATH)["shadow_geometry"]

    bases = config["tripod_support_bases"]
    legs = config["tripod_legs"]

    assert bases["count"] == 3
    assert 15.0 <= float(bases["radial_distance_from_center_m"]) <= 16.0
    assert float(bases["outer_rim_radius_m"]) == 27.0
    assert bool(legs["must_not_extend_to_outer_rim"]) is True
    assert float(legs["end_radius_at_internal_base_m"]) < float(bases["outer_rim_radius_m"])


def test_no_outer_rim_tripod_radius_left_in_shadow_configs() -> None:
    rot54 = load_yaml(ROT54_CONFIG_PATH)
    shadow_v2 = load_yaml(SHADOW_V2_CONFIG_PATH)

    v1_end = float(rot54["shadow_model"]["support_arms"]["end_radius_m"])
    v2_end = float(shadow_v2["shadow_v2"]["support_arm_prisms"]["end_radius_m"])

    assert v1_end != 27.0
    assert v2_end != 27.0
    assert abs(v1_end - 15.5) < 1e-12
    assert abs(v2_end - 15.5) < 1e-12
