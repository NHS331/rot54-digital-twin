from __future__ import annotations

import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def normalize_angle_deg(angle: float) -> float:
    return angle % 360.0


def angle_set_has_south_tripod_geometry(angles: list[float]) -> bool:
    normalized = sorted(normalize_angle_deg(angle) for angle in angles)

    has_south = any(abs(angle - 270.0) <= 1.0e-9 for angle in normalized)
    has_north = any(abs(angle - 90.0) <= 1.0e-9 for angle in normalized)

    gaps = [
        normalized[1] - normalized[0],
        normalized[2] - normalized[1],
        normalized[0] + 360.0 - normalized[2],
    ]

    has_120_spacing = all(abs(gap - 120.0) <= 1.0e-9 for gap in gaps)

    return has_south and not has_north and has_120_spacing


def test_ray_shadow_tripod_uses_south_leg_geometry() -> None:
    config_path = ROOT / "configs" / "ray_shadow_model.yaml"

    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    tripod = config["ray_shadow_model"]["obstacles"]["tripod_legs"]
    angles = [float(value) for value in tripod["angles_deg"]]

    assert angle_set_has_south_tripod_geometry(angles)


def test_shadow_v2_tripod_uses_south_leg_geometry_if_config_exists() -> None:
    config_path = ROOT / "configs" / "shadow_v2_config.yaml"

    if not config_path.exists():
        return

    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    support = config["shadow_v2"]["support_arm_prisms"]
    angles = [float(value) for value in support["angles_deg"]]

    assert angle_set_has_south_tripod_geometry(angles)


def test_south_leg_endpoint_is_on_negative_y_axis() -> None:
    config_path = ROOT / "configs" / "ray_shadow_model.yaml"

    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    tripod = config["ray_shadow_model"]["obstacles"]["tripod_legs"]

    end_radius = float(tripod["end_radius_m"])
    angles = [normalize_angle_deg(float(value)) for value in tripod["angles_deg"]]

    south_angle = next(angle for angle in angles if abs(angle - 270.0) <= 1.0e-9)

    x_end = end_radius * math.cos(math.radians(south_angle))
    y_end = end_radius * math.sin(math.radians(south_angle))

    assert abs(x_end) <= 1.0e-9
    assert y_end < 0.0
    assert abs(y_end + end_radius) <= 1.0e-9
