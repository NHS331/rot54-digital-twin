from __future__ import annotations

import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def circular_angle_difference_deg(a: float, b: float) -> float:
    diff = abs((a - b) % 360.0)
    return min(diff, 360.0 - diff)


def normalize_angle_deg(angle: float) -> float:
    return angle % 360.0


def test_stage_9_tripod_has_one_strictly_south_leg_and_120_degree_spacing() -> None:
    config_path = ROOT / "configs" / "ray_shadow_model.yaml"

    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    tripod = config["ray_shadow_model"]["obstacles"]["tripod_legs"]

    angles = [normalize_angle_deg(float(value)) for value in tripod["angles_deg"]]

    assert len(angles) == 3

    # Coordinate convention:
    # 0 deg   = +x = east
    # 90 deg  = +y = north
    # 180 deg = west
    # 270 deg = -y = south
    assert any(abs(angle - 270.0) <= 1.0e-9 for angle in angles)

    # No support leg is allowed to point strictly north.
    assert not any(abs(angle - 90.0) <= 1.0e-9 for angle in angles)

    sorted_angles = sorted(angles)

    gaps = [
        sorted_angles[1] - sorted_angles[0],
        sorted_angles[2] - sorted_angles[1],
        sorted_angles[0] + 360.0 - sorted_angles[2],
    ]

    for gap in gaps:
        assert abs(gap - 120.0) <= 1.0e-9


def test_stage_9_tripod_south_leg_endpoint_has_negative_y_and_zero_x() -> None:
    config_path = ROOT / "configs" / "ray_shadow_model.yaml"

    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    tripod = config["ray_shadow_model"]["obstacles"]["tripod_legs"]

    end_radius = float(tripod["end_radius_m"])
    angles = [normalize_angle_deg(float(value)) for value in tripod["angles_deg"]]

    south_angle = next(angle for angle in angles if abs(angle - 270.0) <= 1.0e-9)

    angle_rad = math.radians(south_angle)

    x_end = end_radius * math.cos(angle_rad)
    y_end = end_radius * math.sin(angle_rad)

    assert abs(x_end) <= 1.0e-9
    assert y_end < 0.0
    assert abs(y_end + end_radius) <= 1.0e-9


def test_shadow_v2_tripod_has_same_south_leg_geometry_if_config_exists() -> None:
    config_path = ROOT / "configs" / "shadow_v2_config.yaml"

    if not config_path.exists():
        return

    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    support = config["shadow_v2"]["support_arm_prisms"]

    angles = [normalize_angle_deg(float(value)) for value in support["angles_deg"]]

    assert len(angles) == 3
    assert any(abs(angle - 270.0) <= 1.0e-9 for angle in angles)
    assert not any(abs(angle - 90.0) <= 1.0e-9 for angle in angles)

    sorted_angles = sorted(angles)

    gaps = [
        sorted_angles[1] - sorted_angles[0],
        sorted_angles[2] - sorted_angles[1],
        sorted_angles[0] + 360.0 - sorted_angles[2],
    ]

    for gap in gaps:
        assert abs(gap - 120.0) <= 1.0e-9
