from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def normalize(angle: float) -> float:
    return angle % 360.0


def assert_south_tripod_geometry(angles: list[float]) -> None:
    normalized = sorted(normalize(angle) for angle in angles)

    assert len(normalized) == 3
    assert any(abs(angle - 270.0) <= 1.0e-9 for angle in normalized)
    assert not any(abs(angle - 90.0) <= 1.0e-9 for angle in normalized)

    gaps = [
        normalized[1] - normalized[0],
        normalized[2] - normalized[1],
        normalized[0] + 360.0 - normalized[2],
    ]

    for gap in gaps:
        assert abs(gap - 120.0) <= 1.0e-9


def test_ray_shadow_model_tripod_is_south_oriented() -> None:
    path = ROOT / "configs" / "ray_shadow_model.yaml"

    if not path.exists():
        return

    with path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    angles = [
        float(value)
        for value in config["ray_shadow_model"]["obstacles"]["tripod_legs"]["angles_deg"]
    ]

    assert_south_tripod_geometry(angles)


def test_shadow_v2_tripod_is_south_oriented() -> None:
    path = ROOT / "configs" / "shadow_v2_config.yaml"

    if not path.exists():
        return

    with path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    angles = [
        float(value)
        for value in config["shadow_v2"]["support_arm_prisms"]["angles_deg"]
    ]

    assert_south_tripod_geometry(angles)


def test_no_old_north_tripod_angle_set_remains_in_source_text() -> None:
    bad_files = []

    for folder_name in ["configs", "scripts", "src", "tests", "docs"]:
        folder = ROOT / folder_name

        if not folder.exists():
            continue

        for path in folder.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {".py", ".yaml", ".yml", ".md", ".txt"}:
                continue

            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue

            if "[90.0, 210.0, 330.0]" in text:
                bad_files.append(path)

            if "(90.0, 210.0, 330.0)" in text:
                bad_files.append(path)

            if "90.0\n      - 210.0\n      - 330.0" in text:
                bad_files.append(path)

    assert not bad_files, [str(path.relative_to(ROOT)) for path in bad_files]
