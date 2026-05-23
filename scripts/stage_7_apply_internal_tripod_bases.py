from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROT54_CONFIG_PATH = PROJECT_ROOT / "configs" / "rot54_config.yaml"
SHADOW_V2_CONFIG_PATH = PROJECT_ROOT / "configs" / "shadow_v2_config.yaml"
SHADOW_GEOMETRY_PATH = PROJECT_ROOT / "configs" / "shadow_geometry.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "stage_7_tripod_internal_base_update.md"


TRIPOD_BASE_RADIUS_M = 15.5
TRIPOD_BASE_RADIUS_MIN_M = 15.0
TRIPOD_BASE_RADIUS_MAX_M = 16.0
MAIN_REFLECTOR_RADIUS_M = 27.0

V1_ARM_START_RADIUS_M = 2.0
V2_ARM_START_RADIUS_M = 2.2

TRIPOD_ANGLES_DEG = [90.0, 210.0, 330.0]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a dictionary: {path}")

    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def ensure_rot54_shadow_v1_config() -> dict[str, Any]:
    config = load_yaml(ROT54_CONFIG_PATH)

    if "main_reflector" not in config:
        config["main_reflector"] = {}

    config["main_reflector"]["diameter_m"] = float(
        config["main_reflector"].get("diameter_m", 54.0)
    )
    config["main_reflector"]["aperture_radius_m"] = float(
        config["main_reflector"].get("aperture_radius_m", MAIN_REFLECTOR_RADIUS_M)
    )
    config["main_reflector"]["axis_tilt_south_deg"] = float(
        config["main_reflector"].get("axis_tilt_south_deg", 15.0)
    )

    if "shadow_model" not in config:
        config["shadow_model"] = {}

    shadow_model = config["shadow_model"]

    if "support_arms" not in shadow_model or shadow_model["support_arms"] is None:
        shadow_model["support_arms"] = {}

    support = shadow_model["support_arms"]

    support["enabled"] = bool(support.get("enabled", True))
    support["label"] = "support_arms_internal_tripod"
    support["description"] = (
        "Tripod legs are internal supports from the central structure to bases "
        "located at radius 15-16 m, not to the 27 m outer rim."
    )
    support["angles_deg"] = support.get("angles_deg", TRIPOD_ANGLES_DEG)
    support["width_m"] = float(support.get("width_m", 0.70))

    # Critical correction:
    # previous simplified drawings used end_radius_m = 27.0, which incorrectly
    # placed support legs all the way to the outer rim.
    support["start_radius_m"] = V1_ARM_START_RADIUS_M
    support["end_radius_m"] = TRIPOD_BASE_RADIUS_M

    support["tripod_base_radius_m"] = TRIPOD_BASE_RADIUS_M
    support["tripod_base_radius_allowed_range_m"] = [
        TRIPOD_BASE_RADIUS_MIN_M,
        TRIPOD_BASE_RADIUS_MAX_M,
    ]
    support["outer_rim_radius_m"] = MAIN_REFLECTOR_RADIUS_M
    support["confirmed_not_outer_rim"] = True
    support["geometry_status"] = "corrected_internal_base_radius_pending_exact_azimuths"

    if "z_plane_m" not in support:
        support["z_plane_m"] = 0.0

    save_yaml(ROT54_CONFIG_PATH, config)
    return config


def ensure_shadow_v2_config() -> dict[str, Any]:
    config = load_yaml(SHADOW_V2_CONFIG_PATH)

    if "shadow_v2" not in config:
        config["shadow_v2"] = {}

    raw = config["shadow_v2"]

    raw["output_dir"] = raw.get("output_dir", "outputs/shadow_v2")
    raw["figures_dir"] = raw.get("figures_dir", "outputs/figures/shadow_v2")
    raw["morning_offset_minutes_after_front_start"] = int(
        raw.get("morning_offset_minutes_after_front_start", 60)
    )
    raw["evening_offset_minutes_before_front_end"] = int(
        raw.get("evening_offset_minutes_before_front_end", 60)
    )
    raw["numerical_epsilon"] = float(raw.get("numerical_epsilon", 1.0e-9))

    if "circular_cylinders" not in raw:
        raw["circular_cylinders"] = {}

    cylinders = raw["circular_cylinders"]

    cylinders.setdefault(
        "central_hub",
        {
            "enabled": True,
            "label": "central_hub",
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "radius_m": 2.2,
            "z_min_m": -0.5,
            "z_max_m": 2.5,
        },
    )

    cylinders.setdefault(
        "secondary_mirror",
        {
            "enabled": True,
            "label": "secondary_mirror",
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "radius_m": 2.5,
            "z_min_m": 3.5,
            "z_max_m": 4.5,
        },
    )

    cylinders.setdefault(
        "optical_reflector",
        {
            "enabled": True,
            "label": "optical_reflector",
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "radius_m": 1.3,
            "z_min_m": 1.5,
            "z_max_m": 2.5,
        },
    )

    if "support_arm_prisms" not in raw or raw["support_arm_prisms"] is None:
        raw["support_arm_prisms"] = {}

    support = raw["support_arm_prisms"]

    support["enabled"] = bool(support.get("enabled", True))
    support["label"] = "support_arm_prisms_internal_tripod"
    support["description"] = (
        "Volumetric tripod legs from central structure to internal bases at "
        "radius 15-16 m. They must not extend to the 27 m outer rim."
    )
    support["angles_deg"] = support.get("angles_deg", TRIPOD_ANGLES_DEG)
    support["width_m"] = float(support.get("width_m", 1.20))

    # Critical correction for every V2 shadow figure:
    support["start_radius_m"] = V2_ARM_START_RADIUS_M
    support["end_radius_m"] = TRIPOD_BASE_RADIUS_M

    support["tripod_base_radius_m"] = TRIPOD_BASE_RADIUS_M
    support["tripod_base_radius_allowed_range_m"] = [
        TRIPOD_BASE_RADIUS_MIN_M,
        TRIPOD_BASE_RADIUS_MAX_M,
    ]
    support["outer_rim_radius_m"] = MAIN_REFLECTOR_RADIUS_M
    support["confirmed_not_outer_rim"] = True
    support["geometry_status"] = "corrected_internal_base_radius_pending_exact_azimuths"

    support["z_min_m"] = float(support.get("z_min_m", 0.0))
    support["z_max_m"] = float(support.get("z_max_m", 4.5))

    save_yaml(SHADOW_V2_CONFIG_PATH, config)
    return config


def ensure_shadow_geometry_contract() -> dict[str, Any]:
    config = load_yaml(SHADOW_GEOMETRY_PATH)

    if "shadow_geometry" not in config:
        config["shadow_geometry"] = {}

    geometry = config["shadow_geometry"]

    geometry["coordinate_system"] = {
        "origin": "main_reflector_center",
        "x_axis": "east / reflector-local x",
        "y_axis": "north / reflector-local y",
        "z_axis": "reflector-local normal direction",
        "units": "m",
    }

    geometry["main_reflector"] = {
        "diameter_m": 54.0,
        "radius_m": MAIN_REFLECTOR_RADIUS_M,
        "effective_aperture_m": 32.0,
        "fixed_axis_tilt_south_deg": 15.0,
    }

    geometry["tripod_support_bases"] = {
        "description": (
            "Internal support bases from which the tripod legs go toward the "
            "central structure."
        ),
        "critical_correction": (
            "The tripod bases are not located at the outer rim of the 54 m "
            "reflector."
        ),
        "radial_distance_from_center_m": TRIPOD_BASE_RADIUS_M,
        "radial_distance_allowed_range_m": [
            TRIPOD_BASE_RADIUS_MIN_M,
            TRIPOD_BASE_RADIUS_MAX_M,
        ],
        "outer_rim_radius_m": MAIN_REFLECTOR_RADIUS_M,
        "count": 3,
        "azimuths_deg_model_initial": TRIPOD_ANGLES_DEG,
        "azimuth_reference": (
            "0 deg = +x model reference in current scripts; exact measured "
            "azimuths remain pending."
        ),
        "modelling_status": "confirmed_internal_radius_approximate_azimuth_pending",
        "future_ray_shadowing_use": True,
    }

    geometry["tripod_legs"] = {
        "start_radius_near_central_hub_m": V2_ARM_START_RADIUS_M,
        "end_radius_at_internal_base_m": TRIPOD_BASE_RADIUS_M,
        "must_not_extend_to_outer_rim": True,
    }

    save_yaml(SHADOW_GEOMETRY_PATH, config)
    return config


def write_report(
    rot54_config: dict[str, Any],
    shadow_v2_config: dict[str, Any],
) -> None:
    v1_support = rot54_config["shadow_model"]["support_arms"]
    v2_support = shadow_v2_config["shadow_v2"]["support_arm_prisms"]

    report = f"""# Stage 7 — Internal Tripod Base Correction

## Correction applied

The tripod support legs are no longer modelled as radial structures extending to the outer rim of the 54 m reflector.

The corrected engineering model uses internal tripod bases located at:

r_base = {TRIPOD_BASE_RADIUS_M:.2f} m

with the accepted working range:

{TRIPOD_BASE_RADIUS_MIN_M:.1f} m <= r_base <= {TRIPOD_BASE_RADIUS_MAX_M:.1f} m

The outer rim radius remains:

R = {MAIN_REFLECTOR_RADIUS_M:.1f} m

## Updated first-stage shadow model

| Parameter | Value |
|---|---:|
| support_arms.start_radius_m | {float(v1_support["start_radius_m"]):.3f} |
| support_arms.end_radius_m | {float(v1_support["end_radius_m"]):.3f} |
| support_arms.outer_rim_radius_m | {float(v1_support["outer_rim_radius_m"]):.3f} |

## Updated Shadow V2 model

| Parameter | Value |
|---|---:|
| support_arm_prisms.start_radius_m | {float(v2_support["start_radius_m"]):.3f} |
| support_arm_prisms.end_radius_m | {float(v2_support["end_radius_m"]):.3f} |
| support_arm_prisms.outer_rim_radius_m | {float(v2_support["outer_rim_radius_m"]):.3f} |

## Consequence

All previously generated shadow figures and downstream plots based on the old end_radius_m = 27.0 m must be treated as stale.

The following layers must be regenerated:

1. shadow maps;
2. Shadow V2 maps;
3. absorbed solar flux maps;
4. steady temperature maps;
5. thermal sensitivity maps;
6. transient thermal maps;
7. panel temperature non-uniformity;
8. panel displacement response;
9. surface RMS / efficiency tables.

## Remaining uncertainty

The base radius is corrected. The exact azimuths of the three bases are still treated as first-pass symmetric placeholders until measured/passport azimuths are entered.
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    rot54_config = ensure_rot54_shadow_v1_config()
    shadow_v2_config = ensure_shadow_v2_config()
    ensure_shadow_geometry_contract()
    write_report(rot54_config=rot54_config, shadow_v2_config=shadow_v2_config)

    print("Stage 7 tripod internal-base correction applied.")
    print(f"V1 support end radius: {rot54_config['shadow_model']['support_arms']['end_radius_m']} m")
    print(f"V2 support end radius: {shadow_v2_config['shadow_v2']['support_arm_prisms']['end_radius_m']} m")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
