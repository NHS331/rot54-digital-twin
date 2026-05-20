from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rot54.config import load_config


def main() -> None:
    load_config(PROJECT_ROOT / "configs" / "thermal_steady_config.yaml")

    summary_path = PROJECT_ROOT / "outputs" / "thermal_steady" / "steady_temperature_summary.csv"
    validation_path = PROJECT_ROOT / "outputs" / "thermal_steady" / "steady_temperature_validation.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}. "
            "Run scripts/12_compute_steady_temperature_maps.py first."
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation file not found: {validation_path}. "
            "Run scripts/12_compute_steady_temperature_maps.py first."
        )

    summary = pd.read_csv(summary_path)
    validation = pd.read_csv(validation_path)

    print("")
    print("Steady thermal validation:")
    print(validation.to_string(index=False))

    print("")
    print("Steady thermal compact summary:")
    columns = [
        "case_key",
        "time_code",
        "wind_speed_m_s",
        "air_temperature_C",
        "sky_temperature_C",
        "convection_h_W_m2_K",
        "q_abs_max_W_m2",
        "surface_temperature_min_C",
        "surface_temperature_mean_C",
        "surface_temperature_max_C",
        "deltaT_air_max_C",
        "max_abs_thermal_balance_residual_W_m2",
    ]

    existing = [column for column in columns if column in summary.columns]

    print(summary[existing].to_string(index=False))

    if not validation["all_steady_temperature_checks_ok"].all():
        raise ValueError("At least one steady thermal validation check failed.")

    print("")
    print("All steady thermal validation checks passed.")


if __name__ == "__main__":
    main()
