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
    config = load_config(PROJECT_ROOT / "configs" / "rot54_config.yaml")

    validation_path = PROJECT_ROOT / config["shadow_outputs"]["validation_csv"]
    summary_path = PROJECT_ROOT / config["shadow_outputs"]["summary_csv"]

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Shadow validation file not found: {validation_path}. "
            "Run scripts/07_compute_shadow_maps.py first."
        )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Shadow summary file not found: {summary_path}. "
            "Run scripts/07_compute_shadow_maps.py first."
        )

    validation = pd.read_csv(validation_path)
    summary = pd.read_csv(summary_path)

    print("")
    print("Shadow validation table:")
    print(validation.to_string(index=False))

    print("")
    print("Shadow summary table:")
    display_columns = [
        "case_key",
        "potential_sunlit_points",
        "structure_shadow_points",
        "visible_sunlit_points_after_shadowing",
        "shadow_fraction_of_potential_sunlit",
        "visible_fraction_of_aperture",
        "effective_solar_factor_mean_visible",
        "effective_solar_factor_max_visible",
    ]

    existing_columns = [
        column for column in display_columns
        if column in summary.columns
    ]

    print(summary[existing_columns].to_string(index=False))

    if not validation["all_shadow_checks_ok"].all():
        raise ValueError(
            "At least one shadow validation check failed. "
            "Inspect outputs/shadow/shadow_validation.csv"
        )

    print("")
    print("All shadow validation checks passed.")


if __name__ == "__main__":
    main()
