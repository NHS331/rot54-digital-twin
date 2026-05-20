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
    load_config(PROJECT_ROOT / "configs" / "rot54_config.yaml")

    summary_path = PROJECT_ROOT / "outputs" / "irradiation" / "absorbed_flux_summary.csv"
    validation_path = PROJECT_ROOT / "outputs" / "irradiation" / "absorbed_flux_validation.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}. "
            "Run scripts/10_compute_absorbed_flux_maps.py first."
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation file not found: {validation_path}. "
            "Run scripts/10_compute_absorbed_flux_maps.py first."
        )

    summary = pd.read_csv(summary_path)
    validation = pd.read_csv(validation_path)

    print("")
    print("Absorbed flux validation:")
    print(validation.to_string(index=False))

    print("")
    print("Absorbed flux summary:")
    columns = [
        "case_key",
        "time_code",
        "selected_local_time",
        "effective_direct_normal_irradiance_W_m2",
        "absorptivity",
        "visible_points",
        "q_abs_visible_mean_W_m2",
        "q_abs_visible_max_W_m2",
        "q_abs_all_points_mean_W_m2",
    ]

    existing = [col for col in columns if col in summary.columns]
    print(summary[existing].to_string(index=False))

    if not validation["all_absorbed_flux_checks_ok"].all():
        raise ValueError("At least one absorbed flux validation check failed.")

    print("")
    print("All absorbed flux validation checks passed.")


if __name__ == "__main__":
    main()
