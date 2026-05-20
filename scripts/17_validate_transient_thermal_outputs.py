from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "thermal_transient"
    figures_dir = PROJECT_ROOT / "outputs" / "figures" / "thermal_transient"

    summary_csv = output_dir / "transient_summary.csv"
    validation_csv = output_dir / "transient_validation.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(
            f"Transient summary not found: {summary_csv}. "
            "Run scripts/16_compute_transient_thermal_day.py first."
        )

    if not validation_csv.exists():
        raise FileNotFoundError(
            f"Transient validation not found: {validation_csv}. "
            "Run scripts/16_compute_transient_thermal_day.py first."
        )

    summary = pd.read_csv(summary_csv)
    validation = pd.read_csv(validation_csv)

    if summary.empty:
        raise ValueError("transient_summary.csv is empty.")

    if validation.empty:
        raise ValueError("transient_validation.csv is empty.")

    if "all_transient_checks_ok" not in validation.columns:
        raise ValueError("all_transient_checks_ok column missing in validation table.")

    ok = validation["all_transient_checks_ok"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    print("")
    print("Transient validation:")
    print(validation.to_string(index=False))

    print("")
    print("Daily peak summary:")
    columns = [
        "case_key",
        "wind_speed_m_s",
        "daily_peak_local_time",
        "air_temperature_C",
        "sky_temperature_C",
        "q_abs_max_W_m2",
        "surface_temperature_max_C",
        "deltaT_air_max_C",
        "surface_temperature_mean_C",
        "deltaT_air_mean_C",
    ]

    existing = [col for col in columns if col in summary.columns]

    ranked = summary[existing].sort_values(
        by=["deltaT_air_max_C", "surface_temperature_max_C"],
        ascending=False,
    )

    print(ranked.to_string(index=False))

    required_figures = list(figures_dir.glob("transient_deltaT_air_max_*.png"))

    if not required_figures:
        raise FileNotFoundError(
            f"No transient time-series figures found in {figures_dir}"
        )

    if not ok.all():
        failed = validation[~ok]
        raise ValueError(
            "Transient validation failed:\n"
            + failed.to_string(index=False)
        )

    print("")
    print("All transient thermal validation checks passed.")
    print(f"Summary: {summary_csv}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
