from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    analysis_dir = PROJECT_ROOT / "outputs" / "thermal_analysis"
    figures_dir = PROJECT_ROOT / "outputs" / "figures" / "thermal_analysis"

    validation_csv = analysis_dir / "steady_wind_monotonicity_validation.csv"
    worst_csv = analysis_dir / "steady_worst_cases_by_deltaT_air_max.csv"
    sorted_csv = analysis_dir / "steady_wind_sensitivity_sorted.csv"
    report_md = analysis_dir / "steady_wind_report.md"

    required_files = [
        validation_csv,
        worst_csv,
        sorted_csv,
        report_md,
        figures_dir / "deltaT_air_max_vs_wind.png",
        figures_dir / "surface_temperature_max_vs_wind.png",
        figures_dir / "deltaT_air_mean_visible_vs_wind.png",
        figures_dir / "convection_h_vs_wind.png",
    ]

    missing = [path for path in required_files if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Some thermal analysis outputs are missing:\n"
            + "\n".join(str(path) for path in missing)
        )

    validation = pd.read_csv(validation_csv)
    worst = pd.read_csv(worst_csv)
    sorted_summary = pd.read_csv(sorted_csv)

    if validation.empty:
        raise ValueError("Wind monotonicity validation table is empty.")

    if worst.empty:
        raise ValueError("Worst cases table is empty.")

    if sorted_summary.empty:
        raise ValueError("Sorted wind sensitivity table is empty.")

    if "all_wind_sensitivity_checks_ok" not in validation.columns:
        raise ValueError(
            "Column all_wind_sensitivity_checks_ok is missing from validation table."
        )

    ok = validation["all_wind_sensitivity_checks_ok"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    print("")
    print("Thermal wind-sensitivity validation table:")
    print(validation.to_string(index=False))

    print("")
    print("Worst 10 cases by deltaT_air_max_C:")
    columns = [
        "rank",
        "case_key",
        "time_code",
        "wind_speed_m_s",
        "selected_local_time",
        "q_abs_max_W_m2",
        "surface_temperature_max_C",
        "deltaT_air_max_C",
    ]

    existing = [column for column in columns if column in worst.columns]
    print(worst[existing].head(10).to_string(index=False))

    if not ok.all():
        failed = validation[~ok]
        raise ValueError(
            "Wind-sensitivity validation failed for:\n"
            + failed.to_string(index=False)
        )

    print("")
    print("All thermal wind-sensitivity validation checks passed.")
    print(f"Report: {report_md}")


if __name__ == "__main__":
    main()
