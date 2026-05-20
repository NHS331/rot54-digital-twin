from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "surface_rms"
    figures_dir = PROJECT_ROOT / "outputs" / "figures" / "surface_rms"

    summary_csv = output_dir / "surface_rms_summary.csv"
    validation_csv = output_dir / "surface_rms_validation.csv"
    report_md = output_dir / "surface_rms_report.md"

    required = [
        summary_csv,
        validation_csv,
        report_md,
        figures_dir / "sigma_T_upper_worst_cases.png",
        figures_dir / "sigma_total_upper_worst_cases.png",
        figures_dir / "f10_total_upper_worst_cases.png",
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing surface RMS outputs:\n"
            + "\n".join(str(path) for path in missing)
        )

    summary = pd.read_csv(summary_csv)
    validation = pd.read_csv(validation_csv)

    if summary.empty:
        raise ValueError("surface_rms_summary.csv is empty.")

    if validation.empty:
        raise ValueError("surface_rms_validation.csv is empty.")

    if "all_surface_rms_checks_ok" not in validation.columns:
        raise ValueError(
            "all_surface_rms_checks_ok column missing in validation table."
        )

    ok = validation["all_surface_rms_checks_ok"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    print("")
    print("Surface RMS validation:")
    print(validation.to_string(index=False))

    print("")
    print("Worst 20 surface RMS cases:")
    columns = [
        "rank_by_sigma_total_upper",
        "case_key",
        "snapshot_code",
        "wind_code",
        "base_surface_rms_mm",
        "sigma_T_central_min_mm_primary_mm",
        "sigma_T_central_max_mm_primary_mm",
        "sigma_T_upper_mm_primary_mm",
        "sigma_total_central_max_mm_primary_mm",
        "sigma_total_upper_mm_primary_mm",
        "f10_sigma_total_upper_mm_GHz",
        "eta_R_sigma_total_upper_mm_100p0_GHz",
    ]

    existing = [column for column in columns if column in summary.columns]

    print(summary[existing].head(20).to_string(index=False))

    if not ok.all():
        failed = validation[~ok]
        raise ValueError(
            "Surface RMS validation failed:\n"
            + failed.to_string(index=False)
        )

    print("")
    print("All surface RMS validation checks passed.")
    print(f"Summary: {summary_csv}")
    print(f"Report: {report_md}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
