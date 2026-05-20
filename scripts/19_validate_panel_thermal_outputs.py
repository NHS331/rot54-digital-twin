from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "panel_thermal"
    figures_dir = PROJECT_ROOT / "outputs" / "figures" / "panel_thermal"

    summary_csv = output_dir / "panel_thermal_summary.csv"
    validation_csv = output_dir / "panel_thermal_validation.csv"
    report_md = output_dir / "panel_thermal_report.md"

    required = [
        summary_csv,
        validation_csv,
        report_md,
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing panel thermal outputs:\n"
            + "\n".join(str(path) for path in missing)
        )

    summary = pd.read_csv(summary_csv)
    validation = pd.read_csv(validation_csv)

    if summary.empty:
        raise ValueError("panel_thermal_summary.csv is empty.")

    if validation.empty:
        raise ValueError("panel_thermal_validation.csv is empty.")

    if "all_panel_thermal_checks_ok" not in validation.columns:
        raise ValueError(
            "all_panel_thermal_checks_ok column missing in validation table."
        )

    ok = validation["all_panel_thermal_checks_ok"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    print("")
    print("Panel thermal validation:")
    print(validation.to_string(index=False))

    print("")
    print("Worst 15 panel thermal snapshots:")
    columns = [
        "source_snapshot_name",
        "case_key",
        "snapshot_code",
        "wind_code",
        "selected_local_time",
        "equivalent_panel_count",
        "panel_delta_rms_max_C",
        "panel_delta_rms_p95_C",
        "panel_delta_peak_to_peak_max_C",
        "panel_temperature_gradient_max_C_per_m",
    ]

    existing = [column for column in columns if column in summary.columns]

    ranked = summary[existing].sort_values(
        by=[
            "panel_delta_rms_max_C",
            "panel_delta_peak_to_peak_max_C",
        ],
        ascending=False,
    )

    print(ranked.head(15).to_string(index=False))

    figure_files = list(figures_dir.glob("panel_delta_rms_*.png"))

    if not figure_files:
        raise FileNotFoundError(
            f"No panel RMS figures found in {figures_dir}"
        )

    if not ok.all():
        failed = validation[~ok]
        raise ValueError(
            "Panel thermal validation failed:\n"
            + failed.to_string(index=False)
        )

    print("")
    print("All panel thermal validation checks passed.")
    print(f"Summary: {summary_csv}")
    print(f"Report: {report_md}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
