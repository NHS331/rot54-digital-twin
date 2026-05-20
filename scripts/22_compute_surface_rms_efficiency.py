from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rot54.config import load_config
from rot54.surface_rms import (
    build_surface_rms_parameters,
    compute_surface_rms_for_response_map,
    save_surface_rms_summary,
    validate_surface_rms_row,
    write_surface_rms_report,
)


def parse_panel_response_filename(path: Path) -> tuple[str, str, str]:
    """
    Parse:
        panel_response_summer_solstice_peak_v00.csv

    Returns:
        case_key, snapshot_code, wind_code
    """
    pattern = r"^panel_response_(.+)_(morning|axis|evening|peak)_(v.+)$"
    match = re.match(pattern, path.stem)

    if not match:
        raise ValueError(f"Could not parse panel response filename: {path.name}")

    return match.group(1), match.group(2), match.group(3)


def scenario_label(row: pd.Series) -> str:
    """
    Human-readable scenario label for plots.
    """
    return f"{row['case_key']} / {row['snapshot_code']} / {row['wind_code']}"


def plot_worst_bar(
    summary: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    top_n: int,
) -> None:
    """
    Plot top-N worst scenarios by a chosen value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked = summary.sort_values(
        by=value_column,
        ascending=False,
    ).head(top_n).copy()

    labels = [scenario_label(row) for _, row in ranked.iterrows()]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(range(len(ranked)), ranked[value_column])

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(ranked)))
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    main_config = load_config(PROJECT_ROOT / "configs" / "rot54_config.yaml")
    rms_config = load_config(PROJECT_ROOT / "configs" / "surface_rms_config.yaml")

    raw = rms_config["surface_rms"]

    input_dir = PROJECT_ROOT / raw["input_dir"]
    output_dir = PROJECT_ROOT / raw["output_dir"]
    figures_dir = PROJECT_ROOT / raw["figures_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    params = build_surface_rms_parameters(
        main_config=main_config,
        rms_config=rms_config,
    )

    input_files = sorted(input_dir.glob(str(raw["input_glob"])))

    input_files = [
        path for path in input_files
        if path.name.startswith("panel_response_")
        and path.name not in [
            "panel_response_summary.csv",
            "panel_response_validation.csv",
        ]
    ]

    if not input_files:
        raise FileNotFoundError(
            f"No panel response maps found in {input_dir}. "
            "Run scripts/20_compute_panel_response.py first."
        )

    rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for input_path in input_files:
        case_key, snapshot_code, wind_code = parse_panel_response_filename(input_path)

        panel_response = pd.read_csv(input_path)

        row = compute_surface_rms_for_response_map(
            panel_response=panel_response,
            params=params,
            source_name=input_path.name,
            case_key=case_key,
            snapshot_code=snapshot_code,
            wind_code=wind_code,
        )

        validation = validate_surface_rms_row(
            row=row,
            params=params,
        )

        rows.append(row)
        validation_rows.append(validation)

        print(f"Surface RMS calculated: {input_path.name}")
        print(
            f"  sigma_T_upper = {row['sigma_T_upper_mm_primary_mm']:.6f} mm, "
            f"sigma_total_upper = {row['sigma_total_upper_mm_primary_mm']:.6f} mm, "
            f"f10_total_upper = {row['f10_sigma_total_upper_mm_GHz']:.3f} GHz"
        )

    summary = pd.DataFrame(rows)
    validation_df = pd.DataFrame(validation_rows)

    summary = summary.sort_values(
        by=[
            "sigma_total_upper_mm_primary_mm",
            "sigma_T_upper_mm_primary_mm",
        ],
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(0, "rank_by_sigma_total_upper", range(1, len(summary) + 1))

    summary_csv = output_dir / "surface_rms_summary.csv"
    validation_csv = output_dir / "surface_rms_validation.csv"
    worst_csv = output_dir / "surface_rms_worst_cases.csv"
    report_md = output_dir / "surface_rms_report.md"

    save_surface_rms_summary(
        summary=summary,
        output_path=summary_csv,
    )

    validation_df.to_csv(validation_csv, index=False, encoding="utf-8")

    worst = summary.head(params.top_n_worst_cases).copy()
    worst.to_csv(worst_csv, index=False, encoding="utf-8")

    write_surface_rms_report(
        summary=summary,
        validation=validation_df,
        output_path=report_md,
        params=params,
    )

    plot_worst_bar(
        summary=summary,
        value_column="sigma_T_upper_mm_primary_mm",
        output_path=figures_dir / "sigma_T_upper_worst_cases.png",
        title="ROT-54/2.6 Worst Cases by Additional Thermal RMS σ_T Upper",
        ylabel="σ_T upper, mm",
        top_n=params.top_n_worst_cases,
    )

    plot_worst_bar(
        summary=summary,
        value_column="sigma_total_upper_mm_primary_mm",
        output_path=figures_dir / "sigma_total_upper_worst_cases.png",
        title="ROT-54/2.6 Worst Cases by Total RMS σΣ Upper",
        ylabel="σΣ upper, mm",
        top_n=params.top_n_worst_cases,
    )

    plot_worst_bar(
        summary=summary,
        value_column="f10_sigma_total_upper_mm_GHz",
        output_path=figures_dir / "f10_total_upper_worst_cases.png",
        title="ROT-54/2.6 f10 for Worst Cases",
        ylabel="f10 based on σΣ upper, GHz",
        top_n=params.top_n_worst_cases,
    )

    eta_100_column = "eta_R_sigma_total_upper_mm_100p0_GHz"

    if eta_100_column in summary.columns:
        plot_worst_bar(
            summary=summary,
            value_column=eta_100_column,
            output_path=figures_dir / "eta_100GHz_total_upper_worst_cases.png",
            title="ROT-54/2.6 Ruze Efficiency at 100 GHz for Worst Cases",
            ylabel="ηR at 100 GHz, total upper",
            top_n=params.top_n_worst_cases,
        )

    print("")
    print("Surface RMS and Ruze step completed.")
    print(f"Summary: {summary_csv}")
    print(f"Validation: {validation_csv}")
    print(f"Worst cases: {worst_csv}")
    print(f"Report: {report_md}")
    print("")

    columns = [
        "rank_by_sigma_total_upper",
        "case_key",
        "snapshot_code",
        "wind_code",
        "sigma_T_central_max_mm_primary_mm",
        "sigma_T_upper_mm_primary_mm",
        "sigma_total_central_max_mm_primary_mm",
        "sigma_total_upper_mm_primary_mm",
        "f10_sigma_total_upper_mm_GHz",
    ]

    print("Worst 20 cases by σΣ upper:")
    print(summary[columns].head(params.top_n_worst_cases).to_string(index=False))

    failed = validation_df[
        validation_df["all_surface_rms_checks_ok"] != True
    ]

    if not failed.empty:
        raise ValueError(
            "At least one surface RMS validation check failed. "
            "Inspect outputs/surface_rms/surface_rms_validation.csv"
        )


if __name__ == "__main__":
    main()
