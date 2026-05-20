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
from rot54.panel_response import (
    build_panel_response_parameters,
    compute_panel_response_map,
    save_panel_response_map,
    summarize_panel_response_map,
    validate_panel_response_map,
    write_panel_response_report,
)


def parse_panel_thermal_filename(path: Path) -> tuple[str, str, str]:
    """
    Parse:
        panel_thermal_summer_solstice_peak_v00.csv

    Returns:
        case_key
        snapshot_code
        wind_code
    """
    pattern = r"^panel_thermal_(.+)_(morning|axis|evening|peak)_(v.+)$"
    match = re.match(pattern, path.stem)

    if not match:
        raise ValueError(f"Could not parse panel thermal filename: {path.name}")

    return match.group(1), match.group(2), match.group(3)


def plot_panel_response(
    response: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    """
    Plot a panel response quantity over equivalent panel centers.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        response["panel_center_x_m"],
        response["panel_center_y_m"],
        c=response[value_column],
        s=6,
        linewidths=0,
    )

    ax.set_title(title)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    main_config = load_config(PROJECT_ROOT / "configs" / "rot54_config.yaml")
    response_config = load_config(PROJECT_ROOT / "configs" / "panel_response_config.yaml")

    raw = response_config["panel_response"]

    input_dir = PROJECT_ROOT / raw["input_dir"]
    output_dir = PROJECT_ROOT / raw["output_dir"]
    figures_dir = PROJECT_ROOT / raw["figures_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    params = build_panel_response_parameters(
        main_config=main_config,
        response_config=response_config,
    )

    input_files = sorted(input_dir.glob(str(raw["input_glob"])))

    input_files = [
        path for path in input_files
        if path.name.startswith("panel_thermal_")
        and path.name not in [
            "panel_thermal_summary.csv",
            "panel_thermal_validation.csv",
        ]
    ]

    if not input_files:
        raise FileNotFoundError(
            f"No panel thermal maps found in {input_dir}. "
            "Run scripts/18_compute_panel_thermal_nonuniformity.py first."
        )

    summaries: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []

    figure_fields = [str(v) for v in raw["figure_fields"]]

    for input_path in input_files:
        case_key, snapshot_code, wind_code = parse_panel_thermal_filename(input_path)

        panel_thermal = pd.read_csv(input_path)

        response = compute_panel_response_map(
            panel_thermal=panel_thermal,
            params=params,
            case_key=case_key,
            snapshot_code=snapshot_code,
            wind_code=wind_code,
        )

        output_csv = output_dir / (
            f"panel_response_{case_key}_{snapshot_code}_{wind_code}.csv"
        )

        save_panel_response_map(
            response=response,
            output_path=output_csv,
        )

        for field in figure_fields:
            if field not in response.columns:
                raise ValueError(f"Figure field is missing from response map: {field}")

            figure_png = figures_dir / (
                f"{field}_{case_key}_{snapshot_code}_{wind_code}.png"
            )

            plot_panel_response(
                response=response,
                value_column=field,
                output_path=figure_png,
                title=(
                    f"ROT-54/2.6 Panel Normal Thermomechanical Response\n"
                    f"{case_key}, {snapshot_code}, {wind_code}\n"
                    f"{field}"
                ),
                colorbar_label=field,
            )

        summaries.append(
            summarize_panel_response_map(
                source_panel_thermal_name=input_path.name,
                case_key=case_key,
                snapshot_code=snapshot_code,
                wind_code=wind_code,
                response=response,
                params=params,
            )
        )

        validations.append(
            validate_panel_response_map(
                source_panel_thermal_name=input_path.name,
                response=response,
                params=params,
            )
        )

        print(f"Panel response calculated: {input_path.name}")
        print(f"  CSV: {output_csv}")

    summary_df = pd.DataFrame(summaries)
    validation_df = pd.DataFrame(validations)

    summary_csv = output_dir / "panel_response_summary.csv"
    validation_csv = output_dir / "panel_response_validation.csv"
    report_md = output_dir / "panel_response_report.md"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    validation_df.to_csv(validation_csv, index=False, encoding="utf-8")

    write_panel_response_report(
        summary=summary_df,
        validation=validation_df,
        output_path=report_md,
    )

    print("")
    print("Panel response step completed.")
    print(f"Summary: {summary_csv}")
    print(f"Validation: {validation_csv}")
    print(f"Report: {report_md}")
    print("")

    columns = [
        "source_panel_thermal_name",
        "case_key",
        "snapshot_code",
        "wind_code",
        "equivalent_panel_count",
        "u_rms_central_max_mm_max",
        "u_rms_upper_mm_max",
        "u_rms_upper_mm_p95",
        "u_peak_to_peak_upper_mm_max",
        "max_u_rms_upper_panel_id",
    ]

    ranked = summary_df.sort_values(
        by=[
            "u_rms_upper_mm_max",
            "u_peak_to_peak_upper_mm_max",
        ],
        ascending=False,
    )

    print("Worst 15 panel-response snapshots:")
    print(ranked[columns].head(15).to_string(index=False))

    failed = validation_df[
        validation_df["all_panel_response_checks_ok"] != True
    ]

    if not failed.empty:
        raise ValueError(
            "At least one panel response validation check failed. "
            "Inspect outputs/panel_response/panel_response_validation.csv"
        )


if __name__ == "__main__":
    main()
