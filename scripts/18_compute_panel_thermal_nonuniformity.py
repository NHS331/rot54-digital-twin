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
from rot54.panel_thermal import (
    build_panel_thermal_parameters,
    compute_panel_thermal_map,
    save_panel_thermal_map,
    summarize_panel_thermal_map,
    validate_panel_thermal_map,
    write_panel_thermal_report,
)


def parse_snapshot_filename(path: Path) -> tuple[str, str, str]:
    """
    Parse names like:
        transient_snapshot_summer_solstice_peak_v00.csv
        transient_snapshot_spring_equinox_morning_v05.csv

    Returns:
        case_key
        snapshot_code
        wind_code
    """
    stem = path.stem

    pattern = r"^transient_snapshot_(.+)_(morning|axis|evening|peak)_(v.+)$"
    match = re.match(pattern, stem)

    if not match:
        raise ValueError(f"Could not parse transient snapshot filename: {path.name}")

    case_key = match.group(1)
    snapshot_code = match.group(2)
    wind_code = match.group(3)

    return case_key, snapshot_code, wind_code


def plot_panel_quantity(
    panel_map: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    """
    Plot a panel-level quantity over equivalent panel centers.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        panel_map["panel_center_x_m"],
        panel_map["panel_center_y_m"],
        c=panel_map[value_column],
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
    panel_config = load_config(PROJECT_ROOT / "configs" / "panel_thermal_config.yaml")

    raw = panel_config["panel_thermal"]

    input_dir = PROJECT_ROOT / raw["snapshot_input_dir"]
    output_dir = PROJECT_ROOT / raw["output_dir"]
    figures_dir = PROJECT_ROOT / raw["figures_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    params = build_panel_thermal_parameters(
        main_config=main_config,
        panel_config=panel_config,
    )

    snapshot_files = sorted(input_dir.glob(str(raw["snapshot_glob"])))

    snapshot_files = [
        path for path in snapshot_files
        if path.name.startswith("transient_snapshot_")
    ]

    if not snapshot_files:
        raise FileNotFoundError(
            f"No transient snapshot files found in {input_dir}. "
            "Run scripts/16_compute_transient_thermal_day.py first."
        )

    summaries: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []

    for snapshot_path in snapshot_files:
        case_key, snapshot_code, wind_code = parse_snapshot_filename(snapshot_path)

        snapshot = pd.read_csv(snapshot_path)

        case_label = (
            str(snapshot["case_label"].iloc[0])
            if "case_label" in snapshot.columns
            else case_key
        )

        selected_local_time = (
            str(snapshot["selected_local_time"].iloc[0])
            if "selected_local_time" in snapshot.columns
            else ""
        )

        panel_map = compute_panel_thermal_map(
            snapshot=snapshot,
            params=params,
        )

        output_csv = output_dir / (
            f"panel_thermal_{case_key}_{snapshot_code}_{wind_code}.csv"
        )

        save_panel_thermal_map(
            panel_map=panel_map,
            output_path=output_csv,
        )

        title_base = (
            f"ROT-54/2.6 Equivalent Panel Thermal Nonuniformity\n"
            f"{case_label}, {snapshot_code}, {wind_code}, {selected_local_time}"
        )

        rms_png = figures_dir / (
            f"panel_delta_rms_{case_key}_{snapshot_code}_{wind_code}.png"
        )

        ptp_png = figures_dir / (
            f"panel_delta_peak_to_peak_{case_key}_{snapshot_code}_{wind_code}.png"
        )

        mean_png = figures_dir / (
            f"panel_temperature_mean_{case_key}_{snapshot_code}_{wind_code}.png"
        )

        plot_panel_quantity(
            panel_map=panel_map,
            value_column="panel_delta_rms_C",
            output_path=rms_png,
            title=title_base + "\nPanel RMS temperature nonuniformity",
            colorbar_label="Panel ΔT RMS, °C",
        )

        plot_panel_quantity(
            panel_map=panel_map,
            value_column="panel_delta_peak_to_peak_C",
            output_path=ptp_png,
            title=title_base + "\nPanel peak-to-peak temperature difference",
            colorbar_label="Panel ΔT peak-to-peak, °C",
        )

        plot_panel_quantity(
            panel_map=panel_map,
            value_column="panel_temperature_five_point_mean_C",
            output_path=mean_png,
            title=title_base + "\nPanel five-point mean temperature",
            colorbar_label="Panel mean temperature, °C",
        )

        summaries.append(
            summarize_panel_thermal_map(
                source_snapshot_name=snapshot_path.name,
                case_key=case_key,
                case_label=case_label,
                snapshot_code=snapshot_code,
                wind_code=wind_code,
                selected_local_time=selected_local_time,
                panel_map=panel_map,
                params=params,
            )
        )

        validations.append(
            validate_panel_thermal_map(
                source_snapshot_name=snapshot_path.name,
                panel_map=panel_map,
            )
        )

        print(f"Panel thermal map calculated: {snapshot_path.name}")
        print(f"  Equivalent panels: {len(panel_map)}")
        print(f"  CSV: {output_csv}")
        print(f"  RMS PNG: {rms_png}")
        print(f"  Peak-to-peak PNG: {ptp_png}")
        print(f"  Mean temperature PNG: {mean_png}")

    summary_df = pd.DataFrame(summaries)
    validation_df = pd.DataFrame(validations)

    summary_csv = output_dir / "panel_thermal_summary.csv"
    validation_csv = output_dir / "panel_thermal_validation.csv"
    report_md = output_dir / "panel_thermal_report.md"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    validation_df.to_csv(validation_csv, index=False, encoding="utf-8")

    write_panel_thermal_report(
        summary=summary_df,
        validation=validation_df,
        output_path=report_md,
    )

    print("")
    print("Panel thermal nonuniformity step completed.")
    print(f"Summary: {summary_csv}")
    print(f"Validation: {validation_csv}")
    print(f"Report: {report_md}")
    print("")

    print("Worst 10 snapshots by maximum panel RMS temperature nonuniformity:")
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
    ]

    ranked = summary_df.sort_values(
        by=[
            "panel_delta_rms_max_C",
            "panel_delta_peak_to_peak_max_C",
        ],
        ascending=False,
    )

    print(ranked[columns].head(10).to_string(index=False))

    failed = validation_df[
        validation_df["all_panel_thermal_checks_ok"] != True
    ]

    if not failed.empty:
        raise ValueError(
            "At least one panel thermal validation check failed. "
            "Inspect outputs/panel_thermal/panel_thermal_validation.csv"
        )


if __name__ == "__main__":
    main()
