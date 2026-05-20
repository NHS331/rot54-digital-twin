from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rot54.thermal_diagnostics import (
    compute_wind_monotonicity_validation,
    compute_worst_cases,
    load_steady_thermal_summary,
    save_tables,
    validate_previous_steady_solver_output,
    write_markdown_report,
)


def plot_metric_vs_wind(
    summary: pd.DataFrame,
    metric_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Plot one thermal metric against wind speed for all seasonal/time scenarios.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))

    for scenario_id, group in summary.groupby("scenario_id"):
        group = group.sort_values("wind_speed_m_s")
        label = str(group["scenario_label"].iloc[0])

        ax.plot(
            group["wind_speed_m_s"],
            group[metric_column],
            marker="o",
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("Wind speed, m/s")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=7, ncols=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    summary_input_csv = (
        PROJECT_ROOT
        / "outputs"
        / "thermal_steady"
        / "steady_temperature_summary.csv"
    )

    validation_input_csv = (
        PROJECT_ROOT
        / "outputs"
        / "thermal_steady"
        / "steady_temperature_validation.csv"
    )

    output_dir = PROJECT_ROOT / "outputs" / "thermal_analysis"
    figures_dir = PROJECT_ROOT / "outputs" / "figures" / "thermal_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    validate_previous_steady_solver_output(validation_input_csv)

    summary = load_steady_thermal_summary(summary_input_csv)

    validation = compute_wind_monotonicity_validation(summary)
    worst_cases = compute_worst_cases(summary)

    saved = save_tables(
        summary=summary,
        validation=validation,
        worst_cases=worst_cases,
        output_dir=output_dir,
    )

    report_path = output_dir / "steady_wind_report.md"

    write_markdown_report(
        summary=summary,
        validation=validation,
        worst_cases=worst_cases,
        output_path=report_path,
    )

    plot_metric_vs_wind(
        summary=summary,
        metric_column="deltaT_air_max_C",
        ylabel="Maximum Delta T relative to air, °C",
        title="ROT-54/2.6 Steady Thermal Sensitivity: Maximum Delta T vs Wind",
        output_path=figures_dir / "deltaT_air_max_vs_wind.png",
    )

    plot_metric_vs_wind(
        summary=summary,
        metric_column="surface_temperature_max_C",
        ylabel="Maximum surface temperature, °C",
        title="ROT-54/2.6 Steady Thermal Sensitivity: Maximum Surface Temperature vs Wind",
        output_path=figures_dir / "surface_temperature_max_vs_wind.png",
    )

    plot_metric_vs_wind(
        summary=summary,
        metric_column="deltaT_air_mean_visible_C",
        ylabel="Mean visible-zone Delta T relative to air, °C",
        title="ROT-54/2.6 Steady Thermal Sensitivity: Visible-Zone Mean Delta T vs Wind",
        output_path=figures_dir / "deltaT_air_mean_visible_vs_wind.png",
    )

    plot_metric_vs_wind(
        summary=summary,
        metric_column="convection_h_W_m2_K",
        ylabel="Convection coefficient h(v), W/(m² K)",
        title="ROT-54/2.6 Convection Coefficient h(v) vs Wind",
        output_path=figures_dir / "convection_h_vs_wind.png",
    )

    print("")
    print("Steady wind sensitivity analysis completed.")
    print(f"Sorted sensitivity table: {saved['sorted']}")
    print(f"Wind monotonicity validation: {saved['validation']}")
    print(f"Worst cases table: {saved['worst']}")
    print(f"Markdown report: {report_path}")
    print("")
    print("Figures:")
    print(figures_dir / "deltaT_air_max_vs_wind.png")
    print(figures_dir / "surface_temperature_max_vs_wind.png")
    print(figures_dir / "deltaT_air_mean_visible_vs_wind.png")
    print(figures_dir / "convection_h_vs_wind.png")
    print("")

    print("Worst 10 steady thermal cases by maximum Delta T relative to air:")
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

    print(worst_cases[columns].head(10).to_string(index=False))

    failed = validation[
        validation["all_wind_sensitivity_checks_ok"] != True
    ]

    print("")
    print("Wind sensitivity validation:")
    print(validation.to_string(index=False))

    if not failed.empty:
        raise ValueError(
            "At least one wind sensitivity validation check failed. "
            "Inspect outputs/thermal_analysis/steady_wind_monotonicity_validation.csv"
        )


if __name__ == "__main__":
    main()
