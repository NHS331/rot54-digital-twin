from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rot54.config import load_config
from rot54.incidence import IncidenceParameters, compute_incidence_for_grid
from rot54.shadowing_v2 import (
    apply_shadow_v2,
    build_shadow_v2_parameters,
    save_shadow_v2,
    summarize_shadow_v2,
    validate_shadow_v2,
)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def select_three_times(
    solar_case: pd.DataFrame,
    morning_offset_minutes: int,
    evening_offset_minutes: int,
) -> dict[str, pd.Series]:
    """
    Select three representative times from one seasonal day:
        morning: first front-side time + offset
        axis: maximum axis_dot_sun during front-side illumination
        evening: last front-side time - offset
    """
    front_mask = parse_bool_series(solar_case["front_side_illumination"])
    front = solar_case[front_mask].copy()

    if front.empty:
        raise ValueError("No front-side illumination rows found.")

    front = front.reset_index(drop=True)

    # The solar table is currently generated with one-minute resolution.
    # We still infer the nearest row count from the requested minute offset.
    morning_index = min(max(0, morning_offset_minutes), len(front) - 1)
    evening_index = max(0, len(front) - 1 - max(0, evening_offset_minutes))

    original_axis_idx = front["axis_dot_sun"].idxmax()

    return {
        "morning": front.iloc[morning_index],
        "axis": front.loc[original_axis_idx],
        "evening": front.iloc[evening_index],
    }


def plot_binary_map(
    df: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        df["x_m"],
        df["y_m"],
        c=df[value_column],
        s=2,
        linewidths=0,
        vmin=0,
        vmax=1,
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


def plot_factor_map(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        df["x_m"],
        df["y_m"],
        c=df["effective_solar_factor_v2"],
        s=2,
        linewidths=0,
        vmin=0,
        vmax=1,
    )

    ax.set_title(title)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("max(cos(theta_i), 0) * chi_v2")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    main_config = load_config(PROJECT_ROOT / "configs" / "rot54_config.yaml")
    shadow_v2_config = load_config(PROJECT_ROOT / "configs" / "shadow_v2_config.yaml")

    geometry_csv = PROJECT_ROOT / main_config["geometry_grid"]["output_csv"]
    solar_csv = PROJECT_ROOT / main_config["solar_outputs"]["seasonal_positions_csv"]

    if not geometry_csv.exists():
        raise FileNotFoundError(
            f"Geometry grid not found: {geometry_csv}. "
            "Run scripts/03_generate_mirror_grid.py first."
        )

    if not solar_csv.exists():
        raise FileNotFoundError(
            f"Solar table not found: {solar_csv}. "
            "Run scripts/05_generate_solar_tables.py first."
        )

    mirror_grid = pd.read_csv(geometry_csv)
    solar_table = pd.read_csv(solar_csv)

    raw_v2 = shadow_v2_config["shadow_v2"]
    output_dir = PROJECT_ROOT / raw_v2["output_dir"]
    figures_dir = PROJECT_ROOT / raw_v2["figures_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    morning_offset = int(raw_v2["morning_offset_minutes_after_front_start"])
    evening_offset = int(raw_v2["evening_offset_minutes_before_front_end"])

    incidence_params = IncidenceParameters(
        axis_tilt_south_deg=float(main_config["main_reflector"]["axis_tilt_south_deg"])
    )

    shadow_params = build_shadow_v2_parameters(
        main_config=main_config,
        shadow_v2_config=shadow_v2_config,
    )

    summaries: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []

    for case_key, case_data in solar_table.groupby("case_key"):
        case_key = str(case_key)
        case_label = str(case_data["case_label"].iloc[0])

        selected_times = select_three_times(
            solar_case=case_data,
            morning_offset_minutes=morning_offset,
            evening_offset_minutes=evening_offset,
        )

        for time_code, solar_row in selected_times.items():
            date = str(solar_row["date"])
            clock_time = str(solar_row["clock_time"])
            selected_local_time = f"{date} {clock_time} Asia/Yerevan"

            incidence = compute_incidence_for_grid(
                mirror_grid=mirror_grid,
                solar_row=solar_row,
                params=incidence_params,
            )

            incidence.insert(0, "case_key", case_key)
            incidence.insert(1, "case_label", case_label)
            incidence.insert(2, "time_code", time_code)
            incidence.insert(3, "selected_local_time", selected_local_time)

            shadow = apply_shadow_v2(
                incidence=incidence,
                params=shadow_params,
            )

            output_csv = output_dir / f"shadow_v2_{case_key}_{time_code}.csv"

            structure_png = figures_dir / f"shadow_v2_structure_{case_key}_{time_code}.png"
            chi_png = figures_dir / f"shadow_v2_chi_{case_key}_{time_code}.png"
            factor_png = figures_dir / f"shadow_v2_effective_factor_{case_key}_{time_code}.png"

            save_shadow_v2(
                shadow=shadow,
                output_path=output_csv,
            )

            title_base = (
                f"ROT-54/2.6 Shadow V2\n"
                f"{case_label}, {time_code}, {selected_local_time}\n"
                f"alt={float(solar_row['solar_altitude_deg']):.2f} deg, "
                f"az={float(solar_row['solar_azimuth_deg']):.2f} deg"
            )

            plot_binary_map(
                df=shadow.assign(structure_shadow_v2_numeric=shadow["structure_shadow_v2"].astype(int)),
                value_column="structure_shadow_v2_numeric",
                output_path=structure_png,
                title=title_base + "\nStructural shadow only",
                colorbar_label="structure_shadow_v2",
            )

            plot_binary_map(
                df=shadow,
                value_column="visibility_chi_v2",
                output_path=chi_png,
                title=title_base + "\nVisibility chi_v2",
                colorbar_label="chi_v2",
            )

            plot_factor_map(
                df=shadow,
                output_path=factor_png,
                title=title_base + "\nEffective solar factor",
            )

            summaries.append(
                summarize_shadow_v2(
                    case_key=case_key,
                    case_label=case_label,
                    time_code=time_code,
                    selected_local_time=selected_local_time,
                    solar_altitude_deg=float(solar_row["solar_altitude_deg"]),
                    solar_azimuth_deg=float(solar_row["solar_azimuth_deg"]),
                    shadow=shadow,
                )
            )

            validations.append(
                validate_shadow_v2(
                    case_key=case_key,
                    time_code=time_code,
                    shadow=shadow,
                )
            )

            print(f"{case_label} / {time_code}: shadow V2 calculated.")
            print(f"  Time: {selected_local_time}")
            print(f"  Solar altitude: {float(solar_row['solar_altitude_deg']):.2f} deg")
            print(f"  Solar azimuth: {float(solar_row['solar_azimuth_deg']):.2f} deg")
            print(f"  CSV: {output_csv}")
            print(f"  Structure shadow PNG: {structure_png}")
            print(f"  Chi PNG: {chi_png}")
            print(f"  Effective factor PNG: {factor_png}")

    summary_df = pd.DataFrame(summaries)
    validation_df = pd.DataFrame(validations)

    summary_csv = output_dir / "shadow_v2_summary.csv"
    validation_csv = output_dir / "shadow_v2_validation.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    validation_df.to_csv(validation_csv, index=False, encoding="utf-8")

    print("")
    print("Shadow V2 completed.")
    print(f"Summary: {summary_csv}")
    print(f"Validation: {validation_csv}")
    print("")

    for _, row in validation_df.iterrows():
        print(
            f"- {row['case_key']} / {row['time_code']}: "
            f"all_shadow_v2_checks_ok = {row['all_shadow_v2_checks_ok']}, "
            f"max_factor_error = {row['max_factor_equation_error']:.3e}"
        )

    if not validation_df["all_shadow_v2_checks_ok"].all():
        raise ValueError("At least one Shadow V2 validation check failed.")


if __name__ == "__main__":
    main()
