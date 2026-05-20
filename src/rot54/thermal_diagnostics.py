from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CASE_ORDER = {
    "spring_equinox": 0,
    "summer_solstice": 1,
    "winter_solstice": 2,
}

TIME_ORDER = {
    "morning": 0,
    "axis": 1,
    "evening": 2,
}


@dataclass(frozen=True)
class WindSensitivityPaths:
    summary_input_csv: Path
    validation_input_csv: Path
    output_dir: Path
    figures_dir: Path


def parse_bool_column(series: pd.Series) -> pd.Series:
    """
    Robust bool parser for CSV-loaded validation columns.
    """
    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def require_columns(df: pd.DataFrame, required: list[str], source_name: str) -> None:
    """
    Ensure required columns are present.
    """
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def load_steady_thermal_summary(path: Path) -> pd.DataFrame:
    """
    Load and validate steady thermal summary table.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Steady thermal summary was not found: {path}. "
            "Run scripts/12_compute_steady_temperature_maps.py first."
        )

    df = pd.read_csv(path)

    required = [
        "case_key",
        "case_label",
        "time_code",
        "selected_local_time",
        "wind_speed_m_s",
        "air_temperature_C",
        "sky_temperature_C",
        "convection_h_W_m2_K",
        "q_abs_max_W_m2",
        "q_abs_mean_all_W_m2",
        "surface_temperature_min_C",
        "surface_temperature_mean_C",
        "surface_temperature_max_C",
        "deltaT_air_min_C",
        "deltaT_air_mean_C",
        "deltaT_air_max_C",
        "deltaT_air_mean_visible_C",
        "deltaT_aperture_mean_min_C",
        "deltaT_aperture_mean_max_C",
        "max_abs_thermal_balance_residual_W_m2",
    ]

    require_columns(df, required, "steady_temperature_summary.csv")

    df = df.copy()

    df["case_order"] = df["case_key"].map(CASE_ORDER).fillna(99).astype(int)
    df["time_order"] = df["time_code"].map(TIME_ORDER).fillna(99).astype(int)

    df["scenario_id"] = df["case_key"].astype(str) + "__" + df["time_code"].astype(str)
    df["scenario_label"] = (
        df["case_key"].astype(str).str.replace("_", " ", regex=False)
        + " / "
        + df["time_code"].astype(str)
    )

    df = df.sort_values(
        by=[
            "case_order",
            "time_order",
            "wind_speed_m_s",
        ],
        ascending=True,
    ).reset_index(drop=True)

    return df


def validate_previous_steady_solver_output(path: Path) -> pd.DataFrame:
    """
    Load previous steady thermal validation output and ensure it passed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Steady thermal validation was not found: {path}. "
            "Run scripts/12_compute_steady_temperature_maps.py first."
        )

    df = pd.read_csv(path)

    require_columns(
        df,
        ["all_steady_temperature_checks_ok"],
        "steady_temperature_validation.csv",
    )

    ok = parse_bool_column(df["all_steady_temperature_checks_ok"])

    if not ok.all():
        failed = df[~ok].copy()
        raise ValueError(
            "Previous steady thermal validation contains failed rows. "
            f"Failed rows:\n{failed.to_string(index=False)}"
        )

    return df


def check_non_increasing(values: np.ndarray, tolerance: float = 1e-9) -> bool:
    """
    Return True if values are monotonically non-increasing.
    """
    if len(values) <= 1:
        return True

    diffs = np.diff(values)
    return bool(np.all(diffs <= tolerance))


def check_non_decreasing(values: np.ndarray, tolerance: float = 1e-9) -> bool:
    """
    Return True if values are monotonically non-decreasing.
    """
    if len(values) <= 1:
        return True

    diffs = np.diff(values)
    return bool(np.all(diffs >= -tolerance))


def compute_wind_monotonicity_validation(summary: pd.DataFrame) -> pd.DataFrame:
    """
    For each seasonal/time scenario, check whether wind behaves physically.

    Strict checks:
        1. h(v) must increase with wind speed.
        2. surface_temperature_max_C must not increase with wind.
        3. deltaT_air_max_C must not increase with wind.

    We intentionally do not require the full-aperture mean temperature to decrease,
    because shaded cold zones can be pulled closer to air temperature by stronger
    convection, especially in winter cases.
    """
    rows: list[dict[str, object]] = []

    for (case_key, time_code), group in summary.groupby(["case_key", "time_code"]):
        group = group.sort_values("wind_speed_m_s").reset_index(drop=True)

        wind = group["wind_speed_m_s"].to_numpy(dtype=float)
        h = group["convection_h_W_m2_K"].to_numpy(dtype=float)

        surface_max = group["surface_temperature_max_C"].to_numpy(dtype=float)
        delta_air_max = group["deltaT_air_max_C"].to_numpy(dtype=float)

        h_non_decreasing_ok = check_non_decreasing(h)
        surface_max_non_increasing_ok = check_non_increasing(surface_max)
        deltaT_air_max_non_increasing_ok = check_non_increasing(delta_air_max)

        all_ok = bool(
            h_non_decreasing_ok
            and surface_max_non_increasing_ok
            and deltaT_air_max_non_increasing_ok
        )

        rows.append(
            {
                "case_key": case_key,
                "case_label": str(group["case_label"].iloc[0]),
                "time_code": time_code,
                "wind_speed_sequence_m_s": ", ".join(f"{v:.1f}" for v in wind),
                "h_sequence_W_m2_K": ", ".join(f"{v:.6f}" for v in h),
                "surface_temperature_max_sequence_C": ", ".join(
                    f"{v:.6f}" for v in surface_max
                ),
                "deltaT_air_max_sequence_C": ", ".join(
                    f"{v:.6f}" for v in delta_air_max
                ),
                "h_non_decreasing_ok": h_non_decreasing_ok,
                "surface_temperature_max_non_increasing_ok": surface_max_non_increasing_ok,
                "deltaT_air_max_non_increasing_ok": deltaT_air_max_non_increasing_ok,
                "all_wind_sensitivity_checks_ok": all_ok,
            }
        )

    return pd.DataFrame(rows)


def compute_worst_cases(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Rank all steady thermal scenarios by maximum temperature rise over air.
    """
    ranking_columns = [
        "case_key",
        "case_label",
        "time_code",
        "selected_local_time",
        "wind_speed_m_s",
        "air_temperature_C",
        "sky_temperature_C",
        "convection_h_W_m2_K",
        "q_abs_max_W_m2",
        "q_abs_mean_all_W_m2",
        "surface_temperature_max_C",
        "surface_temperature_mean_C",
        "deltaT_air_max_C",
        "deltaT_air_mean_C",
        "deltaT_air_mean_visible_C",
        "deltaT_aperture_mean_min_C",
        "deltaT_aperture_mean_max_C",
        "max_abs_thermal_balance_residual_W_m2",
    ]

    existing = [column for column in ranking_columns if column in summary.columns]

    ranked = summary[existing].copy()
    ranked = ranked.sort_values(
        by=[
            "deltaT_air_max_C",
            "surface_temperature_max_C",
        ],
        ascending=False,
    ).reset_index(drop=True)

    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    return ranked


def write_markdown_report(
    summary: pd.DataFrame,
    validation: pd.DataFrame,
    worst_cases: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Write a compact Markdown report for quick human review.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    worst = worst_cases.iloc[0]

    failed = validation[
        validation["all_wind_sensitivity_checks_ok"] != True
    ].copy()

    lines: list[str] = []

    lines.append("# ROT-54/2.6 Steady Thermal Wind Sensitivity Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This report checks whether the first steady-state thermal maps respond "
        "physically to wind-speed changes before moving to transient thermal modeling."
    )
    lines.append("")
    lines.append("## Worst steady thermal case")
    lines.append("")
    lines.append(f"- Case: `{worst['case_key']}`")
    lines.append(f"- Time code: `{worst['time_code']}`")
    lines.append(f"- Local time: `{worst['selected_local_time']}`")
    lines.append(f"- Wind speed: `{worst['wind_speed_m_s']:.2f} m/s`")
    lines.append(f"- Air temperature: `{worst['air_temperature_C']:.2f} °C`")
    lines.append(f"- Sky temperature: `{worst['sky_temperature_C']:.2f} °C`")
    lines.append(f"- Maximum absorbed flux: `{worst['q_abs_max_W_m2']:.3f} W/m²`")
    lines.append(
        f"- Maximum surface temperature: `{worst['surface_temperature_max_C']:.3f} °C`"
    )
    lines.append(
        f"- Maximum ΔT relative to air: `{worst['deltaT_air_max_C']:.3f} °C`"
    )
    lines.append("")
    lines.append("## Wind monotonicity checks")
    lines.append("")

    if failed.empty:
        lines.append("All wind-sensitivity monotonicity checks passed.")
    else:
        lines.append("Some wind-sensitivity checks failed:")
        lines.append("")
        for _, row in failed.iterrows():
            lines.append(
                f"- `{row['case_key']} / {row['time_code']}` failed one or more checks."
            )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The key expected behavior is that increasing wind speed increases the "
        "convective heat-transfer coefficient and does not increase the maximum "
        "surface temperature or maximum ΔT relative to air for the same solar case."
    )
    lines.append("")
    lines.append(
        "The full-aperture mean temperature is not used as a strict monotonicity "
        "criterion because shaded zones can be radiatively cooled below air temperature "
        "and then pulled back toward air temperature by stronger convection."
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_tables(
    summary: pd.DataFrame,
    validation: pd.DataFrame,
    worst_cases: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """
    Save analysis tables.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_path = output_dir / "steady_wind_sensitivity_sorted.csv"
    validation_path = output_dir / "steady_wind_monotonicity_validation.csv"
    worst_path = output_dir / "steady_worst_cases_by_deltaT_air_max.csv"

    summary.to_csv(sorted_path, index=False, encoding="utf-8")
    validation.to_csv(validation_path, index=False, encoding="utf-8")
    worst_cases.to_csv(worst_path, index=False, encoding="utf-8")

    return {
        "sorted": sorted_path,
        "validation": validation_path,
        "worst": worst_path,
    }
