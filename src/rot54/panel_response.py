from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelResponseParameters:
    """
    Reduced panel thermomechanical response parameters.

    The model is:

        u_n = k_u * DeltaT_panel

    where:
        u_n            is the equivalent normal displacement, mm
        k_u            is the reduced thermomechanical coefficient, mm/degC
        DeltaT_panel   is the panel thermal nonuniformity, degC

    In this step we use:
        panel_delta_rms_C
        panel_delta_peak_to_peak_C
        panel_delta_max_abs_from_mean_C

    The result is a magnitude estimate, not a signed deformation field.
    """

    ku_central_min_mm_per_C: float
    ku_central_max_mm_per_C: float
    ku_upper_mm_per_C: float
    response_thresholds_mm: tuple[float, ...]


def build_panel_response_parameters(
    main_config: dict[str, Any],
    response_config: dict[str, Any],
) -> PanelResponseParameters:
    """
    Build panel-response parameters from YAML configs.
    """
    panel_raw = main_config["panel_model"]
    response_raw = response_config["panel_response"]

    params = PanelResponseParameters(
        ku_central_min_mm_per_C=float(panel_raw["ku_central_min_mm_per_C"]),
        ku_central_max_mm_per_C=float(panel_raw["ku_central_max_mm_per_C"]),
        ku_upper_mm_per_C=float(panel_raw["ku_upper_mm_per_C"]),
        response_thresholds_mm=tuple(
            float(v) for v in response_raw["response_thresholds_mm"]
        ),
    )

    validate_panel_response_parameters(params)

    return params


def validate_panel_response_parameters(params: PanelResponseParameters) -> None:
    """
    Validate response coefficients.
    """
    if params.ku_central_min_mm_per_C < 0.0:
        raise ValueError("ku_central_min_mm_per_C must be non-negative.")

    if params.ku_central_max_mm_per_C < 0.0:
        raise ValueError("ku_central_max_mm_per_C must be non-negative.")

    if params.ku_upper_mm_per_C < 0.0:
        raise ValueError("ku_upper_mm_per_C must be non-negative.")

    if params.ku_central_min_mm_per_C > params.ku_central_max_mm_per_C:
        raise ValueError("ku_central_min must not exceed ku_central_max.")

    if params.ku_central_max_mm_per_C > params.ku_upper_mm_per_C:
        raise ValueError("ku_central_max must not exceed ku_upper.")

    if not params.response_thresholds_mm:
        raise ValueError("At least one response threshold is required.")

    if any(v < 0.0 for v in params.response_thresholds_mm):
        raise ValueError("Response thresholds must be non-negative.")


def require_panel_thermal_columns(panel_thermal: pd.DataFrame) -> None:
    """
    Ensure required input columns are present.
    """
    required = [
        "equivalent_panel_id",
        "panel_center_x_m",
        "panel_center_y_m",
        "panel_center_radius_m",
        "panel_grid_point_count",
        "panel_delta_rms_C",
        "panel_delta_peak_to_peak_C",
        "panel_delta_max_abs_from_mean_C",
        "panel_temperature_gradient_C_per_m",
        "panel_q_abs_mean_W_m2",
        "panel_q_abs_max_W_m2",
    ]

    missing = [column for column in required if column not in panel_thermal.columns]

    if missing:
        raise ValueError(f"Panel thermal map is missing required columns: {missing}")


def compute_panel_response_map(
    panel_thermal: pd.DataFrame,
    params: PanelResponseParameters,
    case_key: str,
    snapshot_code: str,
    wind_code: str,
) -> pd.DataFrame:
    """
    Compute equivalent normal thermomechanical response for every panel.
    """
    require_panel_thermal_columns(panel_thermal)

    result = panel_thermal.copy()

    result.insert(0, "case_key", case_key)
    result.insert(1, "snapshot_code", snapshot_code)
    result.insert(2, "wind_code", wind_code)

    d_rms = result["panel_delta_rms_C"].to_numpy(dtype=float)
    d_ptp = result["panel_delta_peak_to_peak_C"].to_numpy(dtype=float)
    d_max_abs = result["panel_delta_max_abs_from_mean_C"].to_numpy(dtype=float)

    if np.nanmin(d_rms) < -1e-12:
        raise ValueError("panel_delta_rms_C contains negative values.")

    if np.nanmin(d_ptp) < -1e-12:
        raise ValueError("panel_delta_peak_to_peak_C contains negative values.")

    if np.nanmin(d_max_abs) < -1e-12:
        raise ValueError("panel_delta_max_abs_from_mean_C contains negative values.")

    coefficients = {
        "central_min": params.ku_central_min_mm_per_C,
        "central_max": params.ku_central_max_mm_per_C,
        "upper": params.ku_upper_mm_per_C,
    }

    for label, ku in coefficients.items():
        result[f"ku_{label}_mm_per_C"] = ku

        result[f"u_rms_{label}_mm"] = ku * d_rms
        result[f"u_peak_to_peak_{label}_mm"] = ku * d_ptp
        result[f"u_max_abs_from_mean_{label}_mm"] = ku * d_max_abs

    result["u_rms_central_mid_mm"] = 0.5 * (
        result["u_rms_central_min_mm"] + result["u_rms_central_max_mm"]
    )

    result["u_peak_to_peak_central_mid_mm"] = 0.5 * (
        result["u_peak_to_peak_central_min_mm"]
        + result["u_peak_to_peak_central_max_mm"]
    )

    result["u_max_abs_from_mean_central_mid_mm"] = 0.5 * (
        result["u_max_abs_from_mean_central_min_mm"]
        + result["u_max_abs_from_mean_central_max_mm"]
    )

    return result


def summarize_panel_response_map(
    source_panel_thermal_name: str,
    case_key: str,
    snapshot_code: str,
    wind_code: str,
    response: pd.DataFrame,
    params: PanelResponseParameters,
) -> dict[str, object]:
    """
    Summarize panel response map.
    """
    row: dict[str, object] = {
        "source_panel_thermal_name": source_panel_thermal_name,
        "case_key": case_key,
        "snapshot_code": snapshot_code,
        "wind_code": wind_code,
        "equivalent_panel_count": int(len(response)),
        "ku_central_min_mm_per_C": params.ku_central_min_mm_per_C,
        "ku_central_max_mm_per_C": params.ku_central_max_mm_per_C,
        "ku_upper_mm_per_C": params.ku_upper_mm_per_C,
    }

    fields = [
        "u_rms_central_min_mm",
        "u_rms_central_max_mm",
        "u_rms_central_mid_mm",
        "u_rms_upper_mm",
        "u_peak_to_peak_central_max_mm",
        "u_peak_to_peak_upper_mm",
        "u_max_abs_from_mean_central_max_mm",
        "u_max_abs_from_mean_upper_mm",
    ]

    for field in fields:
        values = response[field].to_numpy(dtype=float)

        row[f"{field}_mean"] = float(np.nanmean(values))
        row[f"{field}_p95"] = float(np.nanpercentile(values, 95))
        row[f"{field}_max"] = float(np.nanmax(values))

    upper_rms = response["u_rms_upper_mm"].to_numpy(dtype=float)

    for threshold in params.response_thresholds_mm:
        safe = str(threshold).replace(".", "p")
        count = int(np.sum(upper_rms >= threshold))
        fraction = float(count / len(upper_rms))

        row[f"panels_u_rms_upper_ge_{safe}_mm_count"] = count
        row[f"panels_u_rms_upper_ge_{safe}_mm_fraction"] = fraction

    max_index = int(response["u_rms_upper_mm"].idxmax())
    max_panel = response.loc[max_index]

    row["max_u_rms_upper_panel_id"] = str(max_panel["equivalent_panel_id"])
    row["max_u_rms_upper_panel_x_m"] = float(max_panel["panel_center_x_m"])
    row["max_u_rms_upper_panel_y_m"] = float(max_panel["panel_center_y_m"])
    row["max_u_rms_upper_panel_delta_rms_C"] = float(max_panel["panel_delta_rms_C"])
    row["max_u_rms_upper_panel_u_rms_mm"] = float(max_panel["u_rms_upper_mm"])

    return row


def validate_panel_response_map(
    source_panel_thermal_name: str,
    response: pd.DataFrame,
    params: PanelResponseParameters,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """
    Validate response equations and physical bounds.
    """
    required = [
        "panel_delta_rms_C",
        "panel_delta_peak_to_peak_C",
        "panel_delta_max_abs_from_mean_C",
        "u_rms_central_min_mm",
        "u_rms_central_max_mm",
        "u_rms_upper_mm",
        "u_peak_to_peak_upper_mm",
        "u_max_abs_from_mean_upper_mm",
    ]

    missing = [column for column in required if column not in response.columns]

    if missing:
        raise ValueError(f"Panel response map is missing columns: {missing}")

    numerical = response[required].to_numpy(dtype=float)

    finite_ok = bool(np.isfinite(numerical).all())

    nonnegative_ok = bool((numerical >= -tolerance).all())

    order_ok = bool(
        (
            response["u_rms_central_min_mm"]
            <= response["u_rms_central_max_mm"] + tolerance
        ).all()
        and (
            response["u_rms_central_max_mm"]
            <= response["u_rms_upper_mm"] + tolerance
        ).all()
    )

    expected_min = (
        params.ku_central_min_mm_per_C
        * response["panel_delta_rms_C"].to_numpy(dtype=float)
    )

    expected_max = (
        params.ku_central_max_mm_per_C
        * response["panel_delta_rms_C"].to_numpy(dtype=float)
    )

    expected_upper = (
        params.ku_upper_mm_per_C
        * response["panel_delta_rms_C"].to_numpy(dtype=float)
    )

    error_min = float(
        np.nanmax(np.abs(response["u_rms_central_min_mm"].to_numpy(dtype=float) - expected_min))
    )

    error_max = float(
        np.nanmax(np.abs(response["u_rms_central_max_mm"].to_numpy(dtype=float) - expected_max))
    )

    error_upper = float(
        np.nanmax(np.abs(response["u_rms_upper_mm"].to_numpy(dtype=float) - expected_upper))
    )

    equation_ok = bool(
        error_min <= tolerance
        and error_max <= tolerance
        and error_upper <= tolerance
    )

    panel_count_ok = bool(len(response) > 100)

    all_ok = bool(
        finite_ok
        and nonnegative_ok
        and order_ok
        and equation_ok
        and panel_count_ok
    )

    return {
        "source_panel_thermal_name": source_panel_thermal_name,
        "finite_outputs_ok": finite_ok,
        "nonnegative_outputs_ok": nonnegative_ok,
        "ku_order_ok": order_ok,
        "response_equation_ok": equation_ok,
        "panel_count_ok": panel_count_ok,
        "equivalent_panel_count": int(len(response)),
        "max_u_rms_central_min_equation_error_mm": error_min,
        "max_u_rms_central_max_equation_error_mm": error_max,
        "max_u_rms_upper_equation_error_mm": error_upper,
        "all_panel_response_checks_ok": all_ok,
    }


def save_panel_response_map(
    response: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save panel response map.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    response.to_csv(path, index=False, encoding="utf-8")


def write_panel_response_report(
    summary: pd.DataFrame,
    validation: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Write a compact Markdown report for panel response.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ranked = summary.sort_values(
        by=[
            "u_rms_upper_mm_max",
            "u_peak_to_peak_upper_mm_max",
        ],
        ascending=False,
    ).reset_index(drop=True)

    worst = ranked.iloc[0]

    failed = validation[
        validation["all_panel_response_checks_ok"].astype(str).str.lower()
        != "true"
    ]

    lines: list[str] = []

    lines.append("# ROT-54/2.6 Panel Normal Thermomechanical Response Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This step converts equivalent panel-level thermal nonuniformity into "
        "normal thermomechanical response using the reduced relation "
        "`u_n = k_u · ΔT_panel`."
    )
    lines.append("")
    lines.append("## Worst response snapshot")
    lines.append("")
    lines.append(f"- Source: `{worst['source_panel_thermal_name']}`")
    lines.append(f"- Case: `{worst['case_key']}`")
    lines.append(f"- Snapshot: `{worst['snapshot_code']}`")
    lines.append(f"- Wind: `{worst['wind_code']}`")
    lines.append(
        f"- Maximum upper-bound panel RMS response: "
        f"`{worst['u_rms_upper_mm_max']:.6f} mm`"
    )
    lines.append(
        f"- 95th percentile upper-bound panel RMS response: "
        f"`{worst['u_rms_upper_mm_p95']:.6f} mm`"
    )
    lines.append(
        f"- Maximum central-max panel RMS response: "
        f"`{worst['u_rms_central_max_mm_max']:.6f} mm`"
    )
    lines.append(
        f"- Maximum peak-to-peak upper response: "
        f"`{worst['u_peak_to_peak_upper_mm_max']:.6f} mm`"
    )
    lines.append("")
    lines.append("## Worst panel")
    lines.append("")
    lines.append(f"- Panel ID: `{worst['max_u_rms_upper_panel_id']}`")
    lines.append(f"- x: `{worst['max_u_rms_upper_panel_x_m']:.3f} m`")
    lines.append(f"- y: `{worst['max_u_rms_upper_panel_y_m']:.3f} m`")
    lines.append(
        f"- Panel ΔT RMS: `{worst['max_u_rms_upper_panel_delta_rms_C']:.6f} °C`"
    )
    lines.append(
        f"- Upper RMS response: `{worst['max_u_rms_upper_panel_u_rms_mm']:.6f} mm`"
    )
    lines.append("")
    lines.append("## Validation")
    lines.append("")

    if failed.empty:
        lines.append("All panel response validation checks passed.")
    else:
        lines.append("Some panel response validation checks failed.")
        for _, row in failed.iterrows():
            lines.append(f"- `{row['source_panel_thermal_name']}` failed validation.")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The response values are magnitude estimates. They are suitable for "
        "RMS-budget calculations but are not yet a signed deformation field."
    )
    lines.append("")
    lines.append(
        "The next step will aggregate panel response over the aperture to obtain "
        "`σ_T`, then combine it with the baseline surface RMS `σ_0 = 0.070 mm`."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
