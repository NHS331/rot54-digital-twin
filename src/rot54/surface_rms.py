from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPEED_OF_LIGHT_MM_PER_S = 299_792_458_000.0
MM_PER_M = 1000.0


@dataclass(frozen=True)
class SurfaceRMSParameters:
    """
    Parameters for aperture-level RMS aggregation and Ruze interpretation.

    sigma_T:
        Additional thermally induced RMS surface error, mm.

    sigma_total:
        Combined RMS surface error:
            sigma_total = sqrt(sigma_0^2 + sigma_T^2)

    f10:
        Frequency where Ruze efficiency falls to 0.9.

    Ruze:
        eta = exp[-(4*pi*sigma/lambda)^2]
    """

    base_surface_rms_mm: float
    frequencies_GHz: tuple[float, ...]
    active_panel_u_rms_threshold_mm: float
    primary_weighting: str
    top_n_worst_cases: int


def build_surface_rms_parameters(
    main_config: dict[str, Any],
    rms_config: dict[str, Any],
) -> SurfaceRMSParameters:
    """
    Build surface RMS parameters from configs.
    """
    raw = rms_config["surface_rms"]

    params = SurfaceRMSParameters(
        base_surface_rms_mm=float(
            main_config["main_reflector"]["base_surface_rms_mm"]
        ),
        frequencies_GHz=tuple(float(v) for v in raw["frequencies_GHz"]),
        active_panel_u_rms_threshold_mm=float(
            raw["active_panel_u_rms_threshold_mm"]
        ),
        primary_weighting=str(raw["primary_weighting"]),
        top_n_worst_cases=int(raw["top_n_worst_cases"]),
    )

    validate_surface_rms_parameters(params)

    return params


def validate_surface_rms_parameters(params: SurfaceRMSParameters) -> None:
    """
    Validate RMS parameters.
    """
    if params.base_surface_rms_mm < 0.0:
        raise ValueError("base_surface_rms_mm must be non-negative.")

    if not params.frequencies_GHz:
        raise ValueError("At least one frequency is required.")

    if any(f <= 0.0 for f in params.frequencies_GHz):
        raise ValueError("All frequencies must be positive.")

    if params.active_panel_u_rms_threshold_mm < 0.0:
        raise ValueError("active_panel_u_rms_threshold_mm must be non-negative.")

    if params.primary_weighting not in ["weighted_by_grid_points", "unweighted"]:
        raise ValueError(
            "primary_weighting must be either 'weighted_by_grid_points' or 'unweighted'."
        )

    if params.top_n_worst_cases <= 0:
        raise ValueError("top_n_worst_cases must be positive.")


def require_panel_response_columns(panel_response: pd.DataFrame) -> None:
    """
    Ensure panel response map contains required columns.
    """
    required = [
        "equivalent_panel_id",
        "panel_grid_point_count",
        "u_rms_central_min_mm",
        "u_rms_central_max_mm",
        "u_rms_central_mid_mm",
        "u_rms_upper_mm",
        "u_peak_to_peak_upper_mm",
        "panel_delta_rms_C",
        "panel_q_abs_mean_W_m2",
        "panel_q_abs_max_W_m2",
    ]

    missing = [column for column in required if column not in panel_response.columns]

    if missing:
        raise ValueError(f"Panel response map is missing required columns: {missing}")


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted RMS:
        sqrt(sum(w_i * x_i^2) / sum(w_i))
    """
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    if x.size == 0:
        raise ValueError("Cannot compute RMS of empty array.")

    if w.size != x.size:
        raise ValueError("weights and values must have the same size.")

    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative.")

    weight_sum = float(np.sum(w))

    if weight_sum <= 0.0:
        raise ValueError("Sum of weights must be positive.")

    return float(np.sqrt(np.sum(w * x * x) / weight_sum))


def unweighted_rms(values: np.ndarray) -> float:
    """
    Unweighted RMS:
        sqrt(mean(x_i^2))
    """
    x = np.asarray(values, dtype=float)

    if x.size == 0:
        raise ValueError("Cannot compute RMS of empty array.")

    return float(np.sqrt(np.mean(x * x)))


def combine_rms_quadrature(
    base_rms_mm: float,
    additional_rms_mm: float,
) -> float:
    """
    Quadratic RMS combination.
    """
    return float(np.sqrt(base_rms_mm * base_rms_mm + additional_rms_mm * additional_rms_mm))


def ruze_efficiency(
    sigma_mm: float,
    frequency_GHz: float,
) -> float:
    """
    Ruze efficiency:
        eta = exp[-(4*pi*sigma/lambda)^2]

    sigma:
        surface RMS error in mm

    frequency:
        GHz
    """
    if sigma_mm < 0.0:
        raise ValueError("sigma_mm must be non-negative.")

    if frequency_GHz <= 0.0:
        raise ValueError("frequency_GHz must be positive.")

    wavelength_mm = 299.792458 / frequency_GHz

    exponent = -((4.0 * np.pi * sigma_mm / wavelength_mm) ** 2)

    return float(np.exp(exponent))


def f10_frequency_GHz(
    sigma_mm: float,
) -> float:
    """
    Frequency where Ruze efficiency eta = 0.9.

    eta = exp[-(4*pi*sigma/lambda)^2] = 0.9

    lambda = 4*pi*sigma / sqrt(-ln(0.9))

    f_GHz = 299.792458 / lambda_mm
    """
    if sigma_mm <= 0.0:
        return float("inf")

    constant = 299.792458 * np.sqrt(-np.log(0.9)) / (4.0 * np.pi)

    return float(constant / sigma_mm)


def compute_surface_rms_for_response_map(
    panel_response: pd.DataFrame,
    params: SurfaceRMSParameters,
    source_name: str,
    case_key: str,
    snapshot_code: str,
    wind_code: str,
) -> dict[str, object]:
    """
    Aggregate one panel response map into aperture-level RMS metrics.
    """
    require_panel_response_columns(panel_response)

    response = panel_response.copy()

    weights = response["panel_grid_point_count"].to_numpy(dtype=float)
    unit_weights = np.ones(len(response), dtype=float)

    u_fields = [
        "u_rms_central_min_mm",
        "u_rms_central_max_mm",
        "u_rms_central_mid_mm",
        "u_rms_upper_mm",
    ]

    row: dict[str, object] = {
        "source_panel_response_name": source_name,
        "case_key": case_key,
        "snapshot_code": snapshot_code,
        "wind_code": wind_code,
        "equivalent_panel_count": int(len(response)),
        "base_surface_rms_mm": params.base_surface_rms_mm,
        "active_panel_u_rms_threshold_mm": params.active_panel_u_rms_threshold_mm,
        "primary_weighting": params.primary_weighting,
    }

    if "selected_local_time" in response.columns:
        row["selected_local_time"] = str(response["selected_local_time"].iloc[0])
    else:
        row["selected_local_time"] = ""

    # Reporting: thermal activity count based on upper RMS response.
    upper = response["u_rms_upper_mm"].to_numpy(dtype=float)
    active = upper >= params.active_panel_u_rms_threshold_mm

    row["active_panel_count_upper_threshold"] = int(np.sum(active))
    row["active_panel_fraction_upper_threshold"] = float(np.sum(active) / len(active))

    row["panel_q_abs_mean_W_m2"] = float(response["panel_q_abs_mean_W_m2"].mean())
    row["panel_q_abs_max_W_m2"] = float(response["panel_q_abs_max_W_m2"].max())
    row["panel_delta_rms_mean_C"] = float(response["panel_delta_rms_C"].mean())
    row["panel_delta_rms_max_C"] = float(response["panel_delta_rms_C"].max())
    row["u_rms_upper_panel_mean_mm"] = float(response["u_rms_upper_mm"].mean())
    row["u_rms_upper_panel_p95_mm"] = float(np.percentile(response["u_rms_upper_mm"], 95))
    row["u_rms_upper_panel_max_mm"] = float(response["u_rms_upper_mm"].max())

    for field in u_fields:
        weighted_value = weighted_rms(
            values=response[field].to_numpy(dtype=float),
            weights=weights,
        )

        unweighted_value = unweighted_rms(
            values=response[field].to_numpy(dtype=float),
        )

        if params.primary_weighting == "weighted_by_grid_points":
            primary_value = weighted_value
        else:
            primary_value = unweighted_value

        suffix = field.replace("u_rms_", "")

        row[f"sigma_T_{suffix}_weighted_mm"] = weighted_value
        row[f"sigma_T_{suffix}_unweighted_mm"] = unweighted_value
        row[f"sigma_T_{suffix}_primary_mm"] = primary_value

        row[f"sigma_total_{suffix}_primary_mm"] = combine_rms_quadrature(
            base_rms_mm=params.base_surface_rms_mm,
            additional_rms_mm=primary_value,
        )

        row[f"f10_sigma_T_{suffix}_GHz"] = f10_frequency_GHz(primary_value)
        row[f"f10_sigma_total_{suffix}_GHz"] = f10_frequency_GHz(
            row[f"sigma_total_{suffix}_primary_mm"]
        )

        for frequency in params.frequencies_GHz:
            safe_f = str(frequency).replace(".", "p")

            row[f"eta_R_sigma_T_{suffix}_{safe_f}_GHz"] = ruze_efficiency(
                sigma_mm=primary_value,
                frequency_GHz=frequency,
            )

            row[f"eta_R_sigma_total_{suffix}_{safe_f}_GHz"] = ruze_efficiency(
                sigma_mm=row[f"sigma_total_{suffix}_primary_mm"],
                frequency_GHz=frequency,
            )

    return row


def validate_surface_rms_row(
    row: dict[str, object],
    params: SurfaceRMSParameters,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """
    Validate one surface RMS aggregation row.
    """
    case_key = str(row["case_key"])
    snapshot_code = str(row["snapshot_code"])
    wind_code = str(row["wind_code"])

    sigma_T_min = float(row["sigma_T_central_min_mm_primary_mm"])
    sigma_T_max = float(row["sigma_T_central_max_mm_primary_mm"])
    sigma_T_mid = float(row["sigma_T_central_mid_mm_primary_mm"])
    sigma_T_upper = float(row["sigma_T_upper_mm_primary_mm"])

    sigma_total_min = float(row["sigma_total_central_min_mm_primary_mm"])
    sigma_total_max = float(row["sigma_total_central_max_mm_primary_mm"])
    sigma_total_mid = float(row["sigma_total_central_mid_mm_primary_mm"])
    sigma_total_upper = float(row["sigma_total_upper_mm_primary_mm"])

    finite_values = np.array(
        [
            sigma_T_min,
            sigma_T_max,
            sigma_T_mid,
            sigma_T_upper,
            sigma_total_min,
            sigma_total_max,
            sigma_total_mid,
            sigma_total_upper,
        ],
        dtype=float,
    )

    finite_ok = bool(np.isfinite(finite_values).all())

    nonnegative_ok = bool(np.all(finite_values >= -tolerance))

    order_ok = bool(
        sigma_T_min <= sigma_T_mid + tolerance
        and sigma_T_mid <= sigma_T_max + tolerance
        and sigma_T_max <= sigma_T_upper + tolerance
        and sigma_total_min <= sigma_total_mid + tolerance
        and sigma_total_mid <= sigma_total_max + tolerance
        and sigma_total_max <= sigma_total_upper + tolerance
    )

    total_ge_base_ok = bool(
        sigma_total_min + tolerance >= params.base_surface_rms_mm
        and sigma_total_max + tolerance >= params.base_surface_rms_mm
        and sigma_total_upper + tolerance >= params.base_surface_rms_mm
    )

    quadrature_expected_upper = combine_rms_quadrature(
        base_rms_mm=params.base_surface_rms_mm,
        additional_rms_mm=sigma_T_upper,
    )

    quadrature_error_upper = abs(sigma_total_upper - quadrature_expected_upper)

    quadrature_ok = bool(quadrature_error_upper <= tolerance)

    f10_total_min = float(row["f10_sigma_total_central_min_mm_GHz"])
    f10_total_max = float(row["f10_sigma_total_central_max_mm_GHz"])
    f10_total_upper = float(row["f10_sigma_total_upper_mm_GHz"])

    f10_order_ok = bool(
        f10_total_min + tolerance >= f10_total_max
        and f10_total_max + tolerance >= f10_total_upper
    )

    all_ok = bool(
        finite_ok
        and nonnegative_ok
        and order_ok
        and total_ge_base_ok
        and quadrature_ok
        and f10_order_ok
    )

    return {
        "case_key": case_key,
        "snapshot_code": snapshot_code,
        "wind_code": wind_code,
        "finite_outputs_ok": finite_ok,
        "nonnegative_outputs_ok": nonnegative_ok,
        "sigma_order_ok": order_ok,
        "sigma_total_ge_base_ok": total_ge_base_ok,
        "quadrature_combination_ok": quadrature_ok,
        "f10_order_ok": f10_order_ok,
        "quadrature_error_upper_mm": quadrature_error_upper,
        "all_surface_rms_checks_ok": all_ok,
    }


def save_surface_rms_summary(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save surface RMS summary table.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False, encoding="utf-8")


def write_surface_rms_report(
    summary: pd.DataFrame,
    validation: pd.DataFrame,
    output_path: str | Path,
    params: SurfaceRMSParameters,
) -> None:
    """
    Write compact Markdown report.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ranked = summary.sort_values(
        by=[
            "sigma_total_upper_mm_primary_mm",
            "sigma_T_upper_mm_primary_mm",
        ],
        ascending=False,
    ).reset_index(drop=True)

    worst = ranked.iloc[0]

    failed = validation[
        validation["all_surface_rms_checks_ok"].astype(str).str.lower()
        != "true"
    ]

    lines: list[str] = []

    lines.append("# ROT-54/2.6 Surface RMS and Ruze Efficiency Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This step aggregates panel-level normal thermomechanical response into "
        "aperture-level additional RMS surface error and combines it with the "
        "baseline surface RMS."
    )
    lines.append("")
    lines.append("## Model")
    lines.append("")
    lines.append("- Additional thermally induced RMS: `σ_T = RMS(u_n over panels)`")
    lines.append("- Total RMS: `σ_Σ = sqrt(σ_0^2 + σ_T^2)`")
    lines.append(f"- Baseline RMS: `σ_0 = {params.base_surface_rms_mm:.6f} mm`")
    lines.append("- Ruze efficiency: `η_R = exp[-(4πσ/λ)^2]`")
    lines.append("")
    lines.append("## Worst total-RMS case")
    lines.append("")
    lines.append(f"- Source: `{worst['source_panel_response_name']}`")
    lines.append(f"- Case: `{worst['case_key']}`")
    lines.append(f"- Snapshot: `{worst['snapshot_code']}`")
    lines.append(f"- Wind: `{worst['wind_code']}`")
    lines.append(f"- Local time: `{worst['selected_local_time']}`")
    lines.append(
        f"- `σ_T upper`: `{worst['sigma_T_upper_mm_primary_mm']:.6f} mm`"
    )
    lines.append(
        f"- `σ_Σ upper`: `{worst['sigma_total_upper_mm_primary_mm']:.6f} mm`"
    )
    lines.append(
        f"- `f10 upper total`: `{worst['f10_sigma_total_upper_mm_GHz']:.3f} GHz`"
    )

    for frequency in params.frequencies_GHz:
        safe_f = str(frequency).replace(".", "p")
        column = f"eta_R_sigma_total_upper_mm_{safe_f}_GHz"

        if column in worst.index:
            lines.append(
                f"- `η_R total upper at {frequency:g} GHz`: "
                f"`{worst[column]:.6f}`"
            )

    lines.append("")
    lines.append("## Validation")
    lines.append("")

    if failed.empty:
        lines.append("All surface RMS validation checks passed.")
    else:
        lines.append("Some surface RMS validation checks failed.")
        for _, row in failed.iterrows():
            lines.append(
                f"- `{row['case_key']} / {row['snapshot_code']} / {row['wind_code']}` failed."
            )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The primary aggregation uses panel grid-point count as an approximate "
        "area weight. Unweighted RMS values are also saved for comparison."
    )
    lines.append("")
    lines.append(
        "The next step can generate final article-style tables for selected "
        "seasonal scenarios and wind speeds."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
