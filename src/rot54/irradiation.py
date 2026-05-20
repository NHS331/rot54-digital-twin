from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IrradiationParameters:
    """
    Parameters for absorbed direct solar irradiation.

    The first irradiation layer uses a reduced engineering model:

        I_eff = I_DNI * K_atm * K_cloud * K_humidity

        q_abs = alpha_s * I_eff * F_solar

    where:

        F_solar = max(cos(theta_i), 0) * chi

    Units:
        I_eff: W/m^2
        q_abs: W/m^2
    """

    direct_normal_irradiance_W_m2: float
    absorptivity: float
    atmospheric_transmittance: float
    cloud_factor: float
    humidity_factor: float


def build_irradiation_parameters(config: dict[str, Any]) -> IrradiationParameters:
    """
    Build irradiation parameters from the main YAML configuration.
    """
    raw = config["irradiation"]

    params = IrradiationParameters(
        direct_normal_irradiance_W_m2=float(raw["direct_normal_irradiance_W_m2"]),
        absorptivity=float(raw["absorptivity"]),
        atmospheric_transmittance=float(raw["atmospheric_transmittance"]),
        cloud_factor=float(raw["cloud_factor"]),
        humidity_factor=float(raw["humidity_factor"]),
    )

    validate_irradiation_parameters(params)

    return params


def validate_irradiation_parameters(params: IrradiationParameters) -> None:
    """
    Validate physical ranges of irradiation parameters.
    """
    if params.direct_normal_irradiance_W_m2 < 0.0:
        raise ValueError("direct_normal_irradiance_W_m2 must be non-negative.")

    if not (0.0 <= params.absorptivity <= 1.0):
        raise ValueError("absorptivity must be within [0, 1].")

    if not (0.0 <= params.atmospheric_transmittance <= 1.0):
        raise ValueError("atmospheric_transmittance must be within [0, 1].")

    if not (0.0 <= params.cloud_factor <= 1.0):
        raise ValueError("cloud_factor must be within [0, 1].")

    if not (0.0 <= params.humidity_factor <= 1.0):
        raise ValueError("humidity_factor must be within [0, 1].")


def effective_direct_normal_irradiance_W_m2(
    params: IrradiationParameters,
) -> float:
    """
    Effective direct normal irradiance after first-stage atmospheric/weather factors.
    """
    return float(
        params.direct_normal_irradiance_W_m2
        * params.atmospheric_transmittance
        * params.cloud_factor
        * params.humidity_factor
    )


def compute_absorbed_flux_map(
    shadow_v2: pd.DataFrame,
    params: IrradiationParameters,
) -> pd.DataFrame:
    """
    Compute absorbed solar flux map from the V2 shadowing output.

    Required input column:
        effective_solar_factor_v2

    Output:
        effective_direct_normal_irradiance_W_m2
        absorptivity
        q_abs_W_m2
        q_abs_kW_m2
    """
    required_columns = [
        "x_m",
        "y_m",
        "z_m",
        "cos_incidence",
        "visibility_chi_v2",
        "effective_solar_factor_v2",
    ]

    missing = [
        column for column in required_columns
        if column not in shadow_v2.columns
    ]

    if missing:
        raise ValueError(f"Shadow V2 table is missing required columns: {missing}")

    result = shadow_v2.copy()

    factor = result["effective_solar_factor_v2"].to_numpy(dtype=float)

    if np.nanmin(factor) < -1e-12:
        raise ValueError("effective_solar_factor_v2 contains negative values.")

    if np.nanmax(factor) > 1.0 + 1e-12:
        raise ValueError("effective_solar_factor_v2 exceeds 1.")

    i_eff = effective_direct_normal_irradiance_W_m2(params)

    q_abs = params.absorptivity * i_eff * factor

    result["direct_normal_irradiance_input_W_m2"] = params.direct_normal_irradiance_W_m2
    result["atmospheric_transmittance"] = params.atmospheric_transmittance
    result["cloud_factor"] = params.cloud_factor
    result["humidity_factor"] = params.humidity_factor
    result["effective_direct_normal_irradiance_W_m2"] = i_eff
    result["absorptivity"] = params.absorptivity
    result["q_abs_W_m2"] = q_abs
    result["q_abs_kW_m2"] = q_abs / 1000.0

    return result


def summarize_absorbed_flux(
    case_key: str,
    case_label: str,
    time_code: str,
    selected_local_time: str,
    flux: pd.DataFrame,
) -> dict[str, object]:
    """
    Summarize absorbed solar flux map.
    """
    q = flux["q_abs_W_m2"].to_numpy(dtype=float)
    visible = flux["visibility_chi_v2"].to_numpy(dtype=float) > 0.5

    visible_q = q[visible]

    if visible_q.size == 0:
        q_visible_min = np.nan
        q_visible_mean = np.nan
        q_visible_max = np.nan
        q_visible_std = np.nan
    else:
        q_visible_min = float(np.nanmin(visible_q))
        q_visible_mean = float(np.nanmean(visible_q))
        q_visible_max = float(np.nanmax(visible_q))
        q_visible_std = float(np.nanstd(visible_q))

    if q.size == 0:
        q_all_mean = np.nan
        q_all_max = np.nan
        q_total_relative = np.nan
    else:
        q_all_mean = float(np.nanmean(q))
        q_all_max = float(np.nanmax(q))
        q_total_relative = float(np.nansum(q))

    return {
        "case_key": case_key,
        "case_label": case_label,
        "time_code": time_code,
        "selected_local_time": selected_local_time,
        "direct_normal_irradiance_input_W_m2": float(
            flux["direct_normal_irradiance_input_W_m2"].iloc[0]
        ),
        "effective_direct_normal_irradiance_W_m2": float(
            flux["effective_direct_normal_irradiance_W_m2"].iloc[0]
        ),
        "absorptivity": float(flux["absorptivity"].iloc[0]),
        "visible_points": int(visible.sum()),
        "q_abs_visible_min_W_m2": q_visible_min,
        "q_abs_visible_mean_W_m2": q_visible_mean,
        "q_abs_visible_max_W_m2": q_visible_max,
        "q_abs_visible_std_W_m2": q_visible_std,
        "q_abs_all_points_mean_W_m2": q_all_mean,
        "q_abs_all_points_max_W_m2": q_all_max,
        "q_abs_grid_sum_relative_W_m2_points": q_total_relative,
    }


def validate_absorbed_flux(
    case_key: str,
    time_code: str,
    flux: pd.DataFrame,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    """
    Validate q_abs algebra and physical bounds.

    Checks:
        1. q_abs >= 0.
        2. q_abs <= alpha_s * I_eff.
        3. q_abs = alpha_s * I_eff * effective_solar_factor_v2.
    """
    factor = flux["effective_solar_factor_v2"].to_numpy(dtype=float)
    q_abs = flux["q_abs_W_m2"].to_numpy(dtype=float)

    alpha_s = float(flux["absorptivity"].iloc[0])
    i_eff = float(flux["effective_direct_normal_irradiance_W_m2"].iloc[0])

    expected = alpha_s * i_eff * factor

    max_equation_error = float(np.nanmax(np.abs(q_abs - expected)))

    non_negative_ok = bool(np.nanmin(q_abs) >= -tolerance)
    upper_bound_ok = bool(np.nanmax(q_abs) <= alpha_s * i_eff + tolerance)
    equation_ok = bool(max_equation_error <= tolerance)

    all_ok = bool(non_negative_ok and upper_bound_ok and equation_ok)

    return {
        "case_key": case_key,
        "time_code": time_code,
        "q_abs_non_negative_ok": non_negative_ok,
        "q_abs_upper_bound_ok": upper_bound_ok,
        "q_abs_equation_ok": equation_ok,
        "max_q_abs_equation_error_W_m2": max_equation_error,
        "all_absorbed_flux_checks_ok": all_ok,
    }


def save_absorbed_flux_map(
    flux: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save absorbed solar flux map.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flux.to_csv(path, index=False, encoding="utf-8")
