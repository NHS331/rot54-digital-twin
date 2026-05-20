from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SteadyThermalParameters:
    """
    Parameters for first steady-state thermal surface model.

    Energy balance:
        q_abs = h(v) * (T_s - T_air)
              + epsilon * sigma * (T_s^4 - T_sky^4)

    Units:
        q_abs: W/m^2
        h: W/(m^2 K)
        T: K
        sigma: W/(m^2 K^4)
    """

    emissivity: float
    stefan_boltzmann_W_m2_K4: float
    convection_h0_W_m2_K: float
    convection_h1_W_m2_K_sqrt_m: float
    lower_temperature_K: float
    upper_temperature_K: float
    iterations: int
    residual_tolerance_W_m2: float


@dataclass(frozen=True)
class WeatherCase:
    """
    Weather parameters for one seasonal case.
    """

    air_temperature_C: float
    sky_temperature_C: float
    wind_speed_m_s: float


def celsius_to_kelvin(value_C: float | np.ndarray) -> float | np.ndarray:
    """
    Convert Celsius to Kelvin.
    """
    return np.asarray(value_C) + 273.15


def kelvin_to_celsius(value_K: float | np.ndarray) -> float | np.ndarray:
    """
    Convert Kelvin to Celsius.
    """
    return np.asarray(value_K) - 273.15


def convection_coefficient_W_m2_K(
    wind_speed_m_s: float,
    h0: float,
    h1: float,
) -> float:
    """
    First-stage empirical convection model:

        h(v) = h0 + h1 * sqrt(v)

    This is intentionally simple and transparent.
    It will later be replaced or calibrated if better site-specific
    or structure-specific heat-transfer data are available.
    """
    if wind_speed_m_s < 0.0:
        raise ValueError("wind_speed_m_s must be non-negative.")

    return float(h0 + h1 * np.sqrt(wind_speed_m_s))


def thermal_balance_residual_W_m2(
    surface_temperature_K: np.ndarray,
    q_abs_W_m2: np.ndarray,
    air_temperature_K: float,
    sky_temperature_K: float,
    h_W_m2_K: float,
    emissivity: float,
    sigma_W_m2_K4: float,
) -> np.ndarray:
    """
    Residual of the steady-state surface balance.

    residual = outgoing heat loss - absorbed solar flux

    The solution satisfies:
        residual = 0
    """
    convective_loss = h_W_m2_K * (surface_temperature_K - air_temperature_K)

    radiative_loss = (
        emissivity
        * sigma_W_m2_K4
        * (surface_temperature_K**4 - sky_temperature_K**4)
    )

    return convective_loss + radiative_loss - q_abs_W_m2


def solve_surface_temperature_K(
    q_abs_W_m2: np.ndarray,
    weather: WeatherCase,
    params: SteadyThermalParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the nonlinear steady-state surface temperature by bisection.

    Returns:
        surface_temperature_K
        residual_W_m2
    """
    q = np.asarray(q_abs_W_m2, dtype=float)

    if np.nanmin(q) < -1e-12:
        raise ValueError("q_abs_W_m2 contains negative values.")

    air_K = float(celsius_to_kelvin(weather.air_temperature_C))
    sky_K = float(celsius_to_kelvin(weather.sky_temperature_C))

    if sky_K <= 0.0:
        raise ValueError("sky temperature must be above absolute zero.")

    h = convection_coefficient_W_m2_K(
        wind_speed_m_s=weather.wind_speed_m_s,
        h0=params.convection_h0_W_m2_K,
        h1=params.convection_h1_W_m2_K_sqrt_m,
    )

    low = np.full_like(q, params.lower_temperature_K, dtype=float)
    high = np.full_like(q, params.upper_temperature_K, dtype=float)

    f_low = thermal_balance_residual_W_m2(
        surface_temperature_K=low,
        q_abs_W_m2=q,
        air_temperature_K=air_K,
        sky_temperature_K=sky_K,
        h_W_m2_K=h,
        emissivity=params.emissivity,
        sigma_W_m2_K4=params.stefan_boltzmann_W_m2_K4,
    )

    f_high = thermal_balance_residual_W_m2(
        surface_temperature_K=high,
        q_abs_W_m2=q,
        air_temperature_K=air_K,
        sky_temperature_K=sky_K,
        h_W_m2_K=h,
        emissivity=params.emissivity,
        sigma_W_m2_K4=params.stefan_boltzmann_W_m2_K4,
    )

    if np.any(f_low > 0.0):
        raise ValueError(
            "Lower temperature bound is too high for at least one grid point. "
            "Decrease lower_temperature_K."
        )

    if np.any(f_high < 0.0):
        raise ValueError(
            "Upper temperature bound is too low for at least one grid point. "
            "Increase upper_temperature_K."
        )

    for _ in range(params.iterations):
        mid = 0.5 * (low + high)

        f_mid = thermal_balance_residual_W_m2(
            surface_temperature_K=mid,
            q_abs_W_m2=q,
            air_temperature_K=air_K,
            sky_temperature_K=sky_K,
            h_W_m2_K=h,
            emissivity=params.emissivity,
            sigma_W_m2_K4=params.stefan_boltzmann_W_m2_K4,
        )

        move_low = f_mid < 0.0

        low[move_low] = mid[move_low]
        high[~move_low] = mid[~move_low]

    surface_temperature_K = 0.5 * (low + high)

    residual = thermal_balance_residual_W_m2(
        surface_temperature_K=surface_temperature_K,
        q_abs_W_m2=q,
        air_temperature_K=air_K,
        sky_temperature_K=sky_K,
        h_W_m2_K=h,
        emissivity=params.emissivity,
        sigma_W_m2_K4=params.stefan_boltzmann_W_m2_K4,
    )

    return surface_temperature_K, residual


def build_steady_thermal_parameters(
    main_config: dict[str, Any],
    thermal_config: dict[str, Any],
) -> SteadyThermalParameters:
    """
    Build steady thermal parameters from configs.
    """
    irradiation = main_config["irradiation"]
    thermal_model = main_config["thermal_model"]
    solver = thermal_config["thermal_steady"]["solver"]

    params = SteadyThermalParameters(
        emissivity=float(irradiation["emissivity"]),
        stefan_boltzmann_W_m2_K4=float(
            thermal_model["stefan_boltzmann_W_m2_K4"]
        ),
        convection_h0_W_m2_K=float(
            thermal_model["convection_h0_W_m2_K"]
        ),
        convection_h1_W_m2_K_sqrt_m=float(
            thermal_model["convection_h1_W_m2_K_sqrt_m"]
        ),
        lower_temperature_K=float(solver["lower_temperature_K"]),
        upper_temperature_K=float(solver["upper_temperature_K"]),
        iterations=int(solver["iterations"]),
        residual_tolerance_W_m2=float(
            solver["residual_tolerance_W_m2"]
        ),
    )

    validate_steady_thermal_parameters(params)

    return params


def validate_steady_thermal_parameters(params: SteadyThermalParameters) -> None:
    """
    Validate physical and numerical ranges.
    """
    if not (0.0 <= params.emissivity <= 1.0):
        raise ValueError("emissivity must be within [0, 1].")

    if params.stefan_boltzmann_W_m2_K4 <= 0.0:
        raise ValueError("Stefan-Boltzmann constant must be positive.")

    if params.convection_h0_W_m2_K < 0.0:
        raise ValueError("convection_h0_W_m2_K must be non-negative.")

    if params.convection_h1_W_m2_K_sqrt_m < 0.0:
        raise ValueError("convection_h1_W_m2_K_sqrt_m must be non-negative.")

    if params.lower_temperature_K <= 0.0:
        raise ValueError("lower_temperature_K must be above absolute zero.")

    if params.upper_temperature_K <= params.lower_temperature_K:
        raise ValueError("upper_temperature_K must be greater than lower bound.")

    if params.iterations < 20:
        raise ValueError("At least 20 bisection iterations are recommended.")

    if params.residual_tolerance_W_m2 <= 0.0:
        raise ValueError("residual_tolerance_W_m2 must be positive.")


def get_weather_case_for_flux(
    case_key: str,
    wind_speed_m_s: float,
    thermal_config: dict[str, Any],
) -> WeatherCase:
    """
    Read weather scenario for a case and wind speed.
    """
    weather_map = thermal_config["thermal_steady"]["seasonal_weather"]

    if case_key not in weather_map:
        raise KeyError(
            f"No weather scenario found for case_key={case_key}. "
            f"Available keys: {list(weather_map.keys())}"
        )

    raw = weather_map[case_key]

    return WeatherCase(
        air_temperature_C=float(raw["air_temperature_C"]),
        sky_temperature_C=float(raw["sky_temperature_C"]),
        wind_speed_m_s=float(wind_speed_m_s),
    )


def compute_steady_temperature_map(
    flux: pd.DataFrame,
    weather: WeatherCase,
    params: SteadyThermalParameters,
) -> pd.DataFrame:
    """
    Compute steady-state surface temperature map from absorbed flux.
    """
    required_columns = [
        "x_m",
        "y_m",
        "z_m",
        "q_abs_W_m2",
        "effective_solar_factor_v2",
        "visibility_chi_v2",
    ]

    missing = [
        column for column in required_columns
        if column not in flux.columns
    ]

    if missing:
        raise ValueError(f"Absorbed flux table is missing columns: {missing}")

    result = flux.copy()

    q_abs = result["q_abs_W_m2"].to_numpy(dtype=float)

    surface_temperature_K, residual = solve_surface_temperature_K(
        q_abs_W_m2=q_abs,
        weather=weather,
        params=params,
    )

    surface_temperature_C = kelvin_to_celsius(surface_temperature_K)

    h = convection_coefficient_W_m2_K(
        wind_speed_m_s=weather.wind_speed_m_s,
        h0=params.convection_h0_W_m2_K,
        h1=params.convection_h1_W_m2_K_sqrt_m,
    )

    aperture_mean_C = float(np.nanmean(surface_temperature_C))

    result["air_temperature_C"] = weather.air_temperature_C
    result["sky_temperature_C"] = weather.sky_temperature_C
    result["wind_speed_m_s"] = weather.wind_speed_m_s
    result["convection_h_W_m2_K"] = h
    result["emissivity"] = params.emissivity
    result["surface_temperature_K"] = surface_temperature_K
    result["surface_temperature_C"] = surface_temperature_C
    result["deltaT_air_C"] = surface_temperature_C - weather.air_temperature_C
    result["surface_temperature_aperture_mean_C"] = aperture_mean_C
    result["deltaT_aperture_mean_C"] = surface_temperature_C - aperture_mean_C
    result["thermal_balance_residual_W_m2"] = residual

    return result


def summarize_steady_temperature(
    case_key: str,
    case_label: str,
    time_code: str,
    selected_local_time: str,
    thermal: pd.DataFrame,
) -> dict[str, object]:
    """
    Summarize one steady thermal map.
    """
    visible = thermal["visibility_chi_v2"].to_numpy(dtype=float) > 0.5

    surface_C = thermal["surface_temperature_C"].to_numpy(dtype=float)
    delta_air = thermal["deltaT_air_C"].to_numpy(dtype=float)
    delta_mean = thermal["deltaT_aperture_mean_C"].to_numpy(dtype=float)
    q_abs = thermal["q_abs_W_m2"].to_numpy(dtype=float)
    residual = thermal["thermal_balance_residual_W_m2"].to_numpy(dtype=float)

    if visible.sum() > 0:
        surface_visible = surface_C[visible]
        delta_air_visible = delta_air[visible]
        q_visible = q_abs[visible]
    else:
        surface_visible = np.array([], dtype=float)
        delta_air_visible = np.array([], dtype=float)
        q_visible = np.array([], dtype=float)

    return {
        "case_key": case_key,
        "case_label": case_label,
        "time_code": time_code,
        "selected_local_time": selected_local_time,
        "wind_speed_m_s": float(thermal["wind_speed_m_s"].iloc[0]),
        "air_temperature_C": float(thermal["air_temperature_C"].iloc[0]),
        "sky_temperature_C": float(thermal["sky_temperature_C"].iloc[0]),
        "convection_h_W_m2_K": float(thermal["convection_h_W_m2_K"].iloc[0]),
        "visible_points": int(visible.sum()),
        "q_abs_max_W_m2": float(np.nanmax(q_abs)),
        "q_abs_mean_all_W_m2": float(np.nanmean(q_abs)),
        "q_abs_mean_visible_W_m2": float(np.nanmean(q_visible)) if q_visible.size else np.nan,
        "surface_temperature_min_C": float(np.nanmin(surface_C)),
        "surface_temperature_mean_C": float(np.nanmean(surface_C)),
        "surface_temperature_max_C": float(np.nanmax(surface_C)),
        "surface_temperature_mean_visible_C": float(np.nanmean(surface_visible)) if surface_visible.size else np.nan,
        "deltaT_air_min_C": float(np.nanmin(delta_air)),
        "deltaT_air_mean_C": float(np.nanmean(delta_air)),
        "deltaT_air_max_C": float(np.nanmax(delta_air)),
        "deltaT_air_mean_visible_C": float(np.nanmean(delta_air_visible)) if delta_air_visible.size else np.nan,
        "deltaT_aperture_mean_min_C": float(np.nanmin(delta_mean)),
        "deltaT_aperture_mean_max_C": float(np.nanmax(delta_mean)),
        "max_abs_thermal_balance_residual_W_m2": float(np.nanmax(np.abs(residual))),
    }


def validate_steady_temperature(
    case_key: str,
    time_code: str,
    wind_speed_m_s: float,
    thermal: pd.DataFrame,
    params: SteadyThermalParameters,
) -> dict[str, object]:
    """
    Validate thermal balance and output ranges.
    """
    residual = thermal["thermal_balance_residual_W_m2"].to_numpy(dtype=float)
    surface_K = thermal["surface_temperature_K"].to_numpy(dtype=float)

    max_abs_residual = float(np.nanmax(np.abs(residual)))

    residual_ok = bool(max_abs_residual <= params.residual_tolerance_W_m2)

    temperature_range_ok = bool(
        np.nanmin(surface_K) >= params.lower_temperature_K - 1e-9
        and np.nanmax(surface_K) <= params.upper_temperature_K + 1e-9
    )

    finite_ok = bool(np.isfinite(surface_K).all())

    all_ok = bool(residual_ok and temperature_range_ok and finite_ok)

    return {
        "case_key": case_key,
        "time_code": time_code,
        "wind_speed_m_s": wind_speed_m_s,
        "residual_ok": residual_ok,
        "temperature_range_ok": temperature_range_ok,
        "finite_temperature_ok": finite_ok,
        "max_abs_thermal_balance_residual_W_m2": max_abs_residual,
        "all_steady_temperature_checks_ok": all_ok,
    }


def save_steady_temperature_map(
    thermal: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save steady thermal map.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    thermal.to_csv(path, index=False, encoding="utf-8")
