from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rot54.incidence import IncidenceParameters, compute_incidence_for_grid
from rot54.irradiation import (
    build_irradiation_parameters,
    compute_absorbed_flux_map,
)
from rot54.shadowing_v2 import (
    apply_shadow_v2,
    build_shadow_v2_parameters,
)
from rot54.thermal_steady import (
    WeatherCase,
    celsius_to_kelvin,
    kelvin_to_celsius,
    convection_coefficient_W_m2_K,
    thermal_balance_residual_W_m2,
)


@dataclass(frozen=True)
class TransientMaterialParameters:
    """
    Material and thermal-inertia parameters.

    Areal heat capacity:
        C_A = rho * c_p * h_eff

    Units:
        rho: kg/m^3
        c_p: J/(kg K)
        h_eff: m
        C_A: J/(m^2 K)
    """

    density_kg_m3: float
    specific_heat_J_kg_K: float
    mass_equivalent_thickness_m: float

    @property
    def areal_heat_capacity_J_m2_K(self) -> float:
        return float(
            self.density_kg_m3
            * self.specific_heat_J_kg_K
            * self.mass_equivalent_thickness_m
        )


@dataclass(frozen=True)
class TransientNumericalParameters:
    """
    Numerical bounds and integration settings.
    """

    time_step_minutes: int
    minimum_temperature_K: float
    maximum_temperature_K: float


@dataclass(frozen=True)
class TransientThermalParameters:
    """
    Complete transient thermal model parameters.
    """

    material: TransientMaterialParameters
    numerical: TransientNumericalParameters
    emissivity: float
    stefan_boltzmann_W_m2_K4: float
    convection_h0_W_m2_K: float
    convection_h1_W_m2_K_sqrt_m: float


def parse_bool_series(series: pd.Series) -> pd.Series:
    """
    Robust bool parser for CSV-loaded values.
    """
    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def validate_transient_parameters(params: TransientThermalParameters) -> None:
    """
    Validate physical and numerical ranges.
    """
    material = params.material
    numerical = params.numerical

    if material.density_kg_m3 <= 0.0:
        raise ValueError("density_kg_m3 must be positive.")

    if material.specific_heat_J_kg_K <= 0.0:
        raise ValueError("specific_heat_J_kg_K must be positive.")

    if material.mass_equivalent_thickness_m <= 0.0:
        raise ValueError("mass_equivalent_thickness_m must be positive.")

    if material.areal_heat_capacity_J_m2_K <= 0.0:
        raise ValueError("Areal heat capacity must be positive.")

    if numerical.time_step_minutes <= 0:
        raise ValueError("time_step_minutes must be positive.")

    if numerical.minimum_temperature_K <= 0.0:
        raise ValueError("minimum_temperature_K must be above absolute zero.")

    if numerical.maximum_temperature_K <= numerical.minimum_temperature_K:
        raise ValueError("maximum_temperature_K must be larger than minimum.")

    if not (0.0 <= params.emissivity <= 1.0):
        raise ValueError("emissivity must be within [0, 1].")

    if params.stefan_boltzmann_W_m2_K4 <= 0.0:
        raise ValueError("Stefan-Boltzmann constant must be positive.")


def build_transient_thermal_parameters(
    main_config: dict[str, Any],
    transient_config: dict[str, Any],
) -> TransientThermalParameters:
    """
    Build transient thermal model parameters from YAML configs.
    """
    raw = transient_config["thermal_transient"]
    material_raw = raw["material"]
    solver_raw = raw["solver"]

    material = TransientMaterialParameters(
        density_kg_m3=float(material_raw["density_kg_m3"]),
        specific_heat_J_kg_K=float(material_raw["specific_heat_J_kg_K"]),
        mass_equivalent_thickness_m=float(material_raw["mass_equivalent_thickness_m"]),
    )

    numerical = TransientNumericalParameters(
        time_step_minutes=int(raw["time_step_minutes"]),
        minimum_temperature_K=float(solver_raw["minimum_temperature_K"]),
        maximum_temperature_K=float(solver_raw["maximum_temperature_K"]),
    )

    params = TransientThermalParameters(
        material=material,
        numerical=numerical,
        emissivity=float(main_config["irradiation"]["emissivity"]),
        stefan_boltzmann_W_m2_K4=float(
            main_config["thermal_model"]["stefan_boltzmann_W_m2_K4"]
        ),
        convection_h0_W_m2_K=float(
            main_config["thermal_model"]["convection_h0_W_m2_K"]
        ),
        convection_h1_W_m2_K_sqrt_m=float(
            main_config["thermal_model"]["convection_h1_W_m2_K_sqrt_m"]
        ),
    )

    validate_transient_parameters(params)

    return params


def solve_night_equilibrium_temperature_K(
    weather: WeatherCase,
    params: TransientThermalParameters,
    iterations: int = 80,
) -> float:
    """
    Solve night equilibrium temperature for q_abs = 0.

    Balance:
        0 = h(T - T_air) + epsilon sigma (T^4 - T_sky^4)

    This gives the initial surface temperature at 00:00 local time.
    """
    low = params.numerical.minimum_temperature_K
    high = params.numerical.maximum_temperature_K

    q_zero = np.array([0.0], dtype=float)

    air_K = float(celsius_to_kelvin(weather.air_temperature_C))
    sky_K = float(celsius_to_kelvin(weather.sky_temperature_C))

    h = convection_coefficient_W_m2_K(
        wind_speed_m_s=weather.wind_speed_m_s,
        h0=params.convection_h0_W_m2_K,
        h1=params.convection_h1_W_m2_K_sqrt_m,
    )

    low_array = np.array([low], dtype=float)
    high_array = np.array([high], dtype=float)

    f_low = thermal_balance_residual_W_m2(
        surface_temperature_K=low_array,
        q_abs_W_m2=q_zero,
        air_temperature_K=air_K,
        sky_temperature_K=sky_K,
        h_W_m2_K=h,
        emissivity=params.emissivity,
        sigma_W_m2_K4=params.stefan_boltzmann_W_m2_K4,
    )[0]

    f_high = thermal_balance_residual_W_m2(
        surface_temperature_K=high_array,
        q_abs_W_m2=q_zero,
        air_temperature_K=air_K,
        sky_temperature_K=sky_K,
        h_W_m2_K=h,
        emissivity=params.emissivity,
        sigma_W_m2_K4=params.stefan_boltzmann_W_m2_K4,
    )[0]

    if f_low > 0.0:
        raise ValueError("Night-equilibrium lower bound is too high.")

    if f_high < 0.0:
        raise ValueError("Night-equilibrium upper bound is too low.")

    for _ in range(iterations):
        mid = 0.5 * (low + high)
        mid_array = np.array([mid], dtype=float)

        f_mid = thermal_balance_residual_W_m2(
            surface_temperature_K=mid_array,
            q_abs_W_m2=q_zero,
            air_temperature_K=air_K,
            sky_temperature_K=sky_K,
            h_W_m2_K=h,
            emissivity=params.emissivity,
            sigma_W_m2_K4=params.stefan_boltzmann_W_m2_K4,
        )[0]

        if f_mid < 0.0:
            low = mid
        else:
            high = mid

    return float(0.5 * (low + high))


def transient_temperature_step_K(
    surface_temperature_K: np.ndarray,
    q_abs_W_m2: np.ndarray,
    weather: WeatherCase,
    params: TransientThermalParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Advance one explicit transient time step.

    Returns:
        new_temperature_K
        dT_dt_K_s
        net_flux_W_m2
        heat_loss_W_m2

    Governing equation:
        C_A dT/dt = q_abs - q_conv - q_rad
    """
    T = np.asarray(surface_temperature_K, dtype=float)
    q = np.asarray(q_abs_W_m2, dtype=float)

    air_K = float(celsius_to_kelvin(weather.air_temperature_C))
    sky_K = float(celsius_to_kelvin(weather.sky_temperature_C))

    h = convection_coefficient_W_m2_K(
        wind_speed_m_s=weather.wind_speed_m_s,
        h0=params.convection_h0_W_m2_K,
        h1=params.convection_h1_W_m2_K_sqrt_m,
    )

    convective_loss = h * (T - air_K)

    radiative_loss = (
        params.emissivity
        * params.stefan_boltzmann_W_m2_K4
        * (T**4 - sky_K**4)
    )

    heat_loss = convective_loss + radiative_loss
    net_flux = q - heat_loss

    C_A = params.material.areal_heat_capacity_J_m2_K
    dT_dt = net_flux / C_A

    dt_seconds = float(params.numerical.time_step_minutes * 60.0)

    new_T = T + dt_seconds * dT_dt

    new_T = np.clip(
        new_T,
        params.numerical.minimum_temperature_K,
        params.numerical.maximum_temperature_K,
    )

    return new_T, dT_dt, net_flux, heat_loss


def downsample_solar_case(
    solar_case: pd.DataFrame,
    step_minutes: int,
) -> pd.DataFrame:
    """
    Downsample the one-minute solar table to the transient time step.
    """
    df = solar_case.copy().reset_index(drop=True)

    if "clock_time" not in df.columns:
        raise ValueError("solar_case table must contain clock_time.")

    times = pd.to_datetime(df["local_time"])
    minute_of_day = times.dt.hour * 60 + times.dt.minute

    mask = (minute_of_day % step_minutes) == 0

    sampled = df[mask].copy().reset_index(drop=True)

    if sampled.empty:
        raise ValueError("Downsampled solar table is empty.")

    return sampled


def select_target_snapshot_indices(
    sampled_solar_case: pd.DataFrame,
    full_solar_case: pd.DataFrame,
    morning_offset_minutes: int,
    evening_offset_minutes: int,
) -> dict[str, int]:
    """
    Select nearest sampled indices for morning, axis and evening snapshots.
    """
    front_mask = parse_bool_series(full_solar_case["front_side_illumination"])
    front = full_solar_case[front_mask].copy()

    if front.empty:
        raise ValueError("No front-side illumination rows found.")

    front = front.reset_index(drop=True)

    morning_idx_full = min(max(0, morning_offset_minutes), len(front) - 1)
    evening_idx_full = max(0, len(front) - 1 - max(0, evening_offset_minutes))

    axis_idx_full = front["axis_dot_sun"].idxmax()

    target_times = {
        "morning": str(front.iloc[morning_idx_full]["local_time"]),
        "axis": str(front.loc[axis_idx_full]["local_time"]),
        "evening": str(front.iloc[evening_idx_full]["local_time"]),
    }

    sampled_times = pd.to_datetime(sampled_solar_case["local_time"])

    result: dict[str, int] = {}

    for key, target_time in target_times.items():
        target_dt = pd.to_datetime(target_time)
        diff_seconds = (sampled_times - target_dt).abs().dt.total_seconds()
        nearest_index = int(diff_seconds.idxmin())
        result[key] = nearest_index

    return result


def compute_flux_for_solar_row(
    mirror_grid: pd.DataFrame,
    solar_row: pd.Series,
    incidence_params: IncidenceParameters,
    shadow_params: Any,
    irradiation_params: Any,
) -> pd.DataFrame:
    """
    Compute q_abs map for one solar row using existing incidence, shadow and irradiation modules.
    """
    incidence = compute_incidence_for_grid(
        mirror_grid=mirror_grid,
        solar_row=solar_row,
        params=incidence_params,
    )

    shadow = apply_shadow_v2(
        incidence=incidence,
        params=shadow_params,
    )

    flux = compute_absorbed_flux_map(
        shadow_v2=shadow,
        params=irradiation_params,
    )

    return flux


def make_snapshot_dataframe(
    flux: pd.DataFrame,
    surface_temperature_K: np.ndarray,
    dT_dt_K_s: np.ndarray,
    net_flux_W_m2: np.ndarray,
    heat_loss_W_m2: np.ndarray,
    weather: WeatherCase,
    params: TransientThermalParameters,
    case_key: str,
    case_label: str,
    snapshot_code: str,
    selected_local_time: str,
) -> pd.DataFrame:
    """
    Create full spatial snapshot output table.
    """
    result = flux.copy()

    surface_C = kelvin_to_celsius(surface_temperature_K)
    aperture_mean_C = float(np.nanmean(surface_C))

    h = convection_coefficient_W_m2_K(
        wind_speed_m_s=weather.wind_speed_m_s,
        h0=params.convection_h0_W_m2_K,
        h1=params.convection_h1_W_m2_K_sqrt_m,
    )

    result.insert(0, "case_key", case_key)
    result.insert(1, "case_label", case_label)
    result.insert(2, "snapshot_code", snapshot_code)
    result.insert(3, "selected_local_time", selected_local_time)

    result["wind_speed_m_s"] = weather.wind_speed_m_s
    result["air_temperature_C"] = weather.air_temperature_C
    result["sky_temperature_C"] = weather.sky_temperature_C
    result["convection_h_W_m2_K"] = h
    result["areal_heat_capacity_J_m2_K"] = params.material.areal_heat_capacity_J_m2_K
    result["surface_temperature_K"] = surface_temperature_K
    result["surface_temperature_C"] = surface_C
    result["deltaT_air_C"] = surface_C - weather.air_temperature_C
    result["surface_temperature_aperture_mean_C"] = aperture_mean_C
    result["deltaT_aperture_mean_C"] = surface_C - aperture_mean_C
    result["dT_dt_K_s"] = dT_dt_K_s
    result["net_flux_W_m2"] = net_flux_W_m2
    result["heat_loss_W_m2"] = heat_loss_W_m2

    return result


def summarize_transient_state(
    case_key: str,
    case_label: str,
    local_time: str,
    time_index: int,
    weather: WeatherCase,
    params: TransientThermalParameters,
    flux: pd.DataFrame,
    surface_temperature_K: np.ndarray,
    dT_dt_K_s: np.ndarray,
    net_flux_W_m2: np.ndarray,
) -> dict[str, object]:
    """
    Summarize transient state at one time step.
    """
    surface_C = kelvin_to_celsius(surface_temperature_K)

    visible = flux["visibility_chi_v2"].to_numpy(dtype=float) > 0.5
    q_abs = flux["q_abs_W_m2"].to_numpy(dtype=float)

    if visible.sum() > 0:
        visible_surface = surface_C[visible]
        visible_q = q_abs[visible]
    else:
        visible_surface = np.array([], dtype=float)
        visible_q = np.array([], dtype=float)

    return {
        "case_key": case_key,
        "case_label": case_label,
        "time_index": time_index,
        "local_time": local_time,
        "wind_speed_m_s": weather.wind_speed_m_s,
        "air_temperature_C": weather.air_temperature_C,
        "sky_temperature_C": weather.sky_temperature_C,
        "areal_heat_capacity_J_m2_K": params.material.areal_heat_capacity_J_m2_K,
        "visible_points": int(visible.sum()),
        "q_abs_max_W_m2": float(np.nanmax(q_abs)),
        "q_abs_mean_all_W_m2": float(np.nanmean(q_abs)),
        "q_abs_mean_visible_W_m2": float(np.nanmean(visible_q)) if visible_q.size else np.nan,
        "surface_temperature_min_C": float(np.nanmin(surface_C)),
        "surface_temperature_mean_C": float(np.nanmean(surface_C)),
        "surface_temperature_max_C": float(np.nanmax(surface_C)),
        "surface_temperature_mean_visible_C": float(np.nanmean(visible_surface)) if visible_surface.size else np.nan,
        "deltaT_air_min_C": float(np.nanmin(surface_C - weather.air_temperature_C)),
        "deltaT_air_mean_C": float(np.nanmean(surface_C - weather.air_temperature_C)),
        "deltaT_air_max_C": float(np.nanmax(surface_C - weather.air_temperature_C)),
        "dT_dt_min_K_s": float(np.nanmin(dT_dt_K_s)),
        "dT_dt_mean_K_s": float(np.nanmean(dT_dt_K_s)),
        "dT_dt_max_K_s": float(np.nanmax(dT_dt_K_s)),
        "net_flux_min_W_m2": float(np.nanmin(net_flux_W_m2)),
        "net_flux_mean_W_m2": float(np.nanmean(net_flux_W_m2)),
        "net_flux_max_W_m2": float(np.nanmax(net_flux_W_m2)),
    }


def validate_transient_timeseries(
    case_key: str,
    wind_speed_m_s: float,
    timeseries: pd.DataFrame,
    params: TransientThermalParameters,
) -> dict[str, object]:
    """
    Validate one transient time series.
    """
    required = [
        "surface_temperature_min_C",
        "surface_temperature_max_C",
        "q_abs_max_W_m2",
        "dT_dt_min_K_s",
        "dT_dt_max_K_s",
    ]

    missing = [col for col in required if col not in timeseries.columns]

    if missing:
        raise ValueError(f"Transient timeseries missing columns: {missing}")

    finite_ok = bool(np.isfinite(timeseries[required].to_numpy(dtype=float)).all())

    min_K = float(celsius_to_kelvin(timeseries["surface_temperature_min_C"].min()))
    max_K = float(celsius_to_kelvin(timeseries["surface_temperature_max_C"].max()))

    temperature_bounds_ok = bool(
        min_K >= params.numerical.minimum_temperature_K - 1e-9
        and max_K <= params.numerical.maximum_temperature_K + 1e-9
    )

    q_abs_nonnegative_ok = bool(timeseries["q_abs_max_W_m2"].min() >= -1e-12)

    timestep_count_ok = bool(len(timeseries) >= 10)

    all_ok = bool(
        finite_ok
        and temperature_bounds_ok
        and q_abs_nonnegative_ok
        and timestep_count_ok
    )

    return {
        "case_key": case_key,
        "wind_speed_m_s": wind_speed_m_s,
        "finite_outputs_ok": finite_ok,
        "temperature_bounds_ok": temperature_bounds_ok,
        "q_abs_nonnegative_ok": q_abs_nonnegative_ok,
        "timestep_count_ok": timestep_count_ok,
        "minimum_temperature_K_seen": min_K,
        "maximum_temperature_K_seen": max_K,
        "time_steps": len(timeseries),
        "all_transient_checks_ok": all_ok,
    }


def run_transient_day(
    case_key: str,
    case_label: str,
    solar_case_full: pd.DataFrame,
    mirror_grid: pd.DataFrame,
    weather: WeatherCase,
    main_config: dict[str, Any],
    transient_config: dict[str, Any],
    params: TransientThermalParameters,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, object]]:
    """
    Run one full-day transient calculation for one case and one wind speed.
    """
    raw = transient_config["thermal_transient"]

    step_minutes = params.numerical.time_step_minutes

    sampled_solar = downsample_solar_case(
        solar_case=solar_case_full,
        step_minutes=step_minutes,
    )

    snapshot_raw = raw["snapshots"]

    snapshot_indices = select_target_snapshot_indices(
        sampled_solar_case=sampled_solar,
        full_solar_case=solar_case_full,
        morning_offset_minutes=int(snapshot_raw["morning_offset_minutes_after_front_start"]),
        evening_offset_minutes=int(snapshot_raw["evening_offset_minutes_before_front_end"]),
    )

    incidence_params = IncidenceParameters(
        axis_tilt_south_deg=float(main_config["main_reflector"]["axis_tilt_south_deg"])
    )

    shadow_params = build_shadow_v2_parameters(
        main_config=main_config,
        shadow_v2_config=_load_shadow_v2_config_for_internal_use(),
    )

    irradiation_params = build_irradiation_parameters(main_config)

    initial_T = solve_night_equilibrium_temperature_K(
        weather=weather,
        params=params,
    )

    aperture_points = int(mirror_grid["inside_aperture"].sum())

    T = np.full(aperture_points, initial_T, dtype=float)

    last_dT_dt = np.zeros_like(T)
    last_net_flux = np.zeros_like(T)
    last_heat_loss = np.zeros_like(T)

    timeseries_rows: list[dict[str, object]] = []
    snapshots: dict[str, pd.DataFrame] = {}

    peak_surface_max_C = -np.inf
    peak_snapshot_data: pd.DataFrame | None = None
    peak_snapshot_code = "peak"

    for time_index, (_, solar_row) in enumerate(sampled_solar.iterrows()):
        local_time = str(solar_row["local_time"])

        flux = compute_flux_for_solar_row(
            mirror_grid=mirror_grid,
            solar_row=solar_row,
            incidence_params=incidence_params,
            shadow_params=shadow_params,
            irradiation_params=irradiation_params,
        )

        q_abs = flux["q_abs_W_m2"].to_numpy(dtype=float)

        T, dT_dt, net_flux, heat_loss = transient_temperature_step_K(
            surface_temperature_K=T,
            q_abs_W_m2=q_abs,
            weather=weather,
            params=params,
        )

        last_dT_dt = dT_dt
        last_net_flux = net_flux
        last_heat_loss = heat_loss

        row = summarize_transient_state(
            case_key=case_key,
            case_label=case_label,
            local_time=local_time,
            time_index=time_index,
            weather=weather,
            params=params,
            flux=flux,
            surface_temperature_K=T,
            dT_dt_K_s=dT_dt,
            net_flux_W_m2=net_flux,
        )

        timeseries_rows.append(row)

        current_max_C = float(row["surface_temperature_max_C"])

        if current_max_C > peak_surface_max_C:
            peak_surface_max_C = current_max_C

            peak_snapshot_data = make_snapshot_dataframe(
                flux=flux,
                surface_temperature_K=T.copy(),
                dT_dt_K_s=dT_dt.copy(),
                net_flux_W_m2=net_flux.copy(),
                heat_loss_W_m2=heat_loss.copy(),
                weather=weather,
                params=params,
                case_key=case_key,
                case_label=case_label,
                snapshot_code=peak_snapshot_code,
                selected_local_time=local_time,
            )

        for snapshot_code, snapshot_index in snapshot_indices.items():
            if time_index == snapshot_index:
                snapshots[snapshot_code] = make_snapshot_dataframe(
                    flux=flux,
                    surface_temperature_K=T.copy(),
                    dT_dt_K_s=dT_dt.copy(),
                    net_flux_W_m2=net_flux.copy(),
                    heat_loss_W_m2=heat_loss.copy(),
                    weather=weather,
                    params=params,
                    case_key=case_key,
                    case_label=case_label,
                    snapshot_code=snapshot_code,
                    selected_local_time=local_time,
                )

    if bool(snapshot_raw.get("include_peak_snapshot", True)) and peak_snapshot_data is not None:
        snapshots[peak_snapshot_code] = peak_snapshot_data

    timeseries = pd.DataFrame(timeseries_rows)

    validation = validate_transient_timeseries(
        case_key=case_key,
        wind_speed_m_s=weather.wind_speed_m_s,
        timeseries=timeseries,
        params=params,
    )

    return timeseries, snapshots, validation


def _load_shadow_v2_config_for_internal_use() -> dict[str, Any]:
    """
    Internal helper to load the existing Shadow V2 configuration.
    """
    import yaml

    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "shadow_v2_config.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Shadow V2 config not found: {path}. "
            "Run Step 4.3b first."
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid Shadow V2 config: {path}")

    return data


def save_transient_outputs(
    timeseries: pd.DataFrame,
    snapshots: dict[str, pd.DataFrame],
    output_dir: Path,
    case_key: str,
    wind_code: str,
) -> list[Path]:
    """
    Save transient time series and spatial snapshots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    timeseries_path = output_dir / f"transient_timeseries_{case_key}_{wind_code}.csv"
    timeseries.to_csv(timeseries_path, index=False, encoding="utf-8")
    saved_paths.append(timeseries_path)

    for snapshot_code, snapshot in snapshots.items():
        path = output_dir / f"transient_snapshot_{case_key}_{snapshot_code}_{wind_code}.csv"
        snapshot.to_csv(path, index=False, encoding="utf-8")
        saved_paths.append(path)

    return saved_paths
