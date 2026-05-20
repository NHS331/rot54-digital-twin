from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelThermalParameters:
    """
    Parameters for equivalent panel-level thermal nonuniformity.

    The current implementation creates an equivalent square-cell panelization
    over the projected aperture using L_eq from the reduced panel model.

    This is not a final real panel CAD layout.
    It is a controlled engineering bridge from thermal maps to panel response.
    """

    aperture_radius_m: float
    equivalent_cell_size_m: float
    minimum_grid_points_per_panel: int
    delta_rms_thresholds_C: tuple[float, ...]


def build_panel_thermal_parameters(
    main_config: dict[str, Any],
    panel_config: dict[str, Any],
) -> PanelThermalParameters:
    """
    Build panel thermal parameters from YAML configs.
    """
    raw = panel_config["panel_thermal"]

    aperture_radius_m = float(main_config["main_reflector"]["aperture_radius_m"])

    if bool(raw["use_equivalent_cell_size_from_main_config"]):
        equivalent_cell_size_m = float(
            main_config["panel_model"]["equivalent_cell_size_m"]
        )
    else:
        equivalent_cell_size_m = float(raw["equivalent_cell_size_m"])

    thresholds = tuple(float(v) for v in raw["delta_rms_thresholds_C"])

    params = PanelThermalParameters(
        aperture_radius_m=aperture_radius_m,
        equivalent_cell_size_m=equivalent_cell_size_m,
        minimum_grid_points_per_panel=int(raw["minimum_grid_points_per_panel"]),
        delta_rms_thresholds_C=thresholds,
    )

    validate_panel_thermal_parameters(params)

    return params


def validate_panel_thermal_parameters(params: PanelThermalParameters) -> None:
    """
    Validate panel thermal parameters.
    """
    if params.aperture_radius_m <= 0.0:
        raise ValueError("aperture_radius_m must be positive.")

    if params.equivalent_cell_size_m <= 0.0:
        raise ValueError("equivalent_cell_size_m must be positive.")

    if params.minimum_grid_points_per_panel < 5:
        raise ValueError(
            "minimum_grid_points_per_panel must be at least 5 "
            "because the panel model uses five representative points."
        )

    if not params.delta_rms_thresholds_C:
        raise ValueError("At least one delta RMS threshold is required.")

    if any(value < 0.0 for value in params.delta_rms_thresholds_C):
        raise ValueError("delta_rms_thresholds_C must be non-negative.")


def require_snapshot_columns(snapshot: pd.DataFrame) -> None:
    """
    Ensure that the transient snapshot contains required thermal columns.
    """
    required = [
        "x_m",
        "y_m",
        "z_m",
        "surface_temperature_C",
        "deltaT_air_C",
        "deltaT_aperture_mean_C",
        "q_abs_W_m2",
    ]

    missing = [column for column in required if column not in snapshot.columns]

    if missing:
        raise ValueError(f"Transient snapshot is missing required columns: {missing}")


def assign_equivalent_panel_indices(
    snapshot: pd.DataFrame,
    params: PanelThermalParameters,
) -> pd.DataFrame:
    """
    Assign each thermal grid point to an equivalent projected panel cell.

    Cell coordinates:
        x_index = floor((x + R) / L_eq)
        y_index = floor((y + R) / L_eq)

    where:
        R = aperture radius
        L_eq = equivalent cell size
    """
    require_snapshot_columns(snapshot)

    df = snapshot.copy()

    R = params.aperture_radius_m
    L = params.equivalent_cell_size_m

    x = df["x_m"].to_numpy(dtype=float)
    y = df["y_m"].to_numpy(dtype=float)

    ix = np.floor((x + R) / L).astype(int)
    iy = np.floor((y + R) / L).astype(int)

    df["panel_ix"] = ix
    df["panel_iy"] = iy
    df["equivalent_panel_id"] = [
        f"P_{a:04d}_{b:04d}" for a, b in zip(ix, iy)
    ]

    df["equivalent_panel_center_x_m"] = -R + (ix + 0.5) * L
    df["equivalent_panel_center_y_m"] = -R + (iy + 0.5) * L

    return df


def nearest_value_to_target(
    group: pd.DataFrame,
    target_x: float,
    target_y: float,
    value_column: str,
) -> float:
    """
    Return value at the grid point nearest to a target coordinate.
    """
    dx = group["x_m"].to_numpy(dtype=float) - target_x
    dy = group["y_m"].to_numpy(dtype=float) - target_y

    distance2 = dx * dx + dy * dy

    index = int(np.argmin(distance2))

    return float(group[value_column].iloc[index])


def compute_five_point_temperatures_for_panel(
    group: pd.DataFrame,
    params: PanelThermalParameters,
) -> dict[str, float]:
    """
    Compute five representative temperatures for one equivalent panel.

    Representative points:
        center
        upper-right
        upper-left
        lower-right
        lower-left

    Since the thermal field is available on a dense grid, each representative
    value is taken from the nearest available grid point within the panel cell.
    """
    center_x = float(group["equivalent_panel_center_x_m"].iloc[0])
    center_y = float(group["equivalent_panel_center_y_m"].iloc[0])

    half = 0.5 * params.equivalent_cell_size_m

    targets = {
        "center": (center_x, center_y),
        "upper_right": (center_x + half, center_y + half),
        "upper_left": (center_x - half, center_y + half),
        "lower_right": (center_x + half, center_y - half),
        "lower_left": (center_x - half, center_y - half),
    }

    result: dict[str, float] = {}

    for key, (tx, ty) in targets.items():
        result[f"T_{key}_C"] = nearest_value_to_target(
            group=group,
            target_x=tx,
            target_y=ty,
            value_column="surface_temperature_C",
        )

    return result


def compute_panel_thermal_map(
    snapshot: pd.DataFrame,
    params: PanelThermalParameters,
) -> pd.DataFrame:
    """
    Convert pointwise transient thermal snapshot into equivalent panel-level
    thermal nonuniformity map.
    """
    assigned = assign_equivalent_panel_indices(
        snapshot=snapshot,
        params=params,
    )

    rows: list[dict[str, object]] = []

    for panel_id, group in assigned.groupby("equivalent_panel_id"):
        group = group.copy()

        if len(group) < params.minimum_grid_points_per_panel:
            continue

        center_x = float(group["equivalent_panel_center_x_m"].iloc[0])
        center_y = float(group["equivalent_panel_center_y_m"].iloc[0])
        radius = float(np.sqrt(center_x * center_x + center_y * center_y))

        # Ignore equivalent-cell centers outside the circular aperture.
        # Boundary cells are still represented if their center lies inside aperture.
        if radius > params.aperture_radius_m:
            continue

        temps = compute_five_point_temperatures_for_panel(
            group=group,
            params=params,
        )

        five_values = np.array(
            [
                temps["T_center_C"],
                temps["T_upper_right_C"],
                temps["T_upper_left_C"],
                temps["T_lower_right_C"],
                temps["T_lower_left_C"],
            ],
            dtype=float,
        )

        five_mean = float(np.mean(five_values))
        delta_from_five_mean = five_values - five_mean

        panel_delta_rms = float(np.sqrt(np.mean(delta_from_five_mean**2)))
        panel_peak_to_peak = float(np.max(five_values) - np.min(five_values))
        panel_max_abs_from_mean = float(np.max(np.abs(delta_from_five_mean)))

        grid_temperatures = group["surface_temperature_C"].to_numpy(dtype=float)

        if "visibility_chi_v2" in group.columns:
            visible_fraction = float(
                np.mean(group["visibility_chi_v2"].to_numpy(dtype=float) > 0.5)
            )
        else:
            visible_fraction = np.nan

        row = {
            "equivalent_panel_id": panel_id,
            "panel_ix": int(group["panel_ix"].iloc[0]),
            "panel_iy": int(group["panel_iy"].iloc[0]),
            "panel_center_x_m": center_x,
            "panel_center_y_m": center_y,
            "panel_center_radius_m": radius,
            "panel_grid_point_count": int(len(group)),
            "equivalent_cell_size_m": params.equivalent_cell_size_m,
            "panel_visible_fraction": visible_fraction,
            "panel_q_abs_mean_W_m2": float(group["q_abs_W_m2"].mean()),
            "panel_q_abs_max_W_m2": float(group["q_abs_W_m2"].max()),
            "panel_temperature_grid_mean_C": float(np.mean(grid_temperatures)),
            "panel_temperature_grid_min_C": float(np.min(grid_temperatures)),
            "panel_temperature_grid_max_C": float(np.max(grid_temperatures)),
            "panel_temperature_five_point_mean_C": five_mean,
            "panel_delta_rms_C": panel_delta_rms,
            "panel_delta_peak_to_peak_C": panel_peak_to_peak,
            "panel_delta_max_abs_from_mean_C": panel_max_abs_from_mean,
            "panel_temperature_gradient_C_per_m": (
                panel_peak_to_peak / params.equivalent_cell_size_m
            ),
        }

        row.update(temps)

        rows.append(row)

    if not rows:
        raise ValueError("No equivalent panels were generated from the snapshot.")

    panel_map = pd.DataFrame(rows)

    panel_map = panel_map.sort_values(
        by=["panel_ix", "panel_iy"],
        ascending=True,
    ).reset_index(drop=True)

    return panel_map


def summarize_panel_thermal_map(
    source_snapshot_name: str,
    case_key: str,
    case_label: str,
    snapshot_code: str,
    wind_code: str,
    selected_local_time: str,
    panel_map: pd.DataFrame,
    params: PanelThermalParameters,
) -> dict[str, object]:
    """
    Summarize equivalent panel-level thermal nonuniformity.
    """
    delta_rms = panel_map["panel_delta_rms_C"].to_numpy(dtype=float)
    peak_to_peak = panel_map["panel_delta_peak_to_peak_C"].to_numpy(dtype=float)
    gradient = panel_map["panel_temperature_gradient_C_per_m"].to_numpy(dtype=float)

    row: dict[str, object] = {
        "source_snapshot_name": source_snapshot_name,
        "case_key": case_key,
        "case_label": case_label,
        "snapshot_code": snapshot_code,
        "wind_code": wind_code,
        "selected_local_time": selected_local_time,
        "equivalent_panel_count": int(len(panel_map)),
        "equivalent_cell_size_m": params.equivalent_cell_size_m,
        "panel_delta_rms_min_C": float(np.min(delta_rms)),
        "panel_delta_rms_mean_C": float(np.mean(delta_rms)),
        "panel_delta_rms_p95_C": float(np.percentile(delta_rms, 95)),
        "panel_delta_rms_max_C": float(np.max(delta_rms)),
        "panel_delta_peak_to_peak_mean_C": float(np.mean(peak_to_peak)),
        "panel_delta_peak_to_peak_p95_C": float(np.percentile(peak_to_peak, 95)),
        "panel_delta_peak_to_peak_max_C": float(np.max(peak_to_peak)),
        "panel_temperature_gradient_mean_C_per_m": float(np.mean(gradient)),
        "panel_temperature_gradient_p95_C_per_m": float(np.percentile(gradient, 95)),
        "panel_temperature_gradient_max_C_per_m": float(np.max(gradient)),
        "panel_q_abs_mean_W_m2": float(panel_map["panel_q_abs_mean_W_m2"].mean()),
        "panel_q_abs_max_W_m2": float(panel_map["panel_q_abs_max_W_m2"].max()),
    }

    for threshold in params.delta_rms_thresholds_C:
        safe_name = str(threshold).replace(".", "p")
        count = int(np.sum(delta_rms >= threshold))
        fraction = float(count / len(delta_rms))

        row[f"panels_delta_rms_ge_{safe_name}_C_count"] = count
        row[f"panels_delta_rms_ge_{safe_name}_C_fraction"] = fraction

    return row


def validate_panel_thermal_map(
    source_snapshot_name: str,
    panel_map: pd.DataFrame,
) -> dict[str, object]:
    """
    Validate panel-level thermal quantities.
    """
    required = [
        "panel_delta_rms_C",
        "panel_delta_peak_to_peak_C",
        "panel_delta_max_abs_from_mean_C",
        "panel_temperature_gradient_C_per_m",
        "panel_grid_point_count",
    ]

    missing = [column for column in required if column not in panel_map.columns]

    if missing:
        raise ValueError(f"Panel map is missing required columns: {missing}")

    values = panel_map[required].to_numpy(dtype=float)

    finite_ok = bool(np.isfinite(values).all())

    nonnegative_ok = bool(
        (panel_map["panel_delta_rms_C"] >= -1e-12).all()
        and (panel_map["panel_delta_peak_to_peak_C"] >= -1e-12).all()
        and (panel_map["panel_delta_max_abs_from_mean_C"] >= -1e-12).all()
        and (panel_map["panel_temperature_gradient_C_per_m"] >= -1e-12).all()
    )

    rms_not_larger_than_peak_to_peak_ok = bool(
        (
            panel_map["panel_delta_rms_C"]
            <= panel_map["panel_delta_peak_to_peak_C"] + 1e-12
        ).all()
    )

    panel_count_ok = bool(len(panel_map) > 100)

    all_ok = bool(
        finite_ok
        and nonnegative_ok
        and rms_not_larger_than_peak_to_peak_ok
        and panel_count_ok
    )

    return {
        "source_snapshot_name": source_snapshot_name,
        "finite_outputs_ok": finite_ok,
        "nonnegative_outputs_ok": nonnegative_ok,
        "rms_not_larger_than_peak_to_peak_ok": rms_not_larger_than_peak_to_peak_ok,
        "panel_count_ok": panel_count_ok,
        "equivalent_panel_count": int(len(panel_map)),
        "all_panel_thermal_checks_ok": all_ok,
    }


def save_panel_thermal_map(
    panel_map: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save panel thermal map.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel_map.to_csv(path, index=False, encoding="utf-8")


def write_panel_thermal_report(
    summary: pd.DataFrame,
    validation: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Write compact Markdown report for panel thermal nonuniformity.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ranked = summary.sort_values(
        by=[
            "panel_delta_rms_max_C",
            "panel_delta_peak_to_peak_max_C",
        ],
        ascending=False,
    ).reset_index(drop=True)

    worst = ranked.iloc[0]

    failed = validation[
        validation["all_panel_thermal_checks_ok"].astype(str).str.lower()
        != "true"
    ]

    lines: list[str] = []

    lines.append("# ROT-54/2.6 Panel Thermal Nonuniformity Report")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This step converts pointwise transient thermal snapshots into an "
        "equivalent panel-level temperature nonuniformity map."
    )
    lines.append("")
    lines.append("## Worst panel-level snapshot")
    lines.append("")
    lines.append(f"- Source snapshot: `{worst['source_snapshot_name']}`")
    lines.append(f"- Case: `{worst['case_key']}`")
    lines.append(f"- Snapshot code: `{worst['snapshot_code']}`")
    lines.append(f"- Wind code: `{worst['wind_code']}`")
    lines.append(f"- Local time: `{worst['selected_local_time']}`")
    lines.append(
        f"- Maximum panel RMS temperature nonuniformity: "
        f"`{worst['panel_delta_rms_max_C']:.6f} °C`"
    )
    lines.append(
        f"- 95th percentile panel RMS temperature nonuniformity: "
        f"`{worst['panel_delta_rms_p95_C']:.6f} °C`"
    )
    lines.append(
        f"- Maximum panel peak-to-peak temperature difference: "
        f"`{worst['panel_delta_peak_to_peak_max_C']:.6f} °C`"
    )
    lines.append("")
    lines.append("## Validation")
    lines.append("")

    if failed.empty:
        lines.append("All panel thermal validation checks passed.")
    else:
        lines.append("Some panel thermal validation checks failed.")
        lines.append("")
        for _, row in failed.iterrows():
            lines.append(f"- `{row['source_snapshot_name']}` failed validation.")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The output `panel_delta_rms_C` is the direct thermal bridge to the "
        "later reduced mechanical response model. The next step will use this "
        "quantity to estimate normal panel displacement through the coefficient k_u."
    )
    lines.append("")
    lines.append(
        "This is still an equivalent panel discretization. It must later be "
        "replaced or refined by the real ROT-54/2.6 panel layout if exact CAD or "
        "survey panel coordinates become available."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
