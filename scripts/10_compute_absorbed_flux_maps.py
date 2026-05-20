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
from rot54.irradiation import (
    build_irradiation_parameters,
    compute_absorbed_flux_map,
    save_absorbed_flux_map,
    summarize_absorbed_flux,
    validate_absorbed_flux,
)


def plot_absorbed_flux(
    flux: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot absorbed solar flux q_abs in W/m^2.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        flux["x_m"],
        flux["y_m"],
        c=flux["q_abs_W_m2"],
        s=2,
        linewidths=0,
        vmin=0,
    )

    ax.set_title(title)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("q_abs, W/m^2")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_effective_solar_factor(
    flux: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot the dimensionless factor before alpha_s * I_eff.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        flux["x_m"],
        flux["y_m"],
        c=flux["effective_solar_factor_v2"],
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
    cbar.set_label("effective solar factor")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "rot54_config.yaml")

    shadow_v2_dir = PROJECT_ROOT / "outputs" / "shadow_v2"
    output_dir = PROJECT_ROOT / "outputs" / "irradiation"
    figures_dir = PROJECT_ROOT / "outputs" / "figures" / "irradiation"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    params = build_irradiation_parameters(config)

    shadow_files = sorted(shadow_v2_dir.glob("shadow_v2_*_*.csv"))

    shadow_files = [
        path for path in shadow_files
        if path.name not in [
            "shadow_v2_summary.csv",
            "shadow_v2_validation.csv",
        ]
    ]

    if not shadow_files:
        raise FileNotFoundError(
            f"No Shadow V2 maps found in {shadow_v2_dir}. "
            "Run scripts/09_compute_shadow_v2_time_series.py first."
        )

    summaries: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []

    for shadow_path in shadow_files:
        shadow = pd.read_csv(shadow_path)

        if "case_key" not in shadow.columns:
            raise ValueError(f"case_key column missing in {shadow_path}")

        if "case_label" not in shadow.columns:
            raise ValueError(f"case_label column missing in {shadow_path}")

        if "time_code" not in shadow.columns:
            raise ValueError(f"time_code column missing in {shadow_path}")

        if "selected_local_time" not in shadow.columns:
            raise ValueError(f"selected_local_time column missing in {shadow_path}")

        case_key = str(shadow["case_key"].iloc[0])
        case_label = str(shadow["case_label"].iloc[0])
        time_code = str(shadow["time_code"].iloc[0])
        selected_local_time = str(shadow["selected_local_time"].iloc[0])

        flux = compute_absorbed_flux_map(
            shadow_v2=shadow,
            params=params,
        )

        output_csv = output_dir / f"absorbed_flux_{case_key}_{time_code}.csv"
        flux_png = figures_dir / f"absorbed_flux_{case_key}_{time_code}.png"
        factor_png = figures_dir / f"effective_solar_factor_{case_key}_{time_code}.png"

        save_absorbed_flux_map(
            flux=flux,
            output_path=output_csv,
        )

        title_base = (
            f"ROT-54/2.6 Absorbed Solar Flux\n"
            f"{case_label}, {time_code}, {selected_local_time}\n"
            f"I_eff={float(flux['effective_direct_normal_irradiance_W_m2'].iloc[0]):.1f} W/m^2, "
            f"alpha_s={float(flux['absorptivity'].iloc[0]):.2f}"
        )

        plot_absorbed_flux(
            flux=flux,
            output_path=flux_png,
            title=title_base,
        )

        plot_effective_solar_factor(
            flux=flux,
            output_path=factor_png,
            title=(
                f"ROT-54/2.6 Effective Solar Factor\n"
                f"{case_label}, {time_code}, {selected_local_time}"
            ),
        )

        summaries.append(
            summarize_absorbed_flux(
                case_key=case_key,
                case_label=case_label,
                time_code=time_code,
                selected_local_time=selected_local_time,
                flux=flux,
            )
        )

        validations.append(
            validate_absorbed_flux(
                case_key=case_key,
                time_code=time_code,
                flux=flux,
            )
        )

        print(f"{case_label} / {time_code}: absorbed flux calculated.")
        print(f"  Time: {selected_local_time}")
        print(f"  CSV: {output_csv}")
        print(f"  Flux PNG: {flux_png}")
        print(f"  Factor PNG: {factor_png}")

    summary_df = pd.DataFrame(summaries)
    validation_df = pd.DataFrame(validations)

    summary_csv = output_dir / "absorbed_flux_summary.csv"
    validation_csv = output_dir / "absorbed_flux_validation.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    validation_df.to_csv(validation_csv, index=False, encoding="utf-8")

    print("")
    print("Absorbed solar flux step completed.")
    print(f"Summary: {summary_csv}")
    print(f"Validation: {validation_csv}")
    print("")

    for _, row in validation_df.iterrows():
        print(
            f"- {row['case_key']} / {row['time_code']}: "
            f"all_absorbed_flux_checks_ok = {row['all_absorbed_flux_checks_ok']}, "
            f"max_error = {row['max_q_abs_equation_error_W_m2']:.3e} W/m^2"
        )

    if not validation_df["all_absorbed_flux_checks_ok"].all():
        raise ValueError("At least one absorbed-flux validation check failed.")


if __name__ == "__main__":
    main()
