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
from rot54.shadowing import (
    apply_shadow_model,
    build_shadow_model_parameters,
    save_shadow_result,
    summarize_shadow_result,
    validate_shadow_result,
)


def read_selected_time_map(incidence_summary_path: Path) -> dict[str, str]:
    """
    Read selected representative time for each case from incidence_summary.csv.
    """
    if not incidence_summary_path.exists():
        return {}

    df = pd.read_csv(incidence_summary_path)

    if "case_key" not in df.columns or "selected_time" not in df.columns:
        return {}

    return {
        str(row["case_key"]): str(row["selected_time"])
        for _, row in df.iterrows()
    }


def plot_shadow_mask(
    shadow_result: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot structural visibility mask chi.

    chi = 1 means direct solar visibility after structural shadowing.
    chi = 0 means shadowed or not geometrically sunlit.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        shadow_result["x_m"],
        shadow_result["y_m"],
        c=shadow_result["visibility_chi"],
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
    cbar.set_label("visibility chi")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_effective_solar_factor(
    shadow_result: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot effective solar factor:

        effective_solar_factor = max(cos(theta_i), 0) * chi

    This is the dimensionless multiplier before alpha_s * I_sun.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        shadow_result["x_m"],
        shadow_result["y_m"],
        c=shadow_result["effective_solar_factor"],
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
    cbar.set_label("max(cos(theta_i), 0) * chi")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "rot54_config.yaml"
    config = load_config(config_path)

    incidence_config = config["incidence_model"]
    shadow_config = config["shadow_model"]
    shadow_outputs = config["shadow_outputs"]

    incidence_dir = PROJECT_ROOT / incidence_config["output_dir"]
    incidence_prefix = str(incidence_config["output_prefix"])

    shadow_dir = PROJECT_ROOT / shadow_config["output_dir"]
    figures_dir = PROJECT_ROOT / shadow_config["figures_dir"]
    shadow_prefix = str(shadow_config["output_prefix"])

    shadow_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    params = build_shadow_model_parameters(config)

    incidence_summary_path = incidence_dir / f"{incidence_prefix}_summary.csv"
    selected_time_map = read_selected_time_map(incidence_summary_path)

    incidence_files = sorted(incidence_dir.glob(f"{incidence_prefix}_*.csv"))
    incidence_files = [
        path for path in incidence_files
        if path.name != f"{incidence_prefix}_summary.csv"
    ]

    if not incidence_files:
        raise FileNotFoundError(
            f"No incidence files found in {incidence_dir}. "
            "Run scripts/06_compute_incidence_maps.py first."
        )

    summaries: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []

    for incidence_path in incidence_files:
        case_key = incidence_path.stem.replace(f"{incidence_prefix}_", "")

        incidence = pd.read_csv(incidence_path)

        if "case_label" in incidence.columns:
            case_label = str(incidence["case_label"].iloc[0])
        else:
            case_label = case_key.replace("_", " ").title()

        selected_time = selected_time_map.get(case_key, "")

        shadow_result = apply_shadow_model(
            incidence=incidence,
            params=params,
        )

        shadow_csv = shadow_dir / f"{shadow_prefix}_{case_key}.csv"
        shadow_png = figures_dir / f"shadow_mask_{case_key}.png"
        factor_png = figures_dir / f"effective_solar_factor_{case_key}.png"

        save_shadow_result(
            shadow_result=shadow_result,
            output_path=shadow_csv,
        )

        title_suffix = case_label
        if selected_time:
            title_suffix = f"{case_label}, {selected_time} local time"

        plot_shadow_mask(
            shadow_result=shadow_result,
            output_path=shadow_png,
            title=(
                "ROT-54/2.6 Structural Visibility Mask\n"
                f"{title_suffix}"
            ),
        )

        plot_effective_solar_factor(
            shadow_result=shadow_result,
            output_path=factor_png,
            title=(
                "ROT-54/2.6 Effective Direct Solar Factor\n"
                f"{title_suffix}"
            ),
        )

        summaries.append(
            summarize_shadow_result(
                case_key=case_key,
                case_label=case_label,
                selected_time=selected_time,
                shadow_result=shadow_result,
            )
        )

        validations.append(
            validate_shadow_result(
                case_key=case_key,
                shadow_result=shadow_result,
            )
        )

        print(f"{case_label}: shadowing calculated.")
        print(f"  CSV: {shadow_csv}")
        print(f"  Shadow mask PNG: {shadow_png}")
        print(f"  Effective solar factor PNG: {factor_png}")

    summary_df = pd.DataFrame(summaries)
    validation_df = pd.DataFrame(validations)

    summary_csv = PROJECT_ROOT / shadow_outputs["summary_csv"]
    validation_csv = PROJECT_ROOT / shadow_outputs["validation_csv"]

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    validation_csv.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    validation_df.to_csv(validation_csv, index=False, encoding="utf-8")

    print("")
    print("Structural shadowing step completed successfully.")
    print(f"Shadow summary: {summary_csv}")
    print(f"Shadow validation: {validation_csv}")
    print("")
    print("Validation checks:")
    for _, row in validation_df.iterrows():
        print(
            f"- {row['case_key']}: "
            f"all_shadow_checks_ok = {row['all_shadow_checks_ok']}, "
            f"max_factor_error = {row['max_factor_equation_error']:.3e}"
        )


if __name__ == "__main__":
    main()
