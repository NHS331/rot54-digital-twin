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
from rot54.incidence import (
    IncidenceParameters,
    compute_incidence_for_grid,
    save_incidence_map,
    summarize_incidence_map,
)


def select_representative_solar_row(
    solar_case_table: pd.DataFrame,
    mode: str,
) -> pd.Series:
    """
    Select one representative time for the seasonal incidence map.

    Supported mode:
        max_axis_dot:
            Moment when the Sun is most aligned with the tilted central
            reflector axis.
    """
    if solar_case_table.empty:
        raise ValueError("solar_case_table is empty.")

    front = solar_case_table[
        solar_case_table["front_side_illumination"] == True
    ].copy()

    if front.empty:
        raise ValueError("No front-side illumination rows found.")

    if mode == "max_axis_dot":
        idx = front["axis_dot_sun"].idxmax()
        return solar_case_table.loc[idx]

    raise ValueError(f"Unsupported representative time mode: {mode}")


def plot_incidence_cosine(
    incidence: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """
    Plot cos(theta_i) over the aperture.

    Negative or non-sunlit points are retained in the CSV, but in the figure
    only potential sunlit points are emphasized through their cosine factor.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_data = incidence.copy()
    plot_value = plot_data["cos_incidence"].where(
        plot_data["potential_sunlit"] == True
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        plot_data["x_m"],
        plot_data["y_m"],
        c=plot_value,
        s=2,
        linewidths=0,
    )

    ax.set_title(title)
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.4)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("cos(theta_i)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "rot54_config.yaml"
    config = load_config(config_path)

    geometry_csv = PROJECT_ROOT / config["geometry_grid"]["output_csv"]
    solar_positions_csv = PROJECT_ROOT / config["solar_outputs"]["seasonal_positions_csv"]

    if not geometry_csv.exists():
        raise FileNotFoundError(
            f"Mirror grid file not found: {geometry_csv}. "
            "Run scripts/03_generate_mirror_grid.py first."
        )

    if not solar_positions_csv.exists():
        raise FileNotFoundError(
            f"Solar positions file not found: {solar_positions_csv}. "
            "Run scripts/05_generate_solar_tables.py first."
        )

    mirror_grid = pd.read_csv(geometry_csv)
    solar_table = pd.read_csv(solar_positions_csv)

    incidence_config = config["incidence_model"]
    mirror_config = config["main_reflector"]

    output_dir = PROJECT_ROOT / incidence_config["output_dir"]
    figures_dir = PROJECT_ROOT / incidence_config["figures_dir"]
    output_prefix = str(incidence_config["output_prefix"])
    representative_mode = str(incidence_config["representative_time_mode"])

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    params = IncidenceParameters(
        axis_tilt_south_deg=float(mirror_config["axis_tilt_south_deg"]),
    )

    summaries: list[dict[str, object]] = []

    for case_key, case_table in solar_table.groupby("case_key"):
        case_label = str(case_table["case_label"].iloc[0])

        selected_row = select_representative_solar_row(
            solar_case_table=case_table,
            mode=representative_mode,
        )

        selected_time = str(selected_row["clock_time"])
        selected_date = str(selected_row["date"])

        incidence = compute_incidence_for_grid(
            mirror_grid=mirror_grid,
            solar_row=selected_row,
            params=params,
        )

        incidence_csv = output_dir / f"{output_prefix}_{case_key}.csv"
        figure_png = figures_dir / f"incidence_cos_{case_key}.png"

        save_incidence_map(
            incidence=incidence,
            output_path=incidence_csv,
        )

        plot_incidence_cosine(
            incidence=incidence,
            output_path=figure_png,
            title=(
                f"ROT-54/2.6 Solar Incidence Cosine Map\n"
                f"{case_label}, {selected_date}, {selected_time} local time"
            ),
        )

        summary = summarize_incidence_map(
            case_key=str(case_key),
            case_label=case_label,
            selected_time=f"{selected_date} {selected_time}",
            incidence=incidence,
        )

        summaries.append(summary)

        print(
            f"{case_label}: incidence map generated at "
            f"{selected_date} {selected_time} local time."
        )
        print(f"  CSV: {incidence_csv}")
        print(f"  PNG: {figure_png}")

    summary_df = pd.DataFrame(summaries)
    summary_csv = output_dir / f"{output_prefix}_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    print("")
    print("Incidence maps generated successfully.")
    print(f"Incidence summary: {summary_csv}")


if __name__ == "__main__":
    main()
