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


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "rot54_config.yaml"
    config = load_config(config_path)

    grid_csv = PROJECT_ROOT / config["geometry_grid"]["output_csv"]
    output_png = PROJECT_ROOT / config["geometry_grid"]["output_figure_png"]

    if not grid_csv.exists():
        raise FileNotFoundError(
            f"Grid file was not found: {grid_csv}. "
            "Run scripts/03_generate_mirror_grid.py first."
        )

    grid = pd.read_csv(grid_csv)
    inside = grid[grid["inside_aperture"] == True].copy()

    if inside.empty:
        raise ValueError("No points inside aperture. Cannot plot geometry.")

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        inside["x_m"],
        inside["y_m"],
        c=inside["z_m"],
        s=2,
        linewidths=0,
    )

    ax.set_title("ROT-54/2.6 Main Reflector Geometry Grid")
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("z, m")

    ax.grid(True, linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    print("Main reflector geometry figure generated successfully.")
    print(f"Figure output: {output_png}")


if __name__ == "__main__":
    main()
