from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATH = ROOT / "scripts" / "25_compute_ray_shadow_visibility.py"
VISIBILITY_CSV = ROOT / "outputs" / "shadow_ray" / "control_point_visibility.csv"


def run_stage_9() -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=ROOT,
        check=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_summer_noon_central_region_is_not_fully_illuminated() -> None:
    """
    In summer noon the Sun is close to the reflector-local axis.

    The secondary mirror has radius about 2.5 m.
    Therefore the central region cannot be treated as fully illuminated
    by direct solar radiation.
    """

    run_stage_9()

    rows = read_csv(VISIBILITY_CSV)

    central_rows = []

    for row in rows:
        if row["scenario_key"] != "summer_solstice":
            continue

        if row["time_key"] != "noon":
            continue

        x = float(row["x_m"])
        y = float(row["y_m"])
        radius = (x * x + y * y) ** 0.5

        if radius <= 2.5:
            central_rows.append(row)

    assert len(central_rows) > 0

    shadowed_count = sum(1 for row in central_rows if int(row["chi"]) == 0)
    illuminated_count = sum(1 for row in central_rows if int(row["chi"]) == 1)

    assert shadowed_count > 0
    assert shadowed_count > illuminated_count


def test_summer_noon_secondary_mirror_shadow_is_present() -> None:
    """
    The Stage 9 source classification must explicitly detect the secondary mirror
    in the summer-noon central shadow field.
    """

    run_stage_9()

    rows = read_csv(VISIBILITY_CSV)

    secondary_rows = [
        row
        for row in rows
        if row["scenario_key"] == "summer_solstice"
        and row["time_key"] == "noon"
        and row["shadow_source"] == "secondary_mirror"
    ]

    assert len(secondary_rows) > 0
