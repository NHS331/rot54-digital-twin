from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "solar_statistics"

CSV_STANDARD_PATH = OUTPUT_DIR / "front_side_exposure_2026_standard_refraction.csv"
CSV_TARGET_FIT_PATH = OUTPUT_DIR / "front_side_exposure_2026_target_fit_horizon.csv"
REPORT_PATH = OUTPUT_DIR / "front_side_exposure_validation_report.md"


YEAR = 2026

LATITUDE_DEG = 40.35
REFLECTOR_TILT_SOUTH_DEG = 15.0
FRONT_SIDE_NORMAL_DECLINATION_DEG = LATITUDE_DEG - REFLECTOR_TILT_SOUTH_DEG

STANDARD_REFRACTION_HORIZON_ALTITUDE_DEG = -0.833
GEOMETRIC_HORIZON_ALTITUDE_DEG = 0.0

MANUSCRIPT_SUMMER_SOLSTICE_TARGET_H = 13.58
MANUSCRIPT_WINTER_SOLSTICE_TARGET_H = 9.29
MANUSCRIPT_ANNUAL_TARGET_H = 4275.93

SOLAR_ANGULAR_SPEED_DEG_PER_H = 15.0


@dataclass(frozen=True)
class ExposureRow:
    date_iso: str
    day_of_year: int
    solar_declination_deg: float
    horizon_half_angle_deg: float
    front_side_half_angle_deg: float
    active_half_angle_deg: float
    front_side_exposure_h: float
    limiting_condition: str


@dataclass(frozen=True)
class ExposureSummary:
    horizon_altitude_deg: float
    annual_exposure_h: float
    equinox_exposure_h: float
    summer_solstice_exposure_h: float
    winter_solstice_exposure_h: float


def deg_to_rad(value_deg: float) -> float:
    return math.radians(value_deg)


def rad_to_deg(value_rad: float) -> float:
    return math.degrees(value_rad)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_acos_deg(value: float) -> float:
    return rad_to_deg(math.acos(clamp(value, -1.0, 1.0)))


def solar_declination_deg(day_of_year: int) -> float:
    argument_deg = 360.0 * (284.0 + day_of_year) / 365.0
    return 23.45 * math.sin(deg_to_rad(argument_deg))


def horizon_half_angle_deg(
    latitude_deg: float,
    solar_declination_deg_value: float,
    apparent_horizon_altitude_deg: float,
) -> float:
    latitude_rad = deg_to_rad(latitude_deg)
    declination_rad = deg_to_rad(solar_declination_deg_value)
    horizon_rad = deg_to_rad(apparent_horizon_altitude_deg)

    denominator = math.cos(latitude_rad) * math.cos(declination_rad)

    if abs(denominator) < 1e-15:
        return 0.0

    cosine_hour_angle = (
        math.sin(horizon_rad)
        - math.sin(latitude_rad) * math.sin(declination_rad)
    ) / denominator

    return safe_acos_deg(cosine_hour_angle)


def front_side_half_angle_deg(
    front_side_normal_declination_deg_value: float,
    solar_declination_deg_value: float,
) -> float:
    normal_declination_rad = deg_to_rad(front_side_normal_declination_deg_value)
    solar_declination_rad = deg_to_rad(solar_declination_deg_value)

    cosine_hour_angle = -math.tan(normal_declination_rad) * math.tan(solar_declination_rad)

    return safe_acos_deg(cosine_hour_angle)


def front_side_exposure_for_day(
    day_of_year: int,
    apparent_horizon_altitude_deg: float,
) -> ExposureRow:
    current_date = date(YEAR, 1, 1) + timedelta(days=day_of_year - 1)
    declination_deg = solar_declination_deg(day_of_year)

    horizon_angle_deg = horizon_half_angle_deg(
        latitude_deg=LATITUDE_DEG,
        solar_declination_deg_value=declination_deg,
        apparent_horizon_altitude_deg=apparent_horizon_altitude_deg,
    )

    front_angle_deg = front_side_half_angle_deg(
        front_side_normal_declination_deg_value=FRONT_SIDE_NORMAL_DECLINATION_DEG,
        solar_declination_deg_value=declination_deg,
    )

    active_angle_deg = min(horizon_angle_deg, front_angle_deg)
    exposure_h = 2.0 * active_angle_deg / SOLAR_ANGULAR_SPEED_DEG_PER_H

    if horizon_angle_deg < front_angle_deg:
        limiting_condition = "horizon"
    elif front_angle_deg < horizon_angle_deg:
        limiting_condition = "front_side"
    else:
        limiting_condition = "equal"

    return ExposureRow(
        date_iso=current_date.isoformat(),
        day_of_year=day_of_year,
        solar_declination_deg=declination_deg,
        horizon_half_angle_deg=horizon_angle_deg,
        front_side_half_angle_deg=front_angle_deg,
        active_half_angle_deg=active_angle_deg,
        front_side_exposure_h=exposure_h,
        limiting_condition=limiting_condition,
    )


def generate_rows_for_year(
    apparent_horizon_altitude_deg: float,
) -> list[ExposureRow]:
    return [
        front_side_exposure_for_day(
            day_of_year=day_of_year,
            apparent_horizon_altitude_deg=apparent_horizon_altitude_deg,
        )
        for day_of_year in range(1, 366)
    ]


def summarize_rows(
    rows: list[ExposureRow],
    apparent_horizon_altitude_deg: float,
) -> ExposureSummary:
    by_date = {row.date_iso: row for row in rows}

    return ExposureSummary(
        horizon_altitude_deg=apparent_horizon_altitude_deg,
        annual_exposure_h=sum(row.front_side_exposure_h for row in rows),
        equinox_exposure_h=by_date["2026-03-20"].front_side_exposure_h,
        summer_solstice_exposure_h=by_date["2026-06-22"].front_side_exposure_h,
        winter_solstice_exposure_h=by_date["2026-12-22"].front_side_exposure_h,
    )


def annual_exposure_for_horizon(
    apparent_horizon_altitude_deg: float,
) -> float:
    rows = generate_rows_for_year(
        apparent_horizon_altitude_deg=apparent_horizon_altitude_deg,
    )
    return sum(row.front_side_exposure_h for row in rows)


def find_horizon_for_annual_target(
    target_annual_exposure_h: float,
    lower_deg: float = -2.5,
    upper_deg: float = 0.0,
    iterations: int = 80,
) -> float:
    lower_value = annual_exposure_for_horizon(lower_deg)
    upper_value = annual_exposure_for_horizon(upper_deg)

    if not (lower_value >= target_annual_exposure_h >= upper_value):
        raise ValueError(
            "The annual target is outside the search interval. "
            f"target={target_annual_exposure_h}, "
            f"lower_result={lower_value}, upper_result={upper_value}"
        )

    low = lower_deg
    high = upper_deg

    for _ in range(iterations):
        middle = 0.5 * (low + high)
        middle_value = annual_exposure_for_horizon(middle)

        if middle_value > target_annual_exposure_h:
            low = middle
        else:
            high = middle

    return 0.5 * (low + high)


def write_rows_to_csv(
    rows: list[ExposureRow],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "date",
        "day_of_year",
        "solar_declination_deg",
        "horizon_half_angle_deg",
        "front_side_half_angle_deg",
        "active_half_angle_deg",
        "front_side_exposure_h",
        "limiting_condition",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "date": row.date_iso,
                    "day_of_year": row.day_of_year,
                    "solar_declination_deg": f"{row.solar_declination_deg:.9f}",
                    "horizon_half_angle_deg": f"{row.horizon_half_angle_deg:.9f}",
                    "front_side_half_angle_deg": f"{row.front_side_half_angle_deg:.9f}",
                    "active_half_angle_deg": f"{row.active_half_angle_deg:.9f}",
                    "front_side_exposure_h": f"{row.front_side_exposure_h:.9f}",
                    "limiting_condition": row.limiting_condition,
                }
            )


def format_summary_table(
    title: str,
    summary: ExposureSummary,
) -> str:
    return (
        f"### {title}\n\n"
        "| Quantity | Value |\n"
        "|---|---:|\n"
        f"| Apparent horizon altitude | {summary.horizon_altitude_deg:.6f} deg |\n"
        f"| Equinox exposure, 2026-03-20 | {summary.equinox_exposure_h:.6f} h |\n"
        f"| Summer solstice exposure, 2026-06-22 | {summary.summer_solstice_exposure_h:.6f} h |\n"
        f"| Winter solstice exposure, 2026-12-22 | {summary.winter_solstice_exposure_h:.6f} h |\n"
        f"| Annual exposure, 2026 | {summary.annual_exposure_h:.6f} h |\n"
    )


def write_report(
    standard_summary: ExposureSummary,
    geometric_summary: ExposureSummary,
    target_fit_summary: ExposureSummary,
) -> None:
    annual_difference_standard = (
        standard_summary.annual_exposure_h - MANUSCRIPT_ANNUAL_TARGET_H
    )

    summer_difference_standard = (
        standard_summary.summer_solstice_exposure_h
        - MANUSCRIPT_SUMMER_SOLSTICE_TARGET_H
    )

    winter_difference_standard = (
        standard_summary.winter_solstice_exposure_h
        - MANUSCRIPT_WINTER_SOLSTICE_TARGET_H
    )

    report = f"""# Front-Side Solar Exposure Verification — 2026

## Constants

| Parameter | Value |
|---|---:|
| Latitude | {LATITUDE_DEG:.6f} deg |
| Reflector southward tilt | {REFLECTOR_TILT_SOUTH_DEG:.6f} deg |
| Equivalent front-side normal declination | {FRONT_SIDE_NORMAL_DECLINATION_DEG:.6f} deg |
| Solar angular speed | {SOLAR_ANGULAR_SPEED_DEG_PER_H:.6f} deg/h |

## Manuscript control targets

| Quantity | Target |
|---|---:|
| Summer solstice exposure | {MANUSCRIPT_SUMMER_SOLSTICE_TARGET_H:.6f} h |
| Winter solstice exposure | {MANUSCRIPT_WINTER_SOLSTICE_TARGET_H:.6f} h |
| Annual exposure | {MANUSCRIPT_ANNUAL_TARGET_H:.6f} h |

{format_summary_table("Geometric horizon model", geometric_summary)}

{format_summary_table("Standard-refraction horizon model", standard_summary)}

{format_summary_table("Back-fit horizon required to match the manuscript annual target", target_fit_summary)}

## Diagnostic differences for the standard-refraction model

| Quantity | Difference |
|---|---:|
| Summer solstice minus manuscript target | {summer_difference_standard:.6f} h |
| Winter solstice minus manuscript target | {winter_difference_standard:.6f} h |
| Annual exposure minus manuscript target | {annual_difference_standard:.6f} h |

## Interpretation

The summer and winter control-day values are consistent with the manuscript targets under the standard-refraction horizon convention.

The annual value is more sensitive. In this analytical implementation, the standard-refraction model does not silently reproduce the manuscript annual target of {MANUSCRIPT_ANNUAL_TARGET_H:.2f} h.

To reproduce the annual target exactly with the same analytical assumptions, the apparent horizon would need to be approximately {target_fit_summary.horizon_altitude_deg:.6f} deg.

This does not automatically prove that the manuscript value is wrong. It means that the annual value must be traced to the exact previous implementation: horizon convention, front-side condition, day sampling, and possible time-integration assumptions.

## Decision for the next stage

Do not use the annual value as an unquestioned constant.

Use this report as the control file before moving to panel-grid and ray-based shadowing.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


def print_summary(
    standard_summary: ExposureSummary,
    target_fit_summary: ExposureSummary,
) -> None:
    print("Stage 6 — Front-side solar exposure verification")
    print("=" * 60)
    print(f"Latitude: {LATITUDE_DEG:.6f} deg")
    print(f"Reflector southward tilt: {REFLECTOR_TILT_SOUTH_DEG:.6f} deg")
    print(f"Front-side normal declination: {FRONT_SIDE_NORMAL_DECLINATION_DEG:.6f} deg")
    print()
    print("Standard-refraction model:")
    print(f"  Equinox exposure:          {standard_summary.equinox_exposure_h:.6f} h")
    print(f"  Summer solstice exposure:  {standard_summary.summer_solstice_exposure_h:.6f} h")
    print(f"  Winter solstice exposure:  {standard_summary.winter_solstice_exposure_h:.6f} h")
    print(f"  Annual exposure:           {standard_summary.annual_exposure_h:.6f} h")
    print()
    print("Manuscript annual target diagnostic:")
    print(f"  Target annual exposure:    {MANUSCRIPT_ANNUAL_TARGET_H:.6f} h")
    print(f"  Required apparent horizon: {target_fit_summary.horizon_altitude_deg:.6f} deg")
    print()
    print(f"Saved CSV:    {CSV_STANDARD_PATH}")
    print(f"Saved CSV:    {CSV_TARGET_FIT_PATH}")
    print(f"Saved report: {REPORT_PATH}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    geometric_rows = generate_rows_for_year(
        apparent_horizon_altitude_deg=GEOMETRIC_HORIZON_ALTITUDE_DEG,
    )

    standard_rows = generate_rows_for_year(
        apparent_horizon_altitude_deg=STANDARD_REFRACTION_HORIZON_ALTITUDE_DEG,
    )

    target_fit_horizon_deg = find_horizon_for_annual_target(
        target_annual_exposure_h=MANUSCRIPT_ANNUAL_TARGET_H,
    )

    target_fit_rows = generate_rows_for_year(
        apparent_horizon_altitude_deg=target_fit_horizon_deg,
    )

    geometric_summary = summarize_rows(
        rows=geometric_rows,
        apparent_horizon_altitude_deg=GEOMETRIC_HORIZON_ALTITUDE_DEG,
    )

    standard_summary = summarize_rows(
        rows=standard_rows,
        apparent_horizon_altitude_deg=STANDARD_REFRACTION_HORIZON_ALTITUDE_DEG,
    )

    target_fit_summary = summarize_rows(
        rows=target_fit_rows,
        apparent_horizon_altitude_deg=target_fit_horizon_deg,
    )

    write_rows_to_csv(
        rows=standard_rows,
        output_path=CSV_STANDARD_PATH,
    )

    write_rows_to_csv(
        rows=target_fit_rows,
        output_path=CSV_TARGET_FIT_PATH,
    )

    write_report(
        standard_summary=standard_summary,
        geometric_summary=geometric_summary,
        target_fit_summary=target_fit_summary,
    )

    print_summary(
        standard_summary=standard_summary,
        target_fit_summary=target_fit_summary,
    )


if __name__ == "__main__":
    main()
