from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs" / "solar_statistics"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    targets = load_yaml(CONFIGS / "scientific_targets.yaml")
    scenarios = load_yaml(CONFIGS / "scenarios_2026.yaml")

    paper_targets = targets["paper_targets"]

    print("ROT-54/2.6 solar statistics reproduction scaffold")
    print("=" * 60)
    print(f"Target year: {paper_targets['year']}")
    print(f"Site: {paper_targets['site']['name']}")
    print(f"Panel count: {paper_targets['geometry']['panel_count']}")
    print(f"Passport smoothness: {paper_targets['geometry']['passport_surface_smoothness_mm']} mm")
    print()

    print("Control exposure targets:")
    print(f"  Summer solstice: {paper_targets['front_side_exposure']['summer_solstice_hours']} h")
    print(f"  Winter solstice: {paper_targets['front_side_exposure']['winter_solstice_hours']} h")
    print(f"  Annual 2026: {paper_targets['front_side_exposure']['annual_hours']} h")
    print()

    print("Scenarios:")
    for key, scenario in scenarios["scenarios"].items():
        print(
            f"  - {key}: {scenario['date']} | "
            f"wind = {scenario['wind_speeds_m_per_s']} m/s"
        )

    report_path = OUTPUTS / "stage_5_reproducibility_scaffold.txt"
    report_path.write_text(
        "Stage 5 scaffold created.\n"
        "Scientific control layer is now available.\n"
        "Next technical step: connect existing solar geometry modules to annual front-side exposure calculation.\n",
        encoding="utf-8",
    )

    print()
    print(f"Written: {report_path}")


if __name__ == "__main__":
    main()
