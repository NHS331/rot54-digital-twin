from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TARGET_NAMES = {
    "irradiation",
    "panelthermal",
    "panelresponse",
}

TEXT_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".csv",
    ".json",
}

FORBIDDEN_PATTERNS = [
    "[90.0, 210.0, 330.0]",
    "(90.0, 210.0, 330.0)",
    "angles_deg: [90.0, 210.0, 330.0]",
]


def main() -> None:
    bad_text_files = []

    for folder_name in ["configs", "scripts", "src", "tests", "docs"]:
        folder = ROOT / folder_name

        if not folder.exists():
            continue

        for path in folder.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue

            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue

            for pattern in FORBIDDEN_PATTERNS:
                if pattern in text:
                    bad_text_files.append(path)
                    break

    if bad_text_files:
        print("Forbidden old tripod geometry remains in text files:")
        for path in bad_text_files:
            print(" -", path.relative_to(ROOT))
        raise SystemExit(1)

    generated_folders = []

    for root_name in ["output", "outputs"]:
        root = ROOT / root_name

        if not root.exists():
            continue

        for path in root.rglob("*"):
            if path.is_dir() and path.name.lower() in TARGET_NAMES:
                generated_folders.append(path)

    print("Text audit: OK, no old [90, 210, 330] tripod geometry found.")
    print("Generated target folders:")

    for path in sorted(generated_folders):
        files = [item for item in path.rglob("*") if item.is_file()]
        print(f" - {path.relative_to(ROOT)} | files={len(files)}")

    required = [
        ROOT / "outputs" / "irradiation",
        ROOT / "outputs" / "PanelThermal",
        ROOT / "outputs" / "PanelResponse",
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        print("Missing required regenerated folders:")
        for path in missing:
            print(" -", path.relative_to(ROOT))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
