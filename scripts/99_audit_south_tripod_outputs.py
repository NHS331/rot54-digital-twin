from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OLD_TEXT_PATTERNS = [
    "angles_deg: [90.0, 210.0, 330.0]",
    "[90.0, 210.0, 330.0]",
    "(90.0, 210.0, 330.0)",
]

TEXT_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".csv",
    ".json",
}

GENERATED_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".svg",
    ".pdf",
    ".npz",
    ".csv",
}


def iter_files(base: Path):
    if not base.exists():
        return

    for path in base.rglob("*"):
        if path.is_file():
            yield path


def main() -> None:
    bad_text_files = []

    for folder_name in ["configs", "scripts", "src", "tests", "docs"]:
        folder = ROOT / folder_name

        for path in iter_files(folder):
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue

            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue

            for pattern in OLD_TEXT_PATTERNS:
                if pattern in text:
                    bad_text_files.append(path)
                    break

    if bad_text_files:
        print("Forbidden old tripod geometry still exists in text files:")
        for path in bad_text_files:
            print(" -", path.relative_to(ROOT))
        raise SystemExit(1)

    generated_files = []

    for folder_name in ["output", "outputs"]:
        folder = ROOT / folder_name

        for path in iter_files(folder):
            if path.suffix.lower() in GENERATED_SUFFIXES:
                generated_files.append(path)

    print("South-tripod text audit: OK")
    print(f"Generated artifact count under output/outputs: {len(generated_files)}")

    if generated_files:
        print("Generated artifacts exist. They must have been regenerated after geometry correction.")
        print("Newest 20 generated artifacts:")
        generated_files = sorted(generated_files, key=lambda p: p.stat().st_mtime, reverse=True)
        for path in generated_files[:20]:
            print(" -", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
