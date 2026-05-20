from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load the ROT-54/2.6 project configuration from YAML.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file is empty or invalid: {path}")

    return config


def get_project_root() -> Path:
    """
    Return the project root assuming this file is located at:

        src/rot54/config.py
    """
    return Path(__file__).resolve().parents[2]
