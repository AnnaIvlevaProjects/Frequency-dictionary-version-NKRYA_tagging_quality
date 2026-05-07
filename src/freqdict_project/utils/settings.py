"""Project settings loader without external dependencies."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.isdigit():
        return int(text)

    if text.startswith("[") and text.endswith("]"):
        prepared = text.replace("true", "True").replace("false", "False")
        return ast.literal_eval(prepared)

    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]

    return text


def load_settings(path: str | Path = "config/settings.yaml") -> dict[str, Any]:
    settings_path = Path(path)
    data: dict[str, Any] = {}
    current_section: str | None = None

    for line in settings_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            data[current_section] = {}
            continue

        if current_section and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            data[current_section][key.strip()] = _parse_scalar(value)

    return data
