"""XML text extraction preserving logical node order."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


def _sanitize_xml_text(value: str) -> str:
    cleaned_chars: list[str] = []
    for char in value:
        codepoint = ord(char)
        if codepoint == 0:
            cleaned_chars.append(" ")
            continue
        if 0xD800 <= codepoint <= 0xDFFF:
            cleaned_chars.append(" ")
            continue
        cleaned_chars.append(char)
    return "".join(cleaned_chars)


def extract_clean_text(xml_path: str | Path) -> str:
    root = ET.parse(xml_path).getroot()
    chunks: list[str] = []
    for text in root.itertext():
        normalized = " ".join(_sanitize_xml_text(text).split())
        if normalized:
            chunks.append(normalized)
    return " ".join(chunks)
