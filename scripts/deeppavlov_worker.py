"""DeepPavlov worker for Stage 2 consensus pipeline.

The worker is intended to run under Python 3.9 in a separate environment.
It reads JSON lines from stdin and writes JSON lines to stdout.
"""

from __future__ import annotations

import json
import re
import sys
from contextlib import redirect_stdout
from typing import Any

from deeppavlov import build_model


POS_MAPPING = {
    "NOUN": "NOUN",
    "PROPN": "PROPN",
    "VERB": "VERB",
    "ADJ": "ADJ",
    "NUM": "NUM",
    "ADV": "ADV",
    "PRON": "PRON",
    "ADP": "ADP",
    "PREP": "ADP",
    "CCONJ": "CCONJ",
    "SCONJ": "SCONJ",
    "PART": "PART",
    "INTJ": "INTJ",
    "DET": "DET",
    "AUX": "AUX",
    "PUNCT": "PUNCT",
    "SYM": "SYM",
    "X": "X",
}


def sanitize_text(value: Any) -> str:
    cleaned_chars: list[str] = []
    for char in str(value):
        codepoint = ord(char)
        if codepoint == 0 or 0xD800 <= codepoint <= 0xDFFF:
            cleaned_chars.append(" ")
            continue
        cleaned_chars.append(char)
    return "".join(cleaned_chars)


def canonicalize_feats(feats: str | None) -> str:
    text = str(feats or "").strip()
    if not text or text == "_":
        return ""
    pairs: dict[str, str] = {}
    for item in text.split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        pairs[key.strip()] = value.strip()
    return "|".join(f"{key}={pairs[key]}" for key in sorted(pairs))


def parse_deeppavlov_output(output_text: str) -> list[dict[str, str]]:
    tokens: list[dict[str, str]] = []
    for line in output_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\t+|\s{2,}", line)
        if len(parts) < 6:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        if "-" in parts[0]:
            continue
        tokens.append(
            {
                "surface": sanitize_text(parts[1]),
                "lemma": sanitize_text(parts[2]),
                "pos": POS_MAPPING.get(parts[3], parts[3]),
                "raw_pos": sanitize_text(parts[3]),
                "feats": canonicalize_feats(parts[5]),
            }
        )
    return tokens


def main() -> int:
    with redirect_stdout(sys.stderr):
        model = build_model("morpho_ru_syntagrus_bert", download=True, install=True)
    sys.stdout.write(json.dumps({"ok": True, "type": "ready"}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            command = payload.get("type")
            if command == "shutdown":
                break
            if command not in {"analyze", "analyze_batch"}:
                raise ValueError(f"Unsupported command: {command}")
            if command == "analyze":
                text = sanitize_text(payload.get("text", ""))
                with redirect_stdout(sys.stderr):
                    model_result = model([text])
                output_text = model_result[0] if isinstance(model_result, list) and model_result else ""
                tokens = parse_deeppavlov_output(sanitize_text(output_text))
                response: dict[str, Any] = {"ok": True, "type": "result", "tokens": tokens}
            else:
                texts = [sanitize_text(item) for item in payload.get("texts", [])]
                with redirect_stdout(sys.stderr):
                    model_result = model(texts)
                outputs = model_result if isinstance(model_result, list) else []
                batch_tokens = [parse_deeppavlov_output(sanitize_text(output_text)) for output_text in outputs]
                response = {"ok": True, "type": "result_batch", "tokens_batch": batch_tokens}
        except Exception as exc:  # pragma: no cover - runtime worker path
            response = {
                "ok": False,
                "type": "error",
                "error": sanitize_text(f"{type(exc).__name__}: {exc}"),
            }

        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
