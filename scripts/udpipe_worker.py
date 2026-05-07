"""Standalone UDPipe worker for Stage 2 consensus pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from ufal.udpipe import Model, Pipeline, ProcessingError


def _canonicalize_feats(value: str | None) -> str:
    if not value or value == "_":
        return ""
    parts = [part.strip() for part in str(value).split("|") if part.strip()]
    return "|".join(sorted(parts))


def _adapt_pos(tag: str | None) -> str:
    return str(tag or "X").upper()


class WorkerUDPipeAdapter:
    def __init__(self, model_path: str) -> None:
        model = Model.load(str(model_path))
        if not model:
            raise FileNotFoundError(f"Failed to load UDPipe model: {model_path}")
        self.pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
        self.error = ProcessingError()

    def analyze_sentence(self, text: str) -> list[dict[str, str]]:
        if not isinstance(text, str):
            text = str(text)
        self.error = ProcessingError()
        processed = self.pipeline.process(text, self.error)
        if self.error.occurred():
            raise RuntimeError(self.error.message)

        tokens: list[dict[str, str]] = []
        for line in processed.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6 or "-" in parts[0]:
                continue
            tokens.append(
                {
                    "surface": parts[1],
                    "lemma": parts[2],
                    "pos": _adapt_pos(parts[3]),
                    "raw_pos": parts[3],
                    "feats": _canonicalize_feats(parts[5]),
                }
            )
        return tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    adapter = WorkerUDPipeAdapter(args.model)

    for raw_line in sys.stdin.buffer:
        line = raw_line.decode("utf-8", errors="strict").strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if payload.get("type") != "analyze":
                raise ValueError(f"Unsupported command: {payload.get('type')}")
            text = payload.get("text", "")
            response = {"ok": True, "tokens": adapter.analyze_sentence(text)}
        except Exception as exc:  # pragma: no cover - runtime worker path
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "debug": {"payload_type": type(payload).__name__ if 'payload' in locals() else "unknown"},
            }

        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
