"""Run evaluation on the RNC manually disambiguated sample."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = f"{SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep)

from freqdict_project.eval.rnc_eval import RNCEvaluationEngine, load_gold_documents, save_eval_outputs
from freqdict_project.nlp.consensus_stage2 import build_consensus_engine_from_settings
from freqdict_project.utils.settings import load_settings


def main() -> None:
    settings = load_settings(ROOT / "config" / "settings.yaml")
    paths = settings.get("paths", {})
    eval_settings = settings.get("eval", {})
    texts_root = paths["rnc_eval_texts_root"]
    output_root = eval_settings.get("output_root", "./output/rnc_eval")
    limit_docs = eval_settings.get("limit_docs")
    docs = load_gold_documents(texts_root, limit_docs=limit_docs)
    engine = build_consensus_engine_from_settings(settings, ROOT)
    try:
        evaluator = RNCEvaluationEngine(engine)
        detailed_rows, metrics_rows, report = evaluator.evaluate_documents(docs)
    finally:
        engine.close()
    save_eval_outputs(output_root, detailed_rows, metrics_rows, report)
    print("RNC evaluation complete")
    print(report)


if __name__ == "__main__":
    main()

