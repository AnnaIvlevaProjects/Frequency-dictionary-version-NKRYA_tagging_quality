"""Evaluation pipeline for manually disambiguated RNC sample files."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from freqdict_project.nlp.consensus_stage2 import (
    ConsensusStage2Engine,
    SentenceBatch,
    sanitize_runtime_text,
)
from freqdict_project.nlp.consensus_utils import (
    AlignedTokenCandidate,
    AnalyzerToken,
    align_to_base_tokens,
    calculate_feats_consensus,
    calculate_lemma_consensus,
    calculate_pos_consensus,
    normalize_lemma_for_vote,
)
from freqdict_project.nlp.morph_postprocess import process_token


EXCLUDED_GOLD_POS = {"NONLEX", "INIT", "X", "", "DISTORT", "NORM"}
ANALYZER_NAMES = ("stanza", "natasha", "spacy", "pymystem", "udpipe", "deeppavlov")
SYSTEM_NAMES = ANALYZER_NAMES + ("pwc", "simple_consensus")


@dataclass(slots=True)
class GoldToken:
    surface: str
    gold_lemma: str
    gold_tag_raw: str
    gold_pos_rnc: str
    gold_pos_ud: str
    excluded_from_scoring: bool
    sentence_id: int
    token_id: int


@dataclass(slots=True)
class GoldDocument:
    path: str
    text: str
    sentences: list[SentenceBatch]
    tokens: list[GoldToken]


def normalize_gold_pos(raw_tag: str) -> str:
    text = str(raw_tag or "").strip()
    if not text:
        return ""
    head = text.split(",", 1)[0].strip().replace("(distort)", "").strip()
    if "=" in head:
        head = head.split("=", 1)[0].strip()
    head = head.upper()
    head = head.replace("PRAEDIC-PRO", "PRAEDICPRO")
    head = head.replace("S-PRO", "SPRO")
    if head == "ADPRO":
        head = "ADVPRO"
    if "PRAEDICPRO" in head:
        return "PRAEDICPRO"
    return head


def map_rnc_pos_to_ud(pos_rnc: str) -> str:
    mapping = {
        "S": "NOUN",
        "A": "ADJ",
        "NUM": "NUM",
        "ANUM": "ADJ",
        "V": "VERB",
        "ADV": "ADV",
        "PRAEDIC": "ADV",
        "PARENTH": "ADV",
        "SPRO": "PRON",
        "APRO": "DET",
        "ADVPRO": "ADV",
        "PRAEDICPRO": "ADV",
        "PR": "ADP",
        "CONJ": "CONJ",
        "PART": "PART",
        "INTJ": "INTJ",
    }
    return mapping.get(pos_rnc, "X")


def normalize_eval_pos(pos_ud: str) -> str:
    pos = str(pos_ud or "").strip().upper()
    if pos in {"CCONJ", "SCONJ", "CONJ"}:
        return "CONJ"
    return pos


def normalize_eval_pos_relaxed(pos_ud: str) -> str:
    pos = normalize_eval_pos(pos_ud)
    if pos == "AUX":
        return "VERB"
    if pos == "PROPN":
        return "NOUN"
    return pos


def should_exclude_gold_pos(pos_rnc: str) -> bool:
    return pos_rnc in EXCLUDED_GOLD_POS or pos_rnc.startswith("�")


def _iter_gold_word_elements(root: ET.Element):
    for sentence_id, se in enumerate(root.iter("se"), start=1):
        token_id = 0
        for child in list(se):
            if child.tag != "w":
                continue
            token_id += 1
            yield sentence_id, token_id, child


def parse_gold_document(xml_path: str | Path, *, doc_path: str | None = None) -> GoldDocument:
    path = Path(xml_path)
    root = ET.parse(path).getroot()
    tokens: list[GoldToken] = []
    sentence_tokens: dict[int, list[AnalyzerToken]] = defaultdict(list)

    for sentence_id, token_id, word_el in _iter_gold_word_elements(root):
        ana = word_el.find("ana")
        if ana is None:
            continue
        surface = "".join(word_el.itertext()).strip()
        if not surface:
            continue
        gold_lemma = str(ana.attrib.get("lex", "")).strip() or surface
        gold_tag_raw = str(ana.attrib.get("gr", "")).strip()
        gold_pos_rnc = normalize_gold_pos(gold_tag_raw)
        gold_pos_ud = map_rnc_pos_to_ud(gold_pos_rnc)
        excluded = should_exclude_gold_pos(gold_pos_rnc) or gold_pos_ud == "X"
        token = GoldToken(
            surface=surface,
            gold_lemma=gold_lemma,
            gold_tag_raw=gold_tag_raw,
            gold_pos_rnc=gold_pos_rnc,
            gold_pos_ud=gold_pos_ud,
            excluded_from_scoring=excluded,
            sentence_id=sentence_id,
            token_id=token_id,
        )
        tokens.append(token)
        sentence_tokens[sentence_id].append(
            AnalyzerToken(surface=surface, lemma=gold_lemma, pos=gold_pos_ud, raw_pos=gold_pos_rnc)
        )

    sentences = [
        SentenceBatch(text=sanitize_runtime_text(" ".join(token.surface for token in batch)), tokens=batch)
        for _, batch in sorted(sentence_tokens.items())
    ]
    text = "\n".join(sentence.text for sentence in sentences if sentence.text)
    return GoldDocument(path=doc_path or path.name, text=text, sentences=sentences, tokens=tokens)


def calculate_simple_consensus(
    pos_candidates: dict[str, str],
    lemma_candidates: dict[str, str],
    analyzer_performance: dict[str, dict[str, float]],
) -> tuple[str | None, str | None, float, float]:
    def _vote(values: dict[str, str], weight_key: str) -> tuple[str | None, float]:
        cleaned = {k: v for k, v in values.items() if v}
        if not cleaned:
            return None, 0.0
        counts = Counter(cleaned.values()).most_common()
        if len(counts) == 1:
            return counts[0][0], 1.0
        if counts[0][1] > counts[1][1]:
            return counts[0][0], counts[0][1] / len(cleaned)
        scores: dict[str, float] = defaultdict(float)
        total = 0.0
        for analyzer, value in cleaned.items():
            weight = float(analyzer_performance.get(analyzer, {}).get(weight_key, 0.5))
            scores[value] += weight
            total += weight
        winner, score = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0]
        return winner, (score / total if total else 0.0)

    normalized_pos = {name: normalize_eval_pos(value) for name, value in pos_candidates.items() if value}
    normalized_lemmas = {name: normalize_lemma_for_vote(value) for name, value in lemma_candidates.items() if value}
    pos, pos_conf = _vote(normalized_pos, "overall_pos")
    lemma_norm, lemma_conf = _vote(normalized_lemmas, "overall_lemma")
    raw_lemma = None
    if lemma_norm is not None:
        for analyzer in ANALYZER_NAMES:
            if normalize_lemma_for_vote(lemma_candidates.get(analyzer, "")) == lemma_norm:
                raw_lemma = lemma_candidates.get(analyzer)
                break
    return pos, raw_lemma, pos_conf, lemma_conf


class RNCEvaluationEngine:
    def __init__(self, consensus_engine: ConsensusStage2Engine) -> None:
        self.consensus_engine = consensus_engine
        self.stanza = consensus_engine.stanza
        self.pos_weights = consensus_engine.pos_weights
        self.lemma_weights = consensus_engine.lemma_weights
        self.analyzer_performance = consensus_engine.analyzer_performance

    def evaluate_documents(self, docs: list[GoldDocument]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        detailed_rows: list[dict[str, Any]] = []
        stats = {
            name: {
                "pos_correct": 0,
                "pos_relaxed_correct": 0,
                "pos_total": 0,
                "lemma_correct": 0,
                "lemma_total": 0,
                "excluded_tokens": 0,
            }
            for name in SYSTEM_NAMES
        }
        for doc in docs:
            per_doc_rows = self.evaluate_document(doc)
            detailed_rows.extend(per_doc_rows)
            for row in per_doc_rows:
                for system in SYSTEM_NAMES:
                    if row["excluded_from_scoring"]:
                        stats[system]["excluded_tokens"] += 1
                        continue
                    stats[system]["pos_total"] += 1
                    stats[system]["lemma_total"] += 1
                    stats[system]["pos_correct"] += int(bool(row.get(f"{system}_pos_match")))
                    stats[system]["pos_relaxed_correct"] += int(bool(row.get(f"{system}_pos_match_relaxed")))
                    stats[system]["lemma_correct"] += int(bool(row.get(f"{system}_lemma_match")))

        metrics_rows = []
        for system in SYSTEM_NAMES:
            pos_total = stats[system]["pos_total"]
            lemma_total = stats[system]["lemma_total"]
            metrics_rows.append(
                {
                    "system": system,
                    "pos_accuracy": stats[system]["pos_correct"] / pos_total if pos_total else 0.0,
                    "pos_accuracy_rnc_relaxed": stats[system]["pos_relaxed_correct"] / pos_total if pos_total else 0.0,
                    "lemma_accuracy": stats[system]["lemma_correct"] / lemma_total if lemma_total else 0.0,
                    "scored_tokens_pos": pos_total,
                    "scored_tokens_lemma": lemma_total,
                    "excluded_tokens": stats[system]["excluded_tokens"],
                }
            )
        report = {
            "documents_total": len(docs),
            "tokens_total": len(detailed_rows),
            "scoring_mode": "lexical_clean",
            "pos_modes": ["strict", "rnc_relaxed"],
            "systems": metrics_rows,
        }
        return detailed_rows, metrics_rows, report

    def evaluate_document(self, doc: GoldDocument) -> list[dict[str, Any]]:
        precomputed = self.consensus_engine._analyze_document_sentences(doc.sentences)
        rows: list[dict[str, Any]] = []
        gold_index = 0
        for sent_index, sentence in enumerate(doc.sentences, start=1):
            aligned = self._collect_aligned_to_gold(sentence, precomputed, sent_index - 1)
            for token_index, gold_candidate in enumerate(aligned["gold"], start=1):
                if gold_candidate.token is None:
                    continue
                gold_token = doc.tokens[gold_index]
                gold_index += 1
                rows.append(self._build_row(doc, gold_token, sent_index, token_index, sentence.text, aligned))
        return rows

    def _collect_aligned_to_gold(
        self,
        sentence: SentenceBatch,
        precomputed_batches: dict[str, list[list[AnalyzerToken]]],
        sentence_index: int,
    ) -> dict[str, list[AlignedTokenCandidate]]:
        base_words = [token.surface for token in sentence.tokens]
        stanza_batches = self.stanza.analyze_document(sentence.text)
        stanza_tokens = stanza_batches[0].tokens if stanza_batches else []
        aligned = {
            "gold": [
                AlignedTokenCandidate(token=token, token_found=True, alignment_info=f"base_{index}")
                for index, token in enumerate(sentence.tokens)
            ]
        }
        aligned["stanza"] = align_to_base_tokens(base_words, stanza_tokens)
        for name in ANALYZER_NAMES[1:]:
            aligned[name] = align_to_base_tokens(base_words, precomputed_batches[name][sentence_index])
        return aligned

    def _build_row(
        self,
        doc: GoldDocument,
        gold_token: GoldToken,
        sent_index: int,
        token_index: int,
        sentence_text: str,
        aligned_map: dict[str, list[AlignedTokenCandidate]],
    ) -> dict[str, Any]:
        index = token_index - 1
        row: dict[str, Any] = {
            "path": doc.path,
            "sent_id": sent_index,
            "token_id": token_index,
            "sentence_text": sentence_text,
            "surface": gold_token.surface,
            "gold_lemma": gold_token.gold_lemma,
            "gold_lemma_norm": normalize_lemma_for_vote(gold_token.gold_lemma),
            "gold_tag_raw": gold_token.gold_tag_raw,
            "gold_pos_rnc": gold_token.gold_pos_rnc,
            "gold_pos_ud": gold_token.gold_pos_ud,
            "gold_pos_eval": normalize_eval_pos(gold_token.gold_pos_ud),
            "gold_pos_eval_relaxed": normalize_eval_pos_relaxed(gold_token.gold_pos_ud),
            "excluded_from_scoring": gold_token.excluded_from_scoring,
        }
        pos_candidates: dict[str, str] = {}
        lemma_candidates: dict[str, str] = {}
        feats_candidates: dict[str, str] = {}
        for analyzer_name in ANALYZER_NAMES:
            candidate = aligned_map[analyzer_name][index] if index < len(aligned_map[analyzer_name]) else AlignedTokenCandidate(None, False, "not_aligned")
            token = candidate.token
            pos_value = normalize_eval_pos(token.pos if token else "")
            pos_value_relaxed = normalize_eval_pos_relaxed(token.pos if token else "")
            lemma_value = token.lemma if token else ""
            row[f"{analyzer_name}_token"] = token.surface if token else ""
            row[f"{analyzer_name}_lemma"] = lemma_value
            row[f"{analyzer_name}_pos"] = pos_value
            row[f"{analyzer_name}_pos_relaxed"] = pos_value_relaxed
            row[f"{analyzer_name}_token_found"] = candidate.token_found
            row[f"{analyzer_name}_alignment_info"] = candidate.alignment_info
            if token:
                pos_candidates[analyzer_name] = pos_value
                lemma_candidates[analyzer_name] = lemma_value
                if analyzer_name in {"deeppavlov", "stanza", "udpipe"} and token.feats:
                    feats_candidates[analyzer_name] = token.feats
            row[f"{analyzer_name}_pos_match"] = (pos_value == row["gold_pos_eval"]) and not gold_token.excluded_from_scoring
            row[f"{analyzer_name}_pos_match_relaxed"] = (pos_value_relaxed == row["gold_pos_eval_relaxed"]) and not gold_token.excluded_from_scoring
            row[f"{analyzer_name}_lemma_match"] = (normalize_lemma_for_vote(lemma_value) == row["gold_lemma_norm"]) and not gold_token.excluded_from_scoring

        pwc_pos, pwc_pos_conf = calculate_pos_consensus(pos_candidates, self.pos_weights, self.analyzer_performance)
        pwc_lemma_raw, pwc_lemma_conf = calculate_lemma_consensus(
            lemma_candidates,
            pos_candidates,
            self.lemma_weights,
            self.analyzer_performance,
            estimated_pos=pwc_pos,
        )
        pwc_feats, _ = calculate_feats_consensus(feats_candidates)
        pwc_processed = process_token(
            surface=gold_token.surface,
            lemma=pwc_lemma_raw or gold_token.surface,
            upos=(pwc_pos or "X").replace("CONJ", "CCONJ"),
            feats=pwc_feats,
        )
        row["pwc_pos"] = normalize_eval_pos(pwc_pos or "")
        row["pwc_pos_relaxed"] = normalize_eval_pos_relaxed(pwc_pos or "")
        row["pwc_lemma"] = pwc_processed.lemma_raw
        row["pwc_pos_confidence"] = pwc_pos_conf
        row["pwc_lemma_confidence"] = pwc_lemma_conf
        row["pwc_pos_match"] = (row["pwc_pos"] == row["gold_pos_eval"]) and not gold_token.excluded_from_scoring
        row["pwc_pos_match_relaxed"] = (row["pwc_pos_relaxed"] == row["gold_pos_eval_relaxed"]) and not gold_token.excluded_from_scoring
        row["pwc_lemma_match"] = (normalize_lemma_for_vote(row["pwc_lemma"]) == row["gold_lemma_norm"]) and not gold_token.excluded_from_scoring

        simple_pos, simple_lemma, simple_pos_conf, simple_lemma_conf = calculate_simple_consensus(pos_candidates, lemma_candidates, self.analyzer_performance)
        row["simple_consensus_pos"] = normalize_eval_pos(simple_pos or "")
        row["simple_consensus_pos_relaxed"] = normalize_eval_pos_relaxed(simple_pos or "")
        row["simple_consensus_lemma"] = simple_lemma or ""
        row["simple_consensus_pos_confidence"] = simple_pos_conf
        row["simple_consensus_lemma_confidence"] = simple_lemma_conf
        row["simple_consensus_pos_match"] = (row["simple_consensus_pos"] == row["gold_pos_eval"]) and not gold_token.excluded_from_scoring
        row["simple_consensus_pos_match_relaxed"] = (row["simple_consensus_pos_relaxed"] == row["gold_pos_eval_relaxed"]) and not gold_token.excluded_from_scoring
        row["simple_consensus_lemma_match"] = (normalize_lemma_for_vote(row["simple_consensus_lemma"]) == row["gold_lemma_norm"]) and not gold_token.excluded_from_scoring
        return row


def load_gold_documents(texts_root: str | Path, *, limit_docs: int | None = None) -> list[GoldDocument]:
    root = Path(texts_root)
    files = sorted(root.rglob("*_disamb.xhtml"))
    if limit_docs is not None and limit_docs > 0:
        files = files[:limit_docs]
    return [parse_gold_document(path, doc_path=str(path.relative_to(root)).replace("\\", "/")) for path in files]


def save_eval_outputs(output_root: str | Path, detailed_rows: list[dict[str, Any]], metrics_rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "predictions_detailed.csv", detailed_rows)
    _write_csv(out / "metrics.csv", metrics_rows)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# RNC Evaluation Report",
        "",
        f"- Documents: {report.get('documents_total', 0)}",
        f"- Tokens: {report.get('tokens_total', 0)}",
        f"- Scoring mode: {report.get('scoring_mode', '')}",
        "",
        "| system | pos_accuracy | pos_accuracy_rnc_relaxed | lemma_accuracy | scored_tokens | excluded_tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(metrics_rows, key=lambda item: (-float(item["lemma_accuracy"]), item["system"])):
        lines.append(
            f"| {row['system']} | {float(row['pos_accuracy']):.4f} | {float(row['pos_accuracy_rnc_relaxed']):.4f} | {float(row['lemma_accuracy']):.4f} | {int(row['scored_tokens_lemma'])} | {int(row['excluded_tokens'])} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
