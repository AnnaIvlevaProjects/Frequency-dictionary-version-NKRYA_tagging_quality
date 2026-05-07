"""Utilities for token alignment and consensus voting across analyzers."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ANALYZER_PRIORITY = ("stanza", "udpipe", "deeppavlov", "natasha", "spacy", "pymystem")
FEATS_VOTING_ANALYZERS = ("deeppavlov", "stanza", "udpipe")


@dataclass(slots=True)
class AnalyzerToken:
    surface: str
    lemma: str
    pos: str
    feats: str = ""
    raw_pos: str = ""


@dataclass(slots=True)
class AlignedTokenCandidate:
    token: AnalyzerToken | None
    token_found: bool
    alignment_info: str


def _alignment_module():
    try:
        import spacy_alignments as tokenizations  # type: ignore
    except ModuleNotFoundError:
        import tokenizations  # type: ignore

    return tokenizations


def normalize_lemma_for_vote(value: str | None) -> str:
    return str(value or "").strip().lower().replace("ё", "е")


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
    if not pairs:
        return ""
    return "|".join(f"{key}={pairs[key]}" for key in sorted(pairs))


def load_accuracy_weights(
    path: str | Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    csv_path = Path(path)
    pos_weights: dict[str, dict[str, float]] = defaultdict(dict)
    lemma_weights: dict[str, dict[str, float]] = defaultdict(dict)
    analyzer_performance: dict[str, dict[str, float]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            analyzer = str(row.get("analyzer", "")).strip().lower()
            if not analyzer:
                continue

            analyzer_performance[analyzer] = {
                "overall_lemma": _to_float(row.get("overall_lemma"), default=0.5),
                "overall_pos": _to_float(row.get("overall_pos"), default=0.5),
            }

            for key, raw in row.items():
                if raw in (None, ""):
                    continue
                if key.startswith("pos_") and key != "overall_pos":
                    pos_weights[analyzer][key[4:]] = _to_float(raw, default=0.5)
                elif key.startswith("lemma_") and key != "overall_lemma":
                    lemma_weights[analyzer][key[6:]] = _to_float(raw, default=0.5)

    return dict(pos_weights), dict(lemma_weights), analyzer_performance


def adapt_pos_for_analyzer(pos_value: str | None, analyzer_name: str) -> str:
    pos = str(pos_value or "").strip().upper()
    analyzer = analyzer_name.strip().lower()
    if analyzer == "pymystem" and pos == "CONJ":
        return "CCONJ"
    if pos == "CONJ":
        return "CCONJ"
    return pos


def estimate_pos_simple(pos_candidates: dict[str, str]) -> tuple[str | None, float]:
    cleaned = {name: value for name, value in pos_candidates.items() if value}
    if not cleaned:
        return None, 0.0

    counter = Counter(cleaned.values())
    top_count = max(counter.values())
    winners = [value for value, count in counter.items() if count == top_count]
    winner = _break_tie_by_priority(cleaned, winners)
    confidence = top_count / max(1, len(cleaned))
    return winner, confidence


def calculate_pos_consensus(
    pos_candidates: dict[str, str],
    pos_weights: dict[str, dict[str, float]],
    analyzer_performance: dict[str, dict[str, float]],
) -> tuple[str | None, float]:
    cleaned = {name: adapt_pos_for_analyzer(value, name) for name, value in pos_candidates.items() if value}
    if not cleaned:
        return None, 0.0

    initial_pos, _ = estimate_pos_simple(cleaned)
    votes: dict[str, float] = defaultdict(float)
    total_weight = 0.0

    for analyzer, pos in cleaned.items():
        weight = 0.5
        if initial_pos:
            weight = pos_weights.get(analyzer, {}).get(initial_pos, weight)
        if weight == 0.5:
            weight = analyzer_performance.get(analyzer, {}).get("overall_pos", weight)
        votes[pos] += weight
        total_weight += weight

    if not votes:
        return None, 0.0

    top_weight = max(votes.values())
    winners = [value for value, weight in votes.items() if weight == top_weight]
    winner = _break_tie_by_priority(cleaned, winners)
    confidence = top_weight / total_weight if total_weight > 0 else 0.0
    return winner, confidence


def calculate_lemma_consensus(
    lemma_candidates: dict[str, str],
    pos_candidates: dict[str, str],
    lemma_weights: dict[str, dict[str, float]],
    analyzer_performance: dict[str, dict[str, float]],
    *,
    estimated_pos: str | None,
) -> tuple[str | None, float]:
    cleaned_lemmas = {name: str(value or "").strip() for name, value in lemma_candidates.items() if str(value or "").strip()}
    cleaned_pos = {name: adapt_pos_for_analyzer(value, name) for name, value in pos_candidates.items() if value}
    if not cleaned_lemmas:
        return None, 0.0

    if estimated_pos is None and cleaned_pos:
        estimated_pos, _ = estimate_pos_simple(cleaned_pos)

    votes: dict[str, float] = defaultdict(float)
    total_weight = 0.0
    lemma_forms: dict[str, dict[str, str]] = defaultdict(dict)

    for analyzer, lemma in cleaned_lemmas.items():
        key = normalize_lemma_for_vote(lemma)
        weight = 0.5
        if estimated_pos:
            weight = lemma_weights.get(analyzer, {}).get(estimated_pos, weight)
        if weight == 0.5:
            weight = analyzer_performance.get(analyzer, {}).get("overall_lemma", weight)
        votes[key] += weight
        total_weight += weight
        lemma_forms[key][analyzer] = lemma

    if not votes:
        return None, 0.0

    top_weight = max(votes.values())
    winners = [value for value, weight in votes.items() if weight == top_weight]
    preferred_key = _break_tie_by_priority(
        {analyzer: normalize_lemma_for_vote(lemma) for analyzer, lemma in cleaned_lemmas.items()},
        winners,
    )
    if preferred_key is None:
        return None, 0.0

    winner_forms = lemma_forms[preferred_key]
    raw_winner = _pick_raw_value_by_priority(winner_forms)
    confidence = top_weight / total_weight if total_weight > 0 else 0.0
    return raw_winner, confidence


def calculate_feats_consensus(feats_candidates: dict[str, str]) -> tuple[str, str]:
    cleaned = {
        analyzer: canonicalize_feats(value)
        for analyzer, value in feats_candidates.items()
        if analyzer in FEATS_VOTING_ANALYZERS and canonicalize_feats(value)
    }
    if not cleaned:
        return "", ""

    counter = Counter(cleaned.values())
    winner, count = counter.most_common(1)[0]
    if count >= 2:
        return winner, "majority"

    for analyzer in ("stanza", "udpipe", "deeppavlov"):
        if cleaned.get(analyzer):
            return cleaned[analyzer], analyzer

    return "", ""


def align_to_base_tokens(base_words: list[str], analyzer_tokens: list[AnalyzerToken]) -> list[AlignedTokenCandidate]:
    if not base_words:
        return []
    if not analyzer_tokens:
        return [AlignedTokenCandidate(token=None, token_found=False, alignment_info="not_aligned") for _ in base_words]

    tokenizations = _alignment_module()
    other_words = [token.surface for token in analyzer_tokens]
    try:
        base_to_other, _ = tokenizations.get_alignments(base_words, other_words)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        raise RuntimeError(f"Failed to align tokens: {exc}") from exc

    aligned: list[AlignedTokenCandidate] = []
    for base_index, _ in enumerate(base_words):
        indices = list(base_to_other[base_index]) if base_index < len(base_to_other) else []
        if not indices:
            aligned.append(AlignedTokenCandidate(token=None, token_found=False, alignment_info="not_aligned"))
            continue

        merged = merge_aligned_tokens(analyzer_tokens, indices)
        if len(indices) == 1:
            info = f"1-to-1_{indices[0]}"
        else:
            info = f"1-to-{len(indices)}"
        aligned.append(AlignedTokenCandidate(token=merged, token_found=True, alignment_info=info))
    return aligned


def merge_aligned_tokens(tokens: list[AnalyzerToken], indices: list[int]) -> AnalyzerToken:
    first = tokens[indices[0]]
    if len(indices) == 1:
        return first

    surfaces: list[str] = []
    lemmas: list[str] = []
    for index in indices:
        if index >= len(tokens):
            continue
        token = tokens[index]
        surfaces.append(token.surface)
        lemmas.append(token.lemma)

    return AnalyzerToken(
        surface="".join(surfaces).replace("##", ""),
        lemma="".join(lemmas).replace("-", "").replace("##", ""),
        pos=first.pos,
        feats=first.feats,
        raw_pos=first.raw_pos or first.pos,
    )


def _to_float(value: Any, *, default: float) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _break_tie_by_priority(candidate_map: dict[str, str], winners: list[str]) -> str | None:
    if not winners:
        return None
    winner_set = set(winners)
    for analyzer in ANALYZER_PRIORITY:
        value = candidate_map.get(analyzer)
        if value in winner_set:
            return value
    return sorted(winner_set)[0]


def _pick_raw_value_by_priority(raw_values: dict[str, str]) -> str:
    for analyzer in ANALYZER_PRIORITY:
        value = raw_values.get(analyzer)
        if value:
            return value
    return next(iter(raw_values.values()))
