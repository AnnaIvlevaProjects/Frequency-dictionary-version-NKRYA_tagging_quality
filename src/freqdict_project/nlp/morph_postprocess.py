"""Post-processing rules for lemma/POS normalization."""

from __future__ import annotations

from dataclasses import dataclass

from freqdict_project.nlp.pos_mapping import map_pos_to_dict


@dataclass(slots=True)
class ProcessedToken:
    lemma_raw: str
    lemma_norm: str
    lemma_display: str
    pos_ud: str
    pos_dict: str
    is_propn: bool
    is_abbrev: bool
    is_participle: bool


def parse_feats(feats: str | None) -> dict[str, str]:
    if not feats or feats == "_":
        return {}
    result: dict[str, str] = {}
    for item in feats.split("|"):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def normalize_lemma_yo(lemma: str) -> str:
    return lemma.replace("ё", "е").replace("Ё", "Е")


def detect_abbrev(surface: str, upos: str) -> bool:
    token = surface.strip()
    return upos == "PROPN" and token.isupper() and len(token) > 1


def process_token(surface: str, lemma: str, upos: str, feats: str | None) -> ProcessedToken:
    feat_map = parse_feats(feats)
    lemma_raw = lemma
    lemma_norm = normalize_lemma_yo(lemma_raw)
    is_participle = upos == "VERB" and feat_map.get("VerbForm") == "Part"

    if is_participle:
        lemma_display = f"{lemma_norm}*"
        pos_dict = "PARTICIPLE"
    else:
        lemma_display = lemma_norm
        pos_dict = map_pos_to_dict(upos)

    is_propn = upos == "PROPN"
    is_abbrev = detect_abbrev(surface=surface, upos=upos)

    return ProcessedToken(
        lemma_raw=lemma_raw,
        lemma_norm=lemma_norm,
        lemma_display=lemma_display,
        pos_ud=upos,
        pos_dict=pos_dict,
        is_propn=is_propn,
        is_abbrev=is_abbrev,
        is_participle=is_participle,
    )
