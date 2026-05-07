"""Map UD POS tags to dictionary categories."""

from __future__ import annotations

SERVICE_UPOS = {"ADP", "CCONJ", "SCONJ", "PART", "AUX"}


def map_pos_to_dict(upos: str) -> str:
    upos = upos.upper()
    if upos == "NOUN":
        return "NOUN"
    if upos == "PROPN":
        return "PROPN_ABBREV"
    if upos == "VERB":
        return "VERB"
    if upos == "ADJ":
        return "ADJ"
    if upos == "ADV":
        return "ADV"
    if upos in {"PRON", "DET"}:
        return "PRON"
    if upos == "NUM":
        return "NUM"
    if upos in SERVICE_UPOS:
        return "SERVICE"
    if upos == "INTJ":
        return "INTJ"
    return "OTHER"
