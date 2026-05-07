"""Stanza pipeline factory with one-time initialization per process."""

from __future__ import annotations

from functools import lru_cache


def ensure_stanza_resources(language: str = "ru") -> None:
    import stanza

    stanza.download(language)


@lru_cache(maxsize=1)
def get_stanza_pipeline(language: str = "ru", processors: str = "tokenize,pos,lemma"):
    import stanza

    return stanza.Pipeline(language, processors=processors, download_method=None)
