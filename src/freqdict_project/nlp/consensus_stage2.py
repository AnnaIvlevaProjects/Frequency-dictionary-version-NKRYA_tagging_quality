"""Consensus-based Stage 2 pipeline built on six analyzers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from freqdict_project.corpus.xml_reader import extract_clean_text
from freqdict_project.nlp.consensus_utils import (
    AlignedTokenCandidate,
    AnalyzerToken,
    adapt_pos_for_analyzer,
    align_to_base_tokens,
    calculate_feats_consensus,
    calculate_lemma_consensus,
    calculate_pos_consensus,
    canonicalize_feats,
    load_accuracy_weights,
)
from freqdict_project.nlp.morph_postprocess import process_token
from freqdict_project.nlp.stanza_pipeline import get_stanza_pipeline


@dataclass(slots=True)
class SentenceBatch:
    text: str
    tokens: list[AnalyzerToken]


def sanitize_runtime_text(value: str) -> str:
    cleaned_chars: list[str] = []
    for char in str(value):
        codepoint = ord(char)
        if codepoint == 0 or 0xD800 <= codepoint <= 0xDFFF:
            cleaned_chars.append(" ")
            continue
        cleaned_chars.append(char)
    return "".join(cleaned_chars)


class StanzaBaselineAdapter:
    name = "stanza"

    def __init__(self, *, language: str = "ru", processors: str = "tokenize,pos,lemma") -> None:
        self.pipeline = get_stanza_pipeline(language=language, processors=processors)

    def analyze_document(self, text: str) -> list[SentenceBatch]:
        doc = self.pipeline(text)
        out: list[SentenceBatch] = []
        for sentence in doc.sentences:
            words = list(sentence.words)
            sent_text = getattr(sentence, "text", "") or " ".join(word.text for word in words)
            tokens = [
                AnalyzerToken(
                    surface=word.text,
                    lemma=word.lemma or word.text,
                    pos=adapt_pos_for_analyzer(word.upos or "X", self.name),
                    feats=canonicalize_feats(word.feats),
                    raw_pos=word.upos or "X",
                )
                for word in words
            ]
            out.append(SentenceBatch(text=sent_text, tokens=tokens))
        return out


class NatashaAdapter:
    name = "natasha"

    def __init__(self) -> None:
        from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter

        self.doc_cls = Doc
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        embedding = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(embedding)

    def analyze_sentence(self, text: str) -> list[AnalyzerToken]:
        doc = self.doc_cls(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        return [
            AnalyzerToken(
                surface=token.text,
                lemma=getattr(token, "lemma", token.text),
                pos=adapt_pos_for_analyzer(getattr(token, "pos", "X"), self.name),
                feats="",
                raw_pos=getattr(token, "pos", "X"),
            )
            for token in doc.tokens
        ]


class SpacyAdapter:
    name = "spacy"

    def __init__(self) -> None:
        import spacy

        for model_name in ("ru_core_news_lg", "ru_core_news_md", "ru_core_news_sm"):
            try:
                self.nlp = spacy.load(model_name)
                break
            except OSError:
                continue
        else:  # pragma: no cover - runtime dependency path
            raise ModuleNotFoundError("No Russian spaCy model found. Install ru_core_news_lg/md/sm.")

    def analyze_sentence(self, text: str) -> list[AnalyzerToken]:
        doc = self.nlp(text)
        return [
            AnalyzerToken(
                surface=token.text,
                lemma=token.lemma_,
                pos=adapt_pos_for_analyzer(token.pos_, self.name),
                feats="",
                raw_pos=token.pos_,
            )
            for token in doc
        ]


class PymystemAdapter:
    name = "pymystem"

    POS_MAPPING = {
        "SPRO": "PRON",
        "A": "ADJ",
        "V": "VERB",
        "AUX": "AUX",
        "ADV": "ADV",
        "ADVPRO": "ADV",
        "NUM": "NUM",
        "ANUM": "ADJ",
        "CONJ": "CONJ",
        "PR": "ADP",
        "PART": "PART",
        "INTJ": "INTJ",
        "APRO": "DET",
        "NONLEX": "X",
        "INIT": "X",
        "UNKN": "X",
        "PUNCT": "PUNCT",
    }

    def __init__(self) -> None:
        from pymystem3 import Mystem

        self.mystem = Mystem()

    def analyze_sentence(self, text: str) -> list[AnalyzerToken]:
        items = [item for item in self.mystem.analyze(text) if isinstance(item, dict) and "text" in item]
        return [
            AnalyzerToken(
                surface=item.get("text", ""),
                lemma=self._extract_lemma(item),
                pos=self._extract_pos(item),
                feats="",
                raw_pos=self._extract_raw_pos(item),
            )
            for item in items
        ]

    def _extract_raw_pos(self, item: dict[str, Any]) -> str:
        if not item.get("analysis"):
            return "X"
        return str(item["analysis"][0].get("gr", "X"))

    def _extract_pos(self, item: dict[str, Any]) -> str:
        if not item.get("analysis"):
            return "X"
        gram = str(item["analysis"][0].get("gr", ""))
        if "=" in gram:
            pos_tag = gram.split("=", 1)[0]
        elif "," in gram:
            pos_tag = gram.split(",", 1)[0]
        else:
            pos_tag = gram
        if "," in pos_tag:
            pos_tag = pos_tag.split(",", 1)[0]

        if pos_tag == "S":
            if any(marker in gram for marker in ("имя", "фам", "отч", "гео", "орг", "аббр")):
                return "PROPN"
            return "NOUN"

        mapped = self.POS_MAPPING.get(pos_tag, "X")
        return adapt_pos_for_analyzer(mapped, self.name)

    def _extract_lemma(self, item: dict[str, Any]) -> str:
        if not item.get("analysis"):
            return str(item.get("text", ""))
        return str(item["analysis"][0].get("lex", item.get("text", "")))


class UDPipeAdapter:
    name = "udpipe"

    def __init__(self, model_path: str | Path) -> None:
        from ufal.udpipe import Model, Pipeline

        model = Model.load(str(model_path))
        if not model:  # pragma: no cover - runtime dependency path
            raise FileNotFoundError(f"Failed to load UDPipe model: {model_path}")
        self.model = model
        self.pipeline = Pipeline(self.model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

    def analyze_sentence(self, text: str) -> list[AnalyzerToken]:
        processed = self.pipeline.process(text)
        tokens: list[AnalyzerToken] = []
        for line in processed.strip().splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6 or "-" in parts[0]:
                continue
            tokens.append(
                AnalyzerToken(
                    surface=parts[1],
                    lemma=parts[2],
                    pos=adapt_pos_for_analyzer(parts[3], self.name),
                    feats=canonicalize_feats(parts[5]),
                    raw_pos=parts[3],
                )
            )
        return tokens


class UDPipeWorkerClient:
    name = "udpipe"

    def __init__(self, python_executable: str | Path, worker_script: str | Path, model_path: str | Path) -> None:
        command = [str(python_executable), str(worker_script), "--model", str(model_path)]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def analyze_sentence(self, text: str) -> list[AnalyzerToken]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("UDPipe worker pipes are unavailable.")
        self._ensure_running()
        self.process.stdin.write(json.dumps({"type": "analyze", "text": text}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            raise RuntimeError(f"UDPipe worker produced no response. stderr={stderr}")
        payload = json.loads(line)
        if not payload.get("ok", False):
            debug = payload.get("debug")
            if debug:
                raise RuntimeError(f"UDPipe worker error: {payload.get('error', 'unknown error')} | debug={debug}")
            raise RuntimeError(f"UDPipe worker error: {payload.get('error', 'unknown error')}")
        return [
            AnalyzerToken(
                surface=str(item.get("surface", "")),
                lemma=str(item.get("lemma", "")),
                pos=adapt_pos_for_analyzer(str(item.get("pos", "X")), self.name),
                feats=canonicalize_feats(item.get("feats", "")),
                raw_pos=str(item.get("raw_pos", item.get("pos", "X"))),
            )
            for item in payload.get("tokens", [])
        ]

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                self.process.stdin.flush()
            except OSError:
                pass
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive runtime path
            self.process.kill()

    def _ensure_running(self) -> None:
        if self.process.poll() is not None:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            raise RuntimeError(f"UDPipe worker is not running. stderr={stderr}")


class DeepPavlovWorkerClient:
    name = "deeppavlov"

    def __init__(self, python_executable: str | Path, worker_script: str | Path) -> None:
        command = [str(python_executable), str(worker_script)]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            env=env,
        )
        ready = self._read_json_message(expect_ready=True)
        if not ready.get("ok", False):
            raise RuntimeError(f"DeepPavlov worker failed to start: {ready.get('error', 'unknown error')}")

    def analyze_sentence(self, text: str) -> list[AnalyzerToken]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("DeepPavlov worker pipes are unavailable.")
        self._ensure_running()
        request_bytes = (json.dumps({"type": "analyze", "text": text}, ensure_ascii=False) + "\n").encode("utf-8")
        self.process.stdin.write(request_bytes)
        self.process.stdin.flush()
        payload = self._read_json_message()
        if not payload.get("ok", False):
            error_text = str(payload.get("error", "unknown error"))
            if "shouldn't exceed 512 tokens" in error_text:
                return []
            raise RuntimeError(f"DeepPavlov worker error: {error_text}")
        return [
            AnalyzerToken(
                surface=str(item.get("surface", "")),
                lemma=str(item.get("lemma", "")),
                pos=adapt_pos_for_analyzer(str(item.get("pos", "X")), self.name),
                feats=canonicalize_feats(item.get("feats", "")),
                raw_pos=str(item.get("raw_pos", item.get("pos", "X"))),
            )
            for item in payload.get("tokens", [])
        ]

    def analyze_sentences(self, texts: list[str]) -> list[list[AnalyzerToken]]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("DeepPavlov worker pipes are unavailable.")
        if not texts:
            return []
        self._ensure_running()
        request_bytes = (json.dumps({"type": "analyze_batch", "texts": texts}, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
        self.process.stdin.write(request_bytes)
        self.process.stdin.flush()
        payload = self._read_json_message()
        if not payload.get("ok", False):
            error_text = str(payload.get("error", "unknown error"))
            if "shouldn't exceed 512 tokens" in error_text:
                return [[] for _ in texts]
            raise RuntimeError(f"DeepPavlov worker error: {error_text}")
        result: list[list[AnalyzerToken]] = []
        for batch in payload.get("tokens_batch", []):
            result.append(
                [
                    AnalyzerToken(
                        surface=str(item.get("surface", "")),
                        lemma=str(item.get("lemma", "")),
                        pos=adapt_pos_for_analyzer(str(item.get("pos", "X")), self.name),
                        feats=canonicalize_feats(item.get("feats", "")),
                        raw_pos=str(item.get("raw_pos", item.get("pos", "X"))),
                    )
                    for item in batch
                ]
            )
        return result

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.write((json.dumps({"type": "shutdown"}) + "\n").encode("utf-8"))
                self.process.stdin.flush()
            except OSError:
                pass
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive runtime path
            self.process.kill()

    def _ensure_running(self) -> None:
        if self.process.poll() is not None:
            stderr = self._read_stderr_text()
            raise RuntimeError(f"DeepPavlov worker is not running. stderr={stderr}")

    def _read_json_message(self, *, expect_ready: bool = False) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("DeepPavlov worker stdout is unavailable.")
        noise_lines: list[str] = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                stderr = self._read_stderr_text()
                noise = " | ".join(noise_lines[-5:])
                raise RuntimeError(
                    f"DeepPavlov worker produced no JSON response. noise={noise} stderr={stderr}"
                )
            stripped = line.decode("utf-8", errors="replace").strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                noise_lines.append(stripped)
                continue
            if expect_ready and payload.get("type") != "ready":
                noise_lines.append(stripped)
                continue
            if not expect_ready and payload.get("type") == "ready":
                continue
            return payload

    def _read_stderr_text(self) -> str:
        if self.process.stderr is None:
            return ""
        data = self.process.stderr.read()
        if not data:
            return ""
        return data.decode("utf-8", errors="replace")


class ConsensusStage2Engine:
    analyzer_names = ("stanza", "natasha", "spacy", "pymystem", "udpipe", "deeppavlov")

    def __init__(
        self,
        *,
        accuracy_path: str | Path,
        udpipe_model_path: str | Path,
        udpipe_python: str | Path,
        udpipe_worker_script: str | Path,
        deeppavlov_python39: str | Path,
        deeppavlov_worker_script: str | Path,
        language: str = "ru",
        processors: str = "tokenize,pos,lemma",
    ) -> None:
        self._udpipe_python = Path(udpipe_python)
        self._udpipe_model_path = Path(udpipe_model_path)
        self._udpipe_worker_script = Path(udpipe_worker_script)
        self.pos_weights, self.lemma_weights, self.analyzer_performance = load_accuracy_weights(accuracy_path)
        self.stanza = StanzaBaselineAdapter(language=language, processors=processors)
        natasha = NatashaAdapter()
        spacy = SpacyAdapter()
        pymystem = PymystemAdapter()
        deeppavlov = DeepPavlovWorkerClient(deeppavlov_python39, deeppavlov_worker_script)
        self.sentence_analyzers = {
            "natasha": natasha,
            "spacy": spacy,
            "pymystem": pymystem,
            "udpipe": None,
            "deeppavlov": deeppavlov,
        }

    def analyze_document(self, row: dict[str, str]) -> dict[str, Any]:
        xml_path = row.get("xml_abs_path", "")
        text = extract_clean_text(xml_path)
        if not text:
            return {"status": "empty", "tokens": [], "raw_rows": [], "report": {"sentences": 0, "tokens": 0}}

        sentence_batches = self.stanza.analyze_document(text)
        precomputed_batches = self._analyze_document_sentences(sentence_batches)
        token_rows: list[dict[str, Any]] = []
        raw_rows: list[dict[str, Any]] = []

        for sent_index, sentence in enumerate(sentence_batches, start=1):
            aligned_map = self._collect_aligned_tokens(sentence, precomputed_batches, sent_index - 1)
            for token_index, stanza_candidate in enumerate(aligned_map["stanza"], start=1):
                stanza_token = stanza_candidate.token
                if stanza_token is None:
                    continue
                raw_row, final_row = self._build_rows_for_token(
                    row=row,
                    sentence_text=sentence.text,
                    sent_index=sent_index,
                    token_index=token_index,
                    aligned_map=aligned_map,
                    stanza_token=stanza_token,
                )
                raw_rows.append(raw_row)
                token_rows.append(final_row)

        report = {
            "sentences": len(sentence_batches),
            "tokens": len(token_rows),
            "raw_rows": len(raw_rows),
        }
        return {"status": "processed", "tokens": token_rows, "raw_rows": raw_rows, "report": report}

    def close(self) -> None:
        deeppavlov = self.sentence_analyzers.get("deeppavlov")
        if isinstance(deeppavlov, DeepPavlovWorkerClient):
            deeppavlov.close()

    def _analyze_document_sentences(self, sentence_batches: list[SentenceBatch]) -> dict[str, list[list[AnalyzerToken]]]:
        sentence_texts = [sanitize_runtime_text(sentence.text) for sentence in sentence_batches]
        base_lengths = [len(sentence.tokens) for sentence in sentence_batches]
        precomputed: dict[str, list[list[AnalyzerToken]]] = {
            "natasha": [self.sentence_analyzers["natasha"].analyze_sentence(text) for text in sentence_texts],
            "spacy": [self.sentence_analyzers["spacy"].analyze_sentence(text) for text in sentence_texts],
            "pymystem": [self.sentence_analyzers["pymystem"].analyze_sentence(text) for text in sentence_texts],
            "udpipe": self._run_udpipe_batch_isolated(sentence_texts),
            "deeppavlov": [[] for _ in sentence_batches],
        }
        deeppavlov_adapter = self.sentence_analyzers["deeppavlov"]
        safe_indices = [index for index, token_count in enumerate(base_lengths) if token_count <= 220]
        safe_texts = [sentence_texts[index] for index in safe_indices]
        if safe_texts and isinstance(deeppavlov_adapter, DeepPavlovWorkerClient):
            safe_batches = deeppavlov_adapter.analyze_sentences(safe_texts)
            for index, tokens in zip(safe_indices, safe_batches, strict=False):
                precomputed["deeppavlov"][index] = tokens
        return precomputed

    def _collect_aligned_tokens(
        self,
        sentence: SentenceBatch,
        precomputed_batches: dict[str, list[list[AnalyzerToken]]],
        sentence_index: int,
    ) -> dict[str, list[AlignedTokenCandidate]]:
        base_words = [token.surface for token in sentence.tokens]
        aligned: dict[str, list[AlignedTokenCandidate]] = {
            "stanza": [
                AlignedTokenCandidate(token=token, token_found=True, alignment_info=f"base_{index}")
                for index, token in enumerate(sentence.tokens)
            ]
        }
        for name in self.sentence_analyzers:
            tokens = precomputed_batches[name][sentence_index]
            aligned[name] = align_to_base_tokens(base_words, tokens)
        return aligned

    def _run_udpipe_batch_isolated(self, texts: list[str]) -> list[list[AnalyzerToken]]:
        """Run UDPipe once per document in a short-lived subprocess to avoid crashes and reduce overhead."""
        if not texts:
            return []
        inline_code = textwrap.dedent(
            f"""
            import json
            import sys
            from ufal.udpipe import Model, Pipeline, ProcessingError

            model = Model.load(r\"{self._udpipe_model_path.as_posix()}\")
            if not model:
                raise FileNotFoundError("Failed to load UDPipe model")
            pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
            raw = sys.stdin.buffer.read()
            payload = json.loads(raw.decode("utf-8"))
            texts = payload.get("texts", [])
            if not isinstance(texts, list):
                raise TypeError("texts must be a list")

            def canonicalize(value):
                if not value or value == "_":
                    return ""
                parts = [part.strip() for part in str(value).split("|") if part.strip()]
                return "|".join(sorted(parts))

            tokens_batch = []
            for text in texts:
                if not isinstance(text, str):
                    text = str(text)
                error = ProcessingError()
                processed = pipeline.process(text, error)
                if error.occurred():
                    raise RuntimeError(error.message)

                tokens = []
                for line in processed.splitlines():
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\\t")
                    if len(parts) < 6 or "-" in parts[0]:
                        continue
                    tokens.append(
                        {{
                            "surface": parts[1],
                            "lemma": parts[2],
                            "pos": parts[3],
                            "raw_pos": parts[3],
                            "feats": canonicalize(parts[5]),
                        }}
                    )
                tokens_batch.append(tokens)

            sys.stdout.buffer.write(
                json.dumps({{"ok": True, "tokens_batch": tokens_batch}}, ensure_ascii=False).encode("utf-8")
            )
            """
        )
        completed = subprocess.run(
            [str(self._udpipe_python), "-c", inline_code],
            input=json.dumps({"texts": texts}, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"UDPipe isolated run failed with code {completed.returncode}. stderr={completed.stderr.strip()}"
            )
        payload = json.loads(completed.stdout.strip())
        if not payload.get("ok", False):
            raise RuntimeError(f"UDPipe isolated worker error: {payload.get('error', 'unknown error')}")
        output: list[list[AnalyzerToken]] = []
        for batch in payload.get("tokens_batch", []):
            output.append(
                [
                    AnalyzerToken(
                        surface=str(item.get("surface", "")),
                        lemma=str(item.get("lemma", "")),
                        pos=adapt_pos_for_analyzer(str(item.get("pos", "X")), "udpipe"),
                        feats=canonicalize_feats(item.get("feats", "")),
                        raw_pos=str(item.get("raw_pos", item.get("pos", "X"))),
                    )
                    for item in batch
                ]
            )
        return output

    def _build_rows_for_token(
        self,
        *,
        row: dict[str, str],
        sentence_text: str,
        sent_index: int,
        token_index: int,
        aligned_map: dict[str, list[AlignedTokenCandidate]],
        stanza_token: AnalyzerToken,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        index = token_index - 1
        pos_candidates: dict[str, str] = {}
        lemma_candidates: dict[str, str] = {}
        feats_candidates: dict[str, str] = {}

        raw_row: dict[str, Any] = {
            "path": row.get("path", ""),
            "year": row.get("year", ""),
            "style_3": row.get("style_3", ""),
            "sent_id": sent_index,
            "token_id": token_index,
            "sentence_text": sentence_text,
            "surface": stanza_token.surface,
            "alignment_base": "stanza",
        }

        for analyzer_name, candidates in aligned_map.items():
            candidate = candidates[index] if index < len(candidates) else AlignedTokenCandidate(None, False, "not_aligned")
            token = candidate.token
            raw_row[f"{analyzer_name}_token"] = token.surface if token else ""
            raw_row[f"{analyzer_name}_lemma"] = token.lemma if token else ""
            raw_row[f"{analyzer_name}_pos"] = token.pos if token else ""
            raw_row[f"{analyzer_name}_raw_pos"] = token.raw_pos if token else ""
            raw_row[f"{analyzer_name}_feats"] = token.feats if token else ""
            raw_row[f"{analyzer_name}_token_found"] = candidate.token_found
            raw_row[f"{analyzer_name}_alignment_info"] = candidate.alignment_info

            if token:
                if token.pos:
                    pos_candidates[analyzer_name] = token.pos
                if token.lemma:
                    lemma_candidates[analyzer_name] = token.lemma
                if analyzer_name in {"deeppavlov", "stanza", "udpipe"} and token.feats:
                    feats_candidates[analyzer_name] = token.feats

        consensus_pos, pos_confidence = calculate_pos_consensus(
            pos_candidates,
            self.pos_weights,
            self.analyzer_performance,
        )
        consensus_lemma_raw, lemma_confidence = calculate_lemma_consensus(
            lemma_candidates,
            pos_candidates,
            self.lemma_weights,
            self.analyzer_performance,
            estimated_pos=consensus_pos,
        )
        consensus_feats, feats_source = calculate_feats_consensus(feats_candidates)

        final_pos = consensus_pos or stanza_token.pos or "X"
        final_lemma_raw = consensus_lemma_raw or stanza_token.lemma or stanza_token.surface
        processed = process_token(
            surface=stanza_token.surface,
            lemma=final_lemma_raw,
            upos=final_pos,
            feats=consensus_feats,
        )

        raw_row["consensus_pos"] = final_pos
        raw_row["consensus_lemma_raw"] = final_lemma_raw
        raw_row["consensus_feats"] = consensus_feats
        raw_row["consensus_pos_confidence"] = pos_confidence
        raw_row["consensus_lemma_confidence"] = lemma_confidence
        raw_row["feats_consensus_source"] = feats_source
        raw_row["estimated_pos_for_lemma"] = final_pos
        raw_row["available_analyzers_pos"] = len(pos_candidates)
        raw_row["available_analyzers_lemma"] = len(lemma_candidates)

        final_row = {
            "path": row.get("path", ""),
            "year": row.get("year", ""),
            "style_3": row.get("style_3", ""),
            "sent_id": sent_index,
            "token_id": token_index,
            "surface": stanza_token.surface,
            "lemma_raw": processed.lemma_raw,
            "lemma_norm": processed.lemma_norm,
            "lemma_display": processed.lemma_display,
            "pos_ud": processed.pos_ud,
            "pos_dict": processed.pos_dict,
            "feats": consensus_feats,
            "is_propn": processed.is_propn,
            "is_abbrev": processed.is_abbrev,
            "is_participle": processed.is_participle,
            "consensus_pos_confidence": pos_confidence,
            "consensus_lemma_confidence": lemma_confidence,
            "available_analyzers_pos": len(pos_candidates),
            "available_analyzers_lemma": len(lemma_candidates),
            "alignment_base": "stanza",
        }
        return raw_row, final_row


def build_consensus_engine_from_settings(settings: dict[str, Any], root: str | Path) -> ConsensusStage2Engine:
    root_path = Path(root)
    paths = settings.get("paths", {})
    nlp_settings = settings.get("nlp", {})

    accuracy_path = _resolve_required_path(paths.get("accuracy_weights_csv"), root_path, "paths.accuracy_weights_csv")
    udpipe_model_path = _resolve_required_path(paths.get("udpipe_model_path"), root_path, "paths.udpipe_model_path")
    udpipe_python_value = paths.get("udpipe_python") or sys.executable
    udpipe_python = _resolve_required_path(udpipe_python_value, root_path, "paths.udpipe_python")
    udpipe_worker_script_value = paths.get("udpipe_worker_script") or str(root_path / "scripts" / "udpipe_worker.py")
    udpipe_worker_script = _resolve_required_path(udpipe_worker_script_value, root_path, "paths.udpipe_worker_script")
    deeppavlov_python39 = _resolve_required_path(
        paths.get("deeppavlov_python39"),
        root_path,
        "paths.deeppavlov_python39",
    )
    worker_script_value = paths.get("deeppavlov_worker_script") or str(root_path / "scripts" / "deeppavlov_worker.py")
    deeppavlov_worker_script = _resolve_required_path(worker_script_value, root_path, "paths.deeppavlov_worker_script")

    return ConsensusStage2Engine(
        accuracy_path=accuracy_path,
        udpipe_model_path=udpipe_model_path,
        udpipe_python=udpipe_python,
        udpipe_worker_script=udpipe_worker_script,
        deeppavlov_python39=deeppavlov_python39,
        deeppavlov_worker_script=deeppavlov_worker_script,
        language=str(nlp_settings.get("language", "ru")),
        processors=str(nlp_settings.get("processors", "tokenize,pos,lemma")),
    )


def _resolve_required_path(value: Any, root_path: Path, setting_name: str) -> Path:
    if not value:
        raise ValueError(f"Missing required setting: {setting_name}")
    path = Path(str(value))
    if not path.is_absolute():
        path = root_path / path
    if not path.exists():
        raise FileNotFoundError(f"Configured path does not exist for {setting_name}: {path}")
    return path
