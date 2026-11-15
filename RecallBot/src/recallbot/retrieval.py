import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .embeddings import Embedder, cosine
from .tokens import Tokenizer, tokenizer

WORD = re.compile(r"[a-z0-9]+")
DENSE_MIN_SCORE = 0.25
CODE_MIX = {
    "aamdani": "income",
    "bachat": "savings",
    "bajet": "budget",
    "bhool": "forget",
    "batao": "tell",
    "dikhaiye": "show",
    "kharch": "expense",
    "kiraya": "rent",
    "kitna": "how much",
    "kitni": "how much",
    "mera": "my",
    "meri": "my",
    "pichla": "previous",
    "pichli": "previous",
    "purana": "old",
    "rakam": "amount",
    "yaad": "remember",
    "ab": "now",
    "hai": "is",
    "kya": "what",
    "ke": "of",
    "liye": "for",
    "mein": "in",
    "nahi": "not",
    "rakho": "set",
    "shukriya": "thanks",
    "mujhe": "I",
    "chahiye": "want",
}


def normalize_query(text: str) -> str:
    words = re.findall(r"[\w'-]+", text.lower(), flags=re.UNICODE)
    return " ".join(CODE_MIX.get(word, word) for word in words)


def terms(text: str) -> list[str]:
    return WORD.findall(normalize_query(text).lower())


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    version: str
    valid_from: str
    valid_to: str | None
    text: str
    token_count: int
    vector: list[float] | None = None


class Chunker:
    def __init__(self, token_counter: Tokenizer | None = None) -> None:
        self.token_counter = token_counter or tokenizer()

    def chunk(
        self, text: str, *, size: int, overlap_ratio: float = 0.15
    ) -> list[tuple[str, int]]:
        if size <= 0 or not 0.10 <= overlap_ratio <= 0.20:
            raise ValueError("Use a positive chunk size and 10-20% overlap.")
        tokens = self.token_counter.encode(text)
        overlap = max(1, round(size * overlap_ratio))
        step = max(1, size - overlap)
        chunks: list[tuple[str, int]] = []
        cursor = 0
        while cursor < len(tokens):
            part = tokens[cursor : cursor + size]
            body = self.token_counter.decode(part).strip()
            chunks.append((body, self.token_counter.count(body)))
            if cursor + size >= len(tokens):
                break
            cursor += step
        return chunks

    def directory(self, directory: Path, size: int) -> list[Chunk]:
        records: list[Chunk] = []
        for path in sorted(directory.glob("*.md")):
            metadata, body = parse_document(path.read_text(encoding="utf-8"))
            for index, (text, count) in enumerate(self.chunk(body, size=size)):
                records.append(
                    Chunk(
                        chunk_id=f"{metadata['doc_id']}:{metadata['version']}:{size}:{index}",
                        doc_id=metadata["doc_id"],
                        section=metadata["section"],
                        version=metadata["version"],
                        valid_from=metadata["valid_from"],
                        valid_to=metadata.get("valid_to"),
                        text=text,
                        token_count=count,
                    )
                )
        return records


def parse_document(raw: str) -> tuple[dict[str, Any], str]:
    lines = raw.splitlines()
    metadata: dict[str, Any] = {}
    body_start = 0
    if lines and lines[0].strip() == "---":
        try:
            body_start = lines.index("---", 1) + 1
        except ValueError as error:
            raise ValueError("Unclosed YAML-style metadata block.") from error
        for line in lines[1 : body_start - 1]:
            key, separator, value = line.partition(":")
            if separator:
                value = value.strip()
                metadata[key.strip()] = None if value.lower() == "null" else value
    required = {"doc_id", "section", "version", "valid_from"}
    if required - metadata.keys():
        raise ValueError(f"Document metadata missing: {sorted(required - metadata.keys())}")
    return metadata, "\n".join(lines[body_start:]).strip()


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_terms = [Counter(terms(chunk.text)) for chunk in chunks]
        self.lengths = [sum(counts.values()) for counts in self.doc_terms]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.document_frequency: Counter[str] = Counter()
        for counts in self.doc_terms:
            self.document_frequency.update(counts.keys())

    @staticmethod
    def score_terms(
        query_terms: set[str],
        term_frequencies: Counter[str] | dict[str, int],
        *,
        document_length: int,
        average_length: float,
        total_documents: int,
        document_frequencies: Counter[str] | dict[str, int],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        if total_documents <= 0 or average_length <= 0:
            raise ValueError("BM25 needs a non-empty corpus with a positive average length.")
        score = 0.0
        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if not frequency:
                continue
            df = document_frequencies.get(term, 0)
            inverse = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * document_length / average_length
            )
            score += inverse * frequency * (k1 + 1) / denominator
        return score

    def search(self, query: str, limit: int = 10) -> list[tuple[Chunk, float]]:
        query_terms = set(terms(query))
        scored: list[tuple[Chunk, float]] = []
        total = len(self.chunks)
        for chunk, frequencies, length in zip(self.chunks, self.doc_terms, self.lengths):
            score = self.score_terms(
                query_terms,
                frequencies,
                document_length=length,
                average_length=self.average_length,
                total_documents=total,
                document_frequencies=self.document_frequency,
                k1=self.k1,
                b=self.b,
            )
            if score:
                scored.append((chunk, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))[:limit]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], *, k: int = 60, limit: int = 10
) -> list[tuple[str, float]]:
    if k <= 0:
        raise ValueError("RRF k must be positive.")
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


class KnowledgeIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25Index(chunks)
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def search(
        self,
        query: str,
        *,
        method: str = "hybrid",
        limit: int = 5,
        query_vector: list[float] | None = None,
        as_of: str = "2026-11-15",
    ) -> list[dict[str, Any]]:
        active = [
            chunk
            for chunk in self.chunks
            if chunk.valid_from <= as_of and (chunk.valid_to is None or chunk.valid_to >= as_of)
        ]
        bm25 = BM25Index(active)
        sparse = bm25.search(query, limit=max(limit * 4, 20))
        if method == "bm25":
            return [self._hit(chunk, score, "bm25") for chunk, score in sparse[:limit]]

        dense: list[tuple[Chunk, float]] = []
        if query_vector is not None:
            dense = sorted(
                (
                    (chunk, cosine(query_vector, chunk.vector))
                    for chunk in active
                    if chunk.vector is not None
                ),
                key=lambda item: (-item[1], item[0].chunk_id),
            )[: max(limit * 4, 20)]
            dense = [(chunk, score) for chunk, score in dense if score >= DENSE_MIN_SCORE]
        if method == "dense":
            return [self._hit(chunk, score, "dense") for chunk, score in dense[:limit]]
        if method != "hybrid":
            raise ValueError(f"Unknown retrieval method: {method}")
        if not dense and not sparse:
            return []
        ranks = [
            [chunk.chunk_id for chunk, _ in ranking]
            for ranking in (dense, sparse)
            if ranking
        ]
        fused = reciprocal_rank_fusion(ranks, limit=limit)
        return [
            self._hit(self.by_id[identifier], score, "hybrid")
            for identifier, score in fused
            if identifier in self.by_id
        ]

    @staticmethod
    def _hit(chunk: Chunk, score: float, method: str) -> dict[str, Any]:
        return {"chunk": chunk, "score": score, "method": method}

    def save(self, path: Path, *, chunk_size: int, model: str, pooling: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_size": chunk_size,
            "overlap_ratio": 0.15,
            "embedding_model": model,
            "pooling": pooling,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk(**row) for row in payload["chunks"]]
        if not chunks:
            raise ValueError(f"Knowledge index is empty: {path}")
        return cls(chunks)
