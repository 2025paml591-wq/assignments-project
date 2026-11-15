import json
import math
import urllib.error
import urllib.request
from typing import Protocol

from .call_log import CallLogger
from .tokens import tokenizer


class Embedder(Protocol):
    model: str

    def embed(
        self,
        texts: list[str],
        *,
        logger: CallLogger,
        config: str,
        user: str,
        session: str,
        turn: int,
    ) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(
        self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(
        self,
        texts: list[str],
        *,
        logger: CallLogger,
        config: str,
        user: str,
        session: str,
        turn: int,
    ) -> list[list[float]]:
        started = logger.start()
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Embedding request failed for {self.model} at {self.base_url}: {error}"
            ) from error
        vectors = result.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("Ollama returned an unexpected embedding response.")
        logger.record(
            config=config,
            user=user,
            session=session,
            turn=turn,
            call_type="embed",
            model=self.model,
            input_tokens=sum(tokenizer().count(text) for text in texts),
            output_tokens=0,
            started=started,
        )
        return [[float(value) for value in vector] for vector in vectors]


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install the optional embedding dependencies with "
                "`python -m pip install -e '.[embeddings]'`."
            ) from error
        self.encoder = SentenceTransformer(model)

    def embed(
        self,
        texts: list[str],
        *,
        logger: CallLogger,
        config: str,
        user: str,
        session: str,
        turn: int,
    ) -> list[list[float]]:
        started = logger.start()
        vectors = self.encoder.encode(texts, normalize_embeddings=True)
        logger.record(
            config=config,
            user=user,
            session=session,
            turn=turn,
            call_type="embed",
            model=self.model,
            input_tokens=sum(tokenizer().count(text) for text in texts),
            output_tokens=0,
            started=started,
        )
        return [[float(value) for value in vector] for vector in vectors]


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions do not match.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
