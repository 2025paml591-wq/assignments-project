from pathlib import Path

from .call_log import CallLogger
from .config import Paths
from .embeddings import OllamaEmbedder, SentenceTransformerEmbedder
from .retrieval import Chunker, KnowledgeIndex


def create_embedder(backend: str, model: str, base_url: str):
    if backend == "ollama":
        return OllamaEmbedder(model=model, base_url=base_url)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model=model)
    raise ValueError(f"Unknown embedding backend: {backend}")


def build_indices(
    *,
    paths: Paths,
    backend: str,
    model: str,
    base_url: str,
    logger: CallLogger,
) -> list[Path]:
    embedder = create_embedder(backend, model, base_url)
    chunker = Chunker()
    outputs = []
    for size in (128, 512):
        chunks = chunker.directory(paths.kb, size)
        if not chunks:
            raise RuntimeError(f"No knowledge documents found in {paths.kb}.")
        vectors = embedder.embed(
            [chunk.text for chunk in chunks],
            logger=logger,
            config="hybrid",
            user="U1",
            session="index",
            turn=0,
        )
        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector
        path = paths.index / f"chunks_{size}.json"
        KnowledgeIndex(chunks).save(
            path,
            chunk_size=size,
            model=embedder.model,
            pooling="SentenceTransformer mean pooling (model default)"
            if backend == "sentence-transformers"
            else "Ollama model output; provider-defined pooling",
        )
        outputs.append(path)
    return outputs


def load_embedder_for_index(
    paths: Paths, size: int, backend: str, model: str, base_url: str
):
    path = paths.index / f"chunks_{size}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing index {path}. Run `recallbot build-index` first.")
    return create_embedder(backend, model, base_url)
