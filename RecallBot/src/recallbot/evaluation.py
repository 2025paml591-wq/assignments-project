import csv
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .call_log import CallLogger
from .config import Paths, VARIANT
from .embeddings import Embedder
from .retrieval import KnowledgeIndex, parse_document, terms

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "if", "in", "into", "is", "it", "may", "my", "of", "on",
    "or", "should", "that", "the", "this", "to", "under", "what", "when", "which",
    "with", "would",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_assignment_data(paths: Paths) -> list[str]:
    issues: list[str] = []
    docs = sorted(paths.kb.glob("*.md"))
    parsed_docs = [parse_document(path.read_text(encoding="utf-8")) for path in docs]
    word_count = sum(len(body.split()) for _, body in parsed_docs)
    doc_ids = set()
    versions: defaultdict[str, set[str]] = defaultdict(set)
    codes = set()
    bodies = {}
    for metadata, body in parsed_docs:
        doc_ids.add(metadata["doc_id"])
        versions[metadata["doc_id"]].add(metadata["version"])
        bodies[(metadata["doc_id"], metadata["version"])] = body
        codes.update(re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b", body))
    if len(doc_ids) < 15:
        issues.append(f"Only {len(doc_ids)} unique knowledge-base document IDs.")
    if not 4000 <= word_count <= 8000:
        issues.append(f"Knowledge base has {word_count} words; required range is 4,000-8,000.")
    if len(codes) < 10:
        issues.append(f"Only {len(codes)} distinct domain codes or IDs were found.")
    if not any(len(versions[doc_id]) >= 2 for doc_id in versions):
        issues.append("At least one document must have both an old and a new version.")

    conversation = _read_json(paths.data / "test_conversation.json")
    turns = conversation["turns"]
    session_a = [row for row in turns if row["session"] == "A"]
    session_b = [row for row in turns if row["session"] == "B"]
    if len(turns) < 40 or len(session_a) != 25 or not session_b:
        issues.append("Conversation must have at least 40 turns, with 25 in A and the remainder in B.")
    if len(conversation.get("facts", [])) < 8:
        issues.append("At least eight tagged facts are required.")
    if sum(bool(row["code_mixed"]) for row in turns) / max(1, len(turns)) < 0.30:
        issues.append("Fewer than 30% of scripted turns are tagged code-mixed.")
    if sum(
        any(word in row["text"].lower() for word in ("thanks", "thank you", "hello again"))
        for row in turns
    ) < 5:
        issues.append("Add at least five low-importance turns such as small talk or thanks.")
    updates = [fact for fact in conversation["facts"] if fact.get("updates_turn")]
    if len(updates) < 2:
        issues.append("At least two tagged fact updates are required.")
    if not any("forget" in row["text"].lower() for row in turns):
        issues.append("The scripted conversation needs an explicit forget request.")

    questions = _read_json(paths.data / "kb_questions.json")
    if len(questions) < 25:
        issues.append("At least 25 knowledge-base questions are required.")
    for category, minimum in (("code", 5), ("paraphrase", 5), ("version_new", 2)):
        if sum(row["type"] == category for row in questions) < minimum:
            issues.append(f"Fewer than {minimum} {category} questions.")
    paraphrases = []
    for row in questions:
        body = bodies.get((row["doc_id"], row["version"]), "")
        if row["gold_span"] not in body:
            issues.append(f"{row['id']} gold answer span is not present in its cited document version.")
        if row["type"] == "paraphrase":
            q_terms = {word for word in terms(row["question"]) if word not in STOPWORDS}
            a_terms = {word for word in terms(row["gold_span"]) if word not in STOPWORDS}
            if not q_terms & a_terms:
                paraphrases.append(row["id"])
    if len(paraphrases) < 5:
        issues.append(f"Only {len(paraphrases)} paraphrases have no shared content keyword.")

    probes = _read_json(paths.data / "probes.json")
    categories = {row["category"] for row in probes}
    if len(probes) < 14 or not {"R", "U", "F", "E", "K", "N", "S", "D"} <= categories:
        issues.append("Probes must cover R/U/F/E/K/N/S and include two code-mixed twist cases.")
    if sum(row["category"] == "D" for row in probes) < 2:
        issues.append("At least two code-mixed twist probes are required.")
    return issues


def evaluate_nlu(paths: Paths) -> dict[str, Any]:
    from .nlu import classify

    rows = _read_json(paths.data / "nlu_test.json")
    predictions = []
    for row in rows:
        actual = classify(row["text"]).intent
        predictions.append({**row, "predicted_intent": actual, "correct": actual == row["expected_intent"]})
    correct = sum(row["correct"] for row in predictions)
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / max(1, len(rows)),
        "predictions": predictions,
    }


def evaluate_retrieval(
    *,
    paths: Paths,
    embedder: Embedder,
    logger: CallLogger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions = _read_json(paths.data / "kb_questions.json")
    hit_rows: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    for size in (128, 512):
        index_path = paths.index / f"chunks_{size}.json"
        index = KnowledgeIndex.load(index_path)
        vectors = embedder.embed(
            [row["question"] for row in questions],
            logger=logger,
            config="hybrid",
            user="U1",
            session="kb-eval",
            turn=size,
        )
        for question, vector in zip(questions, vectors):
            question_rankings = {}
            for method in ("dense", "bm25", "hybrid"):
                hits = index.search(
                    question["question"],
                    method=method,
                    limit=5,
                    query_vector=vector,
                )
                question_rankings[method] = [
                    {
                        "chunk_id": hit["chunk"].chunk_id,
                        "doc_id": hit["chunk"].doc_id,
                        "version": hit["chunk"].version,
                        "score": hit["score"],
                    }
                    for hit in hits
                ]
                for k in (1, 3, 5):
                    matching = [
                        hit
                        for hit in hits[:k]
                        if hit["chunk"].doc_id == question["doc_id"]
                        and hit["chunk"].version == question["version"]
                    ]
                    hit_rows.append(
                        {
                            "method": method,
                            "chunk_size": size,
                            "k": k,
                            "question_id": question["id"],
                            "question_type": question["type"],
                            "hit": int(bool(matching)),
                        }
                    )
            rankings.append(
                {
                    "chunk_size": size,
                    "question_id": question["id"],
                    "question": question["question"],
                    "gold_doc_id": question["doc_id"],
                    "gold_version": question["version"],
                    "rankings": question_rankings,
                }
            )
    return hit_rows, rankings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def log_summary(log_path: Path) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if not log_path.exists():
        raise FileNotFoundError(f"No runtime call log found at {log_path}.")
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                groups[(row["config"], row["call_type"])].append(row)
    result = []
    for (config, call_type), rows in sorted(groups.items()):
        input_tokens = [int(row["input_tokens"]) for row in rows]
        result.append(
            {
                "config": config,
                "call_type": call_type,
                "calls": len(rows),
                "input_tokens": sum(input_tokens),
                "output_tokens": sum(int(row["output_tokens"]) for row in rows),
                "retrieved_tokens": sum(int(row["retrieved_tokens"]) for row in rows),
                "median_latency_ms": statistics.median(int(row["latency_ms"]) for row in rows),
            }
        )
    return result


def plot_input_tokens(log_path: Path, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install evaluation extras to create plots.") from error
    rows = []
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["call_type"] in {"chat", "summary"}:
                rows.append(row)
    configs = sorted({row["config"] for row in rows if row["config"] in {"full", "trim", "summary"}})
    figure, axis = plt.subplots(figsize=(11, 5))
    for config in configs:
        selected = sorted(
            (row for row in rows if row["config"] == config),
            key=lambda row: (row["run_id"], row["session"], row["turn"], row["call_type"]),
        )
        axis.plot(
            range(1, len(selected) + 1),
            [row["input_tokens"] for row in selected],
            marker=".",
            label=config,
        )
    axis.set(xlabel="Logged model call within configuration", ylabel="Input tokens", title="RecallBot input tokens from calls.jsonl")
    axis.legend()
    axis.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def estimate_vector_storage(
    *, vectors: int, dimensions: int, hnsw_m: int = 16
) -> dict[str, int]:
    if min(vectors, dimensions, hnsw_m) <= 0:
        raise ValueError("Vector count, dimension, and M must be positive.")
    flat = vectors * dimensions * 4
    hnsw = vectors * (dimensions * 4 + hnsw_m * 2 * 4)
    return {"vectors": vectors, "dimensions": dimensions, "flat_bytes": flat, "hnsw_bytes_estimate": hnsw}


def code_mix_token_comparison(paths: Paths) -> list[dict[str, Any]]:
    from .tokens import tokenizer

    conversation = _read_json(paths.data / "test_conversation.json")
    rows = []
    for turn in conversation["turns"]:
        if not turn["code_mixed"]:
            continue
        mixed = tokenizer().count(turn["text"])
        english = tokenizer().count(turn["english"])
        rows.append(
            {
                "turn": turn["turn"],
                "session": turn["session"],
                "code_mixed_tokens": mixed,
                "english_tokens": english,
                "difference": mixed - english,
            }
        )
    return rows


def evaluate_fact_extraction(paths: Paths, *, config: str, user: str) -> dict[str, Any]:
    from .memory import MemoryStore

    conversation = _read_json(paths.data / "test_conversation.json")
    latest: dict[str, dict[str, Any]] = {}
    for fact in conversation["facts"]:
        if fact["source_turn"] <= 25:
            current = latest.get(fact["key"])
            if current is None or fact["source_turn"] > current["source_turn"]:
                latest[fact["key"]] = fact
    store = MemoryStore(paths.memory / config / user)
    observed = store.active_facts(user)
    matched = []
    false_positives = []
    observed_by_key = {row["key"]: row for row in observed}
    for key, expected in latest.items():
        actual = observed_by_key.get(key)
        if actual and (
            str(expected["value"]).lower() in str(actual["value"]).lower()
            or str(actual["value"]).lower() in str(expected["value"]).lower()
        ):
            matched.append(key)
    for row in observed:
        expected = latest.get(row["key"])
        if expected is None or not (
            str(expected["value"]).lower() in str(row["value"]).lower()
            or str(row["value"]).lower() in str(expected["value"]).lower()
        ):
            false_positives.append(row)
    precision = len(matched) / max(1, len(observed))
    recall = len(matched) / max(1, len(latest))
    return {
        "config": config,
        "user": user,
        "expected_fact_count": len(latest),
        "observed_active_fact_count": len(observed),
        "matched_keys": matched,
        "false_positives": false_positives,
        "precision": precision,
        "recall": recall,
        "source_turns": {key: latest[key]["source_turn"] for key in matched},
    }


def compare_episodic_weightings(
    paths: Paths,
    *,
    config: str,
    user: str,
    embedder: Embedder | None = None,
    logger: CallLogger | None = None,
) -> list[dict[str, Any]]:
    from .memory import MemoryStore

    probes = _read_json(paths.data / "probes.json")
    store = MemoryStore(paths.memory / config / user)
    weightings = {
        "balanced": (0.25, 0.35, 0.40),
        "recency_heavy": (0.50, 0.20, 0.30),
    }
    rows = []
    for probe in (row for row in probes if row["category"] == "E"):
        query_vector = None
        if embedder and logger:
            query_vector = embedder.embed(
                [probe["question"]],
                logger=logger,
                config=config,
                user=user,
                session="episode-eval",
                turn=0,
            )[0]
        for label, weights in weightings.items():
            ranked = store.recall_episodes(
                user,
                probe["question"],
                weights=weights,
                limit=5,
                query_vector=query_vector,
            )
            rows.append(
                {
                    "probe_id": probe["id"],
                    "weighting": label,
                    "weights": json.dumps(weights),
                    "expected": probe["expected"],
                    "ranked_memories": json.dumps(
                        [
                            {
                                "source_turn": row["source_turn"],
                                "kind": row["kind"],
                                "importance": row["importance"],
                                "text": row["text"],
                            }
                            for row in ranked
                        ],
                        ensure_ascii=False,
                    ),
                }
            )
    return rows


def benchmark_real_index(
    *,
    paths: Paths,
    chunk_size: int,
    embedder: Embedder,
    logger: CallLogger,
    min_vectors: int = 100_000,
    query_count: int = 100,
) -> dict[str, Any]:
    try:
        import faiss
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Install benchmark extras with `python -m pip install -e '.[benchmark,evaluation]'`."
        ) from error
    payload = _read_json(paths.index / f"chunks_{chunk_size}.json")
    chunks = payload["chunks"]
    vectors = np.asarray([row["vector"] for row in chunks], dtype=np.float32)
    if not len(vectors):
        raise ValueError("The selected knowledge index is empty.")
    questions = _read_json(paths.data / "kb_questions.json")
    query_texts = [row["question"] for row in questions[:query_count]]
    query_vectors = embedder.embed(
        query_texts,
        logger=logger,
        config="hybrid",
        user="U1",
        session="hnsw-kb-benchmark",
        turn=0,
    )
    queries = np.asarray(query_vectors, dtype=np.float32)
    dimension = vectors.shape[1]
    faiss.normalize_L2(vectors)
    faiss.normalize_L2(queries)
    count = max(min_vectors, len(vectors))
    pad_count = count - len(vectors)
    if pad_count:
        rng = np.random.default_rng(591)
        padding = rng.random((pad_count, dimension), dtype=np.float32)
        faiss.normalize_L2(padding)
        full_vectors = np.vstack([vectors, padding])
    else:
        full_vectors = vectors
    flat = faiss.IndexFlatIP(dimension)
    flat.add(full_vectors)
    started = time.perf_counter()
    _, exact = flat.search(queries, 5)
    flat_ms = (time.perf_counter() - started) * 1000 / len(queries)
    hnsw = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    started = time.perf_counter()
    hnsw.add(full_vectors)
    build_seconds = time.perf_counter() - started
    hnsw.hnsw.efSearch = 64
    started = time.perf_counter()
    _, approximate = hnsw.search(queries, 5)
    hnsw_ms = (time.perf_counter() - started) * 1000 / len(queries)
    recalls = [
        len(set(exact[row]) & set(approximate[row])) / 5
        for row in range(len(queries))
    ]
    return {
        "index_size": chunk_size,
        "real_kb_vectors": len(vectors),
        "padded_random_vectors": pad_count,
        "total_vectors": len(full_vectors),
        "dimensions": dimension,
        "query_count": len(queries),
        "M": 16,
        "ef_construction": 200,
        "ef_search": 64,
        "flat_ms_per_query": flat_ms,
        "hnsw_ms_per_query": hnsw_ms,
        "hnsw_build_seconds": build_seconds,
        "mean_hnsw_recall_at_5_vs_flat": statistics.mean(recalls),
        "query_ids": [row["id"] for row in questions[: len(queries)]],
    }


def estimate_monthly_cost(
    *,
    log_path: Path,
    trim_run_ids: set[str],
    hybrid_run_ids: set[str],
    daily_users: int,
    days: int,
    turns_per_conversation: int,
    chat_input_price_per_million: float,
    chat_output_price_per_million: float,
    cached_input_price_per_million: float,
    embedding_price_per_million: float,
    cache_min_tokens: int,
    system_prompt: str,
) -> dict[str, Any]:
    from .tokens import tokenizer

    if not log_path.exists():
        raise FileNotFoundError(f"No runtime call log found at {log_path}.")
    with log_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    fixed_prefix_tokens = tokenizer().count(system_prompt)
    result: dict[str, Any] = {
        "daily_users": daily_users,
        "days_per_month": days,
        "turns_per_conversation_assumption": turns_per_conversation,
        "fixed_prefix_tokens": fixed_prefix_tokens,
        "cache_min_tokens": cache_min_tokens,
        "prefix_reaches_cache_minimum": fixed_prefix_tokens >= cache_min_tokens,
        "configurations": {},
    }
    for config, run_ids in (("trim", trim_run_ids), ("hybrid_mem", hybrid_run_ids)):
        selected = [row for row in records if row["run_id"] in run_ids]
        if not selected:
            raise ValueError(f"No log rows found for {config} run IDs: {sorted(run_ids)}")
        model_calls = [row for row in selected if row["call_type"] in {"chat", "summary"}]
        embedding_calls = [row for row in selected if row["call_type"] == "embed"]
        turns = {
            (row["user"], row["session"], row["turn"])
            for row in selected
            if row["turn"] > 0
        }
        if not turns:
            raise ValueError(f"No measured turns found for {config}.")
        scale = turns_per_conversation / len(turns) * daily_users * days
        input_tokens = sum(int(row["input_tokens"]) for row in model_calls)
        output_tokens = sum(int(row["output_tokens"]) for row in model_calls)
        cached_prefix = (
            sum(
                min(fixed_prefix_tokens, int(row["input_tokens"]))
                for row in model_calls
                if row["call_type"] == "chat"
            )
            if fixed_prefix_tokens >= cache_min_tokens
            else 0
        )
        embedding_tokens = sum(int(row["input_tokens"]) for row in embedding_calls)
        monthly_input = input_tokens * scale
        monthly_output = output_tokens * scale
        monthly_cached = cached_prefix * scale
        monthly_embeddings = embedding_tokens * scale
        uncached_cost = (
            monthly_input * chat_input_price_per_million
            + monthly_output * chat_output_price_per_million
            + monthly_embeddings * embedding_price_per_million
        ) / 1_000_000
        cached_cost = (
            (monthly_input - monthly_cached) * chat_input_price_per_million
            + monthly_cached * cached_input_price_per_million
            + monthly_output * chat_output_price_per_million
            + monthly_embeddings * embedding_price_per_million
        ) / 1_000_000
        result["configurations"][config] = {
            "observed_turns": len(turns),
            "measured_chat_input_tokens": input_tokens,
            "measured_chat_output_tokens": output_tokens,
            "measured_embedding_input_tokens": embedding_tokens,
            "monthly_uncached_cost": uncached_cost,
            "monthly_with_prefix_caching_cost": cached_cost,
            "monthly_cached_prefix_tokens": monthly_cached,
        }
    return result


def benchmark_faiss(*, vectors: int = 100_000, dimensions: int = 384, queries: int = 100) -> dict[str, Any]:
    try:
        import faiss
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install benchmark extras with `python -m pip install -e '.[benchmark,evaluation]'`.") from error
    rng = np.random.default_rng(591)
    data = rng.random((vectors, dimensions), dtype=np.float32)
    query = rng.random((queries, dimensions), dtype=np.float32)
    faiss.normalize_L2(data)
    faiss.normalize_L2(query)
    exact = faiss.IndexFlatIP(dimensions)
    exact.add(data)
    started = time.perf_counter()
    exact_scores, exact_ids = exact.search(query, 10)
    flat_ms = (time.perf_counter() - started) * 1000 / queries
    approximate = faiss.IndexHNSWFlat(dimensions, 16, faiss.METRIC_INNER_PRODUCT)
    approximate.hnsw.efConstruction = 200
    started = time.perf_counter()
    approximate.add(data)
    build_seconds = time.perf_counter() - started
    approximate.hnsw.efSearch = 64
    started = time.perf_counter()
    _, approximate_ids = approximate.search(query, 10)
    hnsw_ms = (time.perf_counter() - started) * 1000 / queries
    recalls = [
        len(set(exact_ids[row]) & set(approximate_ids[row])) / 10 for row in range(queries)
    ]
    return {
        "vectors": vectors,
        "dimensions": dimensions,
        "queries": queries,
        "M": 16,
        "ef_construction": 200,
        "ef_search": 64,
        "flat_ms_per_query": flat_ms,
        "hnsw_ms_per_query": hnsw_ms,
        "hnsw_build_seconds": build_seconds,
        "mean_recall_at_10": statistics.mean(recalls),
    }
