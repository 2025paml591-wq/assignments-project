import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .agent import RecallBotAgent
from .call_log import CallLogger
from .chat import OllamaChat, TemplateChat
from .config import CONFIGS, Paths, VARIANT
from .embeddings import OllamaEmbedder, SentenceTransformerEmbedder
from .evaluation import (
    benchmark_faiss,
    benchmark_real_index,
    code_mix_token_comparison,
    compare_episodic_weightings,
    estimate_vector_storage,
    estimate_monthly_cost,
    evaluate_fact_extraction,
    evaluate_nlu,
    evaluate_retrieval,
    log_summary,
    plot_input_tokens,
    validate_assignment_data,
    write_csv,
)
from .indexing import build_indices, create_embedder


def _paths(root: str | None) -> Paths:
    return Paths(Path(root).resolve()) if root else Paths()


def _logger(paths: Paths, run_id: str | None) -> CallLogger:
    return CallLogger(paths.logs / "calls.jsonl", run_id=run_id)


def _embedder(args: argparse.Namespace):
    return create_embedder(args.embedding_backend, args.embedding_model, args.ollama_url)


def command_validate(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    issues = validate_assignment_data(paths)
    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        return 1
    print(
        json.dumps(
            {
                "status": "pass",
                "variant": VARIANT,
                "knowledge_documents": len(list(paths.kb.glob("*.md"))),
                "questions": len(json.loads((paths.data / "kb_questions.json").read_text(encoding="utf-8"))),
                "probes": len(json.loads((paths.data / "probes.json").read_text(encoding="utf-8"))),
            },
            indent=2,
        )
    )
    return 0


def command_build_index(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    logger = _logger(paths, args.run_id)
    outputs = build_indices(
        paths=paths,
        backend=args.embedding_backend,
        model=args.embedding_model,
        base_url=args.ollama_url,
        logger=logger,
    )
    for path in outputs:
        print(path)
    return 0


def command_run(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    if args.config not in CONFIGS:
        raise ValueError(f"Unknown configuration {args.config}. Choose from {sorted(CONFIGS)}.")
    if args.session == "B" and args.config in {"hybrid_sem", "hybrid_mem"}:
        memory_path = paths.memory / args.config / args.user / "semantic.json"
        episode_path = paths.memory / args.config / args.user / "episodic.json"
        if not memory_path.exists() and not episode_path.exists():
            raise RuntimeError(
                f"Session B needs persisted Session A memory. Run Session A first for {args.config}."
            )
    chat = TemplateChat() if args.engine == "template" else OllamaChat(args.model, args.ollama_url)
    embedder = _embedder(args)
    logger = _logger(paths, args.run_id)
    agent = RecallBotAgent(
        paths=paths,
        user=args.user,
        config=args.config,
        chat=chat,
        embedder=embedder,
        logger=logger,
        chunk_size=args.chunk_size,
    )
    script_path = Path(args.script) if args.script else paths.data / "test_conversation.json"
    scripted = json.loads(script_path.read_text(encoding="utf-8"))
    turns = [row for row in scripted["turns"] if row["session"] == args.session]
    for row in turns:
        result = agent.process_turn(
            row["text"],
            session=args.session,
            turn=row["turn"],
        )
        output = {
            "run_id": logger.run_id,
            "config": args.config,
            "user_id": args.user,
            "session": args.session,
            "turn": row["turn"],
            "user": row["text"],
            "intent": result["intent"],
            "answer": result["answer"],
            "citations": result["citations"],
            "tool_evidence": result["tool_evidence"],
            "state": result["state"],
            "ranked_retrieval": result["ranked_retrieval"],
            "semantic_memory": result["semantic_memory"],
            "episodes": result["episodes"],
            "prompt_excerpt": result["prompt_excerpt"],
        }
        paths.results.mkdir(parents=True, exist_ok=True)
        with (paths.results / "dialogue_outputs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
        print(json.dumps(output, ensure_ascii=False))
    if args.session == "A":
        agent.finish_session(session="A", turn=max(row["turn"] for row in turns))
    return 0


def command_evaluate_nlu(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    result = evaluate_nlu(paths)
    rows = [
        {
            "text": row["text"],
            "expected_intent": row["expected_intent"],
            "predicted_intent": row["predicted_intent"],
            "correct": int(row["correct"]),
        }
        for row in result["predictions"]
    ]
    write_csv(paths.results / "nlu_accuracy.csv", rows)
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, indent=2))
    return 0


def command_evaluate_kb(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    logger = _logger(paths, args.run_id)
    embedder = _embedder(args)
    hits, rankings = evaluate_retrieval(paths=paths, embedder=embedder, logger=logger)
    write_csv(
        paths.results / "hit_rate.csv",
        hits,
        ["method", "chunk_size", "k", "question_id", "question_type", "hit"],
    )
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "rankings.json").write_text(
        json.dumps(rankings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(hits)} hit-rate rows and ranked results to {paths.results}.")
    return 0


def command_run_probes(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    probes = json.loads((paths.data / "probes.json").read_text(encoding="utf-8"))
    if args.stage:
        probes = [probe for probe in probes if probe["stage"] == args.stage]
    chat = TemplateChat() if args.engine == "template" else OllamaChat(args.model, args.ollama_url)
    embedder = _embedder(args)
    logger = _logger(paths, args.run_id)
    output_rows = []
    for probe in probes:
        agent = RecallBotAgent(
            paths=paths,
            user=args.user,
            config=args.config,
            chat=chat,
            embedder=embedder,
            logger=logger,
            chunk_size=args.chunk_size,
        )
        agent.state = dict(probe.get("state", {}))
        result = agent.process_turn(
            probe["question"],
            session="probe",
            turn=0,
        )
        output_rows.append(
            {
                "run_id": logger.run_id,
                "config": args.config,
                "probe_id": probe["id"],
                "category": probe["category"],
                "expected": probe["expected"],
                "answer": result["answer"],
                "correct": "",
                "judged_by": "",
                "code_mixed": int(probe.get("code_mixed", False)),
                "stage": probe["stage"],
                "citations": json.dumps(result["citations"]),
                "state": json.dumps(result["state"], ensure_ascii=False),
                "ranked_retrieval": json.dumps(result["ranked_retrieval"], ensure_ascii=False),
                "prompt_excerpt": result["prompt_excerpt"],
            }
        )
    paths.results.mkdir(parents=True, exist_ok=True)
    output_path = paths.results / "probe_outputs.csv"
    write_csv(output_path, output_rows)
    result_path = paths.results / "probe_results.csv"
    write_csv(
        result_path,
        output_rows,
        [
            "run_id",
            "config",
            "probe_id",
            "category",
            "expected",
            "answer",
            "correct",
            "judged_by",
            "code_mixed",
            "stage",
            "citations",
            "state",
            "ranked_retrieval",
            "prompt_excerpt",
        ],
    )
    print(f"Raw answers saved to {output_path}. Review them with `recallbot review-probes`.")
    return 0


def command_review_probes(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    result_path = paths.results / "probe_results.csv"
    if not result_path.exists():
        raise FileNotFoundError("Run `recallbot run-probes` first.")
    with result_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["correct"] in {"0", "1"}:
            continue
        print(f"\n{row['probe_id']} [{row['category']}] expected: {row['expected']}")
        print(f"answer: {row['answer']}")
        while True:
            value = input("Correct? [y/n]: ").strip().lower()
            if value in {"y", "n"}:
                row["correct"] = "1" if value == "y" else "0"
                row["judged_by"] = "manual"
                break
            print("Enter y or n.")
    write_csv(result_path, rows)
    print(f"Saved manually reviewed labels to {result_path}.")
    return 0


def command_metrics(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    log_path = paths.logs / "calls.jsonl"
    summary = log_summary(log_path)
    write_csv(paths.results / "call_summary.csv", summary)
    plot_input_tokens(log_path, paths.results / "plots" / "input_tokens.png")
    print(json.dumps(summary, indent=2))
    return 0


def command_vector_sizing(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    index_path = paths.index / f"chunks_{args.chunk_size}.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Build the {args.chunk_size}-token index first.")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    dimensions = len(payload["chunks"][0]["vector"])
    kb_vectors = len(payload["chunks"])
    daily_users = VARIANT["daily_users"]
    semantic_facts_per_user = args.facts_per_user
    episodes_per_user_daily = args.episodes_per_user_daily
    annual_count = (
        kb_vectors
        + daily_users * 365 * (semantic_facts_per_user + episodes_per_user_daily)
    )
    result = estimate_vector_storage(vectors=annual_count, dimensions=dimensions)
    result.update(
        {
            "kb_vectors": kb_vectors,
            "daily_users": daily_users,
            "semantic_facts_per_user_assumption": semantic_facts_per_user,
            "episodic_memories_per_user_per_day_assumption": episodes_per_user_daily,
            "flat_gib": result["flat_bytes"] / (1024**3),
            "hnsw_gib_estimate": result["hnsw_bytes_estimate"] / (1024**3),
        }
    )
    print(json.dumps(result, indent=2))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    result = benchmark_faiss(vectors=args.vectors, dimensions=args.dimensions, queries=args.queries)
    paths = _paths(args.root)
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "index_benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


def command_benchmark_real(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    logger = _logger(paths, args.run_id)
    result = benchmark_real_index(
        paths=paths,
        chunk_size=args.chunk_size,
        embedder=_embedder(args),
        logger=logger,
        min_vectors=args.vectors,
        query_count=args.queries,
    )
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "kb_index_benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


def command_code_mix(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    rows = code_mix_token_comparison(paths)
    write_csv(paths.results / "code_mixed_token_comparison.csv", rows)
    turns = json.loads((paths.data / "test_conversation.json").read_text(encoding="utf-8"))["turns"]
    ratio = sum(bool(row["code_mixed"]) for row in turns) / len(turns)
    print(json.dumps({"code_mixed_turns": sum(bool(row["code_mixed"]) for row in turns), "total_turns": len(turns), "share": ratio, "paired_rows": len(rows)}, indent=2))
    return 0


def command_fact_metrics(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    result = evaluate_fact_extraction(paths, config=args.config, user=args.user)
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "fact_extraction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_episode_comparison(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    logger = _logger(paths, args.run_id)
    rows = compare_episodic_weightings(
        paths,
        config=args.config,
        user=args.user,
        embedder=_embedder(args),
        logger=logger,
    )
    write_csv(paths.results / "episodic_weighting_comparison.csv", rows)
    print(f"Wrote {len(rows)} ranked comparisons.")
    return 0


def command_cost(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    system_prompt = (paths.root / "skills" / "system_prompt_v1.md").read_text(encoding="utf-8")
    result = estimate_monthly_cost(
        log_path=paths.logs / "calls.jsonl",
        trim_run_ids=set(args.trim_run_ids.split(",")),
        hybrid_run_ids=set(args.hybrid_run_ids.split(",")),
        daily_users=args.daily_users,
        days=args.days,
        turns_per_conversation=args.turns,
        chat_input_price_per_million=args.input_price,
        chat_output_price_per_million=args.output_price,
        cached_input_price_per_million=args.cached_input_price,
        embedding_price_per_million=args.embedding_price,
        cache_min_tokens=args.cache_min_tokens,
        system_prompt=system_prompt,
    )
    result["price_row"] = args.price_row
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "monthly_cost_estimate.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


def command_review_dialogue(args: argparse.Namespace) -> int:
    paths = _paths(args.root)
    output_path = paths.results / "dialogue_outputs.jsonl"
    if not output_path.exists():
        raise FileNotFoundError("Run the scripted sessions first.")
    with output_path.open(encoding="utf-8") as handle:
        rows = [
            json.loads(line)
            for line in handle
            if line.strip()
            and (row := json.loads(line))["config"] == args.config
            and row["session"] == "B"
            and (not args.user or row["user_id"] == args.user)
        ]
    rows = rows[: args.count]
    if len(rows) < args.count:
        raise ValueError(f"Only {len(rows)} Session B replies found; need {args.count}.")
    ratings = []
    for row in rows:
        print(f"\n{row['config']} turn {row['turn']}: {row['user']}")
        print(f"Reply: {row['answer']}")
        rating = {"run_id": row["run_id"], "config": row["config"], "turn": row["turn"]}
        for dimension in ("coherence", "relevance", "groundedness", "safety"):
            while True:
                value = input(f"{dimension} (1-5): ").strip()
                if value.isdigit() and 1 <= int(value) <= 5:
                    rating[dimension] = int(value)
                    break
                print("Enter a number from 1 to 5.")
        rating["judged_by"] = "manual"
        ratings.append(rating)
    rating_path = paths.results / "dialogue_ratings.csv"
    prior = []
    if rating_path.exists():
        with rating_path.open(newline="", encoding="utf-8") as handle:
            prior = list(csv.DictReader(handle))
    write_csv(
        rating_path,
        prior + ratings,
        ["run_id", "config", "turn", "coherence", "relevance", "groundedness", "safety", "judged_by"],
    )
    print(f"Saved ratings to {rating_path}.")
    return 0


def _add_model_arguments(
    parser: argparse.ArgumentParser, *, include_chunk_size: bool = True
) -> None:
    parser.add_argument("--engine", choices=("ollama", "template"), default="ollama")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--embedding-backend",
        choices=("ollama", "sentence-transformers"),
        default="ollama",
    )
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    if include_chunk_size:
        parser.add_argument("--chunk-size", type=int, default=128, choices=(128, 512))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recallbot")
    parser.add_argument("--root", help="Project root; defaults to the installed package root.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data")
    validate.set_defaults(handler=command_validate)

    build = subparsers.add_parser("build-index")
    _add_model_arguments(build)
    build.add_argument("--run-id")
    build.set_defaults(handler=command_build_index)

    run = subparsers.add_parser("run")
    run.add_argument("--session", choices=("A", "B"), required=True)
    run.add_argument("--config", choices=sorted(CONFIGS), default="hybrid_mem")
    run.add_argument("--user", default="U1")
    run.add_argument("--script")
    run.add_argument("--run-id")
    _add_model_arguments(run)
    run.set_defaults(handler=command_run)

    nlu = subparsers.add_parser("evaluate-nlu")
    nlu.set_defaults(handler=command_evaluate_nlu)

    kb = subparsers.add_parser("evaluate-kb")
    kb.add_argument("--run-id")
    _add_model_arguments(kb)
    kb.set_defaults(handler=command_evaluate_kb)

    probes = subparsers.add_parser("run-probes")
    probes.add_argument("--config", choices=sorted(CONFIGS), default="hybrid_mem")
    probes.add_argument(
        "--stage",
        choices=("session_a", "session_b", "after_forget_restart"),
    )
    probes.add_argument("--user", default="U1")
    probes.add_argument("--run-id")
    _add_model_arguments(probes)
    probes.set_defaults(handler=command_run_probes)

    review = subparsers.add_parser("review-probes")
    review.set_defaults(handler=command_review_probes)

    metrics = subparsers.add_parser("metrics")
    metrics.set_defaults(handler=command_metrics)

    sizing = subparsers.add_parser("vector-sizing")
    sizing.add_argument("--chunk-size", type=int, default=128, choices=(128, 512))
    sizing.add_argument("--facts-per-user", type=int, default=8)
    sizing.add_argument("--episodes-per-user-daily", type=int, default=1)
    sizing.set_defaults(handler=command_vector_sizing)

    benchmark = subparsers.add_parser("benchmark-index")
    benchmark.add_argument("--vectors", type=int, default=100_000)
    benchmark.add_argument("--dimensions", type=int, default=384)
    benchmark.add_argument("--queries", type=int, default=100)
    benchmark.set_defaults(handler=command_benchmark)

    real_benchmark = subparsers.add_parser("benchmark-kb-index")
    real_benchmark.add_argument("--chunk-size", type=int, default=128, choices=(128, 512))
    real_benchmark.add_argument("--vectors", type=int, default=100_000)
    real_benchmark.add_argument("--queries", type=int, default=100)
    real_benchmark.add_argument("--run-id")
    _add_model_arguments(real_benchmark, include_chunk_size=False)
    real_benchmark.set_defaults(handler=command_benchmark_real)

    twist = subparsers.add_parser("compare-code-mix")
    twist.set_defaults(handler=command_code_mix)

    facts = subparsers.add_parser("evaluate-facts")
    facts.add_argument("--config", choices=("hybrid_sem", "hybrid_mem"), default="hybrid_sem")
    facts.add_argument("--user", default="U1")
    facts.set_defaults(handler=command_fact_metrics)

    episodes = subparsers.add_parser("compare-episodes")
    episodes.add_argument("--config", choices=("hybrid_mem",), default="hybrid_mem")
    episodes.add_argument("--user", default="U1")
    episodes.add_argument("--run-id")
    _add_model_arguments(episodes)
    episodes.set_defaults(handler=command_episode_comparison)

    cost = subparsers.add_parser("cost-estimate")
    cost.add_argument("--trim-run-ids", required=True, help="Comma-separated run IDs for trim Session A and B.")
    cost.add_argument("--hybrid-run-ids", required=True, help="Comma-separated run IDs for hybrid_mem Session A and B.")
    cost.add_argument("--price-row", required=True)
    cost.add_argument("--input-price", type=float, required=True, help="USD per million uncached input tokens.")
    cost.add_argument("--output-price", type=float, required=True, help="USD per million output tokens.")
    cost.add_argument("--cached-input-price", type=float, required=True, help="USD per million cached input tokens.")
    cost.add_argument("--embedding-price", type=float, required=True, help="USD per million embedding input tokens.")
    cost.add_argument("--cache-min-tokens", type=int, required=True)
    cost.add_argument("--daily-users", type=int, default=VARIANT["daily_users"])
    cost.add_argument("--days", type=int, default=30)
    cost.add_argument("--turns", type=int, default=42)
    cost.set_defaults(handler=command_cost)

    review_dialogue = subparsers.add_parser("review-dialogue")
    review_dialogue.add_argument("--config", choices=("no_rag", "hybrid", "hybrid_sem", "hybrid_mem"), required=True)
    review_dialogue.add_argument("--user")
    review_dialogue.add_argument("--count", type=int, default=15)
    review_dialogue.set_defaults(handler=command_review_dialogue)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
