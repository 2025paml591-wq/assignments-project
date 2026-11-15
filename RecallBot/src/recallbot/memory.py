import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .call_log import CallLogger
from .embeddings import Embedder, cosine
from .retrieval import terms


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return data


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


class MemoryStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.semantic_path = directory / "semantic.json"
        self.episodic_path = directory / "episodic.json"
        self.reflection_path = directory / "reflections.json"

    def extract_and_store(
        self,
        user: str,
        text: str,
        turn: int,
        *,
        embedder: Embedder | None = None,
        logger: CallLogger | None = None,
        config: str = "hybrid_mem",
        session: str = "A",
        record_episode: bool = True,
    ) -> list[dict[str, Any]]:
        from .nlu import FACTS

        found: list[dict[str, Any]] = []
        rows = _read(self.semantic_path)
        for key, pattern in FACTS.items():
            match = pattern.search(text)
            if not match:
                continue
            value = " ".join(match.group(1).split()).strip(" .")
            value = value.replace(",", "")
            for record in rows:
                if record["user"] == user and record["key"] == key and record["status"] == "active":
                    record["status"] = "superseded"
            row = {
                "user": user,
                "key": key,
                "value": value,
                "source_turn": turn,
                "date": datetime.now(timezone.utc).isoformat(),
                "status": "active",
            }
            rows.append(row)
            found.append(row)
        if found and embedder and logger:
            vectors = embedder.embed(
                [f"{row['key']}: {row['value']}" for row in found],
                logger=logger,
                config=config,
                user=user,
                session=session,
                turn=turn,
            )
            for row, vector in zip(found, vectors):
                row["vector"] = vector
        if found:
            _write(self.semantic_path, rows)
        if record_episode:
            self.record_episode(
                user,
                text,
                turn,
                embedder=embedder,
                logger=logger,
                config=config,
                session=session,
            )
        return found

    def record_episode(
        self,
        user: str,
        text: str,
        turn: int,
        *,
        kind: str = "event",
        embedder: Embedder | None = None,
        logger: CallLogger | None = None,
        config: str = "hybrid_mem",
        session: str = "A",
    ) -> None:
        lowered = text.lower()
        high_value = any(
            word in lowered
            for word in ("confirm", "book", "goal", "complaint", "problem", "decided", "forget")
        )
        if not high_value:
            return
        rows = _read(self.episodic_path)
        row = {
                "user": user,
                "source_turn": turn,
                "date": datetime.now(timezone.utc).isoformat(),
                "text": text,
                "kind": kind,
                "importance": 8 if "confirm" in lowered or "goal" in lowered else 6,
                "vector": None,
            }
        if embedder and logger:
            row["vector"] = embedder.embed(
                [text],
                logger=logger,
                config=config,
                user=user,
                session=session,
                turn=turn,
            )[0]
        rows.append(row)
        _write(self.episodic_path, rows)

    def active_facts(self, user: str) -> list[dict[str, Any]]:
        return [
            row
            for row in _read(self.semantic_path)
            if row["user"] == user and row["status"] == "active"
        ]

    def recall_facts(
        self,
        user: str,
        query: str,
        limit: int = 5,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        query_terms = set(terms(query))
        rows = self.active_facts(user)

        def score(row: dict[str, Any]) -> tuple[float, int]:
            fact_terms = set(terms(f"{row['key']} {row['value']}"))
            lexical = len(query_terms & fact_terms) / max(1, len(query_terms))
            vector = 0.0
            if query_vector and row.get("vector"):
                vector = max(0.0, cosine(query_vector, row["vector"]))
            return (0.5 * lexical + 0.5 * vector, int(row["source_turn"]))

        return sorted(
            (row for row in rows if score(row)[0] > 0.05),
            key=score,
            reverse=True,
        )[:limit]

    def add_reflection(
        self,
        user: str,
        session: str,
        source_turn: int,
        *,
        embedder: Embedder | None = None,
        logger: CallLogger | None = None,
        config: str = "hybrid_mem",
    ) -> dict[str, Any] | None:
        episodes = [row for row in _read(self.episodic_path) if row["user"] == user]
        if not episodes:
            return None
        summary = "Session reflection: " + "; ".join(
            f"turn {row['source_turn']}: {row['text']}" for row in episodes[-3:]
        )
        row = {
            "user": user,
            "session": session,
            "source_turn": source_turn,
            "date": datetime.now(timezone.utc).isoformat(),
            "text": summary,
            "kind": "reflection",
            "importance": 7,
            "vector": None,
        }
        if embedder and logger:
            row["vector"] = embedder.embed(
                [summary],
                logger=logger,
                config=config,
                user=user,
                session=session,
                turn=source_turn,
            )[0]
        rows = _read(self.reflection_path)
        rows.append(row)
        _write(self.reflection_path, rows)
        return row

    def recall_episodes(
        self,
        user: str,
        query: str,
        *,
        weights: tuple[float, float, float] = (0.25, 0.35, 0.40),
        limit: int = 5,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        from datetime import datetime

        rows = [
            row
            for path in (self.episodic_path, self.reflection_path)
            for row in _read(path)
            if row["user"] == user
        ]
        query_terms = set(terms(query))
        now = datetime.now(timezone.utc)
        scored = []
        for row in rows:
            overlap = len(query_terms & set(terms(row["text"]))) / max(1, len(query_terms))
            if query_vector and row.get("vector"):
                overlap = 0.5 * overlap + 0.5 * max(0.0, cosine(query_vector, row["vector"]))
            timestamp = datetime.fromisoformat(row["date"])
            age_days = max(0.0, (now - timestamp).total_seconds() / 86400)
            recency = 0.5 ** (age_days / 30.0)
            importance = max(0, min(10, int(row["importance"]))) / 10
            relevance = overlap
            score = weights[0] * recency + weights[1] * importance + weights[2] * relevance
            scored.append((score, row))
        return [row for _, row in sorted(scored, key=lambda pair: pair[0], reverse=True)[:limit]]

    def forget(self, user: str, key: str, value: str | None = None) -> int:
        removed = 0
        for path in (self.semantic_path, self.episodic_path, self.reflection_path):
            rows = _read(path)
            kept = []
            for row in rows:
                if row["user"] != user:
                    kept.append(row)
                    continue
                mentions = key == "*" or key.lower() in json.dumps(row, ensure_ascii=False).lower()
                if value:
                    mentions = mentions or value.lower() in json.dumps(row, ensure_ascii=False).lower()
                if mentions:
                    removed += 1
                else:
                    kept.append(row)
            _write(path, kept)
        return removed

    def user_facts_text(self, user: str, query: str) -> str:
        rows = self.recall_facts(user, query)
        return "\n".join(f"{row['key']}={row['value']} (turn {row['source_turn']})" for row in rows)


def infer_forget_key(text: str) -> str:
    lowered = text.lower()
    candidates = {
        "rent": ("rent", "kiraya"),
        "income": ("income", "salary", "aamdani"),
        "city": ("city", "address", "location"),
        "preference": ("preference", "weekly", "daily"),
        "grocery_budget": ("grocery", "groceries", "food"),
        "family_support": ("parents", "family"),
        "savings_goal": ("goal", "savings"),
    }
    for key, words in candidates.items():
        if any(word in lowered for word in words):
            return key
    return "*"
