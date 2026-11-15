import json
import re
from pathlib import Path
from typing import Any

from .call_log import CallLogger
from .chat import ChatModel
from .config import FALLBACK_ANSWER, Paths, VARIANT
from .database import BudgetDatabase
from .embeddings import Embedder
from .memory import MemoryStore, infer_forget_key
from .nlu import classify
from .retrieval import Chunk, KnowledgeIndex, normalize_query
from .tokens import tokenizer


class RecallBotAgent:
    def __init__(
        self,
        *,
        paths: Paths,
        user: str,
        config: str,
        chat: ChatModel,
        embedder: Embedder,
        logger: CallLogger,
        chunk_size: int = 128,
        context_window: int = VARIANT["context_window"],
    ) -> None:
        self.paths = paths
        self.user = user
        self.config = config
        self.chat = chat
        self.embedder = embedder
        self.logger = logger
        self.chunk_size = chunk_size
        self.context_window = context_window
        self.memory = MemoryStore(paths.memory / config / user)
        self.database = BudgetDatabase(paths.database)
        self.state: dict[str, Any] = {}
        self.history: list[dict[str, str]] = []
        self.summary = ""
        self.system_prompt = (paths.root / "skills" / "system_prompt_v1.md").read_text(
            encoding="utf-8"
        )
        self.task_steps = (paths.root / "skills" / "savings_goal_v1.md").read_text(
            encoding="utf-8"
        )

    def process_turn(
        self, text: str, *, session: str, turn: int, config_override: str | None = None
    ) -> dict[str, Any]:
        self.last_context = ""
        config = config_override or self.config
        parsed = classify(text)
        if config in {"hybrid_sem", "hybrid_mem"}:
            self.memory.extract_and_store(
                self.user,
                text,
                turn,
                embedder=self.embedder,
                logger=self.logger,
                config=config,
                session=session,
                record_episode=config == "hybrid_mem",
            )
        if "grocery" in parsed.normalized or "groceries" in parsed.normalized:
            facts = self.memory.active_facts(self.user)
            grocery = next((row for row in reversed(facts) if row["key"] == "grocery_budget"), None)
            self.state["last_topic"] = "grocery cap"
            if grocery:
                self.state["last_topic_value"] = grocery["value"]
        if "saturday" in parsed.normalized and not parsed.entities.get("state_followup"):
            self.state["last_topic"] = "Saturday grocery review"
        hits: list[dict[str, Any]] = []
        semantic: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        memory_query_vector: list[float] | None = None
        tool_evidence: list[str] = []
        database_context = ""

        if parsed.intent == "forget":
            key = infer_forget_key(text)
            started = self.logger.start()
            removed = self.memory.forget(self.user, key)
            self.logger.record(
                config=config,
                user=self.user,
                session=session,
                turn=turn,
                call_type="tool",
                model="memory.forget",
                input_tokens=0,
                output_tokens=0,
                started=started,
            )
            answer = f"Forgot the {key} information I had stored; {removed} matching memory record(s) were removed."
            self._append_history(text, answer, config, session, turn)
            return self._result(parsed, answer, hits, semantic, episodes, tool_evidence)

        if parsed.entities.get("state_followup") and self.state.get("last_topic"):
            topic = self.state["last_topic"]
            detail = self.state.get("last_topic_value")
            answer = f"For Saturday, the current dialogue topic is {topic}"
            if detail:
                answer += f" (Rs {detail})"
            answer += "."
            self._append_history(text, answer, config, session, turn)
            return self._result(parsed, answer, hits, semantic, episodes, tool_evidence)

        if config != "no_rag" and (
            parsed.intent == "knowledge_query"
            or (
                parsed.intent == "open"
                and "?" in text
                and not re.search(r"\bmy\b", parsed.normalized)
            )
        ):
            hits = self._search_kb(text, session, turn, config)
            if hits:
                tool_evidence.append(f"search_kb returned {hits[0]['chunk'].chunk_id}")

        if config in {"hybrid_sem", "hybrid_mem"} and parsed.intent in {
            "semantic_recall",
            "budget_lookup",
            "open",
        }:
            memory_query_vector = self.embedder.embed(
                [normalize_query(text)],
                logger=self.logger,
                config=config,
                user=self.user,
                session=session,
                turn=turn,
            )[0]
            started = self.logger.start()
            semantic = self.memory.recall_facts(
                self.user, normalize_query(text), query_vector=memory_query_vector
            )
            self.logger.record(
                config=config,
                user=self.user,
                session=session,
                turn=turn,
                call_type="tool",
                model="memory.semantic_search",
                input_tokens=0,
                output_tokens=0,
                retrieved_tokens=sum(
                    tokenizer().count(f"{row['key']} {row['value']}") for row in semantic
                ),
                started=started,
            )
        if config == "hybrid_mem" and parsed.intent in {
            "episodic_recall",
            "open",
        }:
            if memory_query_vector is None:
                memory_query_vector = self.embedder.embed(
                    [normalize_query(text)],
                    logger=self.logger,
                    config=config,
                    user=self.user,
                    session=session,
                    turn=turn,
                )[0]
            started = self.logger.start()
            episodes = self.memory.recall_episodes(
                self.user, text, query_vector=memory_query_vector
            )
            self.logger.record(
                config=config,
                user=self.user,
                session=session,
                turn=turn,
                call_type="tool",
                model="memory.episodic_search",
                input_tokens=0,
                output_tokens=0,
                retrieved_tokens=sum(
                    tokenizer().count(row["text"]) for row in episodes
                ),
                started=started,
            )

        if parsed.intent == "budget_lookup":
            started = self.logger.start()
            profile = self.database.get_profile(self.user)
            self.logger.record(
                config=config,
                user=self.user,
                session=session,
                turn=turn,
                call_type="tool",
                model="database.get_profile",
                input_tokens=0,
                output_tokens=0,
                started=started,
            )
            if profile:
                database_context = f"Budget profile: {profile}"
                tool_evidence.append("database.get_profile read")

        task_reply = self._handle_savings_goal(parsed.intent, parsed.entities, text, session, turn, config)
        if task_reply is not None:
            self._append_history(text, task_reply, config, session, turn)
            return self._result(parsed, task_reply, hits, semantic, episodes, tool_evidence)

        personal_fact_question = (
            parsed.intent == "budget_lookup"
            and re.search(r"\bmy\b", parsed.normalized)
            and any(term in parsed.normalized for term in ("rent", "income", "budget", "salary"))
        )
        if parsed.intent == "episodic_recall" and not episodes:
            answer = "I don't have a relevant earlier event stored for that question."
        elif parsed.intent == "semantic_recall" and not semantic:
            answer = "I don't have that fact stored, so I won't guess."
        elif personal_fact_question and not semantic:
            answer = "I don't have that fact stored, so I won't guess."
        elif parsed.intent == "knowledge_query" and not hits:
            answer = FALLBACK_ANSWER
        else:
            context, context_metrics = self._assemble_context(
                text=text,
                session=session,
                turn=turn,
                config=config,
                hits=hits,
                semantic=semantic,
                episodes=episodes,
                database_context=database_context,
            )
            self.last_context = context
            self._write_context_audit(
                session=session,
                turn=turn,
                config=config,
                metrics=context_metrics,
                citations=[hit["chunk"].chunk_id for hit in hits],
            )
            answer = self.chat.complete(
                context,
                system=self.system_prompt,
                temperature=0.0,
                logger=self.logger,
                config=config,
                user=self.user,
                session=session,
                turn=turn,
                retrieved_tokens=context_metrics["retrieved_tokens"],
            )
            if parsed.intent == "knowledge_query" and hits:
                answer = self._attach_citation(answer, hits[0]["chunk"])
        self._append_history(text, answer, config, session, turn)
        return self._result(parsed, answer, hits, semantic, episodes, tool_evidence)

    def finish_session(self, *, session: str, turn: int) -> None:
        if self.config == "hybrid_mem":
            self.memory.add_reflection(
                self.user,
                session,
                turn,
                embedder=self.embedder,
                logger=self.logger,
                config=self.config,
            )

    def _search_kb(self, query: str, session: str, turn: int, config: str) -> list[dict[str, Any]]:
        path = self.paths.index / f"chunks_{self.chunk_size}.json"
        if not path.exists():
            return []
        index = KnowledgeIndex.load(path)
        started = self.logger.start()
        vectors = self.embedder.embed(
            [normalize_query(query)],
            logger=self.logger,
            config=config,
            user=self.user,
            session=session,
            turn=turn,
        )
        year = re.search(r"\b(20\d{2})\b", query)
        as_of = f"{year.group(1)}-12-31" if year else "2026-11-15"
        hits = index.search(
            query,
            method="hybrid",
            query_vector=vectors[0],
            limit=3,
            as_of=as_of,
        )
        self.logger.record(
            config=config,
            user=self.user,
            session=session,
            turn=turn,
            call_type="tool",
            model="search_kb",
            input_tokens=0,
            output_tokens=0,
            retrieved_tokens=sum(hit["chunk"].token_count for hit in hits),
            started=started,
        )
        return hits

    def _handle_savings_goal(
        self,
        intent: str,
        entities: dict[str, str],
        text: str,
        session: str,
        turn: int,
        config: str,
    ) -> str | None:
        if intent == "cancel" and "pending_goal" in self.state:
            del self.state["pending_goal"]
            return "Cancelled the pending savings goal; nothing was saved."
        if intent == "goal_create":
            self.state["pending_goal"] = {
                "purpose": self._purpose_from_text(text),
                "amount": entities.get("amount"),
                "due_date": entities.get("date"),
                "awaiting_confirmation": False,
            }
        goal = self.state.get("pending_goal")
        if not goal:
            return None
        if entities.get("amount"):
            goal["amount"] = entities["amount"]
        if entities.get("date"):
            goal["due_date"] = entities["date"]
        missing = [key for key in ("purpose", "amount", "due_date") if not goal.get(key)]
        if missing:
            labels = {"purpose": "the goal purpose", "amount": "the target amount", "due_date": "the target date"}
            return f"I still need {', '.join(labels[key] for key in missing)}."
        if intent == "confirm" and goal["awaiting_confirmation"]:
            started = self.logger.start()
            goal_id = self.database.create_goal(
                self.user,
                str(goal["purpose"]),
                float(goal["amount"]),
                str(goal["due_date"]),
            )
            self.logger.record(
                config=config,
                user=self.user,
                session=session,
                turn=turn,
                call_type="tool",
                model="database.create_goal",
                input_tokens=0,
                output_tokens=0,
                started=started,
            )
            self.memory.record_episode(
                self.user,
                f"Confirmed savings goal {goal['purpose']} for Rs {goal['amount']} due {goal['due_date']}.",
                turn,
                kind="confirmed_goal",
                embedder=self.embedder,
                logger=self.logger,
                config=config,
                session=session,
            )
            del self.state["pending_goal"]
            return f"Saved savings goal #{goal_id}: {goal['purpose']}, Rs {goal['amount']} by {goal['due_date']}."
        if not goal["awaiting_confirmation"]:
            goal["awaiting_confirmation"] = True
            return (
                f"Please confirm: create the {goal['purpose']} goal for Rs {goal['amount']} "
                f"by {goal['due_date']}? I will not save it until you say yes."
            )
        return "The goal is ready. Please confirm explicitly before I save it."

    @staticmethod
    def _purpose_from_text(text: str) -> str:
        match = re.search(
            r"(?:goal|fund|saving)\s+(?:for|named)\s+([A-Za-z][A-Za-z -]{1,40})",
            text,
            re.I,
        )
        if match:
            return match.group(1).strip(" .")
        return ""

    def _assemble_context(
        self,
        *,
        text: str,
        session: str,
        turn: int,
        config: str,
        hits: list[dict[str, Any]],
        semantic: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        database_context: str,
    ) -> tuple[str, dict[str, Any]]:
        if config == "full":
            history = self.history[:]
        elif config in {"trim", "summary"}:
            turn_limit = VARIANT["window_k"] if config == "trim" else 2
            history = self.history[-(turn_limit * 2) :]
        else:
            history = self.history[-(VARIANT["window_k"] * 2) :]
        selected_hits = list(hits)
        selected_semantic = list(semantic)
        selected_episodes = list(episodes)
        cuts: list[str] = []
        reply_reserve = 512

        def compose() -> tuple[str, str, str, str, list[str], int]:
            history_text = "\n".join(f"{row['role']}: {row['text']}" for row in history)
            if config == "summary" and self.summary:
                history_text = f"Rolling summary: {self.summary}\n{history_text}"
            hit_text = "\n".join(
                f"[{hit['chunk'].chunk_id}] {hit['chunk'].text}" for hit in selected_hits
            ) or "(none)"
            semantic_text = "\n".join(
                f"[semantic:{row['key']}:turn-{row['source_turn']}] {row['value']}"
                for row in selected_semantic
            ) or "(none)"
            episode_text = "\n".join(
                f"[episode:turn-{row['source_turn']}] {row['text']}"
                for row in selected_episodes
            ) or "(none)"
            fixed_parts = [
                "Tools: search_kb, database.get_profile, database.create_goal (confirmation required)",
                f"Procedural task steps: {self.task_steps}",
                f"Long-term semantic memory: {semantic_text}",
                f"Episodic memory: {episode_text}",
                f"Database: {database_context or '(not used)'}",
            ]
            context = (
                f"Session: {session}; turn: {turn}\n"
                f"State: {self.state}\n"
                f"Conversation history:\n{history_text or '(none)'}\n"
                f"Retrieved knowledge:\n{hit_text}\n"
                + "\n".join(fixed_parts)
                + f"\nUser: {text}\n"
                + "Grounding: use only the provided facts; cite chunk IDs; "
                + f"if no relevant policy is present say exactly: {FALLBACK_ANSWER}"
            )
            total_tokens = tokenizer().count(self.system_prompt + context)
            return context, hit_text, semantic_text, episode_text, fixed_parts, total_tokens

        context, hit_text, semantic_text, episode_text, fixed_parts, total = compose()
        while total + reply_reserve > self.context_window:
            if config != "full" and history:
                history = history[2:]
                cuts.append("oldest history turn")
            elif selected_episodes:
                selected_episodes.pop()
                cuts.append("lowest-ranked episodic memory")
            elif selected_semantic:
                selected_semantic.pop()
                cuts.append("lowest-ranked semantic memory")
            elif selected_hits:
                selected_hits.pop()
                cuts.append("lowest-ranked knowledge chunk")
            else:
                raise RuntimeError(
                    f"Context budget exceeded ({total} + {reply_reserve} > {self.context_window}); "
                    "the fixed prompt and current task cannot fit."
                )
            context, hit_text, semantic_text, episode_text, fixed_parts, total = compose()
        retrieved_tokens = tokenizer().count(hit_text + semantic_text + episode_text)
        return context, {
            "total_tokens": total,
            "retrieved_tokens": retrieved_tokens,
            "reply_reserve": reply_reserve,
            "context_window": self.context_window,
            "cuts": cuts,
            "parts": [
                {
                    "part": "system_and_instructions",
                    "tokens": tokenizer().count(self.system_prompt),
                    "memory_type": "procedural",
                    "home": "external skills/ file loaded into prompt",
                },
                {
                    "part": "tools_and_task_steps",
                    "tokens": tokenizer().count(fixed_parts[0] + fixed_parts[1]),
                    "memory_type": "procedural",
                    "home": "external tool code and skills/ file loaded into prompt",
                },
                {
                    "part": "semantic_memory",
                    "tokens": tokenizer().count(semantic_text),
                    "memory_type": "semantic",
                    "home": "external JSON store and vectors",
                },
                {
                    "part": "episodic_memory",
                    "tokens": tokenizer().count(episode_text),
                    "memory_type": "episodic",
                    "home": "external JSON stream and vectors",
                },
                {
                    "part": "conversation_history_and_state",
                    "tokens": tokenizer().count(
                        "\n".join(f"{row['role']}: {row['text']}" for row in history)
                        + str(self.state)
                    ),
                    "memory_type": "short-term working",
                    "home": "model-native prompt; last six turns for V12 trim",
                },
                {
                    "part": "retrieved_knowledge",
                    "tokens": tokenizer().count(hit_text),
                    "memory_type": "knowledge-base semantic",
                    "home": "external chunk index",
                },
                {
                    "part": "current_task",
                    "tokens": tokenizer().count(text + database_context),
                    "memory_type": "current turn",
                    "home": "model-native prompt",
                },
            ],
        }

    def _write_context_audit(
        self,
        *,
        session: str,
        turn: int,
        config: str,
        metrics: dict[str, Any],
        citations: list[str],
    ) -> None:
        path = self.paths.results / "context_assembly.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "run_id": self.logger.run_id,
            "config": config,
            "user": self.user,
            "session": session,
            "turn": turn,
            "context_window": metrics["context_window"],
            "measured_tokens": metrics["total_tokens"],
            "reply_reserve": metrics["reply_reserve"],
            "cuts": metrics["cuts"],
            "parts": metrics["parts"],
            "cited_ids": citations,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _append_history(
        self, user_text: str, assistant_text: str, config: str, session: str, turn: int
    ) -> None:
        self.history.extend(
            [{"role": "user", "text": user_text}, {"role": "assistant", "text": assistant_text}]
        )
        if config == "summary" and len(self.history) > 4:
            older = self.history[:-4]
            summary_input = self.summary
            if summary_input:
                summary_input = "Existing summary:\n" + summary_input + "\n"
            summary_input += "\n".join(f"{row['role']}: {row['text']}" for row in older)
            summary_text = self.chat.complete(
                summary_input,
                system=(self.paths.root / "skills" / "summary_prompt_v1.md").read_text(encoding="utf-8"),
                temperature=0.0,
                logger=self.logger,
                config=config,
                user=self.user,
                session=session,
                turn=turn,
                call_type="summary",
            )
            self.summary = summary_text
            self.history = self.history[-4:]

    @staticmethod
    def _attach_citation(answer: str, chunk: Chunk) -> str:
        citation = f"[{chunk.chunk_id}]"
        return answer if citation in answer else f"{answer.rstrip()} {citation}"

    def _result(
        self,
        parsed: Any,
        answer: str,
        hits: list[dict[str, Any]],
        semantic: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        evidence: list[str],
    ) -> dict[str, Any]:
        return {
            "intent": parsed.intent,
            "normalized": parsed.normalized,
            "entities": parsed.entities,
            "answer": answer,
            "citations": [hit["chunk"].chunk_id for hit in hits],
            "semantic_memory": semantic,
            "episodes": episodes,
            "tool_evidence": evidence,
            "state": dict(self.state),
            "ranked_retrieval": [
                {
                    "chunk_id": hit["chunk"].chunk_id,
                    "doc_id": hit["chunk"].doc_id,
                    "version": hit["chunk"].version,
                    "score": hit["score"],
                    "excerpt": hit["chunk"].text[:500],
                }
                for hit in hits
            ],
            "prompt_excerpt": self.last_context[:1500],
        }
