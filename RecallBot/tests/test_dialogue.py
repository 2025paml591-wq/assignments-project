import json
import tempfile
import unittest
from pathlib import Path

from recallbot.agent import RecallBotAgent
from recallbot.call_log import CallLogger
from recallbot.chat import TemplateChat
from recallbot.config import Paths
from recallbot.database import BudgetDatabase
from recallbot.retrieval import Chunk


class FakeEmbedder:
    model = "test-vector"

    def embed(self, texts, **kwargs):
        return [[1.0, 0.0] for _ in texts]


class DialogueTests(unittest.TestCase):
    def test_goal_write_waits_for_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "skills").mkdir()
            (root / "skills" / "system_prompt_v1.md").write_text("Test prompt", encoding="utf-8")
            (root / "skills" / "savings_goal_v1.md").write_text("Goal steps", encoding="utf-8")
            paths = Paths(root)
            agent = RecallBotAgent(
                paths=paths,
                user="U1",
                config="hybrid_mem",
                chat=TemplateChat(),
                embedder=FakeEmbedder(),
                logger=CallLogger(paths.logs / "calls.jsonl", run_id="test"),
            )
            agent.process_turn("Please create a savings goal for emergency fund.", session="A", turn=1)
            agent.process_turn("Target Rs 5,000.", session="A", turn=2)
            prompt = agent.process_turn("The due date is 2027-01-31.", session="A", turn=3)
            self.assertIn("Please confirm", prompt["answer"])
            self.assertEqual(BudgetDatabase(paths.database).goals("U1"), [])
            saved = agent.process_turn("Yes, confirm it.", session="A", turn=4)
            self.assertIn("Saved savings goal", saved["answer"])
            goals = BudgetDatabase(paths.database).goals("U1")
            self.assertEqual(len(goals), 1)
            self.assertEqual(goals[0]["target_amount"], 5000.0)
            self.assertEqual(goals[0]["due_date"], "2027-01-31")

    def test_cancel_discards_pending_goal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "skills").mkdir()
            (root / "skills" / "system_prompt_v1.md").write_text("Test prompt", encoding="utf-8")
            (root / "skills" / "savings_goal_v1.md").write_text("Goal steps", encoding="utf-8")
            paths = Paths(root)
            agent = RecallBotAgent(
                paths=paths,
                user="U1",
                config="hybrid_mem",
                chat=TemplateChat(),
                embedder=FakeEmbedder(),
                logger=CallLogger(paths.logs / "calls.jsonl", run_id="cancel-test"),
            )
            agent.state["pending_goal"] = {
                "name": "Emergency fund",
                "target_amount": 5000,
                "due_date": "2027-01-31",
            }
            result = agent.process_turn("Cancel that goal.", session="A", turn=1)
            self.assertIn("nothing was saved", result["answer"])
            self.assertNotIn("pending_goal", agent.state)
            self.assertEqual(BudgetDatabase(paths.database).goals("U1"), [])

    def test_followup_uses_dialogue_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "skills").mkdir()
            (root / "skills" / "system_prompt_v1.md").write_text("Test prompt", encoding="utf-8")
            (root / "skills" / "savings_goal_v1.md").write_text("Goal steps", encoding="utf-8")
            agent = RecallBotAgent(
                paths=Paths(root),
                user="U1",
                config="hybrid_mem",
                chat=TemplateChat(),
                embedder=FakeEmbedder(),
                logger=CallLogger(root / "logs" / "calls.jsonl", run_id="test"),
            )
            agent.state["last_topic"] = "grocery cap"
            agent.state["last_topic_value"] = "9000"
            result = agent.process_turn("And for Saturday?", session="A", turn=1)
            self.assertIn("grocery cap", result["answer"])
            self.assertIn("9000", result["answer"])

    def test_context_budget_records_what_was_cut(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "skills").mkdir()
            (root / "skills" / "system_prompt_v1.md").write_text("Test prompt", encoding="utf-8")
            (root / "skills" / "savings_goal_v1.md").write_text("Goal steps", encoding="utf-8")
            agent = RecallBotAgent(
                paths=Paths(root),
                user="U1",
                config="hybrid_mem",
                chat=TemplateChat(),
                embedder=FakeEmbedder(),
                logger=CallLogger(root / "logs" / "calls.jsonl", run_id="test"),
                context_window=800,
            )
            agent.history = [
                {"role": role, "text": "old context " * 20}
                for _ in range(8)
                for role in ("user", "assistant")
            ]
            chunks = [
                {
                    "chunk": Chunk(f"D:{index}", "D", "s", "1", "2026-01-01", None, "policy " * 100, 100),
                    "score": 1.0,
                    "method": "hybrid",
                }
                for index in range(3)
            ]
            context, metrics = agent._assemble_context(
                text="What does the policy say?",
                session="B",
                turn=30,
                config="hybrid_mem",
                hits=chunks,
                semantic=[],
                episodes=[],
                database_context="",
            )
            self.assertLessEqual(
                metrics["total_tokens"] + metrics["reply_reserve"],
                metrics["context_window"],
            )
            self.assertTrue(metrics["cuts"])
            self.assertIn("User: What does the policy say?", context)

    def test_summary_strategy_logs_rolling_summary_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "skills").mkdir()
            (root / "skills" / "system_prompt_v1.md").write_text("Test prompt", encoding="utf-8")
            (root / "skills" / "savings_goal_v1.md").write_text("Goal steps", encoding="utf-8")
            (root / "skills" / "summary_prompt_v1.md").write_text("Summarize", encoding="utf-8")
            log_path = root / "logs" / "calls.jsonl"
            agent = RecallBotAgent(
                paths=Paths(root),
                user="U1",
                config="summary",
                chat=TemplateChat(),
                embedder=FakeEmbedder(),
                logger=CallLogger(log_path, run_id="summary-test"),
            )
            for turn in range(1, 4):
                agent.process_turn(f"Small talk turn {turn}.", session="A", turn=turn)
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(row["call_type"] == "summary" for row in rows), 1)


if __name__ == "__main__":
    unittest.main()
