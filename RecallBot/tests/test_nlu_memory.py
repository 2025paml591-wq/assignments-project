import tempfile
import unittest
from pathlib import Path

from recallbot.config import Paths
from recallbot.evaluation import evaluate_nlu, validate_assignment_data
from recallbot.memory import MemoryStore
from recallbot.nlu import classify, parse_amount


class NluMemoryTests(unittest.TestCase):
    def test_code_mixed_query_normalizes_and_classifies(self):
        result = classify("Mera purana kiraya kya tha?")
        self.assertEqual(result.intent, "budget_lookup")
        self.assertIn("rent", result.normalized)

    def test_date_is_not_parsed_as_money(self):
        self.assertIsNone(parse_amount("The due date is 2027-01-31."))

    def test_memory_updates_scope_and_forget_survive_reopen(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryStore(Path(temporary))
            store.extract_and_store("U1", "My monthly rent is Rs 24,000.", 1, record_episode=False)
            store.extract_and_store("U1", "Actually rent is now Rs 26,000.", 2, record_episode=False)
            store.extract_and_store("U2", "My monthly rent is Rs 45,000.", 1, record_episode=False)
            store.record_episode("U1", "I complained about my rent.", 3)
            store.add_reflection("U1", "A", 3)
            reloaded = MemoryStore(Path(temporary))
            self.assertEqual(reloaded.active_facts("U1")[0]["value"], "26000")
            self.assertEqual(reloaded.active_facts("U2")[0]["value"], "45000")
            removed = reloaded.forget("U1", "rent")
            self.assertGreaterEqual(removed, 2)
            self.assertEqual(reloaded.active_facts("U1"), [])
            self.assertEqual(len(reloaded.active_facts("U2")), 1)
            self.assertEqual(reloaded.recall_episodes("U1", "rent"), [])

    def test_variant_dataset_meets_data_requirements(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(validate_assignment_data(Paths(root)), [])

    def test_rule_nlu_accuracy_is_measured_on_labelled_examples(self):
        root = Path(__file__).resolve().parents[1]
        result = evaluate_nlu(Paths(root))
        self.assertEqual(result["total"], 20)
        self.assertGreaterEqual(result["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
