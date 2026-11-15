import math
import unittest

from recallbot.retrieval import BM25Index, Chunk, Chunker, reciprocal_rank_fusion


class RetrievalTests(unittest.TestCase):
    def test_bm25_matches_hand_calculated_example(self):
        chunks = [
            Chunk("a", "D1", "s", "1", "2026-01-01", None, "budget now", 2),
            Chunk("b", "D2", "s", "1", "2026-01-01", None, "rent", 1),
        ]
        score = BM25Index(chunks).search("budget")[0][1]
        self.assertAlmostEqual(score, math.log(2) * (2.5 / 2.875), places=8)

    def test_rrf_matches_course_tie_and_order_example(self):
        rankings = [
            ["A", "C", "B", "E", "D"],
            ["B", "A", "D", "F", "C"],
        ]
        result = reciprocal_rank_fusion(rankings, k=60, limit=6)
        self.assertEqual([identifier for identifier, _ in result], ["A", "B", "C", "D", "E", "F"])
        self.assertAlmostEqual(result[4][1], result[5][1])

    def test_bm25_matches_cs4_slide_example(self):
        score = BM25Index.score_terms(
            {"machine", "learning"},
            {"machine": 2, "learning": 3},
            document_length=12,
            average_length=50,
            total_documents=1000,
            document_frequencies={"machine": 300, "learning": 400},
        )
        self.assertEqual(round(score, 3), 4.160)

    def test_chunker_counts_requested_tokens_with_overlap(self):
        chunks = Chunker().chunk(" ".join(["budgeting"] * 400), size=128, overlap_ratio=0.15)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(count <= 128 for _, count in chunks))

    def test_chunker_rejects_non_assignment_overlap(self):
        with self.assertRaises(ValueError):
            Chunker().chunk("some text", size=128, overlap_ratio=0.25)


if __name__ == "__main__":
    unittest.main()
