import unittest

from services.reliability import compute_confidence, confidence_label


class TestReliability(unittest.TestCase):
    def test_compute_confidence_zero_when_no_sources(self) -> None:
        self.assertEqual(compute_confidence([], []), 0.0)

    def test_compute_confidence_with_sparse_retrieval(self) -> None:
        confidence = compute_confidence(["doc1"], [0.9])
        self.assertAlmostEqual(confidence, 0.56, places=2)

    def test_compute_confidence_caps_at_one(self) -> None:
        confidence = compute_confidence(["a", "b", "c", "d"], [1.5, 1.2, 1.1, 1.0])
        self.assertLessEqual(confidence, 1.0)
        self.assertAlmostEqual(confidence, 1.0, places=2)

    def test_confidence_label_low(self) -> None:
        self.assertEqual(confidence_label(0.2), "LOW")

    def test_confidence_label_medium(self) -> None:
        self.assertEqual(confidence_label(0.5), "MEDIUM")

    def test_confidence_label_high(self) -> None:
        self.assertEqual(confidence_label(0.9), "HIGH")


if __name__ == "__main__":
    unittest.main()
