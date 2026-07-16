import unittest

from scripts.run_mixed_corpus_experiment import order_from, run_experiment


def _prose():  # ascending-friendly under identity: ids rise within clause
    return iter([[(0, 0), (0, 1), (0, 2)] for _ in range(20)])


def _code():   # different regularity: ids fall
    return iter([[(0, 2), (0, 1), (0, 0)] for _ in range(20)])


class OrderFromTests(unittest.TestCase):
    def test_order_from_returns_positions_and_fraction(self):
        positions, frac = order_from(_prose, vocab_size=3, max_passes=5,
                                     ils_restarts=1, ils_generations=1, jobs=1)
        self.assertEqual(sorted(positions.tolist()), [0, 1, 2])
        self.assertGreaterEqual(frac, 0.0)


class RunExperimentTests(unittest.TestCase):
    def test_report_has_bespoke_mixed_and_kway_per_domain(self):
        report = run_experiment(_prose, _prose, _code, _code, vocab_size=3,
                                top_n=3, k_max=4, max_passes=5, ils_restarts=1,
                                ils_generations=1, jobs=1, max_chain_len=9)
        for domain in ("prose", "code"):
            self.assertIn("bespoke", report[domain])
            self.assertIn("mixed", report[domain])
            self.assertIn("mixed_kway", report[domain])
            self.assertGreater(report[domain]["bespoke"]["chains"], 0)
        self.assertIn("kway_extra_slots", report)


if __name__ == "__main__":
    unittest.main()
