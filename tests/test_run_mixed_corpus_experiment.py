import unittest

from scripts.run_mixed_corpus_experiment import every_nth, order_from, run_experiment


def _prose():  # ascending-friendly under identity: ids rise within clause
    return iter([[(0, 0), (0, 1), (0, 2)] for _ in range(20)])


def _code():   # different regularity: ids fall
    return iter([[(0, 2), (0, 1), (0, 0)] for _ in range(20)])


class OrderFromTests(unittest.TestCase):
    def test_order_from_returns_positions_fraction_and_pairs(self):
        positions, frac, total_pairs = order_from(_prose, vocab_size=3, max_passes=5,
                                                  ils_restarts=1, ils_generations=1, jobs=1)
        self.assertEqual(sorted(positions.tolist()), [0, 1, 2])
        self.assertGreaterEqual(frac, 0.0)
        # 20 clauses of 3 ascending tokens -> 2 within-clause pairs each = 40
        self.assertEqual(total_pairs, 40)


class EveryNthTests(unittest.TestCase):
    def test_keeps_every_nth_stream_deterministically(self):
        base = lambda: iter([["a"], ["b"], ["c"], ["d"], ["e"]])
        thinned = every_nth(base, 2)
        self.assertEqual(list(thinned()), [["a"], ["c"], ["e"]])
        # factory re-yields fresh each call
        self.assertEqual(list(thinned()), [["a"], ["c"], ["e"]])

    def test_keep_every_one_is_identity(self):
        base = lambda: iter([["a"], ["b"]])
        self.assertEqual(list(every_nth(base, 1)()), [["a"], ["b"]])


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
        # duplication breakdown surfaced: which ids got how many copies
        self.assertIn("duplication", report)
        for entry in report["duplication"]:
            self.assertEqual(set(entry), {"old_index", "k", "targets"})
            self.assertGreaterEqual(entry["k"], 2)
            self.assertEqual(len(entry["targets"]), entry["k"])
        # per-corpus pair volumes surfaced (the size-confound quantity)
        self.assertEqual(set(report["pairs"]), {"prose", "code", "mixed", "mixed_kway"})
        self.assertEqual(report["pairs"]["mixed"], report["pairs"]["prose"] + report["pairs"]["code"])

    def test_balanced_mix_thins_prose_toward_code_pairs(self):
        # prose contributes 4x the pairs of code; balancing should thin prose ~4:1
        prose_heavy = lambda: iter([[(0, 0), (0, 1), (0, 2)] for _ in range(40)])
        code_light = lambda: iter([[(0, 2), (0, 1), (0, 0)] for _ in range(10)])
        report = run_experiment(prose_heavy, prose_heavy, code_light, code_light, vocab_size=3,
                                top_n=3, k_max=4, max_passes=5, ils_restarts=1,
                                ils_generations=1, jobs=1, max_chain_len=9, balance_pairs=True)
        self.assertIn("balance", report)
        self.assertGreater(report["balance"]["prose_keep_every"], 1)
        # balanced mixed pairs are far closer to code than the unbalanced prose total
        self.assertLess(report["pairs"]["mixed"], report["pairs"]["prose"])


class PredrankMixedTests(unittest.TestCase):
    def test_report_has_mixed_kway_predrank_per_domain(self):
        report = run_experiment(_prose, _prose, _code, _code, vocab_size=3,
                                top_n=3, k_max=4, max_passes=5, ils_restarts=1,
                                ils_generations=1, jobs=1, max_chain_len=9)
        for domain in ("prose", "code"):
            self.assertIn("mixed_kway_predrank", report[domain])
            self.assertGreater(report[domain]["mixed_kway_predrank"]["chains"], 0)


if __name__ == "__main__":
    unittest.main()
