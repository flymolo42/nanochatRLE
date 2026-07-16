# tests/test_run_kway_experiment.py
import numpy as np
import unittest

from scripts.run_kway_experiment import run_arm
from scripts.plan_kway_duplicates import build_plan


class RunArmTests(unittest.TestCase):
    def test_no_dup_arm_reports_chains(self):
        # 6 identical 3-token clauses; ids in [0,3)
        streams = [[(i, 0), (i, 1), (i, 2)] for i in range(6)]
        plan = build_plan([], np.zeros((3, 20), dtype=np.int64), vocab_size=3, fixed_k=None)  # empty -> no copies
        result = run_arm("nodup", lambda: iter(streams), lambda: iter(streams[:2]),
                         vocab_size=3, plan=plan, out_dir=None, max_passes=5,
                         ils_restarts=1, ils_generations=1, jobs=1, max_chain_len=9)
        self.assertEqual(result["arm"], "nodup")
        self.assertEqual(result["extra_slots"], 0)
        self.assertGreater(result["validation_chains"]["chains"], 0)
        self.assertIn("ascending_fraction", result)

    def test_kway_arm_expands_vocab(self):
        streams = [[(i, 0), (i, 1), (i, 0)] for i in range(8)]
        hist = np.zeros((2, 20), dtype=np.int64)
        hist[0, 1] = 20
        hist[0, 18] = 20  # bimodal -> k=2 for token 0
        plan = build_plan([0], hist, vocab_size=2, k_max=6)
        result = run_arm("kway", lambda: iter(streams), lambda: iter(streams[:2]),
                         vocab_size=2, plan=plan, out_dir=None, max_passes=5,
                         ils_restarts=1, ils_generations=1, jobs=1, max_chain_len=9)
        self.assertEqual(result["extra_slots"], 1)  # token 0 -> 2 copies = +1 slot


if __name__ == "__main__":
    unittest.main()
