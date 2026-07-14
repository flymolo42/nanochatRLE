import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.plan_token_duplicates import (
    build_plan,
    conflict_losses,
    renumber_array,
    select_duplicates,
)


def _pair_arrays(vocab_size, pair_counts):
    codes = np.array(sorted(left * vocab_size + right for left, right in pair_counts), dtype=np.int64)
    counts = np.array([pair_counts[(int(c) // vocab_size, int(c) % vocab_size)] for c in codes], dtype=np.int64)
    return codes, counts


class ConflictLossTests(unittest.TestCase):
    def test_loss_is_min_mass_per_conflicted_pair(self):
        vocab_size = 4
        codes, counts = _pair_arrays(vocab_size, {(0, 1): 5, (1, 0): 3, (2, 3): 7})
        loss, directed = conflict_losses(codes, counts, vocab_size)
        # pair {0,1} conflicted with min 3 -> both endpoints get 3; pair (2,3) one-directional -> 0
        self.assertEqual(loss.tolist(), [3.0, 3.0, 0.0, 0.0])
        # directed mass: token 0: out 5 + in 3 = 8; token 2: out 7; token 3: in 7
        self.assertEqual(directed.tolist(), [8.0, 8.0, 7.0, 7.0])

    def test_select_excludes_low_share_and_ranks_by_loss(self):
        vocab_size = 6
        # token 0<->1 heavy conflict; token 2->3 huge one-way plus tiny conflict (share < 5%)
        codes, counts = _pair_arrays(vocab_size, {
            (0, 1): 10, (1, 0): 8,
            (2, 3): 1000, (3, 2): 4,
            (4, 5): 6, (5, 4): 6,
        })
        parents = select_duplicates(codes, counts, vocab_size, num_duplicates=3, min_share=0.05)
        # loss: t0=8, t1=8, t2=4(share 4/1004<0.05 excluded), t3 likewise, t4=6, t5=6
        self.assertEqual(parents, [0, 1, 4])  # top-3 by loss among eligible, sorted


class RenumberTests(unittest.TestCase):
    def test_renumber_inserts_copy_slots_after_parents(self):
        new = renumber_array(6, [1, 3])
        # old:      0  1  2  3  4  5
        # parents:     ^     ^
        # new ids:  0  1  3  4  6  7   (late copies occupy 2 and 5)
        self.assertEqual(new.tolist(), [0, 1, 3, 4, 6, 7])

    def test_renumber_no_parents_is_identity(self):
        self.assertEqual(renumber_array(4, []).tolist(), [0, 1, 2, 3])


class BuildPlanTests(unittest.TestCase):
    def test_build_plan_end_to_end(self):
        vocab_size = 6
        codes, counts = _pair_arrays(vocab_size, {(0, 1): 10, (1, 0): 8, (4, 5): 6, (5, 4): 6})
        vocab = [{"token": f"tok{i}", "index": i, "count": 100} for i in range(vocab_size)]
        with tempfile.TemporaryDirectory() as tmpdir:
            pc_path = Path(tmpdir) / "pair_counts.npz"
            np.savez(pc_path, codes=codes, counts=counts, vocab_size=np.int64(vocab_size), stories=np.int64(1))
            vocab_path = Path(tmpdir) / "vocab.json"
            vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
            plan = build_plan(pc_path, vocab_path, num_duplicates=2, min_share=0.05)
        self.assertEqual(plan["format"], "duplicates_plan_v1")
        self.assertEqual(plan["vocab_size_old"], 6)
        self.assertEqual(plan["vocab_size_new"], 8)
        self.assertEqual([p["old_index"] for p in plan["parents"]], [0, 1])
        first = plan["parents"][0]
        self.assertEqual(first["token"], "tok0")
        self.assertEqual(first["early_new_index"], 0)
        self.assertEqual(first["late_new_index"], 1)
        second = plan["parents"][1]
        self.assertEqual(second["early_new_index"], 2)
        self.assertEqual(second["late_new_index"], 3)


if __name__ == "__main__":
    unittest.main()
