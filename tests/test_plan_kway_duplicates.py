import json
import numpy as np
import unittest

from scripts.plan_kway_duplicates import (
    collect_position_histograms,
    fixed_centroids,
    select_candidates,
    select_k,
)


def _pair_arrays(vocab_size, pair_counts):
    codes = np.array(sorted(l * vocab_size + r for l, r in pair_counts), dtype=np.int64)
    counts = np.array([pair_counts[(int(c) // vocab_size, int(c) % vocab_size)] for c in codes], dtype=np.int64)
    return codes, counts


class PositionHistogramTests(unittest.TestCase):
    def test_relative_positions_binned(self):
        # one clause of 3 tokens (ids 5, 5, 9): positions 0, .5, 1
        streams = [[(0, 5), (0, 5), (0, 9)]]
        hist = collect_position_histograms(streams, vocab_size=10, bins=4)
        # token 5 seen at rel-pos 0.0 (bin 0) and 0.5 (bin 2)
        self.assertEqual(hist[5].tolist(), [1, 0, 1, 0])
        # token 9 at rel-pos 1.0 -> last bin
        self.assertEqual(hist[9][-1], 1)

    def test_singleton_clause_is_position_zero(self):
        hist = collect_position_histograms([[(0, 3)]], vocab_size=5, bins=4)
        self.assertEqual(hist[3][0], 1)


class SelectKTests(unittest.TestCase):
    def test_uniform_distribution_gives_k_max(self):
        row = np.ones(20, dtype=np.int64) * 5
        k, centroids = select_k(row, k_max=6)
        self.assertEqual(k, 6)
        self.assertEqual(len(centroids), 6)
        self.assertTrue(all(0.0 <= c <= 1.0 for c in centroids))

    def test_bimodal_gives_two(self):
        row = np.zeros(20, dtype=np.int64)
        row[1] = 50   # ~0.05
        row[18] = 50  # ~0.9
        k, centroids = select_k(row, k_max=6)
        self.assertEqual(k, 2)
        self.assertLess(centroids[0], 0.3)
        self.assertGreater(centroids[1], 0.7)

    def test_monomodal_gives_one(self):
        row = np.zeros(20, dtype=np.int64)
        row[9] = 100
        k, centroids = select_k(row, k_max=6)
        self.assertEqual(k, 1)

    def test_empty_row_gives_one(self):
        k, centroids = select_k(np.zeros(20, dtype=np.int64), k_max=6)
        self.assertEqual(k, 1)


class FixedAndCandidateTests(unittest.TestCase):
    def test_fixed_centroids_evenly_spaced(self):
        self.assertEqual(fixed_centroids(2), [0.25, 0.75])
        self.assertEqual(fixed_centroids(4), [0.125, 0.375, 0.625, 0.875])

    def test_select_candidates_by_conflict_mass(self):
        vocab_size = 5
        codes, counts = _pair_arrays(vocab_size, {(0, 1): 10, (1, 0): 8, (2, 3): 6, (3, 2): 6})
        # loss: t0=8,t1=8,t2=6,t3=6 -> top-3 = [0,1,2] (stable)
        self.assertEqual(select_candidates(codes, counts, vocab_size, top_n=3), [0, 1, 2])


from scripts.plan_kway_duplicates import apply_kway, build_plan


class BuildAndApplyTests(unittest.TestCase):
    def test_build_plan_fixed_k_renumbers_with_gaps(self):
        hist = np.zeros((4, 20), dtype=np.int64)
        plan = build_plan([1], hist, vocab_size=4, fixed_k=3)
        self.assertEqual(plan["vocab_size_old"], 4)
        self.assertEqual(plan["vocab_size_new"], 6)  # +2 extra copies for parent 1
        parent = plan["parents"][1]
        self.assertEqual(parent["base_new_index"], 1)          # parent keeps first slot
        self.assertEqual([c["new_index"] for c in parent["copies"]], [1, 2, 3])
        self.assertEqual([round(c["target"], 3) for c in parent["copies"]], [0.167, 0.5, 0.833])
        # non-parent id 2 renumbers to 2+2=4 (two extra copies inserted before it)
        stream = [(0, 3), (0, 2)]
        self.assertEqual([nid for _, nid in apply_kway(stream, plan)], [5, 4])

    def test_apply_assigns_nearest_target(self):
        # parent 0 with copies at targets ~0.167/0.5/0.833; a 3-token clause of
        # token 0 at rel-pos 0, .5, 1 -> nearest copies
        hist = np.zeros((1, 20), dtype=np.int64)
        plan = build_plan([0], hist, vocab_size=1, fixed_k=3)
        stream = [(0, 0), (0, 0), (0, 0)]
        got = [nid for _, nid in apply_kway(stream, plan)]
        # targets in new-index order: copy0=idx0(0.167), copy1=idx1(0.5), copy2=idx2(0.833)
        self.assertEqual(got, [0, 1, 2])

    def test_apply_kway_normalizes_json_stringified_parent_keys(self):
        # Regression test: verify that apply_kway correctly normalizes plan["parents"]
        # when keys are strings (from JSON round-trip) vs ints.
        hist = np.zeros((1, 20), dtype=np.int64)
        plan = build_plan([0], hist, vocab_size=1, fixed_k=3)
        stream = [(0, 0), (0, 0), (0, 0)]
        
        # Get result from original int-keyed plan
        result_int_keys = [nid for _, nid in apply_kway(stream, plan)]
        
        # Round-trip through JSON (which stringifies integer keys)
        json_str = json.dumps(plan)
        plan_str_keys = json.loads(json_str)
        
        # Verify keys are now strings
        self.assertTrue(all(isinstance(k, str) for k in plan_str_keys["parents"].keys()))
        
        # Apply to str-keyed plan
        result_str_keys = [nid for _, nid in apply_kway(stream, plan_str_keys)]
        
        # Both should produce identical output (apply_kway normalizes keys internally)
        self.assertEqual(result_int_keys, result_str_keys)
        self.assertEqual(result_str_keys, [0, 1, 2])

    def test_data_driven_k_uses_selected_centroids(self):
        hist = np.zeros((2, 20), dtype=np.int64)
        hist[1, 1] = 50
        hist[1, 18] = 50   # bimodal -> k=2
        plan = build_plan([1], hist, vocab_size=2, k_max=6)
        self.assertEqual(len(plan["parents"][1]["copies"]), 2)


if __name__ == "__main__":
    unittest.main()
