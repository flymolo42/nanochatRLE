import unittest

from scripts.eval_chain_split_ablation import explode_last_slot, paired_deltas, split_last_slot


class SplitVariantTests(unittest.TestCase):
    def test_split_last_slot_at_midpoint_preserves_order(self):
        context = [[9], [1, 4, 6, 7, 8]]
        self.assertEqual(split_last_slot(context), [[9], [1, 4, 6], [7, 8]])

    def test_split_last_slot_length_two(self):
        self.assertEqual(split_last_slot([[3, 5]]), [[3], [5]])

    def test_split_leaves_original_untouched(self):
        context = [[1, 2, 3]]
        split_last_slot(context)
        self.assertEqual(context, [[1, 2, 3]])

    def test_explode_last_slot_to_singletons(self):
        self.assertEqual(explode_last_slot([[9], [1, 4, 6]]), [[9], [1], [4], [6]])


class PairedDeltaTests(unittest.TestCase):
    def test_paired_deltas_bucketed_by_length(self):
        records = [
            {"length": 3, "intact_top1": 1, "variant_top1": 0, "intact_ce": 1.0, "variant_ce": 2.0},
            {"length": 3, "intact_top1": 1, "variant_top1": 1, "intact_ce": 1.0, "variant_ce": 1.0},
            {"length": 5, "intact_top1": 0, "variant_top1": 1, "intact_ce": 3.0, "variant_ce": 1.0},
        ]
        buckets = paired_deltas(records)
        self.assertEqual(buckets["3"]["count"], 2)
        self.assertAlmostEqual(buckets["3"]["delta_top1"], 0.5)   # intact better by 0.5
        self.assertAlmostEqual(buckets["3"]["delta_ce"], -0.5)    # mean of -1.0 and 0.0
        self.assertAlmostEqual(buckets["5"]["delta_top1"], -1.0)  # split better
        self.assertIn("all", buckets)
        self.assertEqual(buckets["all"]["count"], 3)


if __name__ == "__main__":
    unittest.main()
