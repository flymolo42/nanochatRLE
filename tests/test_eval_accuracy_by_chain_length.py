import unittest

import torch

from scripts.eval_accuracy_by_chain_length import bucket_by_last_slot, bucket_label


class BucketTests(unittest.TestCase):
    def test_bucket_label_caps_at_ten_plus(self):
        self.assertEqual(bucket_label(1), "1")
        self.assertEqual(bucket_label(9), "9")
        self.assertEqual(bucket_label(10), "10+")
        self.assertEqual(bucket_label(15), "10+")

    def test_bucket_by_last_slot_aggregates_topk_and_ce(self):
        contexts = [
            [[1, 2, 3], [4]],        # last slot len 1
            [[1, 2, 3]],             # last slot len 3
            [[5], [1, 2, 3]],        # last slot len 3
        ]
        # vocab 8; targets: 0, 7, 2
        logits = torch.full((3, 8), -10.0)
        logits[0, 0] = 10.0   # correct top-1
        logits[1, 6] = 10.0   # wrong; target 7 ranks below
        logits[1, 7] = 9.0    # target 7 in top-5
        logits[2, 2] = 10.0   # correct top-1
        targets = [0, 7, 2]
        buckets = bucket_by_last_slot(contexts, logits, targets)
        self.assertEqual(buckets["1"]["count"], 1)
        self.assertEqual(buckets["1"]["top1"], 1.0)
        self.assertEqual(buckets["3"]["count"], 2)
        self.assertEqual(buckets["3"]["top1"], 0.5)
        self.assertEqual(buckets["3"]["top5"], 1.0)
        self.assertIn("mean_ce", buckets["3"])


if __name__ == "__main__":
    unittest.main()
