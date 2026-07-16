import unittest

import torch

from scripts.eval_packed_vs_single import (
    PairedExample,
    build_paired_examples,
    compute_pair_metrics,
)


class EvalPackedVsSingleTests(unittest.TestCase):
    def test_build_paired_examples_compares_last_singleton_to_same_phrase_pack(self):
        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "record_type": "single", "indices": [0], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "record_type": "single", "indices": [1], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "record_type": "packed", "indices": [0, 1]},
            {"split": "train", "story_id": 0, "phrase_id": 1, "record_type": "single", "indices": [2], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 1, "record_type": "packed", "indices": [2, 3]},
        ]

        pairs = build_paired_examples(records)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].single_indices, [1])
        self.assertEqual(pairs[0].packed_indices, [0, 1])
        self.assertEqual(pairs[0].target_index, 2)

    def test_build_paired_examples_accepts_legacy_phrase_rows(self):
        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "tokens": ["a", "cat"], "indices": [0, 1]},
            {"split": "train", "story_id": 0, "phrase_id": 1, "tokens": ["sat", "."], "indices": [2, 3]},
        ]

        pairs = build_paired_examples(records)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].single_indices, [1])
        self.assertEqual(pairs[0].packed_indices, [0, 1])
        self.assertEqual(pairs[0].target_index, 2)

    def test_compute_pair_metrics_reports_accuracy_agreement_and_win_rates(self):
        pairs = [
            PairedExample([0], [0, 1], 2),
            PairedExample([1], [1, 2], 3),
            PairedExample([2], [2, 3], 4),
            PairedExample([3], [3, 4], 5),
        ]
        single_logits = torch.tensor([
            [0.0, 0.0, 9.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 9.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
        ])
        packed_logits = torch.tensor([
            [0.0, 0.0, 9.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 9.0, 0.0],
            [0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
        ])

        metrics = compute_pair_metrics(pairs, single_logits, packed_logits)

        self.assertEqual(metrics["paired_examples"], 4)
        self.assertEqual(metrics["seq_accuracy"], 0.5)
        self.assertEqual(metrics["multihot_accuracy"], 0.5)
        self.assertEqual(metrics["same_prediction_rate"], 0.5)
        self.assertEqual(metrics["seq_wins"], 0.25)
        self.assertEqual(metrics["multihot_wins"], 0.25)
        self.assertEqual(metrics["both_correct"], 0.25)
        self.assertEqual(metrics["both_wrong"], 0.25)

    def test_compute_pair_metrics_handles_empty_pairs(self):
        metrics = compute_pair_metrics([], torch.empty((0, 0)), torch.empty((0, 0)))

        self.assertEqual(metrics["paired_examples"], 0)
        self.assertEqual(metrics["seq_accuracy"], 0.0)
        self.assertEqual(metrics["multihot_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
