import tempfile
import unittest
from pathlib import Path

import torch

from scripts.train_phrase_vectors import (
    EarlyStoppingConfig,
    EarlyStoppingState,
    PhraseTokenPredictor,
    build_training_examples,
    choose_device,
    collate_phrase_examples,
    normalize_phrase_records,
    load_vocab,
    run_epoch,
)


class TrainPhraseVectorTests(unittest.TestCase):
    def test_load_vocab_uses_ordered_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            vocab_path.write_text(
                '[{"token": "a", "index": 0}, {"token": "cat", "index": 1}, {"token": ".", "index": 2}]\n',
                encoding="utf-8",
            )

            vocab = load_vocab(vocab_path)

        self.assertEqual(vocab.token_to_index, {"a": 0, "cat": 1, ".": 2})
        self.assertEqual(vocab.size, 3)

    def test_build_training_examples_targets_each_next_token_in_following_phrase(self):
        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "record_type": "packed", "indices": [0, 1]},
            {"split": "train", "story_id": 0, "phrase_id": 1, "record_type": "packed", "indices": [2, 3]},
            {"split": "train", "story_id": 1, "phrase_id": 0, "record_type": "packed", "indices": [4]},
        ]

        examples = build_training_examples(records)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].input_indices, [0, 1])
        self.assertEqual(examples[0].target_index, 2)
        self.assertEqual(examples[1].input_indices, [0, 1])
        self.assertEqual(examples[1].target_index, 3)

    def test_normalize_phrase_records_expands_legacy_rows_to_singletons_and_valid_pack(self):
        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "tokens": ["a", "cat"], "indices": [0, 1]},
            {"split": "train", "story_id": 0, "phrase_id": 1, "tokens": ["cat", "a"], "indices": [1, 0]},
        ]

        normalized = normalize_phrase_records(records)

        self.assertEqual(
            [(record["record_type"], record["indices"]) for record in normalized],
            [
                ("single", [0]),
                ("single", [1]),
                ("packed", [0, 1]),
                ("single", [1]),
                ("single", [0]),
            ],
        )

    def test_collate_phrase_examples_builds_embeddingbag_inputs_and_token_targets(self):
        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "record_type": "packed", "indices": [0, 2]},
            {"split": "train", "story_id": 0, "phrase_id": 1, "record_type": "packed", "indices": [1, 3]},
            {"split": "train", "story_id": 0, "phrase_id": 2, "record_type": "packed", "indices": [2]},
        ]
        examples = build_training_examples(records)

        batch = collate_phrase_examples(examples, device="cpu")

        self.assertEqual(batch.flat_indices.tolist(), [0, 2, 0, 2, 1, 3])
        self.assertEqual(batch.offsets.tolist(), [0, 2, 4])
        self.assertEqual(batch.targets.tolist(), [1, 3, 2])

    def test_phrase_token_predictor_outputs_one_vocab_logit_row_per_example(self):
        model = PhraseTokenPredictor(vocab_size=5, hidden_size=8)
        flat_indices = torch.tensor([0, 2, 4], dtype=torch.long)
        offsets = torch.tensor([0, 2], dtype=torch.long)

        logits = model(flat_indices, offsets)

        self.assertEqual(logits.shape, (2, 5))

    def test_run_epoch_reports_progress_when_requested(self):
        examples = build_training_examples([
            {"split": "train", "story_id": 0, "phrase_id": 0, "record_type": "packed", "indices": [0, 1]},
            {"split": "train", "story_id": 0, "phrase_id": 1, "record_type": "packed", "indices": [2]},
            {"split": "train", "story_id": 0, "phrase_id": 2, "record_type": "packed", "indices": [3]},
        ])
        model = PhraseTokenPredictor(vocab_size=4, hidden_size=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        lines = []

        run_epoch(
            model,
            examples,
            optimizer=optimizer,
            batch_size=1,
            device="cpu",
            progress_every=1,
            progress_label="train epoch 1",
            output_fn=lines.append,
        )

        self.assertTrue(any("train epoch 1 batch 1/2" in line for line in lines))
        self.assertTrue(all("loss=" in line and "acc=" in line for line in lines))

    def test_choose_device_uses_cpu_when_only_mps_is_available(self):
        original_cuda_available = torch.cuda.is_available
        original_mps_available = torch.backends.mps.is_available
        torch.cuda.is_available = lambda: False
        torch.backends.mps.is_available = lambda: True
        try:
            self.assertEqual(choose_device(""), "cpu")
        finally:
            torch.cuda.is_available = original_cuda_available
            torch.backends.mps.is_available = original_mps_available

    def test_early_stopping_tracks_best_validation_loss_and_patience(self):
        config = EarlyStoppingConfig(patience=1, min_delta=0.01)
        state = EarlyStoppingState()

        state.update({"loss": 1.0, "accuracy": 0.1}, epoch=1, config=config)
        self.assertTrue(state.is_best)
        self.assertFalse(state.should_stop)

        state.update({"loss": 0.995, "accuracy": 0.2}, epoch=2, config=config)
        self.assertFalse(state.is_best)
        self.assertTrue(state.should_stop)
        self.assertEqual(state.stop_reason, "validation loss did not improve for 1 epoch")

    def test_early_stopping_stops_when_target_accuracy_is_reached(self):
        config = EarlyStoppingConfig(target_val_accuracy=0.75)
        state = EarlyStoppingState()

        state.update({"loss": 1.0, "accuracy": 0.8}, epoch=3, config=config)

        self.assertTrue(state.should_stop)
        self.assertEqual(state.stop_reason, "target validation accuracy reached")


if __name__ == "__main__":
    unittest.main()
