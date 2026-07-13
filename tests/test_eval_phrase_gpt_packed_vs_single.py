import json
import tempfile
import unittest
from pathlib import Path

import torch

from nanochat.gpt import GPT, GPTConfig
from scripts.eval_packed_vs_single import PairedExample
from scripts.eval_phrase_gpt_packed_vs_single import examples_from_pairs, load_model, predict_logits


class FakePhraseGPT(torch.nn.Module):
    def __init__(self, vocab_size=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.seen_phrase_indices = []

    def forward(self, idx, phrase_indices=None, phrase_offsets=None, phrase_batch_positions=None):
        self.seen_phrase_indices.append(list(phrase_indices.cpu().tolist()))
        logits = torch.zeros((idx.size(0), idx.size(1), self.vocab_size), device=idx.device)
        for row in range(idx.size(0)):
            start = phrase_offsets[row].item()
            end = phrase_offsets[row + 1].item() if row + 1 < phrase_offsets.numel() else phrase_indices.numel()
            pred = int(phrase_indices[start:end].sum().item()) % self.vocab_size
            logits[row, 0, pred] = 10.0
        return logits


class PositionAwareFakePhraseGPT(torch.nn.Module):
    """Predicts (sum of the phrase indices at each position) % vocab_size, at every position."""

    def __init__(self, vocab_size=8):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, idx, phrase_indices=None, phrase_offsets=None, phrase_batch_positions=None):
        logits = torch.zeros((idx.size(0), idx.size(1), self.vocab_size), device=idx.device)
        ends = torch.cat([phrase_offsets[1:], torch.tensor([phrase_indices.numel()])])
        for (batch_idx, time_idx), start, end in zip(phrase_batch_positions.tolist(), phrase_offsets.tolist(), ends.tolist()):
            pred = int(phrase_indices[start:end].sum().item()) % self.vocab_size
            logits[batch_idx, time_idx, pred] = 10.0
        return logits


class RequiresTwoTokenFakePhraseGPT(FakePhraseGPT):
    def forward(self, idx, phrase_indices=None, phrase_offsets=None, phrase_batch_positions=None):
        if idx.size(1) < 2:
            raise AssertionError("Training forward pass should have T > 1")
        return super().forward(
            idx,
            phrase_indices=phrase_indices,
            phrase_offsets=phrase_offsets,
            phrase_batch_positions=phrase_batch_positions,
        )


def _story_records(story_id, phrases):
    """phrases: list of lists of token indices, one list per phrase_id."""
    records = []
    position = 0
    for phrase_id, tokens in enumerate(phrases):
        for token_pos, token in enumerate(tokens):
            records.append({
                "split": "train", "story_id": story_id, "phrase_id": phrase_id,
                "label": "punctuation", "record_type": "single", "indices": [token],
                "token_pos": token_pos, "start": position, "end": position + len(tokens),
            })
        records.append({
            "split": "train", "story_id": story_id, "phrase_id": phrase_id,
            "label": "punctuation", "record_type": "packed", "indices": list(tokens),
            "start": position, "end": position + len(tokens),
        })
        position += len(tokens)
    return records


class EvalPhraseGPTPackedVsSingleTests(unittest.TestCase):
    def test_build_context_examples_windows_phrases_and_targets_next_phrase(self):
        from scripts.eval_phrase_gpt_packed_vs_single import build_context_examples

        records = _story_records(0, [[1, 2], [3], [4, 5], [6]])

        examples = build_context_examples(records, context_phrases=2)

        # windows of 2 phrases, target = first token of the following phrase
        self.assertEqual(
            [(ex.single_steps, ex.packed_steps, ex.target_index) for ex in examples],
            [
                ([[1], [2], [3]], [[1, 2], [3]], 4),
                ([[3], [4], [5]], [[3], [4, 5]], 6),
            ],
        )

    def test_build_context_examples_synthesizes_packed_step_from_singles(self):
        from scripts.eval_phrase_gpt_packed_vs_single import build_context_examples

        records = _story_records(0, [[1, 2], [3], [4]])
        # drop the packed record for phrase 1 (single-token phrases often have none)
        records = [
            record for record in records
            if not (record["phrase_id"] == 1 and record["record_type"] == "packed")
        ]

        examples = build_context_examples(records, context_phrases=2)

        self.assertEqual(
            [(ex.single_steps, ex.packed_steps, ex.target_index) for ex in examples],
            [([[1], [2], [3]], [[1, 2], [3]], 4)],
        )

    def test_build_context_examples_does_not_cross_stories(self):
        from scripts.eval_phrase_gpt_packed_vs_single import build_context_examples

        records = _story_records(0, [[1], [2]]) + _story_records(1, [[3], [4], [5]])

        examples = build_context_examples(records, context_phrases=2)

        # story 0 has no window of 2 phrases with a following phrase; story 1 has one
        self.assertEqual(
            [(ex.single_steps, ex.target_index) for ex in examples],
            [([[3], [4]], 5)],
        )

    def test_predict_context_predictions_scores_last_position_in_original_order(self):
        from scripts.eval_phrase_gpt_packed_vs_single import ContextExample, predict_context_predictions

        examples = [
            ContextExample(single_steps=[[1], [2]], packed_steps=[[1, 2]], target_index=2),
            ContextExample(single_steps=[[3], [4], [5]], packed_steps=[[3], [4, 5]], target_index=1),
        ]
        model = PositionAwareFakePhraseGPT(vocab_size=8)

        # batch_size=1 forces length-sorting to reorder internally; result must stay in input order
        single = predict_context_predictions(model, examples, mode="single", batch_size=1, device="cpu")
        packed = predict_context_predictions(model, examples, mode="packed", batch_size=1, device="cpu")

        # single mode: last steps are [2] and [5] -> preds 2 and 5
        self.assertEqual(single.tolist(), [2, 5])
        # packed mode: last steps are [1,2] (sum 3) and [4,5] (sum 9 % 8 = 1) -> preds 3 and 1
        self.assertEqual(packed.tolist(), [3, 1])

    def test_compute_context_metrics_matches_pair_metrics_shape(self):
        from scripts.eval_phrase_gpt_packed_vs_single import ContextExample, compute_context_metrics

        examples = [
            ContextExample(single_steps=[[1]], packed_steps=[[1]], target_index=2),
            ContextExample(single_steps=[[3]], packed_steps=[[3]], target_index=3),
        ]
        single_preds = torch.tensor([2, 1])   # first correct, second wrong
        packed_preds = torch.tensor([2, 3])   # both correct

        metrics = compute_context_metrics(examples, single_preds, packed_preds)

        self.assertEqual(metrics["paired_examples"], 2)
        self.assertAlmostEqual(metrics["seq_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["multihot_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["multihot_wins"], 0.5)
    def test_examples_from_pairs_uses_single_or_packed_indices(self):
        pairs = [
            PairedExample([1], [1, 2], 3),
            PairedExample([4], [4, 5], 6),
        ]

        single_examples = examples_from_pairs(pairs, mode="single")
        packed_examples = examples_from_pairs(pairs, mode="packed")

        self.assertEqual([example.input_indices for example in single_examples], [[[1]], [[4]]])
        self.assertEqual([example.input_indices for example in packed_examples], [[[1, 2]], [[4, 5]]])
        self.assertEqual([example.targets for example in packed_examples], [[3], [6]])

    def test_predict_logits_returns_position_zero_logits_for_gpt_pairs(self):
        pairs = [
            PairedExample([1], [1, 2], 3),
            PairedExample([4], [4, 5], 6),
        ]
        model = FakePhraseGPT(vocab_size=8)

        logits = predict_logits(model, pairs, mode="packed", batch_size=2, sequence_len=4, dummy_token_id=0, device="cpu")

        self.assertEqual(logits.shape, (2, 8))
        self.assertEqual(logits.argmax(dim=1).tolist(), [3, 1])
        self.assertEqual(model.seen_phrase_indices[0], [1, 2, 4, 5])

    def test_predict_logits_default_sequence_len_satisfies_nanochat_smear_minimum(self):
        pairs = [
            PairedExample([1], [1, 2], 3),
        ]
        model = RequiresTwoTokenFakePhraseGPT(vocab_size=8)

        logits = predict_logits(model, pairs, mode="single", batch_size=1, dummy_token_id=0, device="cpu")

        self.assertEqual(logits.shape, (1, 8))

    def test_remap_paired_examples_maps_inputs_and_targets(self):
        from scripts.eval_phrase_gpt_packed_vs_single import remap_paired_examples

        pairs = [
            PairedExample([1], [1, 2], 3),
            PairedExample([0], [0, 3], 2),
        ]
        # old->new: 0 and 3 pruned to unk (=2), 1->0, 2->1
        lookup = torch.tensor([2, 0, 1, 2], dtype=torch.long)

        remapped = remap_paired_examples(pairs, lookup)

        self.assertEqual(
            [(pair.single_indices, pair.packed_indices, pair.target_index) for pair in remapped],
            [([0], [0, 1], 2), ([2], [2, 2], 1)],
        )
        self.assertEqual(pairs[0].single_indices, [1])  # input untouched

    def test_resolve_vocab_remap_reads_top_k_from_checkpoint_config(self):
        from scripts.eval_phrase_gpt_packed_vs_single import resolve_vocab_remap

        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            vocab_path.write_text(json.dumps([
                {"token": f"t{i}", "index": i, "count": 100 - i} for i in range(4)
            ]), encoding="utf-8")

            lookup = resolve_vocab_remap({"vocab": str(vocab_path), "vocab_top_k": 2})
            self.assertEqual(lookup.tolist(), [0, 1, 2, 2])

            self.assertIsNone(resolve_vocab_remap({"vocab": str(vocab_path), "vocab_top_k": None}))
            self.assertIsNone(resolve_vocab_remap({}))

            override = resolve_vocab_remap({"vocab": "does/not/exist.json", "vocab_top_k": 2}, vocab_override=str(vocab_path))
            self.assertEqual(override.tolist(), [0, 1, 2, 2])

    def test_load_model_ignores_checkpoint_config_keys_that_are_not_gpt_config(self):
        with self.subTest("checkpoint with saved cli args"):
            config = GPTConfig(
                sequence_len=4,
                vocab_size=8,
                n_layer=1,
                n_head=2,
                n_kv_head=2,
                n_embd=32,
                window_pattern="L",
                phrase_vocab_size=6,
            )
            model = GPT(config)
            model.init_weights()
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config.__dict__ | {
                    "vocab": "phrase_vectors_out/vocab.json",
                    "records": "phrase_vectors_out/phrase_index.jsonl",
                    "out_dir": "phrase_gpt_out",
                },
                "metrics": {},
            }
            path = "/tmp/test_phrase_gpt_extra_config.pt"
            torch.save(checkpoint, path)

            loaded, loaded_config = load_model(path, device="cpu")

            self.assertEqual(loaded_config.vocab_size, 8)
            self.assertEqual(loaded_config.phrase_vocab_size, 6)
            self.assertIsInstance(loaded, GPT)


if __name__ == "__main__":
    unittest.main()


class ApplyIndexMapTests(unittest.TestCase):
    def test_remaps_singles_and_sorts_packed(self):
        from scripts.eval_phrase_gpt_packed_vs_single import apply_index_map
        records = [
            {"record_type": "single", "indices": [1], "split": "validation", "story_id": 0, "phrase_id": 0},
            {"record_type": "packed", "indices": [1, 2], "split": "validation", "story_id": 0, "phrase_id": 0},
        ]
        mapped = apply_index_map(records, [0, 9, 4])
        self.assertEqual(mapped[0]["indices"], [9])
        self.assertEqual(mapped[1]["indices"], [4, 9])
        # originals untouched
        self.assertEqual(records[1]["indices"], [1, 2])
