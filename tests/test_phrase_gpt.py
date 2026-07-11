import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import nanochat.flash_attention as fa_module
from nanochat.gpt import GPT, GPTConfig
from scripts.train_phrase_gpt import (
    EarlyStoppingConfig,
    EarlyStoppingState,
    PhraseSequenceExample,
    build_phrase_sequence_examples,
    build_phrase_sequence_examples_streaming,
    collate_phrase_sequences,
    examples_to_tensor_shard,
    tensor_shard_to_examples,
    load_shard_manifest,
    iter_shard_example_sets,
)


def _force_sdpa():
    fa_module._override_impl = "sdpa"
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()


def _tiny_phrase_gpt():
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
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()
    return model


def _write_shard_fixture(tmp, train_shards, val_examples=None, sequence_len=4, vocab_size=6):
    tmp = Path(tmp)
    (tmp / "vocab.json").write_text(json.dumps(
        [{"token": f"t{i}", "index": i, "count": 100 - i} for i in range(vocab_size)]
    ), encoding="utf-8")
    shards = []
    splits = {"train": {"num_shards": len(train_shards), "num_examples": sum(len(s) for s in train_shards)}}
    for shard_idx, shard_examples in enumerate(train_shards):
        filename = f"train_shard_{shard_idx:05d}.pt"
        torch.save(examples_to_tensor_shard(shard_examples, sequence_len=sequence_len), tmp / filename)
        shards.append({"file": filename, "split": "train", "num_examples": len(shard_examples)})
    if val_examples:
        torch.save(examples_to_tensor_shard(val_examples, sequence_len=sequence_len), tmp / "validation_shard_00000.pt")
        shards.append({"file": "validation_shard_00000.pt", "split": "validation", "num_examples": len(val_examples)})
        splits["validation"] = {"num_shards": 1, "num_examples": len(val_examples)}
    (tmp / "manifest.json").write_text(json.dumps({
        "format": "phrase_gpt_tensor_shard_manifest_v1",
        "sequence_len": sequence_len,
        "num_examples": sum(len(s) for s in train_shards) + len(val_examples or []),
        "splits": splits,
        "shards": shards,
    }), encoding="utf-8")
    return str(tmp / "vocab.json"), str(tmp / "manifest.json")


def _main_argv(vocab, manifest, out_dir, extra=()):
    return [
        "train_phrase_gpt",
        "--vocab", vocab,
        "--shards", manifest,
        "--out-dir", out_dir,
        "--sequence-len", "4",
        "--depth", "1",
        "--n-embd", "32",
        "--n-head", "2",
        "--batch-size", "2",
        "--device", "cpu",
        "--progress-every", "0",
        *extra,
    ]


class PhraseGPTTests(unittest.TestCase):
    def test_phrase_vectors_change_logits_when_token_ids_are_dummy(self):
        _force_sdpa()
        torch.manual_seed(123)
        model = _tiny_phrase_gpt()
        idx = torch.zeros((1, 4), dtype=torch.long)

        logits_a = model(
            idx,
            phrase_indices=torch.tensor([1, 2], dtype=torch.long),
            phrase_offsets=torch.tensor([0], dtype=torch.long),
            phrase_batch_positions=torch.tensor([[0, 1]], dtype=torch.long),
        )
        logits_b = model(
            idx,
            phrase_indices=torch.tensor([3, 4], dtype=torch.long),
            phrase_offsets=torch.tensor([0], dtype=torch.long),
            phrase_batch_positions=torch.tensor([[0, 1]], dtype=torch.long),
        )

        self.assertFalse(torch.allclose(logits_a[:, 1, :], logits_b[:, 1, :]))

    def test_phrase_vectors_can_be_the_only_content_for_training_loss(self):
        _force_sdpa()
        torch.manual_seed(123)
        model = _tiny_phrase_gpt()
        idx = torch.zeros((2, 4), dtype=torch.long)
        targets = torch.tensor(
            [
                [1, 2, 3, -1],
                [2, 3, 4, -1],
            ],
            dtype=torch.long,
        )

        loss = model(
            idx,
            targets=targets,
            phrase_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            phrase_offsets=torch.tensor([0, 2], dtype=torch.long),
            phrase_batch_positions=torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        )

        self.assertTrue(torch.isfinite(loss))

    def test_collate_phrase_sequences_builds_dummy_ids_targets_and_sparse_phrase_inputs(self):
        examples = [
            PhraseSequenceExample(
                input_indices=[[0], [1, 2], [3]],
                targets=[1, 2, 3],
            ),
            PhraseSequenceExample(
                input_indices=[[2, 3], [4]],
                targets=[4, 5],
            ),
        ]

        batch = collate_phrase_sequences(examples, sequence_len=4, dummy_token_id=0, device="cpu")

        self.assertEqual(batch.idx.tolist(), [[0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(batch.targets.tolist(), [[1, 2, 3, -1], [4, 5, -1, -1]])
        self.assertEqual(batch.phrase_indices.tolist(), [0, 1, 2, 3, 2, 3, 4])
        self.assertEqual(batch.phrase_offsets.tolist(), [0, 1, 3, 4, 6])
        self.assertEqual(batch.phrase_batch_positions.tolist(), [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]])

    def test_tensor_shard_round_trip_preserves_sequence_examples(self):
        examples = [
            PhraseSequenceExample(input_indices=[[0], [1, 2], [3]], targets=[1, 2, 3]),
            PhraseSequenceExample(input_indices=[[2, 3], [4]], targets=[4, 5]),
        ]

        shard = examples_to_tensor_shard(examples, sequence_len=4)
        restored = tensor_shard_to_examples(shard)

        self.assertEqual(shard["targets"].shape, (2, 4))
        self.assertEqual(shard["vector_offsets"].shape, (2, 5))
        self.assertEqual(
            [(example.input_indices, example.targets) for example in restored],
            [(example.input_indices, example.targets) for example in examples],
        )

    def test_iter_shard_example_sets_loads_tensor_shards_from_manifest(self):
        import json
        import tempfile
        from pathlib import Path

        examples = [
            PhraseSequenceExample(input_indices=[[0], [1]], targets=[1, 2]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            shard = examples_to_tensor_shard(examples, sequence_len=4)
            torch.save(shard, Path(tmpdir) / "shard_00000.pt")
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(json.dumps({
                "format": "phrase_gpt_tensor_shard_manifest_v1",
                "sequence_len": 4,
                "shards": [{"file": "shard_00000.pt", "num_examples": 1}],
            }), encoding="utf-8")

            manifest = load_shard_manifest(str(manifest_path))
            loaded_sets = list(iter_shard_example_sets(manifest, seed=123, shuffle=False))

        self.assertEqual(len(loaded_sets), 1)
        self.assertEqual(loaded_sets[0][0].targets, [1, 2])

    def test_build_phrase_sequence_examples_uses_canonical_story_token_order(self):
        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 2, "record_type": "single", "indices": [10], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 2, "record_type": "single", "indices": [11], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 2, "record_type": "packed", "indices": [10, 11]},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "noun_chunk", "start": 0, "end": 2, "record_type": "single", "indices": [10], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "noun_chunk", "start": 0, "end": 2, "record_type": "single", "indices": [11], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 2, "label": "punctuation", "start": 2, "end": 4, "record_type": "single", "indices": [12], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 2, "label": "punctuation", "start": 2, "end": 4, "record_type": "single", "indices": [13], "token_pos": 1},
            {"split": "train", "story_id": 1, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 1, "record_type": "single", "indices": [99], "token_pos": 0},
        ]

        examples = build_phrase_sequence_examples(records, sequence_len=2)

        self.assertEqual(
            [(example.input_indices, example.targets) for example in examples],
            [
                ([[10], [11]], [11, 12]),
                ([[12]], [13]),
            ],
        )

    def test_build_phrase_sequence_examples_streaming_flushes_each_story_without_full_grouping(self):
        records = iter([
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 2, "record_type": "single", "indices": [10], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 2, "record_type": "single", "indices": [11], "token_pos": 1},
            {"split": "train", "story_id": 1, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [20], "token_pos": 0},
            {"split": "train", "story_id": 1, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [21], "token_pos": 1},
            {"split": "train", "story_id": 1, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [22], "token_pos": 2},
        ])

        examples = build_phrase_sequence_examples_streaming(records, sequence_len=2)

        self.assertEqual(
            [(example.input_indices, example.targets) for example in examples],
            [
                ([[10]], [11]),
                ([[20], [21]], [21, 22]),
            ],
        )

    def test_build_phrase_sequence_examples_streaming_honors_max_examples_early(self):
        def records():
            yield {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 2, "record_type": "single", "indices": [10], "token_pos": 0}
            yield {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 2, "record_type": "single", "indices": [11], "token_pos": 1}
            yield {"split": "train", "story_id": 1, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 2, "record_type": "single", "indices": [20], "token_pos": 0}
            yield {"split": "train", "story_id": 1, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 2, "record_type": "single", "indices": [21], "token_pos": 1}
            raise AssertionError("streaming builder read past the requested example limit")

        examples = build_phrase_sequence_examples_streaming(records(), sequence_len=2, max_examples=1)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].targets, [11])

    def test_examples_from_story_records_token_mode_matches_legacy_helper(self):
        from scripts.train_phrase_gpt import examples_from_story_records, _examples_from_story_records

        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [10], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [11], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [12], "token_pos": 2},
        ]

        via_dispatch = examples_from_story_records(records, sequence_len=2, chain_mode="token")
        via_legacy = _examples_from_story_records(records, sequence_len=2)

        self.assertEqual(
            [(e.input_indices, e.targets) for e in via_dispatch],
            [(e.input_indices, e.targets) for e in via_legacy],
        )
        self.assertEqual([(e.input_indices, e.targets) for e in via_dispatch], [([[10], [11]], [11, 12])])

    def _chain_story_records(self):
        return [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [1], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [3], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [2], "token_pos": 2},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [4], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [5], "token_pos": 1},
        ]

    def test_phrase_mode_breaks_on_order_and_clause_boundary(self):
        from scripts.train_phrase_gpt import examples_from_story_records

        examples = examples_from_story_records(self._chain_story_records(), sequence_len=10, chain_mode="phrase")

        # clause 0 -> [1,3] then break on out-of-order 2 -> [2]; clause boundary breaks before [4,5]
        # chains: [1,3], [2], [4,5]; steps target = next chain's first token
        self.assertEqual(
            [(e.input_indices, e.targets) for e in examples],
            [([[1, 3], [2]], [2, 4])],
        )

    def test_phrase_mode_monotone_clause_is_single_chain(self):
        from scripts.train_phrase_gpt import examples_from_story_records

        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [10], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [11], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [12], "token_pos": 2},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [20], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [21], "token_pos": 1},
        ]

        examples = examples_from_story_records(records, sequence_len=10, chain_mode="phrase")

        # monotone clause 0 -> single chain [10,11,12]; target = first token of clause-1 chain
        self.assertEqual([(e.input_indices, e.targets) for e in examples], [([[10, 11, 12]], [20])])

    def test_phrase_mode_respects_sequence_len_chunking(self):
        from scripts.train_phrase_gpt import examples_from_story_records

        examples = examples_from_story_records(self._chain_story_records(), sequence_len=1, chain_mode="phrase")

        self.assertEqual(
            [(e.input_indices, e.targets) for e in examples],
            [([[1, 3]], [2]), ([[2]], [4])],
        )

    def test_cross_phrase_mode_merges_in_order_run_across_clauses(self):
        from scripts.train_phrase_gpt import examples_from_story_records

        examples = examples_from_story_records(self._chain_story_records(), sequence_len=10, chain_mode="cross-phrase")

        # stream 1,3,2,4,5: break only on out-of-order (2<=3) -> chains [1,3] and [2,4,5]
        # (2->4->5 merges across the clause boundary because there is no clause reset)
        self.assertEqual(
            [(e.input_indices, e.targets) for e in examples],
            [([[1, 3]], [2])],
        )

    def test_phrase_boundary_positions_marks_clause_starts_and_ends(self):
        from scripts.train_phrase_gpt import _canonical_token_stream, _phrase_boundary_positions
        stream = _canonical_token_stream(self._chain_story_records())
        self.assertEqual(_phrase_boundary_positions(stream), [0, 3, 5])

    def test_hybrid_steps_at_split_endpoints_and_middle(self):
        from scripts.train_phrase_gpt import _canonical_token_stream, _hybrid_steps_at_split
        stream = _canonical_token_stream(self._chain_story_records())  # indices 1,3,2,4,5; clauses 0,0,0,1,1

        # split=N (5): fully compressed == phrase mode
        self.assertEqual(_hybrid_steps_at_split(stream, 5), [([1, 3], 2), ([2], 4)])
        # split=0: fully 1-hot, every token predicts the next
        self.assertEqual(_hybrid_steps_at_split(stream, 0), [([1], 3), ([3], 2), ([2], 4), ([4], 5)])
        # split=3: front [1,3,2] compressed -> [[1,3],[2]]; tail [4],[5] 1-hot
        self.assertEqual(_hybrid_steps_at_split(stream, 3), [([1, 3], 2), ([2], 4), ([4], 5)])

    def test_choose_split_is_reproducible_and_in_bounds(self):
        from scripts.train_phrase_gpt import _choose_split
        boundaries = [0, 3, 5]
        a = _choose_split(boundaries, seed=42, story_id=7)
        b = _choose_split(boundaries, seed=42, story_id=7)
        self.assertEqual(a, b)
        self.assertIn(a, boundaries)

    def test_hybrid_mode_dispatch_matches_endpoint_modes(self):
        from scripts.train_phrase_gpt import examples_from_story_records, _canonical_token_stream, _hybrid_steps_at_split, _chunk_steps_into_examples
        records = self._chain_story_records()
        # whatever split the seed picks, the hybrid output must equal _hybrid_steps_at_split at that split
        stream = _canonical_token_stream(records)
        from scripts.train_phrase_gpt import _phrase_boundary_positions, _choose_split
        split = _choose_split(_phrase_boundary_positions(stream), seed=123, story_id=int(stream[0].get("story_id", 0)))
        expected = _chunk_steps_into_examples(_hybrid_steps_at_split(stream, split), sequence_len=10)
        got = examples_from_story_records(records, sequence_len=10, chain_mode="hybrid", seed=123)
        self.assertEqual(
            [(e.input_indices, e.targets) for e in got],
            [(e.input_indices, e.targets) for e in expected],
        )

    def test_early_stopping_tracks_best_validation_loss_and_patience(self):
        config = EarlyStoppingConfig(patience=1, min_delta=0.01)
        state = EarlyStoppingState()

        state.update({"loss": 3.0, "accuracy": 0.1}, epoch=1, config=config)
        self.assertTrue(state.is_best)
        self.assertFalse(state.should_stop)

        state.update({"loss": 2.995, "accuracy": 0.2}, epoch=2, config=config)
        self.assertFalse(state.is_best)
        self.assertTrue(state.should_stop)
        self.assertEqual(state.best_epoch, 1)
        self.assertEqual(state.stop_reason, "validation loss did not improve for 1 epoch")

    @unittest.skipUnless(
        torch.backends.mps.is_available() and not torch.cuda.is_available(),
        "requires a machine with MPS and without CUDA",
    )
    def test_choose_device_prefers_mps_when_cuda_unavailable(self):
        from scripts.train_phrase_gpt import choose_device

        self.assertEqual(choose_device(""), "mps")
        self.assertEqual(choose_device("cpu"), "cpu")

    def test_collate_shard_batch_matches_example_collation(self):
        from scripts.train_phrase_gpt import collate_shard_batch

        examples = [
            PhraseSequenceExample(input_indices=[[0], [1, 2], [3]], targets=[1, 2, 3]),
            PhraseSequenceExample(input_indices=[[2, 3], [4]], targets=[4, 5]),
            PhraseSequenceExample(input_indices=[[5], [], [1]], targets=[2, 3, 4]),
        ]
        shard = examples_to_tensor_shard(examples, sequence_len=4)
        order = [2, 0, 1]

        batch = collate_shard_batch(shard, order, dummy_token_id=0, device="cpu")
        expected = collate_phrase_sequences(
            [examples[i] for i in order], sequence_len=4, dummy_token_id=0, device="cpu",
        )

        self.assertEqual(batch.idx.tolist(), expected.idx.tolist())
        self.assertEqual(batch.targets.tolist(), expected.targets.tolist())
        self.assertEqual(batch.phrase_indices.tolist(), expected.phrase_indices.tolist())
        self.assertEqual(batch.phrase_offsets.tolist(), expected.phrase_offsets.tolist())
        self.assertEqual(batch.phrase_batch_positions.tolist(), expected.phrase_batch_positions.tolist())

    def test_main_tracks_best_epoch_and_early_stopping_in_shard_mode(self):
        from scripts.train_phrase_gpt import main

        _force_sdpa()
        train_examples = [
            PhraseSequenceExample(input_indices=[[0], [1], [2]], targets=[1, 2, 3]),
            PhraseSequenceExample(input_indices=[[1], [2]], targets=[2, 3]),
            PhraseSequenceExample(input_indices=[[3], [4], [5]], targets=[4, 5, 1]),
            PhraseSequenceExample(input_indices=[[2], [3]], targets=[3, 4]),
        ]
        val_examples = [
            PhraseSequenceExample(input_indices=[[0], [1]], targets=[1, 2]),
            PhraseSequenceExample(input_indices=[[4], [5]], targets=[5, 1]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "vocab.json").write_text(json.dumps(
                [{"token": f"t{i}", "index": i} for i in range(6)]
            ), encoding="utf-8")
            torch.save(examples_to_tensor_shard(train_examples, sequence_len=4), tmp / "train_shard_00000.pt")
            torch.save(examples_to_tensor_shard(val_examples, sequence_len=4), tmp / "validation_shard_00000.pt")
            (tmp / "manifest.json").write_text(json.dumps({
                "format": "phrase_gpt_tensor_shard_manifest_v1",
                "sequence_len": 4,
                "num_examples": 6,
                "splits": {
                    "train": {"num_shards": 1, "num_examples": 4},
                    "validation": {"num_shards": 1, "num_examples": 2},
                },
                "shards": [
                    {"file": "train_shard_00000.pt", "split": "train", "num_examples": 4},
                    {"file": "validation_shard_00000.pt", "split": "validation", "num_examples": 2},
                ],
            }), encoding="utf-8")
            argv = [
                "train_phrase_gpt",
                "--vocab", str(tmp / "vocab.json"),
                "--shards", str(tmp / "manifest.json"),
                "--out-dir", str(tmp / "out"),
                "--sequence-len", "4",
                "--depth", "1",
                "--n-embd", "32",
                "--n-head", "2",
                "--batch-size", "2",
                "--epochs", "1",
                "--device", "cpu",
                "--progress-every", "0",
            ]
            with mock.patch("sys.argv", argv):
                main()
            metrics = json.loads((tmp / "out" / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["best_epoch"], 1)
        self.assertIsNotNone(metrics["best_val_loss"])
        self.assertTrue(math.isfinite(metrics["epochs"][0]["train"]["loss"]))
        self.assertTrue(math.isfinite(metrics["epochs"][0]["val"]["loss"]))

    def test_load_vocab_top_k_remap_keeps_most_frequent_and_maps_rest_to_unk(self):
        from scripts.train_phrase_gpt import load_vocab_top_k_remap

        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            vocab_path.write_text(json.dumps([
                {"token": "rare", "index": 0, "count": 2},
                {"token": "common", "index": 1, "count": 100},
                {"token": "mid", "index": 2, "count": 50},
                {"token": "rarest", "index": 3, "count": 1},
            ]), encoding="utf-8")
            lookup, tokens = load_vocab_top_k_remap(str(vocab_path), top_k=2)

        self.assertEqual(tokens, ["common", "mid", "<unk>"])
        # kept tokens get ranks by frequency; the rest collapse to the unk index
        self.assertEqual(lookup.tolist(), [2, 0, 1, 2])

    def test_load_vocab_top_k_remap_preserves_original_index_order(self):
        from scripts.train_phrase_gpt import load_vocab_top_k_remap

        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            # frequency order (b, d, c) disagrees with index order (b, c, d)
            vocab_path.write_text(json.dumps([
                {"token": "a", "index": 0, "count": 5},
                {"token": "b", "index": 1, "count": 100},
                {"token": "c", "index": 2, "count": 7},
                {"token": "d", "index": 3, "count": 50},
            ]), encoding="utf-8")
            lookup, tokens = load_vocab_top_k_remap(str(vocab_path), top_k=3)

        self.assertEqual(tokens, ["b", "c", "d", "<unk>"])
        self.assertEqual(lookup.tolist(), [3, 0, 1, 2])
        # relative order of kept indices is monotone: old 1 < 2 < 3 -> new 0 < 1 < 2

    def test_remap_tensor_shard_remaps_indices_and_preserves_padding(self):
        from scripts.train_phrase_gpt import remap_tensor_shard

        examples = [
            PhraseSequenceExample(input_indices=[[1], [2, 3]], targets=[2, 0]),
            PhraseSequenceExample(input_indices=[[0]], targets=[3]),
        ]
        shard = examples_to_tensor_shard(examples, sequence_len=3)
        lookup = torch.tensor([2, 0, 1, 2], dtype=torch.long)

        remapped = remap_tensor_shard(shard, lookup)

        self.assertEqual(remapped["phrase_indices"].tolist(), [0, 1, 2, 2])
        self.assertEqual(remapped["targets"].tolist(), [[1, 2, -1], [2, -1, -1]])
        self.assertEqual(shard["targets"].tolist(), [[2, 0, -1], [3, -1, -1]])  # input untouched

    def test_main_applies_vocab_top_k_in_shard_mode(self):
        from scripts.train_phrase_gpt import main

        _force_sdpa()
        train_examples = [
            PhraseSequenceExample(input_indices=[[0], [1], [2]], targets=[1, 2, 3]),
            PhraseSequenceExample(input_indices=[[3], [4], [5]], targets=[4, 5, 1]),
        ]
        val_examples = [
            PhraseSequenceExample(input_indices=[[0], [5]], targets=[5, 1]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "vocab.json").write_text(json.dumps(
                [{"token": f"t{i}", "index": i, "count": 100 - i} for i in range(6)]
            ), encoding="utf-8")
            torch.save(examples_to_tensor_shard(train_examples, sequence_len=4), tmp / "train_shard_00000.pt")
            torch.save(examples_to_tensor_shard(val_examples, sequence_len=4), tmp / "validation_shard_00000.pt")
            (tmp / "manifest.json").write_text(json.dumps({
                "format": "phrase_gpt_tensor_shard_manifest_v1",
                "sequence_len": 4,
                "num_examples": 3,
                "splits": {
                    "train": {"num_shards": 1, "num_examples": 2},
                    "validation": {"num_shards": 1, "num_examples": 1},
                },
                "shards": [
                    {"file": "train_shard_00000.pt", "split": "train", "num_examples": 2},
                    {"file": "validation_shard_00000.pt", "split": "validation", "num_examples": 1},
                ],
            }), encoding="utf-8")
            argv = [
                "train_phrase_gpt",
                "--vocab", str(tmp / "vocab.json"),
                "--shards", str(tmp / "manifest.json"),
                "--out-dir", str(tmp / "out"),
                "--sequence-len", "4",
                "--depth", "1",
                "--n-embd", "32",
                "--n-head", "2",
                "--batch-size", "2",
                "--epochs", "1",
                "--device", "cpu",
                "--progress-every", "0",
                "--vocab-top-k", "4",
            ]
            with mock.patch("sys.argv", argv):
                main()
            metrics = json.loads((tmp / "out" / "metrics.json").read_text(encoding="utf-8"))
            pruned = json.loads((tmp / "out" / "vocab_top_k.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics["vocab_size"], 5)  # top 4 + <unk>
        self.assertEqual(metrics["vocab_top_k"], 4)
        self.assertTrue(math.isfinite(metrics["epochs"][0]["train"]["loss"]))
        self.assertTrue(math.isfinite(metrics["epochs"][0]["val"]["loss"]))
        self.assertEqual([row["token"] for row in pruned], ["t0", "t1", "t2", "t3", "<unk>"])

    def test_checkpoint_contains_optimizer_state_and_completed_epoch(self):
        from scripts.train_phrase_gpt import main

        _force_sdpa()
        train = [PhraseSequenceExample(input_indices=[[0], [1], [2]], targets=[1, 2, 3])]
        val = [PhraseSequenceExample(input_indices=[[0], [1]], targets=[1, 2])]
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab, manifest = _write_shard_fixture(tmpdir, [train], val_examples=val)
            out_dir = str(Path(tmpdir) / "out")
            with mock.patch("sys.argv", _main_argv(vocab, manifest, out_dir, extra=["--epochs", "1"])):
                main()
            checkpoint = torch.load(Path(out_dir) / "phrase_gpt.pt", map_location="cpu", weights_only=False)

        self.assertIn("optimizer_state_dict", checkpoint)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertIn("epochs_without_improvement", checkpoint)

    def test_main_resumes_training_from_checkpoint(self):
        from scripts.train_phrase_gpt import main

        _force_sdpa()
        train = [
            PhraseSequenceExample(input_indices=[[0], [1], [2]], targets=[1, 2, 3]),
            PhraseSequenceExample(input_indices=[[3], [4]], targets=[4, 5]),
        ]
        val = [PhraseSequenceExample(input_indices=[[0], [1]], targets=[1, 2])]
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab, manifest = _write_shard_fixture(tmpdir, [train], val_examples=val)
            out_dir = str(Path(tmpdir) / "out")
            with mock.patch("sys.argv", _main_argv(vocab, manifest, out_dir, extra=["--epochs", "1"])):
                main()
            first_metrics = json.loads((Path(out_dir) / "metrics.json").read_text(encoding="utf-8"))
            resume_args = ["--epochs", "2", "--resume", str(Path(out_dir) / "phrase_gpt.pt")]
            with mock.patch("sys.argv", _main_argv(vocab, manifest, out_dir, extra=resume_args)):
                main()
            metrics = json.loads((Path(out_dir) / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual([row["epoch"] for row in metrics["epochs"]], [1, 2])
        # epoch 1 was not re-run: its row is carried over from the checkpoint
        self.assertEqual(metrics["epochs"][0], first_metrics["epochs"][0])
        self.assertTrue(math.isfinite(metrics["epochs"][1]["train"]["loss"]))

    def test_run_epoch_on_shards_resumes_mid_epoch_from_start_shard(self):
        from scripts.train_phrase_gpt import run_epoch_on_shards, load_shard_manifest

        _force_sdpa()
        torch.manual_seed(0)
        model = _tiny_phrase_gpt()
        shard_a = [
            PhraseSequenceExample(input_indices=[[0], [1], [2]], targets=[1, 2, 3]),
            PhraseSequenceExample(input_indices=[[3], [4]], targets=[4, 5]),
        ]
        shard_b = [PhraseSequenceExample(input_indices=[[0], [1]], targets=[1, 2])]
        with tempfile.TemporaryDirectory() as tmpdir:
            _, manifest_path = _write_shard_fixture(tmpdir, [shard_a, shard_b])
            manifest = load_shard_manifest(manifest_path)
            common = dict(batch_size=2, device="cpu", progress_every=0, split="train")

            full = run_epoch_on_shards(model, manifest, **common)
            seen = []
            partial = run_epoch_on_shards(
                model, manifest, start_shard=1,
                on_shard_end=lambda shard_idx, rows: seen.append(shard_idx), **common,
            )
            prior = [{"loss": 1.0, "accuracy": 0.5, "tokens": 5}]
            resumed = run_epoch_on_shards(model, manifest, start_shard=1, prior_rows=prior, **common)

        self.assertEqual(full["tokens"], 7)      # both shards: 5 + 2 tokens
        self.assertEqual(partial["tokens"], 2)   # only the second shard
        self.assertEqual(seen, [2])              # global shard numbering is preserved
        self.assertEqual(resumed["tokens"], 7)   # prior rows fold into the totals

    def test_early_stopping_stops_when_target_validation_accuracy_is_reached(self):
        config = EarlyStoppingConfig(target_val_accuracy=0.5)
        state = EarlyStoppingState()

        state.update({"loss": 3.0, "accuracy": 0.55}, epoch=1, config=config)

        self.assertTrue(state.should_stop)
        self.assertEqual(state.stop_reason, "target validation accuracy reached")

    def test_run_training_sweep_appends_entry_and_restores_train_mode(self):
        from scripts.train_phrase_gpt import _run_training_sweep
        from scripts.hybrid_sweep import build_sweep_probes
        import argparse

        _force_sdpa()
        model = _tiny_phrase_gpt()
        model.train(True)
        records = [
            {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [1], "token_pos": 0},
            {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [2], "token_pos": 1},
            {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [3], "token_pos": 2},
        ]
        probes = build_sweep_probes(records, min_history=1, split="validation")
        args = argparse.Namespace(sweep_x_values="0", sweep_d_values="1", sweep_batch_size=4,
                                  sweep_bootstrap=0, sweep_seed=0, sweep_eval_split="validation")
        trajectory = []
        _run_training_sweep(model, probes, args, None, "cpu", epoch=3, shard=None, trajectory=trajectory)

        self.assertEqual(len(trajectory), 1)
        self.assertEqual(trajectory[0]["epoch"], 3)
        self.assertIsNone(trajectory[0]["shard"])
        self.assertIn("x_sweep", trajectory[0]["sweep"])
        self.assertTrue(model.training)  # restored

    def test_main_records_sweep_trajectory_per_epoch(self):
        from scripts.train_phrase_gpt import main

        _force_sdpa()
        train = [
            PhraseSequenceExample(input_indices=[[0], [1], [2]], targets=[1, 2, 3]),
            PhraseSequenceExample(input_indices=[[3], [4], [5]], targets=[4, 5, 1]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            vocab, manifest = _write_shard_fixture(tmp, [train])
            sweep_records = tmp / "val_records.jsonl"
            rows = [
                {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [1], "token_pos": 0},
                {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [2], "token_pos": 1},
                {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [3], "token_pos": 2},
            ]
            sweep_records.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            out_dir = str(tmp / "out")
            extra = ["--epochs", "2", "--sweep-eval-records", str(sweep_records),
                     "--sweep-eval-split", "validation", "--sweep-x-values", "0",
                     "--sweep-d-values", "1", "--sweep-max-probes", "5", "--sweep-bootstrap", "0"]
            with mock.patch("sys.argv", _main_argv(vocab, manifest, out_dir, extra=extra)):
                main()
            metrics = json.loads((Path(out_dir) / "metrics.json").read_text(encoding="utf-8"))

        traj = metrics["sweep_trajectory"]
        self.assertEqual([e["epoch"] for e in traj], [1, 2])
        self.assertTrue(all(e["shard"] is None for e in traj))
        self.assertTrue(all("x_sweep" in e["sweep"] for e in traj))
        counts = {e["sweep"]["num_probes"] for e in traj}
        self.assertEqual(len(counts), 1)  # same probe set every epoch


if __name__ == "__main__":
    unittest.main()
