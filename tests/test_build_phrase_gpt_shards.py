import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.build_phrase_gpt_shards import build_shards_from_records, remap_record_indices
from scripts.train_phrase_gpt import tensor_shard_to_examples


class BuildPhraseGPTShardsTests(unittest.TestCase):
    def test_build_shards_from_records_writes_tensor_shards_and_manifest(self):
        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [10], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [11], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [12], "token_pos": 2},
            {"split": "train", "story_id": 1, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [20], "token_pos": 0},
            {"split": "train", "story_id": 1, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [21], "token_pos": 1},
            {"split": "train", "story_id": 1, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [22], "token_pos": 2},
            {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 2, "record_type": "single", "indices": [30], "token_pos": 0},
            {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 2, "record_type": "single", "indices": [31], "token_pos": 1},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_shards_from_records(
                records=iter(records),
                out_dir=tmpdir,
                sequence_len=2,
                examples_per_shard=1,
                records_path="records.jsonl.gz",
                vocab_path="vocab.json",
            )
            manifest_path = Path(tmpdir) / "manifest.json"
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            first_train = next(shard for shard in saved_manifest["shards"] if shard["split"] == "train")
            first_shard = torch.load(Path(tmpdir) / first_train["file"], map_location="cpu", weights_only=False)

        self.assertEqual(manifest["num_examples"], 3)
        self.assertEqual(saved_manifest["splits"]["train"]["num_examples"], 2)
        self.assertEqual(saved_manifest["splits"]["validation"]["num_examples"], 1)
        self.assertEqual(first_shard["format"], "phrase_gpt_tensor_shard_v1")
        self.assertEqual(first_shard["split"], "train")
        self.assertEqual(tensor_shard_to_examples(first_shard)[0].targets, [11, 12])

    def test_build_shards_phrase_mode_records_mode_and_multihot_examples(self):
        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [1], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [3], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [2], "token_pos": 2},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [4], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [5], "token_pos": 1},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_shards_from_records(
                records=iter(records),
                out_dir=tmpdir,
                sequence_len=10,
                examples_per_shard=50,
                chain_mode="phrase",
            )
            saved = json.loads((Path(tmpdir) / "manifest.json").read_text(encoding="utf-8"))
            shard = torch.load(Path(tmpdir) / saved["shards"][0]["file"], map_location="cpu", weights_only=False)

        self.assertEqual(saved["chain_mode"], "phrase")
        example = tensor_shard_to_examples(shard)[0]
        self.assertEqual((example.input_indices, example.targets), ([[1, 3], [2]], [2, 4]))

    def test_build_shards_hybrid_mode_records_mode_and_seed(self):
        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [1], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [3], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [2], "token_pos": 2},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [4], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [5], "token_pos": 1},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_shards_from_records(
                records=iter(records), out_dir=tmpdir, sequence_len=10,
                examples_per_shard=50, chain_mode="hybrid", split_seed=123,
            )
            saved = json.loads((Path(tmpdir) / "manifest.json").read_text(encoding="utf-8"))
            shard = torch.load(Path(tmpdir) / saved["shards"][0]["file"], map_location="cpu", weights_only=False)

        self.assertEqual(saved["chain_mode"], "hybrid")
        self.assertEqual(saved["split_seed"], 123)
        # deterministic given the seed: matches examples_from_story_records at seed=123
        from scripts.train_phrase_gpt import examples_from_story_records, tensor_shard_to_examples
        expected = examples_from_story_records(records, sequence_len=10, chain_mode="hybrid", seed=123)
        got = tensor_shard_to_examples(shard)
        self.assertEqual(
            [(e.input_indices, e.targets) for e in got],
            [(e.input_indices, e.targets) for e in expected],
        )


class IndexMapTests(unittest.TestCase):
    def test_remap_record_indices_maps_and_sorts_multitoken_lists(self):
        record = {"record_type": "packed", "indices": [3, 5], "split": "train", "story_id": 0, "phrase_id": 0}
        mapped = remap_record_indices(record, [0, 1, 2, 9, 4, 2])
        self.assertEqual(mapped["indices"], [2, 9])
        self.assertEqual(record["indices"], [3, 5])

    def test_build_shards_applies_index_map(self):
        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [10], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [11], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [12], "token_pos": 2},
        ]
        index_map = list(range(13))
        index_map[10], index_map[11], index_map[12] = 0, 5, 3

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_shards_from_records(
                records=iter(records),
                out_dir=tmpdir,
                sequence_len=4,
                examples_per_shard=10,
                index_map=index_map,
            )
            shard = torch.load(Path(tmpdir) / manifest["shards"][0]["file"], map_location="cpu", weights_only=False)

        example = tensor_shard_to_examples(shard)[0]
        self.assertEqual(example.input_indices, [[0], [5]])
        self.assertEqual(example.targets, [5, 3])


if __name__ == "__main__":
    unittest.main()
