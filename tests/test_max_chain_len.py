import unittest

from scripts.train_phrase_gpt import _cap_chains, _chains_from_token_records, _hybrid_steps_at_split


def _records(indices, clause=0):
    return [
        {"record_type": "single", "label": "punctuation", "indices": [index],
         "phrase_id": clause, "token_pos": pos, "start": 0, "story_id": 0, "split": "train"}
        for pos, index in enumerate(indices)
    ]


class CapChainsTests(unittest.TestCase):
    def test_short_chains_untouched(self):
        self.assertEqual(_cap_chains([[1, 2, 3]], 9), [[1, 2, 3]])

    def test_balanced_split_length_ten_cap_nine(self):
        chains = _cap_chains([list(range(10))], 9)
        self.assertEqual(chains, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    def test_balanced_split_length_fifteen(self):
        chains = _cap_chains([list(range(15))], 9)
        self.assertEqual([len(c) for c in chains], [8, 7])
        self.assertEqual(sum(chains, []), list(range(15)))

    def test_three_way_split(self):
        chains = _cap_chains([list(range(19))], 9)
        self.assertEqual([len(c) for c in chains], [7, 6, 6])

    def test_none_cap_is_identity(self):
        self.assertEqual(_cap_chains([list(range(15))], None), [list(range(15))])

    def test_chains_from_token_records_applies_cap(self):
        token_records = _records(list(range(12)))
        chains = _chains_from_token_records(token_records, reset_on_clause=False, max_chain_len=9)
        self.assertEqual([len(c) for c in chains], [6, 6])

    def test_hybrid_steps_cap_front_only(self):
        token_records = _records(list(range(12)))
        steps = _hybrid_steps_at_split(token_records, split=10, reset_on_clause=False, max_chain_len=9)
        # front run of 10 splits 5+5; back tokens 10, 11 are 1-hot
        self.assertEqual(steps[0], ([0, 1, 2, 3, 4], 5))
        self.assertEqual(steps[1], ([5, 6, 7, 8, 9], 10))
        self.assertEqual(steps[2], ([10], 11))


class SweepCapTests(unittest.TestCase):
    def test_context_steps_cap(self):
        from scripts.hybrid_sweep import SweepProbe, context_steps_for_probe
        probe = SweepProbe(token_indices=list(range(11)), clause_ids=[0] * 11, target_pos=10, is_opener=False)
        context = context_steps_for_probe(probe, x=0, depth=None, reset_on_clause=False, max_chain_len=9)
        self.assertEqual([len(slot) for slot in context], [5, 5])


class ShardBuilderCapTests(unittest.TestCase):
    def test_manifest_records_cap_and_examples_split(self):
        import tempfile
        from pathlib import Path
        from scripts.build_phrase_gpt_shards import build_shards_from_records
        records = _records(list(range(12)))
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_shards_from_records(
                records=iter(records), out_dir=tmpdir, sequence_len=16,
                examples_per_shard=10, chain_mode="cross-phrase", max_chain_len=9,
            )
        self.assertEqual(manifest["max_chain_len"], 9)


if __name__ == "__main__":
    unittest.main()
