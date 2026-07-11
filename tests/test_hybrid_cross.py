import unittest

from scripts.hybrid_sweep import build_sweep_probes, context_steps_for_probe, _probe_contexts, SweepProbe
from scripts.train_phrase_gpt import (
    _canonical_token_stream,
    _hybrid_steps,
    _hybrid_steps_at_split,
    examples_from_story_records,
)
from scripts.train_phrase_vectors import normalize_phrase_records


def _story(story_id, clause_tokens):
    records = []
    pos = 0
    for clause_id, toks in clause_tokens:
        for tp, tok in enumerate(toks):
            records.append({
                "split": "train", "story_id": story_id, "phrase_id": clause_id,
                "label": "punctuation", "record_type": "single", "indices": [tok],
                "token_pos": tp, "start": pos, "end": pos + len(toks),
            })
        pos += len(toks)
    return records


class HybridCrossStepsTests(unittest.TestCase):
    def _token_records(self, clause_tokens):
        return _canonical_token_stream(normalize_phrase_records(_story(0, clause_tokens)))

    def test_reset_true_breaks_front_chains_at_clause_boundary(self):
        token_records = self._token_records([(0, [1, 3]), (1, [5, 2])])
        steps = _hybrid_steps_at_split(token_records, split=4, reset_on_clause=True)
        # front chains: [1,3] | [5] | [2]
        self.assertEqual(steps, [([1, 3], 5), ([5], 2)])

    def test_reset_false_merges_ascending_front_chains_across_boundary(self):
        token_records = self._token_records([(0, [1, 3]), (1, [5, 2])])
        steps = _hybrid_steps_at_split(token_records, split=4, reset_on_clause=False)
        # front chains: [1,3,5] | [2]
        self.assertEqual(steps, [([1, 3, 5], 2)])

    def test_default_matches_reset_true(self):
        token_records = self._token_records([(0, [1, 3]), (1, [5, 2])])
        self.assertEqual(
            _hybrid_steps_at_split(token_records, split=4),
            _hybrid_steps_at_split(token_records, split=4, reset_on_clause=True),
        )

    def test_hybrid_cross_chain_mode_dispatches(self):
        records = _story(0, [(0, [1, 3]), (1, [5, 2])])
        examples = examples_from_story_records(records, sequence_len=10, chain_mode="hybrid-cross", seed=7)
        expected_steps = _hybrid_steps(records, seed=7, reset_on_clause=False)
        self.assertEqual(
            [(e.input_indices, e.targets) for e in examples],
            [([indices for indices, _ in expected_steps], [target for _, target in expected_steps])] if expected_steps else [],
        )

    def test_plain_hybrid_unchanged(self):
        records = _story(0, [(0, [1, 3]), (1, [5, 2])])
        self.assertEqual(
            _hybrid_steps(records, seed=7),
            _hybrid_steps(records, seed=7, reset_on_clause=True),
        )


class SweepCrossClauseTests(unittest.TestCase):
    def test_context_steps_cross_clause_merges_front_chains(self):
        probe = SweepProbe(token_indices=[1, 3, 5, 2, 4], clause_ids=[0, 0, 1, 1, 1], target_pos=4, is_opener=False)
        reset_ctx = context_steps_for_probe(probe, x=1, depth=None, reset_on_clause=True)
        cross_ctx = context_steps_for_probe(probe, x=1, depth=None, reset_on_clause=False)
        self.assertEqual(reset_ctx, [[1, 3], [5], [2]])
        self.assertEqual(cross_ctx, [[1, 3, 5], [2]])

    def test_probe_contexts_threads_reset_flag(self):
        probe = SweepProbe(token_indices=[1, 3, 5, 2, 4], clause_ids=[0, 0, 1, 1, 1], target_pos=4, is_opener=False)
        reset_contexts = _probe_contexts([probe], x=1, depth=None, remap=None, reset_on_clause=True)
        cross_contexts = _probe_contexts([probe], x=1, depth=None, remap=None, reset_on_clause=False)
        self.assertEqual(reset_contexts, [[[1, 3], [5], [2]]])
        self.assertEqual(cross_contexts, [[[1, 3, 5], [2]]])


class SweepIndexMapTests(unittest.TestCase):
    def test_build_sweep_probes_applies_index_map(self):
        records = _story(0, [(0, [1, 3]), (1, [0, 2])])
        index_map = [9, 4, 6, 5]  # old -> new
        probes = build_sweep_probes(iter(records), min_history=1, index_map=index_map)
        self.assertTrue(probes)
        self.assertEqual(probes[0].token_indices, [4, 5, 9, 6])

    def test_build_sweep_probes_without_map_unchanged(self):
        records = _story(0, [(0, [1, 3]), (1, [0, 2])])
        probes = build_sweep_probes(iter(records), min_history=1)
        self.assertEqual(probes[0].token_indices, [1, 3, 0, 2])


if __name__ == "__main__":
    unittest.main()
