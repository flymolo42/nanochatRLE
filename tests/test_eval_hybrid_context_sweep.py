import math
import unittest

import torch

from scripts.eval_hybrid_context_sweep import (
    SweepProbe, build_sweep_probes, context_steps_for_probe, topk_and_ce,
)


def _story(story_id, clause_tokens):
    """clause_tokens: list of (clause_id, [token indices]) in story order."""
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


class HybridSweepPureTests(unittest.TestCase):
    def test_build_sweep_probes_marks_openers_and_history(self):
        records = _story(0, [(0, [1, 3, 2]), (1, [4, 5])])
        probes = build_sweep_probes(records, min_history=1)
        # targets are positions 1..4 (pos 0 has no history); token at pos 3 opens clause 1
        self.assertEqual([p.target_pos for p in probes], [1, 2, 3, 4])
        self.assertEqual([p.token_indices for p in probes][0], [1, 3, 2, 4, 5])
        opener = next(p for p in probes if p.target_pos == 3)
        self.assertTrue(opener.is_opener)
        interior = next(p for p in probes if p.target_pos == 2)
        self.assertFalse(interior.is_opener)

    def test_context_steps_x_controls_recent_one_hot_tail(self):
        # A monotone single clause is where X visibly matters: X=0 merges the whole
        # history into one chain; X=2 leaves the last two tokens 1-hot.
        probe = SweepProbe(token_indices=[1, 2, 3, 4], clause_ids=[0, 0, 0, 0], target_pos=3, is_opener=False)
        self.assertEqual(context_steps_for_probe(probe, x=0, depth=None), [[1, 2, 3]])
        self.assertEqual(context_steps_for_probe(probe, x=2, depth=None), [[1], [2], [3]])
        # When order/clause breaks already singletonize the tail, X=0 and X=2 coincide.
        probe2 = SweepProbe(token_indices=[1, 3, 2, 4, 5], clause_ids=[0, 0, 0, 1, 1], target_pos=4, is_opener=True)
        self.assertEqual(context_steps_for_probe(probe2, x=0, depth=None), [[1, 3], [2], [4]])
        self.assertEqual(context_steps_for_probe(probe2, x=2, depth=None), [[1, 3], [2], [4]])

    def test_context_steps_depth_caps_compressed_history(self):
        probe = SweepProbe(token_indices=[1, 2, 3, 4, 5, 6], clause_ids=[0, 0, 1, 1, 2, 2], target_pos=5, is_opener=False)
        full = context_steps_for_probe(probe, x=0, depth=None)
        capped = context_steps_for_probe(probe, x=0, depth=1)
        self.assertEqual(capped, full[-1:])

    def test_topk_and_ce_computes_hits_and_cross_entropy(self):
        logits = torch.tensor([0.0, 5.0, 1.0, 2.0, 0.0, 0.0])  # argmax = index 1
        hits, ce = topk_and_ce(logits, target=1, ks=(1, 5))
        self.assertEqual(hits[1], 1)
        self.assertEqual(hits[5], 1)
        hits2, ce2 = topk_and_ce(logits, target=4, ks=(1, 5))
        self.assertEqual(hits2[1], 0)   # 4 is not the argmax
        self.assertEqual(hits2[5], 1)   # but within top-5
        expected_ce = -math.log(torch.softmax(logits, dim=0)[1].item())
        self.assertAlmostEqual(ce, expected_ce, places=5)


if __name__ == "__main__":
    unittest.main()
