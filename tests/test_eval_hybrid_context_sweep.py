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

    def test_build_sweep_probes_clamps_min_history_to_avoid_empty_context(self):
        # min_history=0 would otherwise emit a probe at position 0 with no history
        # (an empty context), which silently scores the padding row. min_history
        # should be clamped to at least 1, so this must match min_history=1.
        records = _story(0, [(0, [1, 3, 2]), (1, [4, 5])])
        probes_zero = build_sweep_probes(records, min_history=0)
        probes_one = build_sweep_probes(records, min_history=1)
        self.assertEqual([p.target_pos for p in probes_zero], [p.target_pos for p in probes_one])
        self.assertNotIn(0, [p.target_pos for p in probes_zero])

    def test_build_sweep_probes_split_filter(self):
        train_records = _story(0, [(0, [1, 3, 2]), (1, [4, 5])])
        val_records = [dict(r, split="validation") for r in _story(1, [(0, [6, 7])])]
        records = train_records + val_records
        probes = build_sweep_probes(records, min_history=1, split="validation")
        self.assertTrue(len(probes) > 0)
        self.assertTrue(all(idx in (6, 7) for p in probes for idx in p.token_indices))
        self.assertEqual([p.target_pos for p in probes], [1])

    def test_build_sweep_probes_streams_filters_split_and_stops_early(self):
        from scripts.hybrid_sweep import build_sweep_probes

        def records():
            for r in _story(0, [(0, [1, 2, 3])]):            # train (filtered out)
                yield r
            for r in _story(1, [(0, [4, 5, 6])]):            # validation — the wanted story
                r = dict(r); r["split"] = "validation"
                yield r
            # one record of a later story so story 1's boundary is detectable,
            # then a trap that a correct early-stopping builder never reaches
            yield {"split": "train", "story_id": 2, "phrase_id": 0, "label": "punctuation",
                   "record_type": "single", "indices": [7], "token_pos": 0, "start": 0, "end": 1}
            raise AssertionError("build_sweep_probes read past the story it needed")

        probes = build_sweep_probes(records(), min_history=1, max_probes=1, split="validation")
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0].token_indices, [4, 5, 6])

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

    def test_run_sweep_scores_same_probes_across_configs(self):
        import torch
        from nanochat.gpt import GPT, GPTConfig
        import nanochat.flash_attention as fa_module
        from scripts.eval_hybrid_context_sweep import build_sweep_probes, run_sweep

        fa_module._override_impl = "sdpa"
        fa_module.USE_FA3 = fa_module._resolve_use_fa3()
        torch.manual_seed(0)
        config = GPTConfig(sequence_len=8, vocab_size=8, n_layer=1, n_head=2, n_kv_head=2,
                           n_embd=32, window_pattern="L", phrase_vocab_size=8)
        model = GPT(config, pad_vocab_size_to=1)
        model.init_weights()

        records = _story(0, [(0, [1, 3, 2]), (1, [4, 5])]) + _story(1, [(0, [1, 2]), (1, [3, 4, 5])])
        probes = build_sweep_probes(records, min_history=1)

        result = run_sweep(model, probes, x_values=[0, 2], d_values=[1, None],
                           fixed_x_for_depth=0, remap=None, batch_size=4, device="cpu")

        # comparability: every X config scored the SAME number of probes
        x0 = result["x_sweep"]["0"]["all"]["count"]
        x2 = result["x_sweep"]["2"]["all"]["count"]
        self.assertEqual(x0, x2)
        self.assertEqual(x0, len(probes))
        # metric keys present
        for key in ("top1", "top5", "top10", "mean_ce", "perplexity", "count"):
            self.assertIn(key, result["x_sweep"]["0"]["all"])
        # opener/interior buckets present and sum to all
        opener = result["x_sweep"]["0"]["opener"]["count"]
        interior = result["x_sweep"]["0"]["interior"]["count"]
        self.assertEqual(opener + interior, x0)

    def test_run_sweep_bootstrap_cis_present_reproducible_and_bracket_point(self):
        import torch
        from nanochat.gpt import GPT, GPTConfig
        import nanochat.flash_attention as fa_module
        from scripts.hybrid_sweep import build_sweep_probes, run_sweep

        fa_module._override_impl = "sdpa"
        fa_module.USE_FA3 = fa_module._resolve_use_fa3()
        torch.manual_seed(0)
        config = GPTConfig(sequence_len=8, vocab_size=8, n_layer=1, n_head=2, n_kv_head=2,
                           n_embd=32, window_pattern="L", phrase_vocab_size=8)
        model = GPT(config, pad_vocab_size_to=1); model.init_weights()
        records = _story(0, [(0, [1, 3, 2]), (1, [4, 5])]) + _story(1, [(0, [1, 2]), (1, [3, 4, 5])])
        probes = build_sweep_probes(records, min_history=1)

        kw = dict(x_values=[0], d_values=[None], fixed_x_for_depth=0, remap=None, batch_size=4, device="cpu")
        a = run_sweep(model, probes, bootstrap=200, bootstrap_seed=7, **kw)
        b = run_sweep(model, probes, bootstrap=200, bootstrap_seed=7, **kw)
        none = run_sweep(model, probes, bootstrap=0, **kw)

        cell = a["x_sweep"]["0"]["all"]
        for key in ("top1_ci", "top5_ci", "top10_ci", "mean_ce_ci", "perplexity_ci"):
            self.assertIn(key, cell)
            self.assertEqual(len(cell[key]), 2)
            self.assertLessEqual(cell[key][0], cell[key][1])
        # CI brackets the point estimate
        self.assertLessEqual(cell["top1_ci"][0], cell["top1"])
        self.assertLessEqual(cell["top1"], cell["top1_ci"][1])
        # reproducible for a fixed seed
        self.assertEqual(a["x_sweep"]["0"]["all"]["top1_ci"], b["x_sweep"]["0"]["all"]["top1_ci"])
        # bootstrap=0 omits CI keys
        self.assertNotIn("top1_ci", none["x_sweep"]["0"]["all"])


if __name__ == "__main__":
    unittest.main()
