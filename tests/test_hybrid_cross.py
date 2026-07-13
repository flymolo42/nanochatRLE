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


class ClassicContextTests(unittest.TestCase):
    def test_classic_context_steps_is_full_one_hot_history(self):
        from scripts.hybrid_sweep import classic_context_steps
        probe = SweepProbe(token_indices=[1, 3, 5, 2, 4], clause_ids=[0, 0, 1, 1, 1], target_pos=3, is_opener=False)
        self.assertEqual(classic_context_steps(probe), [[1], [3], [5]])

    def test_run_sweep_reports_classic_matching_full_tail_x(self):
        import torch
        import nanochat.flash_attention as fa_module
        from nanochat.gpt import GPT, GPTConfig
        from scripts.hybrid_sweep import build_sweep_probes, run_sweep

        fa_module._override_impl = "sdpa"
        fa_module.USE_FA3 = fa_module._resolve_use_fa3()
        torch.manual_seed(0)
        config = GPTConfig(sequence_len=8, vocab_size=8, n_layer=1, n_head=2, n_kv_head=2,
                           n_embd=32, window_pattern="L", phrase_vocab_size=8)
        model = GPT(config, pad_vocab_size_to=1)
        model.init_weights()

        records = _story(0, [(0, [1, 3, 2]), (1, [4, 5])]) + _story(1, [(0, [1, 2]), (1, [3, 4, 5])])
        probes = build_sweep_probes(iter(records), min_history=1)
        result = run_sweep(model, probes, x_values=[99], d_values=[None],
                           fixed_x_for_depth=0, remap=None, batch_size=4, device="cpu")

        self.assertIn("classic_1hot", result)
        classic = result["classic_1hot"]["all"]
        self.assertEqual(classic["count"], len(probes))
        for key in ("top1", "top5", "top10", "mean_ce", "perplexity"):
            self.assertIn(key, classic)
        # x=99 tail covers every probe's whole history -> identical contexts -> identical metrics
        full_tail = result["x_sweep"]["99"]["all"]
        self.assertEqual(classic["top1"], full_tail["top1"])
        self.assertAlmostEqual(classic["mean_ce"], full_tail["mean_ce"], places=6)


class SAEFrontEncoderTests(unittest.TestCase):
    def test_sae_front_encoder_replaces_front_chains(self):
        import torch
        from scripts.build_sae_context_shards import sae_front_encoder
        from scripts.hybrid_sweep import _probe_contexts
        from scripts.sae import TopKSAE
        torch.manual_seed(0)
        sae = TopKSAE(input_dim=8, latent_dim=16, k=2)
        encoder = sae_front_encoder(sae, mode="chain", window=4, latent_offset=8, lookup=torch.arange(8), index_map=None)
        probe = SweepProbe(token_indices=[1, 3, 5, 2, 4], clause_ids=[0, 0, 1, 1, 1], target_pos=4, is_opener=False)
        contexts = _probe_contexts([probe], x=1, depth=None, remap=None, front_encoder=encoder)
        (context,) = contexts
        # front tokens [1,3,5] -> chains [[1,3,5]] -> one SAE slot of k=2 latent ids; tail [2] stays a token
        self.assertEqual(len(context), 2)
        self.assertTrue(all(latent_id >= 8 for latent_id in context[0]))
        self.assertEqual(context[1], [2])

    def test_front_encoder_branch_is_not_remapped_with_nonnull_remap(self):
        # Regression test for the Task 7 review finding: the CLI must be able to
        # keep a real (non-None) top-k remap alongside --sae, because `remap` also
        # scores the TARGET token (_aggregate) and the classic_1hot baseline. So
        # _probe_contexts's front_encoder branch must never push its steps (front
        # latent ids >= latent_offset, tail already mapped via tail_lookup) back
        # through _remap_steps -- a non-None remap here would previously either
        # IndexError (remap sized for the 8 token ids, front ids run 8..23) or
        # silently corrupt the front slot's latent ids.
        import torch
        from scripts.build_sae_context_shards import sae_front_encoder
        from scripts.hybrid_sweep import _probe_contexts
        from scripts.sae import TopKSAE
        torch.manual_seed(0)
        sae = TopKSAE(input_dim=8, latent_dim=16, k=2)
        encoder = sae_front_encoder(sae, mode="chain", window=4, latent_offset=8, lookup=torch.arange(8), index_map=None)
        probe = SweepProbe(token_indices=[1, 3, 5, 2, 4], clause_ids=[0, 0, 1, 1, 1], target_pos=4, is_opener=False)
        remap = torch.arange(8)  # only large enough for original token ids, NOT latent ids >= 8
        contexts = _probe_contexts([probe], x=1, depth=None, remap=remap, front_encoder=encoder)
        (context,) = contexts
        # Would previously raise IndexError (or corrupt the front slot) once remap
        # was non-None -- front latent ids must pass through untouched.
        self.assertEqual(len(context), 2)
        self.assertTrue(all(latent_id >= 8 for latent_id in context[0]))
        self.assertEqual(context[1], [2])

    def test_run_sweep_with_sae_front_encoder_and_remap_matches_classic_without_encoder(self):
        # Integration regression test: run_sweep must complete end-to-end when a
        # front_encoder AND a real (non-None) remap are both supplied -- the shape
        # the fixed CLI now produces for --sae runs. classic_1hot never touches
        # front_encoder, so it must be identical with or without one.
        import torch
        import nanochat.flash_attention as fa_module
        from nanochat.gpt import GPT, GPTConfig
        from scripts.build_sae_context_shards import sae_front_encoder
        from scripts.hybrid_sweep import build_sweep_probes, run_sweep
        from scripts.sae import TopKSAE

        fa_module._override_impl = "sdpa"
        fa_module.USE_FA3 = fa_module._resolve_use_fa3()
        torch.manual_seed(0)
        # phrase_vocab_size = 8 token ids + 16 SAE latent ids (offset 8) so latent
        # ids up to 23 fit the phrase-multihot embedding table; vocab_size=8 stays
        # the target/output-head space (post top-k remap).
        config = GPTConfig(sequence_len=8, vocab_size=8, n_layer=1, n_head=2, n_kv_head=2,
                           n_embd=32, window_pattern="L", phrase_vocab_size=8 + 16)
        model = GPT(config, pad_vocab_size_to=1)
        model.init_weights()

        records = _story(0, [(0, [1, 3, 2]), (1, [4, 5])]) + _story(1, [(0, [1, 2]), (1, [3, 4, 5])])
        probes = build_sweep_probes(iter(records), min_history=1)

        sae = TopKSAE(input_dim=8, latent_dim=16, k=2)
        encoder = sae_front_encoder(sae, mode="chain", window=4, latent_offset=8, lookup=torch.arange(8), index_map=None)
        remap = torch.arange(8)

        result_with_encoder = run_sweep(model, probes, x_values=[1], d_values=[None],
                                         fixed_x_for_depth=0, remap=remap, batch_size=4, device="cpu",
                                         front_encoder=encoder)
        result_without_encoder = run_sweep(model, probes, x_values=[1], d_values=[None],
                                            fixed_x_for_depth=0, remap=remap, batch_size=4, device="cpu",
                                            front_encoder=None)

        self.assertIn("x_sweep", result_with_encoder)
        self.assertIn("d_sweep", result_with_encoder)
        self.assertIn("classic_1hot", result_with_encoder)
        # classic is the uncompressed baseline and is defined independently of
        # front_encoder -- it must match byte-for-byte whether or not a front
        # encoder was supplied.
        self.assertEqual(result_with_encoder["classic_1hot"], result_without_encoder["classic_1hot"])


if __name__ == "__main__":
    unittest.main()
