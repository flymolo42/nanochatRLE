import unittest

import torch

from scripts.train_phrase_gpt import extend_phrase_vocab_state


class ExtendPhraseVocabTests(unittest.TestCase):
    def _checkpoint(self):
        weight = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        return {
            "model": {"phrase_wte.weight": weight.clone(), "other.weight": torch.ones(3, 2)},
            "optimizer": {
                "state": {
                    0: {"exp_avg": torch.ones(6, 2), "exp_avg_sq": torch.ones(6, 2), "step": torch.tensor(5.0)},
                    1: {"exp_avg": torch.ones(3, 2), "exp_avg_sq": torch.ones(3, 2), "step": torch.tensor(5.0)},
                },
                "param_groups": [],
            },
        }

    def test_pads_weight_and_optimizer_rows(self):
        checkpoint = extend_phrase_vocab_state(self._checkpoint(), extra_rows=4, n_embd=2, seed=0)
        weight = checkpoint["model"]["phrase_wte.weight"]
        self.assertEqual(weight.shape, (10, 2))
        self.assertTrue(torch.equal(weight[:6], torch.arange(12, dtype=torch.float32).reshape(6, 2)))
        # new rows are small random, not zeros (they must break symmetry)
        self.assertGreater(weight[6:].abs().sum().item(), 0.0)
        self.assertLess(weight[6:].abs().max().item(), 0.2)
        state = checkpoint["optimizer"]["state"]
        self.assertEqual(state[0]["exp_avg"].shape, (10, 2))
        self.assertTrue(torch.equal(state[0]["exp_avg"][6:], torch.zeros(4, 2)))
        # non-phrase params untouched
        self.assertEqual(state[1]["exp_avg"].shape, (3, 2))

    def test_untouched_when_extra_rows_zero(self):
        checkpoint = extend_phrase_vocab_state(self._checkpoint(), extra_rows=0, n_embd=2, seed=0)
        self.assertEqual(checkpoint["model"]["phrase_wte.weight"].shape, (6, 2))


class SurgeryIntegrationTests(unittest.TestCase):
    def test_extended_model_predicts_identically_on_token_inputs(self):
        import nanochat.flash_attention as fa_module
        from nanochat.gpt import GPT, GPTConfig
        fa_module._override_impl = "sdpa"
        fa_module.USE_FA3 = fa_module._resolve_use_fa3()
        torch.manual_seed(0)
        # n_embd must be >= 24: GPT.forward's smear_gate is a hardcoded Linear(24, 1)
        # applied to x[:, :, :24] regardless of config.n_embd (pre-existing constraint,
        # unrelated to this task; the brief's n_embd=8 example does not satisfy it).
        config = GPTConfig(sequence_len=8, vocab_size=16, n_layer=1, n_head=2, n_kv_head=2,
                           n_embd=24, window_pattern="L", phrase_vocab_size=16)
        model = GPT(config, pad_vocab_size_to=1)
        model.init_weights()
        checkpoint = {"model": model.state_dict(), "optimizer": {"state": {}, "param_groups": []}}
        extended = extend_phrase_vocab_state(checkpoint, extra_rows=4, n_embd=24, seed=0)
        config2 = GPTConfig(sequence_len=8, vocab_size=16, n_layer=1, n_head=2, n_kv_head=2,
                            n_embd=24, window_pattern="L", phrase_vocab_size=20)
        model2 = GPT(config2, pad_vocab_size_to=1)
        model2.load_state_dict(extended["model"])
        idx = torch.zeros(1, 4, dtype=torch.long)
        phrase_indices = torch.tensor([1, 2, 3, 4, 5, 6, 7])
        phrase_offsets = torch.tensor([0, 2, 4, 6])
        # collate_phrase_sequences always emits real [batch_idx, time_idx] pairs;
        # _encode_phrase_inputs raises ValueError if phrase_batch_positions is None
        # while phrase_indices/phrase_offsets are set, so build it the same way.
        phrase_batch_positions = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=torch.long)
        with torch.no_grad():
            a = model(idx, phrase_indices=phrase_indices, phrase_offsets=phrase_offsets, phrase_batch_positions=phrase_batch_positions)
            b = model2(idx, phrase_indices=phrase_indices, phrase_offsets=phrase_offsets, phrase_batch_positions=phrase_batch_positions)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
