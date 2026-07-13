import unittest

import torch

from scripts.train_phrase_gpt import extend_phrase_vocab_state, pad_phrase_optimizer_state


def _use_sdpa():
    import nanochat.flash_attention as fa_module
    fa_module._override_impl = "sdpa"
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()


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

    def test_pads_weight_rows(self):
        # extend_phrase_vocab_state only pads the phrase_wte.weight model tensor
        # (by name). Optimizer moment padding is handled separately, by identity,
        # in pad_phrase_optimizer_state -- see PadPhraseOptimizerStateTests below.
        checkpoint = extend_phrase_vocab_state(self._checkpoint(), extra_rows=4, n_embd=2, seed=0)
        weight = checkpoint["model"]["phrase_wte.weight"]
        self.assertEqual(weight.shape, (10, 2))
        self.assertTrue(torch.equal(weight[:6], torch.arange(12, dtype=torch.float32).reshape(6, 2)))
        # new rows are small random, not zeros (they must break symmetry)
        self.assertGreater(weight[6:].abs().sum().item(), 0.0)
        self.assertLess(weight[6:].abs().max().item(), 0.2)
        # extend_phrase_vocab_state must NOT touch optimizer state at all anymore
        # (that used to be a shape-based guess that could corrupt unrelated params).
        state = checkpoint["optimizer"]["state"]
        self.assertEqual(state[0]["exp_avg"].shape, (6, 2))
        self.assertEqual(state[1]["exp_avg"].shape, (3, 2))

    def test_untouched_when_extra_rows_zero(self):
        checkpoint = extend_phrase_vocab_state(self._checkpoint(), extra_rows=0, n_embd=2, seed=0)
        self.assertEqual(checkpoint["model"]["phrase_wte.weight"].shape, (6, 2))

    def test_uses_model_state_dict_key_fallback(self):
        # Real on-disk checkpoints use "model_state_dict", not "model". This was
        # previously only exercised indirectly; test it directly.
        weight = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        checkpoint = {
            "model_state_dict": {"phrase_wte.weight": weight.clone(), "other.weight": torch.ones(3, 2)},
            "optimizer_state_dict": {"state": {}, "param_groups": []},
        }
        result = extend_phrase_vocab_state(checkpoint, extra_rows=4, n_embd=2, seed=0)
        padded = result["model_state_dict"]["phrase_wte.weight"]
        self.assertEqual(padded.shape, (10, 2))
        self.assertTrue(torch.equal(padded[:6], weight))
        # unrelated key untouched
        self.assertTrue(torch.equal(result["model_state_dict"]["other.weight"], torch.ones(3, 2)))


class PadPhraseOptimizerStateTests(unittest.TestCase):
    def _model_and_optimizer(self, phrase_vocab_size, vocab_size=8, n_embd=24):
        from nanochat.gpt import GPT, GPTConfig
        config = GPTConfig(sequence_len=8, vocab_size=vocab_size, n_layer=1, n_head=2, n_kv_head=2,
                            n_embd=n_embd, window_pattern="L", phrase_vocab_size=phrase_vocab_size)
        model = GPT(config, pad_vocab_size_to=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, optimizer

    def _param_index(self, optimizer, param):
        params = [p for group in optimizer.param_groups for p in group["params"]]
        return next(i for i, p in enumerate(params) if p is param)

    def test_pads_only_phrase_wte_by_identity_even_with_shape_collision(self):
        # model already reflects the EXTENDED phrase_vocab_size (12 = old 8 + extra 4),
        # as it does in the trainer's real resume path where the model is built with
        # the extended config before the checkpoint is loaded. vocab_size=8 means
        # transformer.wte's optimizer moments have the SAME old row count (8) as
        # phrase_wte's old row count -- the exact collision that caused the bug.
        model, optimizer = self._model_and_optimizer(phrase_vocab_size=12, vocab_size=8)
        phrase_index = self._param_index(optimizer, model.phrase_wte.weight)
        wte_index = self._param_index(optimizer, model.transformer.wte.weight)
        old_rows = 8
        checkpoint = {
            "model_state_dict": {},
            "optimizer_state_dict": {
                "state": {
                    phrase_index: {
                        "exp_avg": torch.ones(old_rows, 24),
                        "exp_avg_sq": torch.ones(old_rows, 24),
                        "step": torch.tensor(5.0),
                    },
                    wte_index: {
                        "exp_avg": torch.full((old_rows, 24), 2.0),
                        "exp_avg_sq": torch.full((old_rows, 24), 2.0),
                        "step": torch.tensor(5.0),
                    },
                },
                "param_groups": [],
            },
        }
        result = pad_phrase_optimizer_state(checkpoint, model, optimizer, extra_rows=4)
        state = result["optimizer_state_dict"]["state"]
        # phrase_wte's entry padded to the new row count, zeros appended
        self.assertEqual(state[phrase_index]["exp_avg"].shape, (12, 24))
        self.assertTrue(torch.equal(state[phrase_index]["exp_avg"][8:], torch.zeros(4, 24)))
        self.assertTrue(torch.equal(state[phrase_index]["exp_avg"][:8], torch.ones(8, 24)))
        # wte's entry, despite having the exact same old shape, is left untouched
        self.assertEqual(state[wte_index]["exp_avg"].shape, (old_rows, 24))
        self.assertTrue(torch.equal(state[wte_index]["exp_avg"], torch.full((old_rows, 24), 2.0)))
        # scalar step untouched/unpadded
        self.assertEqual(state[phrase_index]["step"].shape, torch.Size([]))

    def test_untouched_when_extra_rows_zero(self):
        model, optimizer = self._model_and_optimizer(phrase_vocab_size=12, vocab_size=8)
        checkpoint = {"model_state_dict": {}, "optimizer_state_dict": {"state": {}, "param_groups": []}}
        result = pad_phrase_optimizer_state(checkpoint, model, optimizer, extra_rows=0)
        self.assertEqual(result["optimizer_state_dict"]["state"], {})

    def test_noop_when_phrase_index_has_no_saved_state(self):
        model, optimizer = self._model_and_optimizer(phrase_vocab_size=12, vocab_size=8)
        checkpoint = {"model_state_dict": {}, "optimizer_state_dict": {"state": {}, "param_groups": []}}
        # should not raise even though there's nothing to pad
        result = pad_phrase_optimizer_state(checkpoint, model, optimizer, extra_rows=4)
        self.assertEqual(result["optimizer_state_dict"]["state"], {})


class SurgeryIntegrationTests(unittest.TestCase):
    def test_extended_model_predicts_identically_on_token_inputs(self):
        _use_sdpa()
        from nanochat.gpt import GPT, GPTConfig
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


class OptimizerCollisionRegressionTests(unittest.TestCase):
    """Reproduces the real bug: vocab_size == phrase_vocab_size means wte,
    lm_head, and phrase_wte optimizer moments all start out with identical
    shapes. Shape-based padding corrupts wte/lm_head; identity-based padding
    (the fix) must touch only phrase_wte and must survive a real optimizer.step().
    """

    def _step(self, model, optimizer, vocab_size, phrase_vocab_size):
        idx = torch.randint(0, vocab_size, (1, 4))
        targets = torch.randint(0, vocab_size, (1, 4))
        phrase_indices = torch.randint(0, phrase_vocab_size, (7,))
        phrase_offsets = torch.tensor([0, 2, 4, 6])
        phrase_batch_positions = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=torch.long)
        loss = model(idx, targets=targets, phrase_indices=phrase_indices,
                     phrase_offsets=phrase_offsets, phrase_batch_positions=phrase_batch_positions)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_surgery_pad_load_and_step_survive_the_shape_collision(self):
        _use_sdpa()
        from nanochat.gpt import GPT, GPTConfig
        torch.manual_seed(0)
        vocab_size = 32
        old_phrase_rows = 32  # == vocab_size: the exact collision condition
        n_embd = 24
        extra_rows = 4

        config = GPTConfig(sequence_len=8, vocab_size=vocab_size, n_layer=1, n_head=2, n_kv_head=2,
                            n_embd=n_embd, window_pattern="L", phrase_vocab_size=old_phrase_rows)
        model = GPT(config, pad_vocab_size_to=1)
        model.init_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Build real AdamW state (exp_avg/exp_avg_sq) via one real training step,
        # exactly as the trainer would before saving a checkpoint.
        self._step(model, optimizer, vocab_size, old_phrase_rows)

        checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
        checkpoint = extend_phrase_vocab_state(checkpoint, extra_rows=extra_rows, n_embd=n_embd, seed=0)

        config2 = GPTConfig(sequence_len=8, vocab_size=vocab_size, n_layer=1, n_head=2, n_kv_head=2,
                             n_embd=n_embd, window_pattern="L", phrase_vocab_size=old_phrase_rows + extra_rows)
        model2 = GPT(config2, pad_vocab_size_to=1)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        # Pad the optimizer state BEFORE loading it, as the trainer's resume path does.
        checkpoint = pad_phrase_optimizer_state(checkpoint, model2, optimizer2, extra_rows=extra_rows)
        model2.load_state_dict(checkpoint["model_state_dict"])
        optimizer2.load_state_dict(checkpoint["optimizer_state_dict"])

        # wte/lm_head moments must have kept their ORIGINAL row count -- the bug
        # padded them too because they shared phrase_wte's old shape.
        wte_state = optimizer2.state[model2.transformer.wte.weight]
        self.assertEqual(wte_state["exp_avg"].shape[0], vocab_size)
        self.assertEqual(wte_state["exp_avg_sq"].shape[0], vocab_size)
        lm_head_state = optimizer2.state[model2.lm_head.weight]
        self.assertEqual(lm_head_state["exp_avg"].shape[0], vocab_size)
        phrase_state = optimizer2.state[model2.phrase_wte.weight]
        self.assertEqual(phrase_state["exp_avg"].shape[0], old_phrase_rows + extra_rows)

        # And the extended model/optimizer must actually be usable: a further
        # optimizer.step() must not crash with a shape-mismatch RuntimeError.
        self._step(model2, optimizer2, vocab_size, old_phrase_rows + extra_rows)


if __name__ == "__main__":
    unittest.main()
