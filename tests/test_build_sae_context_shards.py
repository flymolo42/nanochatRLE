# tests/test_build_sae_context_shards.py
import unittest

import torch

from scripts.build_sae_context_shards import sae_steps_for_story
from scripts.sae import TopKSAE


class SAEStepsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.sae = TopKSAE(input_dim=8, latent_dim=16, k=2)
        self.lookup = torch.arange(8)  # identity top-k remap for an 8-token space

    def test_front_slots_are_latent_ids_and_tail_is_tokens(self):
        # stream indices ascend 1,3 then descend to 2: chains [1,3] | [2,5] under cross-clause
        stream = [(0, 1), (0, 3), (1, 2), (1, 5)]
        steps = sae_steps_for_story(
            stream, index_map=None, lookup=self.lookup, sae=self.sae,
            mode="chain", window=4, split_seed=None, story_id=0, latent_offset=8,
            force_split=2,  # front = first 2 tokens, tail = rest
        )
        # step targets walk the remaining stream: front slot(s) predict token 2, then 2 predicts 5
        self.assertEqual([target for _, target in steps], [2, 5])
        front_ids = steps[0][0]
        self.assertTrue(all(8 <= latent_id < 8 + 16 for latent_id in front_ids))
        self.assertEqual(len(front_ids), 2)  # k=2 active latents
        self.assertEqual(steps[1][0], [2])   # tail 1-hot token id

    def test_window_mode_uses_fixed_windows(self):
        stream = [(0, 1), (0, 3), (1, 2), (1, 5)]
        steps = sae_steps_for_story(
            stream, index_map=None, lookup=self.lookup, sae=self.sae,
            mode="window", window=2, split_seed=None, story_id=0, latent_offset=8,
            force_split=2,
        )
        self.assertEqual([target for _, target in steps], [2, 5])

    def test_deterministic_split_choice(self):
        stream = [(0, 1), (0, 3), (1, 2), (1, 5)]
        a = sae_steps_for_story(stream, None, self.lookup, self.sae, "chain", 4, split_seed=42, story_id=7, latent_offset=8)
        b = sae_steps_for_story(stream, None, self.lookup, self.sae, "chain", 4, split_seed=42, story_id=7, latent_offset=8)
        self.assertEqual(a, b)

    def test_front_to_front_step_targets_next_bags_first_token(self):
        # front segments into TWO chain bags: [1,3] then [2,5] (index 2 <= current
        # last value 3 resets the chain); tail is a single chain [4,6]. This mirrors
        # _steps_from_chains in train_phrase_gpt.py, which the spec says these SAE
        # shards must match: every slot position emits a step, and a front slot's
        # target is the next slot's first token -- even when that next slot is
        # itself another (compressed) front slot, not just a 1-hot tail token.
        stream = [(0, 1), (0, 3), (0, 2), (0, 5), (1, 4), (1, 6)]
        steps = sae_steps_for_story(
            stream, index_map=None, lookup=self.lookup, sae=self.sae,
            mode="chain", window=4, split_seed=None, story_id=0, latent_offset=8,
            force_split=4,  # front = first 4 tokens (2 chain bags), tail = last 2
        )
        # front bag 0 -> front bag 1's first token (2); front bag 1 -> tail's
        # first token (4); tail(4) -> tail(6).
        self.assertEqual([target for _, target in steps], [2, 4, 6])
        self.assertEqual(len(steps), 3)
        # step 0 is a front->front step: its source is bag-0's latent-id slot,
        # not a raw token id.
        self.assertTrue(all(8 <= latent_id < 8 + 16 for latent_id in steps[0][0]))
        # step 1 is also a front->front-adjacent step: its source is bag-1's
        # latent-id slot.
        self.assertTrue(all(8 <= latent_id < 8 + 16 for latent_id in steps[1][0]))
        # step 2 is the tail->tail step: its source is the 1-hot tail token.
        self.assertEqual(steps[2][0], [4])


if __name__ == "__main__":
    unittest.main()
