import random
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.sae import TopKSAE, load_sae, save_sae, bags_to_dense, set_recovery, train_sae


class TopKSAETests(unittest.TestCase):
    def test_encode_is_exactly_k_sparse_and_nonnegative(self):
        torch.manual_seed(0)
        sae = TopKSAE(input_dim=32, latent_dim=64, k=4)
        x = torch.rand(8, 32)
        values, indices = sae.encode(x)
        self.assertEqual(values.shape, (8, 4))
        self.assertEqual(indices.shape, (8, 4))
        self.assertTrue((values >= 0).all())

    def test_forward_reconstructs_shape_and_uses_topk(self):
        torch.manual_seed(0)
        sae = TopKSAE(input_dim=32, latent_dim=64, k=4)
        x = torch.rand(8, 32)
        x_hat, values, indices = sae(x)
        self.assertEqual(x_hat.shape, x.shape)
        manual = sae.decode_from_topk(values, indices)
        self.assertTrue(torch.allclose(x_hat, manual))

    def test_renorm_decoder_unit_columns(self):
        torch.manual_seed(0)
        sae = TopKSAE(input_dim=32, latent_dim=64, k=4)
        sae.renorm_decoder()
        norms = sae.decoder.weight.norm(dim=0)
        self.assertTrue(torch.allclose(norms, torch.ones(64), atol=1e-5))

    def test_save_load_roundtrip(self):
        torch.manual_seed(0)
        sae = TopKSAE(input_dim=32, latent_dim=64, k=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sae.pt"
            save_sae(sae, path)
            loaded = load_sae(path)
        self.assertEqual(loaded.k, 4)
        x = torch.rand(3, 32)
        self.assertTrue(torch.allclose(sae(x)[0], loaded(x)[0]))


def _planted_bags(num_bags, num_concepts=16, tokens_per_concept=4, input_dim=64, seed=3):
    """Each concept is a fixed disjoint set of tokens; each bag is one concept's set.
    A d>=num_concepts SAE can represent this exactly, so recovery should be ~1.0."""
    rng = random.Random(seed)
    concepts = [
        list(range(c * tokens_per_concept, (c + 1) * tokens_per_concept))
        for c in range(num_concepts)
    ]
    return [sorted(concepts[rng.randrange(num_concepts)]) for _ in range(num_bags)]


class TrainSAETests(unittest.TestCase):
    def test_bags_to_dense(self):
        dense = bags_to_dense([[0, 2], [1]], input_dim=4)
        self.assertEqual(dense.tolist(), [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def test_planted_concepts_are_recovered(self):
        bags = _planted_bags(2000)
        sae = train_sae(bags, input_dim=64, latent_dim=32, k=2, epochs=40, batch_size=256, seed=0)
        metrics = set_recovery(sae, _planted_bags(200, seed=9), input_dim=64)
        self.assertGreater(metrics["f1"], 0.95)
        self.assertGreater(metrics["exact_set_rate"], 0.9)
        self.assertIn(4, metrics["by_len"])

    def test_set_recovery_is_deterministic(self):
        bags = _planted_bags(500)
        sae = train_sae(bags, input_dim=64, latent_dim=32, k=2, epochs=5, batch_size=256, seed=0)
        a = set_recovery(sae, bags[:100], input_dim=64)
        b = set_recovery(sae, bags[:100], input_dim=64)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
