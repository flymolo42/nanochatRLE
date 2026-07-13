import tempfile
import unittest
from pathlib import Path

import torch

from scripts.sae import TopKSAE, load_sae, save_sae


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


if __name__ == "__main__":
    unittest.main()
