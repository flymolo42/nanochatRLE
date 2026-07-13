"""
Top-k sparse autoencoder over sparse token bags (top-8k vocab space).

Latent ids double as vocabulary entries downstream: a bag's SAE code is the
set of active latent indices, offset past the token vocab.
"""

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    def __init__(self, input_dim, latent_dim, k):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

    def encode(self, x):
        pre = torch.relu(self.encoder(x))
        values, indices = pre.topk(self.k, dim=-1)
        return values, indices

    def decode_from_topk(self, values, indices):
        sparse = torch.zeros(values.shape[0], self.latent_dim, device=values.device, dtype=values.dtype)
        sparse.scatter_(1, indices, values)
        return self.decoder(sparse)

    def forward(self, x):
        values, indices = self.encode(x)
        return self.decode_from_topk(values, indices), values, indices

    @torch.no_grad()
    def renorm_decoder(self):
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.div_(norms)


def save_sae(sae, path):
    torch.save({
        "state_dict": sae.state_dict(),
        "input_dim": sae.input_dim,
        "latent_dim": sae.latent_dim,
        "k": sae.k,
        "format": "topk_sae_v1",
    }, path)


def load_sae(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(payload["input_dim"], payload["latent_dim"], payload["k"])
    sae.load_state_dict(payload["state_dict"])
    return sae
