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


def bags_to_dense(indices_list, input_dim):
    dense = torch.zeros(len(indices_list), input_dim)
    for row, indices in enumerate(indices_list):
        dense[row, list(indices)] = 1.0
    return dense


def train_sae(bags, input_dim, latent_dim, k, epochs=5, batch_size=1024, lr=1e-3, seed=0, device="cpu", progress_every=0):
    torch.manual_seed(seed)
    sae = TopKSAE(input_dim, latent_dim, k).to(device)
    sae.renorm_decoder()
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    order = torch.randperm(len(bags), generator=torch.Generator().manual_seed(seed))
    for epoch in range(epochs):
        total = 0.0
        for start in range(0, len(bags), batch_size):
            batch_bags = [bags[i] for i in order[start:start + batch_size].tolist()]
            x = bags_to_dense(batch_bags, input_dim).to(device)
            x_hat, _, _ = sae(x)
            loss = torch.nn.functional.mse_loss(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sae.renorm_decoder()
            total += loss.item()
        if progress_every and (epoch + 1) % progress_every == 0:
            print(f"sae epoch {epoch + 1}/{epochs} loss {total:.5f}", flush=True)
    return sae


@torch.no_grad()
def set_recovery(sae, bags, input_dim, batch_size=1024, device="cpu"):
    sae.eval()
    tp = fp = fn = exact = 0
    by_len_tp = {}
    by_len_fp = {}
    by_len_fn = {}
    for start in range(0, len(bags), batch_size):
        batch_bags = bags[start:start + batch_size]
        x = bags_to_dense(batch_bags, input_dim).to(device)
        x_hat, _, _ = sae(x)
        for row, bag in enumerate(batch_bags):
            true_set = set(bag)
            m = len(true_set)
            predicted = set(x_hat[row].topk(m).indices.tolist())
            hit = len(true_set & predicted)
            tp += hit
            fp += m - hit
            fn += m - hit
            exact += int(predicted == true_set)
            by_len_tp[m] = by_len_tp.get(m, 0) + hit
            by_len_fp[m] = by_len_fp.get(m, 0) + (m - hit)
            by_len_fn[m] = by_len_fn.get(m, 0) + (m - hit)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    by_len = {}
    for m in sorted(by_len_tp):
        p = by_len_tp[m] / max(by_len_tp[m] + by_len_fp[m], 1)
        r = by_len_tp[m] / max(by_len_tp[m] + by_len_fn[m], 1)
        by_len[m] = round(2 * p * r / max(p + r, 1e-9), 4)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_set_rate": round(exact / max(len(bags), 1), 4),
        "by_len": by_len,
        "count": len(bags),
    }
