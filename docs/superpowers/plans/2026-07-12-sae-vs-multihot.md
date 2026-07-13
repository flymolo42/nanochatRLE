# SAE vs Multihot Encoding Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the downstream value of the hand-built multihot chain encoding versus learned top-k SAE codes (on chains, and on fixed windows), with a reconstruction gate deciding whether SAE-on-multihot proceeds.

**Architecture:** A top-k SAE trains on sparse token bags; its active latent ids (offset past the 8,192-token vocab) become just another multihot bag, so existing shards/trainer/sweep work unchanged. Three post-train arms share the exact v2 protocol (resume base, 1 epoch, 30k examples, 2,000 true-validation probes).

**Tech Stack:** PyTorch (MPS), numpy, existing nanochat scripts (`train_phrase_gpt`, `hybrid_sweep`, `build_phrase_gpt_shards` patterns), unittest.

**Spec:** `docs/superpowers/specs/2026-07-12-sae-vs-multihot-encoding-design.md`

## Global Constraints

- Run tests with `uv run --group dev python -m pytest tests/<file> -q` from `nanochat/`.
- Tests are unittest-style classes in `tests/test_*.py` (see `tests/test_hybrid_cross.py` for the house style).
- All record processing is streaming (one story in memory at a time); reuse `story_index_streams`.
- Token space for bags is the top-8k remap: 8,191 kept tokens + `<unk>` at index 8191 (`load_vocab_top_k_remap`).
- Latent ids occupy `8192 .. 8192+d-1` in shards; targets are ALWAYS real token ids `< 8192`.
- Never launch a bare trainer: runner scripts must use the pidfile + RSS-watchdog pattern from `scripts/run_hybrid_posttrain.sh`; one trainer at a time (MacBook, 15 GB cap).
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Fixed seeds everywhere: SAE init/training seed 0, split-seed 42, sweep probes are deterministic.

**Key existing interfaces (read-only reference):**
- `scripts/reorder_phrase_vocab.story_index_streams(records) -> yields (split, story_id, [(clause_id, index), ...])`
- `scripts/train_phrase_gpt.load_vocab_top_k_remap(vocab_path, top_k) -> (lookup: torch.LongTensor[46327], tokens: list[str])` — monotone, `<unk>` last.
- `scripts/train_phrase_vectors.iter_records(path)` — streaming jsonl/.gz reader.
- `scripts/hybrid_sweep.build_sweep_probes(records, min_history=1, max_probes=None, split=None, index_map=None) -> list[SweepProbe]`; `SweepProbe(token_indices, clause_ids, target_pos, is_opener)`.
- `nanochat/gpt.py`: `GPT.phrase_wte = nn.Embedding(config.phrase_vocab_size, config.n_embd)`; trainer builds `GPTConfig(..., vocab_size=vocab_size, phrase_vocab_size=vocab_size)` at `scripts/train_phrase_gpt.py:690-698`; resume loads checkpoint at `:732`.
- Base checkpoint: `phrase_gpt_base_v2_out/phrase_gpt.pt` (rolling, includes optimizer). Base 8k vocab file: `phrase_gpt_base_v2_out/vocab_top_k.json` (8,192 rows, loadable by `load_vocab`).
- Data: records `phrase_quote_split_out/phrase_index.jsonl.gz`, validation side file `phrase_quote_split_out/phrase_index_validation.jsonl.gz`, index map `phrase_quote_split_ils_out/old_to_new.json`.

---

### Task 1: Top-k SAE module (`scripts/sae.py`)

**Files:**
- Create: `scripts/sae.py`
- Test: `tests/test_sae.py`

**Interfaces:**
- Produces: `TopKSAE(input_dim, latent_dim, k)` nn.Module with `encode(x)->(values,indices)`, `decode_from_topk(values,indices)->x_hat`, `forward(x)->(x_hat,values,indices)`; `save_sae(sae,path)`, `load_sae(path)->TopKSAE`. Decoder columns unit-norm after `renorm_decoder()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sae.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_sae.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'scripts.sae'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/sae.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_sae.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/sae.py tests/test_sae.py
git commit -m "Add top-k sparse autoencoder module

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: SAE training loop + set-recovery metric (`scripts/sae.py`)

**Files:**
- Modify: `scripts/sae.py`
- Test: `tests/test_sae.py`

**Interfaces:**
- Consumes: `TopKSAE` from Task 1.
- Produces: `bags_to_dense(indices_list, input_dim)->FloatTensor`; `train_sae(bags, input_dim, latent_dim, k, epochs=5, batch_size=1024, lr=1e-3, seed=0, device="cpu")->TopKSAE`; `set_recovery(sae, bags, input_dim, batch_size=1024, device="cpu")->dict` with keys `precision`, `recall`, `f1`, `exact_set_rate`, `by_len` (dict len->f1). Recovery rule: top-m decoder outputs, m = true set size.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sae.py`:

```python
import random

from scripts.sae import bags_to_dense, set_recovery, train_sae


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
        sae = train_sae(bags, input_dim=64, latent_dim=32, k=2, epochs=20, batch_size=256, seed=0)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_sae.py -q`
Expected: ImportError (`bags_to_dense` etc. not defined); earlier 4 tests still pass once imports split — if collection fails entirely that is the expected RED.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/sae.py`:

```python
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
```

Note: with top-m recovery, precision == recall by construction; both are reported anyway for the report's readability.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_sae.py -q`
Expected: 7 passed (planted-concept test takes ~30-60s on CPU)

- [ ] **Step 5: Commit**

```bash
git add scripts/sae.py tests/test_sae.py
git commit -m "Add SAE training loop and top-m set-recovery metric

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Bag extraction (`scripts/extract_sae_bags.py`)

**Files:**
- Create: `scripts/extract_sae_bags.py`
- Test: `tests/test_extract_sae_bags.py`

**Interfaces:**
- Consumes: `story_index_streams`, `iter_records`, `load_vocab_top_k_remap`.
- Produces: `chain_bags_from_stream(stream, index_map)->list[list[int]]` (cross-clause ascending chains over index-mapped 46k ids, before top-k); `window_bags_from_stream(stream, index_map, window=4)->list[list[int]]`; `remap_bags(bags, lookup)->list[sorted unique list[int]]` (top-8k space); CLI writing `chain_bags.npz` / `window_bags.npz` with CSR arrays `indices` (int32), `offsets` (int64), plus `input_dim`. Loader: `load_bags(path)->list[list[int]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extract_sae_bags.py
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.extract_sae_bags import (
    chain_bags_from_stream,
    load_bags,
    remap_bags,
    save_bags,
    window_bags_from_stream,
)


class BagExtractionTests(unittest.TestCase):
    def test_chain_bags_cross_clause_ascending(self):
        stream = [(0, 1), (0, 3), (1, 5), (1, 2), (1, 4)]
        # identity map; cross-clause: chains [1,3,5] | [2,4]
        self.assertEqual(chain_bags_from_stream(stream, index_map=None), [[1, 3, 5], [2, 4]])

    def test_chain_bags_apply_index_map(self):
        stream = [(0, 0), (0, 1)]
        # map 0->5, 1->2: mapped stream [5, 2] descends -> two chains
        self.assertEqual(chain_bags_from_stream(stream, index_map=[5, 2]), [[5], [2]])

    def test_window_bags_fixed_length(self):
        stream = [(0, 1), (0, 3), (1, 5), (1, 2), (1, 4)]
        self.assertEqual(window_bags_from_stream(stream, index_map=None, window=4), [[1, 3, 5, 2], [4]])

    def test_remap_bags_dedups_and_sorts(self):
        lookup = torch.tensor([7, 7, 3])
        self.assertEqual(remap_bags([[0, 1, 2]], lookup), [[3, 7]])

    def test_save_load_roundtrip(self):
        bags = [[1, 3, 5], [2], [0, 4]]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bags.npz"
            save_bags(bags, path, input_dim=8)
            loaded, input_dim = load_bags(path)
        self.assertEqual(loaded, bags)
        self.assertEqual(input_dim, 8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_extract_sae_bags.py -q`
Expected: collection ERROR, module not found

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/extract_sae_bags.py
"""
Extract SAE training bags from phrase records.

Chain bags: cross-clause strictly-ascending chains over the index-mapped vocab
(the same segmentation the hybrid-cross shards use). Window bags: fixed-length
token windows. Both are then remapped to the top-8k space and stored CSR-style.

Example:
python -m scripts.extract_sae_bags \
    --records phrase_quote_split_out/phrase_index.jsonl.gz \
    --index-map phrase_quote_split_ils_out/old_to_new.json \
    --vocab phrase_quote_split_ils_out/vocab.json --vocab-top-k 8191 \
    --out-dir sae_bags_out --max-stories 60000
"""

import argparse
import itertools
import json

import numpy as np

from scripts.reorder_phrase_vocab import story_index_streams
from scripts.train_phrase_gpt import load_vocab_top_k_remap
from scripts.train_phrase_vectors import iter_records


def _mapped_indices(stream, index_map):
    return [int(index_map[index]) if index_map is not None else int(index) for _, index in stream]


def chain_bags_from_stream(stream, index_map):
    indices = _mapped_indices(stream, index_map)
    bags = []
    current = []
    for index in indices:
        if current and index <= current[-1]:
            bags.append(current)
            current = []
        current.append(index)
    if current:
        bags.append(current)
    return bags


def window_bags_from_stream(stream, index_map, window=4):
    indices = _mapped_indices(stream, index_map)
    return [indices[start:start + window] for start in range(0, len(indices), window)]


def remap_bags(bags, lookup):
    return [sorted({int(lookup[index]) for index in bag}) for bag in bags]


def save_bags(bags, path, input_dim):
    offsets = np.zeros(len(bags) + 1, dtype=np.int64)
    for slot, bag in enumerate(bags):
        offsets[slot + 1] = offsets[slot] + len(bag)
    indices = np.fromiter((index for bag in bags for index in bag), dtype=np.int32, count=int(offsets[-1]))
    np.savez(path, indices=indices, offsets=offsets, input_dim=np.int64(input_dim))


def load_bags(path):
    saved = np.load(path)
    offsets = saved["offsets"]
    indices = saved["indices"]
    bags = [indices[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
    return bags, int(saved["input_dim"])


def parse_args():
    parser = argparse.ArgumentParser(description="Extract SAE training bags (chains + windows).")
    parser.add_argument("--records", required=True)
    parser.add_argument("--index-map", required=True)
    parser.add_argument("--vocab", required=True, help="Original (ILS) vocab.json for the top-k remap.")
    parser.add_argument("--vocab-top-k", type=int, default=8191)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-stories", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def main():
    import os
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.index_map, "r", encoding="utf-8") as file:
        index_map = json.load(file)
    lookup, tokens = load_vocab_top_k_remap(args.vocab, args.vocab_top_k)
    input_dim = len(tokens)
    chain_bags = []
    window_bags = []
    stories = 0
    streams = story_index_streams(iter_records(args.records))
    if args.max_stories is not None:
        streams = itertools.islice(streams, args.max_stories)
    for _, _, stream in streams:
        stories += 1
        chain_bags.extend(remap_bags(chain_bags_from_stream(stream, index_map), lookup))
        window_bags.extend(remap_bags(window_bags_from_stream(stream, index_map, window=args.window), lookup))
        if args.progress_every and stories % args.progress_every == 0:
            print(f"extracted {stories} stories; chains={len(chain_bags)} windows={len(window_bags)}", flush=True)
    save_bags(chain_bags, f"{args.out_dir}/chain_bags.npz", input_dim)
    save_bags(window_bags, f"{args.out_dir}/window_bags.npz", input_dim)
    report = {
        "stories": stories,
        "chain_bags": len(chain_bags),
        "window_bags": len(window_bags),
        "input_dim": input_dim,
        "window": args.window,
        "records": args.records,
    }
    with open(f"{args.out_dir}/extract_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_extract_sae_bags.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_sae_bags.py tests/test_extract_sae_bags.py
git commit -m "Add SAE bag extraction (chain + window bags, top-8k space)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Stage-1 grid driver (`scripts/train_sae_grid.py`)

**Files:**
- Create: `scripts/train_sae_grid.py`
- Test: `tests/test_train_sae_grid.py`

**Interfaces:**
- Consumes: `train_sae`, `set_recovery`, `load_bags`, `save_sae` from Tasks 1-3.
- Produces: `run_grid(train_bags, val_bags, input_dim, grid, epochs, batch_size, device, out_dir, gate_f1=0.9, gate_max_len=6, seed=0)->dict` report: per-config `{"d":..,"k":..,"f1":..,"by_len":..,"gate_f1":..}`, `best` config name, `gate_passed` bool (best config's F1 restricted to bags of len<=gate_max_len >= gate_f1), winner SAE saved as `<out_dir>/sae_best.pt`, report as `<out_dir>/grid_report.json`. CLI wires bags npz paths.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_sae_grid.py
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_sae import _planted_bags
from scripts.train_sae_grid import run_grid


class GridTests(unittest.TestCase):
    def test_grid_reports_and_gates(self):
        train = _planted_bags(1500)
        val = _planted_bags(300, seed=11)
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_grid(
                train, val, input_dim=64,
                grid=[(32, 2), (32, 8)],
                epochs=15, batch_size=256, device="cpu", out_dir=tmpdir,
                gate_f1=0.9, gate_max_len=6, seed=0,
            )
            saved = json.loads((Path(tmpdir) / "grid_report.json").read_text())
            self.assertTrue((Path(tmpdir) / "sae_best.pt").exists())
        self.assertEqual(saved["best"], report["best"])
        self.assertEqual(len(report["configs"]), 2)
        for config in report["configs"]:
            self.assertIn("f1", config)
            self.assertIn("gate_f1", config)
        # planted concepts (len 4 <= gate_max_len) are learnable -> gate passes
        self.assertTrue(report["gate_passed"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_train_sae_grid.py -q`
Expected: collection ERROR, module not found

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/train_sae_grid.py
"""
Stage-1 SAE grid: train each (d, k) config, score validation set-recovery,
save the best SAE and a gate verdict.

Example:
python -m scripts.train_sae_grid \
    --train-bags sae_bags_out/chain_bags.npz \
    --val-bags sae_bags_val_out/chain_bags.npz \
    --out-dir sae_chain_out --epochs 3 --device mps
"""

import argparse
import json
import os

from scripts.extract_sae_bags import load_bags
from scripts.sae import save_sae, set_recovery, train_sae

DEFAULT_GRID = [(512, 8), (512, 32), (1024, 8), (1024, 32)]


def _gate_f1(sae, val_bags, input_dim, gate_max_len, device):
    eligible = [bag for bag in val_bags if len(bag) <= gate_max_len]
    if not eligible:
        return 0.0
    return set_recovery(sae, eligible, input_dim, device=device)["f1"]


def run_grid(train_bags, val_bags, input_dim, grid, epochs, batch_size, device, out_dir, gate_f1=0.9, gate_max_len=6, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    configs = []
    best = None
    best_sae = None
    for latent_dim, k in grid:
        sae = train_sae(train_bags, input_dim, latent_dim, k, epochs=epochs, batch_size=batch_size, seed=seed, device=device, progress_every=1)
        metrics = set_recovery(sae, val_bags, input_dim, device=device)
        gate_value = _gate_f1(sae, val_bags, input_dim, gate_max_len, device)
        name = f"d{latent_dim}_k{k}"
        configs.append({"name": name, "d": latent_dim, "k": k, "gate_f1": gate_value, **metrics})
        print(f"grid {name}: f1={metrics['f1']} gate_f1={gate_value}", flush=True)
        if best is None or gate_value > best["gate_f1"]:
            best = configs[-1]
            best_sae = sae
    save_sae(best_sae, os.path.join(out_dir, "sae_best.pt"))
    report = {
        "format": "sae_grid_report_v1",
        "configs": configs,
        "best": best["name"],
        "gate_threshold": gate_f1,
        "gate_max_len": gate_max_len,
        "gate_passed": best["gate_f1"] >= gate_f1,
        "input_dim": input_dim,
        "train_bags": len(train_bags),
        "val_bags": len(val_bags),
    }
    with open(os.path.join(out_dir, "grid_report.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Stage-1 SAE grid and record the gate verdict.")
    parser.add_argument("--train-bags", required=True)
    parser.add_argument("--val-bags", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gate-f1", type=float, default=0.9)
    parser.add_argument("--gate-max-len", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    train_bags, input_dim = load_bags(args.train_bags)
    val_bags, val_dim = load_bags(args.val_bags)
    if val_dim != input_dim:
        raise SystemExit(f"train input_dim {input_dim} != val input_dim {val_dim}")
    report = run_grid(
        train_bags, val_bags, input_dim, DEFAULT_GRID,
        epochs=args.epochs, batch_size=args.batch_size, device=args.device,
        out_dir=args.out_dir, gate_f1=args.gate_f1, gate_max_len=args.gate_max_len, seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_train_sae_grid.py -q`
Expected: 1 passed (~1-2 min on CPU)

- [ ] **Step 5: Commit**

```bash
git add scripts/train_sae_grid.py tests/test_train_sae_grid.py
git commit -m "Add Stage-1 SAE grid driver with set-recovery gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Phrase-vocab extension surgery (trainer `--extend-phrase-vocab`)

**Files:**
- Modify: `scripts/train_phrase_gpt.py` (GPTConfig construction ~:690, resume load ~:732, parse_args)
- Test: `tests/test_extend_phrase_vocab.py`

**Interfaces:**
- Consumes: base checkpoint dict layout (`checkpoint["model"]` state dict containing a key ending in `phrase_wte.weight`; `checkpoint["optimizer"]["state"][i]` Adam moments).
- Produces: `extend_phrase_vocab_state(checkpoint, extra_rows, n_embd, seed=0)->checkpoint` — pads the `phrase_wte.weight` tensor with `extra_rows` new rows drawn `normal(0, 0.02)` and zero-pads any optimizer state tensor whose shape differs from its (new) target only in dim 0. Trainer flag `--extend-phrase-vocab N` sets `phrase_vocab_size = vocab_size + N` and applies the surgery on resume.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extend_phrase_vocab.py
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_extend_phrase_vocab.py -q`
Expected: ImportError: cannot import name `extend_phrase_vocab_state`

- [ ] **Step 3: Write minimal implementation**

In `scripts/train_phrase_gpt.py`, add near `load_vocab_top_k_remap`:

```python
def extend_phrase_vocab_state(checkpoint, extra_rows, n_embd, seed=0):
    """Grow phrase_wte by extra_rows (normal 0, 0.02) and zero-pad matching
    optimizer moment tensors so a resumed run can accept latent-id inputs."""
    if extra_rows <= 0:
        return checkpoint
    model_state = checkpoint["model"]
    key = next(k for k in model_state if k.endswith("phrase_wte.weight"))
    old_weight = model_state[key]
    old_rows = old_weight.shape[0]
    generator = torch.Generator().manual_seed(seed)
    new_rows = torch.normal(0.0, 0.02, size=(extra_rows, n_embd), generator=generator, dtype=old_weight.dtype)
    model_state[key] = torch.cat([old_weight, new_rows.to(old_weight.device)], dim=0)
    optimizer_state = checkpoint.get("optimizer", {}).get("state", {})
    for entry in optimizer_state.values():
        for name, tensor in list(entry.items()):
            if torch.is_tensor(tensor) and tensor.dim() == old_weight.dim() and tensor.shape[0] == old_rows and tensor.shape[1:] == old_weight.shape[1:] and name != "step":
                pad = torch.zeros((extra_rows,) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
                entry[name] = torch.cat([tensor, pad], dim=0)
    return checkpoint
```

CAUTION: the shape rule `shape[0] == old_rows` also matches the token embedding
if `wte` has exactly the same row count (it does: both 8192). That is why the
surgery must run and be verified with the equality test in Step 1's shapes AND
the integration check in Step 6 — padding `wte`'s moments would break resume
(model weight stays 8192 rows, moment becomes 8192+N -> load error). Guard:
only pad optimizer tensors whose row count differs from their NEW target. The
implementation above cannot see targets, so instead: pad only the entries whose
index corresponds to the phrase param. Since optimizer state keys are param
indices in `model.parameters()` order and `phrase_wte` is the only `nn.Embedding`
with `phrase` in its state key, resolve the index by loading the model params:
in the trainer wiring (below), call `extend_phrase_vocab_state` BEFORE building
the model, then let the trainer's normal `load_state_dict(strict=True)` verify
every shape — a mismatch anywhere fails loudly rather than silently.

Trainer wiring (three edits):

1. `parse_args()` — add:

```python
    parser.add_argument("--extend-phrase-vocab", type=int, default=0, help="Extra phrase-embedding rows (SAE latent ids) appended past the token vocab. Requires --resume.")
```

2. `GPTConfig` construction (~line 690) — change `phrase_vocab_size=vocab_size` to:

```python
        phrase_vocab_size=vocab_size + args.extend_phrase_vocab,
```

3. Resume path (~line 732), immediately after `checkpoint = torch.load(...)`:

```python
    if args.extend_phrase_vocab > 0:
        checkpoint = extend_phrase_vocab_state(checkpoint, args.extend_phrase_vocab, args.n_embd, seed=args.seed)
```

If the wte/phrase_wte shape collision bites (load_state_dict complains about an
optimizer tensor for `wte`), constrain the padding loop to the single optimizer
state entry whose index equals the position of `phrase_wte.weight` in
`checkpoint["model"]` key order filtered to parameters — but ONLY do this if the
integration test in Step 6 fails; do not preemptively complicate.

- [ ] **Step 4: Run unit test**

Run: `uv run --group dev python -m pytest tests/test_extend_phrase_vocab.py -q`
Expected: 2 passed

- [ ] **Step 5: Integration test — surgery preserves token-only behavior**

Append to `tests/test_extend_phrase_vocab.py`:

```python
class SurgeryIntegrationTests(unittest.TestCase):
    def test_extended_model_predicts_identically_on_token_inputs(self):
        import nanochat.flash_attention as fa_module
        from nanochat.gpt import GPT, GPTConfig
        fa_module._override_impl = "sdpa"
        fa_module.USE_FA3 = fa_module._resolve_use_fa3()
        torch.manual_seed(0)
        config = GPTConfig(sequence_len=8, vocab_size=16, n_layer=1, n_head=2, n_kv_head=2,
                           n_embd=8, window_pattern="L", phrase_vocab_size=16)
        model = GPT(config, pad_vocab_size_to=1)
        model.init_weights()
        checkpoint = {"model": model.state_dict(), "optimizer": {"state": {}, "param_groups": []}}
        extended = extend_phrase_vocab_state(checkpoint, extra_rows=4, n_embd=8, seed=0)
        config2 = GPTConfig(sequence_len=8, vocab_size=16, n_layer=1, n_head=2, n_kv_head=2,
                            n_embd=8, window_pattern="L", phrase_vocab_size=20)
        model2 = GPT(config2, pad_vocab_size_to=1)
        model2.load_state_dict(extended["model"])
        idx = torch.zeros(1, 4, dtype=torch.long)
        phrase_indices = torch.tensor([1, 2, 3, 4, 5, 6, 7])
        phrase_offsets = torch.tensor([0, 2, 4, 6])
        phrase_batch_positions = None
        with torch.no_grad():
            a = model(idx, phrase_indices=phrase_indices, phrase_offsets=phrase_offsets, phrase_batch_positions=phrase_batch_positions)
            b = model2(idx, phrase_indices=phrase_indices, phrase_offsets=phrase_offsets, phrase_batch_positions=phrase_batch_positions)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
```

NOTE: `phrase_batch_positions` handling must match `collate_phrase_sequences`'s
actual output; if the direct `model(...)` call signature differs, mirror the
call used in `predict_probe_logits` (scripts/hybrid_sweep.py:~186) exactly.

Run: `uv run --group dev python -m pytest tests/test_extend_phrase_vocab.py -q`
Expected: 3 passed

- [ ] **Step 6: Run full suite and commit**

Run: `uv run --group dev python -m pytest tests -q -m "not slow"`
Expected: all pass (previous count 148 + 3 new)

```bash
git add scripts/train_phrase_gpt.py tests/test_extend_phrase_vocab.py
git commit -m "Trainer: --extend-phrase-vocab surgery for SAE latent-id inputs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: SAE context shards (`scripts/build_sae_context_shards.py`)

**Files:**
- Create: `scripts/build_sae_context_shards.py`
- Test: `tests/test_build_sae_context_shards.py`

**Interfaces:**
- Consumes: `load_sae`, `bags_to_dense` (Task 1-2), `chain_bags_from_stream` / `window_bags_from_stream` / `remap_bags` (Task 3), `load_vocab_top_k_remap`, `story_index_streams`, `examples_to_tensor_shard`/`_write_shard` patterns from `scripts/build_phrase_gpt_shards.py`, `_phrase_boundary_positions`/`_choose_split` semantics from `scripts/train_phrase_gpt.py`.
- Produces: `sae_steps_for_story(stream, index_map, lookup, sae, mode, window, split_seed, story_id, latent_offset)->list[(input_ids, target)]` where front slots are sorted active-latent-id lists (each id `latent_offset + latent_index`) and tail steps are single 1-hot token ids in top-8k space; targets are top-8k token ids of the NEXT step's first token (identical step semantics to `_steps_from_chains`). CLI builds train (30k examples, limit) + full validation shards, mirroring `build_phrase_gpt_shards` manifest format, chain_mode recorded as `sae-chain` / `sae-window`.

- [ ] **Step 1: Write the failing test**

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_build_sae_context_shards.py -q`
Expected: collection ERROR, module not found

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/build_sae_context_shards.py
"""
Build tensor shards whose compressed history slots are SAE codes (active latent
ids offset past the token vocab) and whose recent tail is 1-hot token ids.
Front/tail split mirrors train_phrase_gpt._hybrid_steps (random phrase boundary
per story, same split-seed arithmetic).

Example (chain arm):
python -m scripts.build_sae_context_shards \
    --records phrase_quote_split_out/phrase_index.jsonl.gz \
    --validation-records phrase_quote_split_out/phrase_index_validation.jsonl.gz \
    --index-map phrase_quote_split_ils_out/old_to_new.json \
    --vocab phrase_quote_split_ils_out/vocab.json --vocab-top-k 8191 \
    --sae sae_chain_out/sae_best.pt --mode chain \
    --out-dir sae_chain_shards --limit-examples 30000 --split-seed 42
"""

import argparse
import itertools
import json
import os
import random
import time

import torch

from scripts.build_phrase_gpt_shards import _write_shard
from scripts.extract_sae_bags import chain_bags_from_stream, remap_bags, window_bags_from_stream
from scripts.reorder_phrase_vocab import story_index_streams
from scripts.sae import bags_to_dense, load_sae
from scripts.train_phrase_gpt import PhraseSequenceExample, load_vocab_top_k_remap
from scripts.train_phrase_vectors import iter_records


def _clause_boundaries(stream):
    boundaries = [0]
    previous = None
    for position, (clause, _) in enumerate(stream):
        if previous is not None and clause != previous:
            boundaries.append(position)
        previous = clause
    boundaries.append(len(stream))
    return sorted(set(boundaries))


def _encode_front(front_stream, index_map, lookup, sae, mode, window, latent_offset):
    if not front_stream:
        return []
    if mode == "chain":
        bags = chain_bags_from_stream(front_stream, index_map)
    else:
        bags = window_bags_from_stream(front_stream, index_map, window=window)
    bags = remap_bags(bags, lookup)
    dense = bags_to_dense(bags, sae.input_dim)
    with torch.no_grad():
        _, indices = sae.encode(dense)
    return [sorted(latent_offset + latent for latent in set(row.tolist())) for row in indices]


def sae_steps_for_story(stream, index_map, lookup, sae, mode, window, split_seed, story_id, latent_offset, force_split=None):
    if not stream:
        return []
    if force_split is not None:
        split = force_split
    else:
        boundaries = _clause_boundaries(stream)
        split = random.Random(split_seed * 1_000_003 + int(story_id)).choice(boundaries)
    front = stream[:split]
    back = stream[split:]
    tail_tokens = [int(lookup[int(index_map[idx]) if index_map is not None else int(idx)]) for _, idx in back]
    slots = _encode_front(front, index_map, lookup, sae, mode, window, latent_offset)
    slots.extend([token] for token in tail_tokens)
    return [(slots[position], slots[position + 1][0] if len(slots[position + 1]) == 1 else tail_tokens[0])
            for position in range(len(slots) - 1)] if False else _steps(slots, tail_tokens, len(front))


def _steps(slots, tail_tokens, front_len):
    # target of each step is the FIRST token of the next slot; only tail slots
    # (1-hot) can be targets, and every slot after the front is a tail slot.
    steps = []
    num_front = len(slots) - len(tail_tokens)
    for position in range(len(slots) - 1):
        next_slot_tail_index = position + 1 - num_front
        if next_slot_tail_index < 0:
            continue  # next slot is still a compressed front slot: no token target
        steps.append((slots[position], tail_tokens[next_slot_tail_index]))
    return steps


def build(records, out_dir, sae, lookup, index_map, mode, window, sequence_len, examples_per_shard, split_seed, latent_offset, max_examples=None, progress_every=10000, validation_records=None):
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    shards = []
    shard_counts = {}
    pending = {}
    examples_seen = 0

    def flush(split, force=False):
        rows = pending.get(split, [])
        while rows and (len(rows) >= examples_per_shard or (force and rows)):
            batch, rows = rows[:examples_per_shard], rows[examples_per_shard:]
            index = shard_counts.get(split, 0)
            shards.append(_write_shard(out_dir, index, batch, sequence_len, split=split))
            shard_counts[split] = index + 1
        pending[split] = rows

    def consume(stream_iter, limit):
        nonlocal examples_seen
        stories = 0
        for split, story_id, stream in stream_iter:
            stories += 1
            steps = sae_steps_for_story(stream, index_map, lookup, sae, mode, window, split_seed, story_id, latent_offset)
            for start in range(0, len(steps), sequence_len):
                chunk = steps[start:start + sequence_len]
                if not chunk:
                    continue
                pending.setdefault(split, []).append(PhraseSequenceExample(
                    input_indices=[ids for ids, _ in chunk],
                    targets=[target for _, target in chunk],
                ))
                examples_seen += 1
                if limit is not None and examples_seen >= limit:
                    flush(split)
                    return
            flush(split)
            if progress_every and stories % progress_every == 0:
                print(f"sae-sharded {stories} stories; examples={examples_seen}", flush=True)

    consume(story_index_streams(records), max_examples)
    if validation_records is not None:
        consume(story_index_streams(validation_records), None)
    for split in list(pending):
        flush(split, force=True)
    manifest = {
        "format": "phrase_gpt_tensor_shard_manifest_v1",
        "chain_mode": f"sae-{mode}",
        "sequence_len": sequence_len,
        "examples_per_shard": examples_per_shard,
        "split_seed": split_seed,
        "latent_offset": latent_offset,
        "sae_latent_dim": sae.latent_dim,
        "sae_k": sae.k,
        "num_shards": len(shards),
        "num_examples": sum(s["num_examples"] for s in shards),
        "splits": {
            split: {
                "num_shards": sum(1 for s in shards if s["split"] == split),
                "num_examples": sum(s["num_examples"] for s in shards if s["split"] == split),
            }
            for split in sorted({s["split"] for s in shards})
        },
        "num_tokens": sum(s["num_tokens"] for s in shards),
        "elapsed_seconds": time.time() - started,
        "shards": shards,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Build SAE-coded context shards.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--validation-records", default=None)
    parser.add_argument("--index-map", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--vocab-top-k", type=int, default=8191)
    parser.add_argument("--sae", required=True)
    parser.add_argument("--mode", choices=["chain", "window"], required=True)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sequence-len", type=int, default=128)
    parser.add_argument("--examples-per-shard", type=int, default=30000)
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.index_map, "r", encoding="utf-8") as file:
        index_map = json.load(file)
    lookup, tokens = load_vocab_top_k_remap(args.vocab, args.vocab_top_k)
    sae = load_sae(args.sae)
    if sae.input_dim != len(tokens):
        raise SystemExit(f"SAE input_dim {sae.input_dim} != top-k vocab size {len(tokens)}")
    manifest = build(
        records=iter_records(args.records),
        out_dir=args.out_dir,
        sae=sae, lookup=lookup, index_map=index_map,
        mode=args.mode, window=args.window,
        sequence_len=args.sequence_len, examples_per_shard=args.examples_per_shard,
        split_seed=args.split_seed, latent_offset=len(tokens),
        max_examples=args.limit_examples, progress_every=args.progress_every,
        validation_records=iter_records(args.validation_records) if args.validation_records else None,
    )
    print(json.dumps({k: manifest[k] for k in manifest if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
```

Remove the dead `if False` branch in `sae_steps_for_story` before committing —
the function body should be:

```python
    slots = _encode_front(front, index_map, lookup, sae, mode, window, latent_offset)
    slots.extend([token] for token in tail_tokens)
    return _steps(slots, tail_tokens, len(front))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_build_sae_context_shards.py -q`
Expected: 3 passed

- [ ] **Step 5: Run full suite and commit**

Run: `uv run --group dev python -m pytest tests -q -m "not slow"`
Expected: all pass

```bash
git add scripts/build_sae_context_shards.py tests/test_build_sae_context_shards.py
git commit -m "Add SAE-coded context shard builder (chain and window arms)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: SAE front-encoder in sweep eval

**Files:**
- Modify: `scripts/hybrid_sweep.py` (`_probe_contexts`, `run_sweep`), `scripts/eval_hybrid_context_sweep.py` (CLI)
- Test: `tests/test_hybrid_cross.py` (append)

**Interfaces:**
- Consumes: `SweepProbe`, `context_steps_for_probe`, Task 6's `_encode_front` semantics.
- Produces: `run_sweep(..., front_encoder=None)` — when set, a callable `(front_token_indices: list[int], front_clause_ids: list[int]) -> list[list[int]]` replacing chain-front construction; `sae_front_encoder(sae, mode, window, latent_offset)->callable` in `scripts/build_sae_context_shards.py`; eval CLI flags `--sae`, `--sae-mode`, `--sae-window`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hybrid_cross.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_hybrid_cross.py -q`
Expected: FAIL / TypeError (`front_encoder` unknown; `sae_front_encoder` missing)

- [ ] **Step 3: Write minimal implementation**

In `scripts/build_sae_context_shards.py` add:

```python
def sae_front_encoder(sae, mode, window, latent_offset, lookup, index_map):
    """Returns callable(front_tokens, front_clauses) -> list of latent-id slots,
    for use as hybrid_sweep run_sweep(front_encoder=...)."""
    def encode(front_tokens, front_clauses):
        stream = list(zip(front_clauses, front_tokens))
        return _encode_front(stream, index_map, lookup, sae, mode, window, latent_offset)
    return encode
```

NOTE: the sweep probes are built WITH `index_map` already applied
(`build_sweep_probes(..., index_map=...)`), so when wiring the CLI pass
`index_map=None` to `sae_front_encoder` and let `lookup` do only the top-8k
remap. The unit test above passes `index_map=None` for the same reason.

In `scripts/hybrid_sweep.py`, change `_probe_contexts` and `run_sweep`:

```python
def _probe_contexts(probes, x, depth, remap, reset_on_clause=True, front_encoder=None):
    contexts = []
    for p in probes:
        if front_encoder is None:
            steps = context_steps_for_probe(p, x=x, depth=depth, reset_on_clause=reset_on_clause)
        else:
            tail_start = max(0, p.target_pos - x)
            front = front_encoder(p.token_indices[:tail_start], p.clause_ids[:tail_start])
            if depth is not None:
                front = front[-depth:]
            tail = [[p.token_indices[i]] for i in range(tail_start, p.target_pos)]
            steps = front + tail
        contexts.append(_remap_steps(steps, remap))
    return contexts
```

and thread `front_encoder=None` through `run_sweep`'s signature and both
`_probe_contexts` calls (x_sweep and d_sweep); `classic_1hot` NEVER uses the
front encoder (it is the uncompressed baseline by definition).

CAUTION on double-remap: for SAE arms the CLI must NOT let `_remap_steps`
re-remap latent ids. The sweep's `remap` lookup is sized 46327 for original ids;
latent ids (>= 8192) would index out of bounds. For SAE arms, probes are built
with `index_map` applied and the front encoder's `lookup` does top-8k, so pass
`remap=None` to `run_sweep` and instead apply the top-8k lookup to the TAIL
inside the encoder path: extend `_probe_contexts`'s front_encoder branch:

```python
            tail = [[int(front_encoder.tail_lookup[p.token_indices[i]])] for i in range(tail_start, p.target_pos)]
```

with `sae_front_encoder` attaching `encode.tail_lookup = lookup` before
returning (`encode` is a function object; attribute assignment is fine). Update
the unit test expectation accordingly: with identity lookup, `context[1] == [2]`
still holds.

In `scripts/eval_hybrid_context_sweep.py` add flags and wiring:

```python
    parser.add_argument("--sae", default=None, help="sae_best.pt; enables SAE front encoding.")
    parser.add_argument("--sae-mode", choices=["chain", "window"], default="chain")
    parser.add_argument("--sae-window", type=int, default=4)
```

```python
    front_encoder = None
    if args.sae:
        from scripts.build_sae_context_shards import sae_front_encoder
        from scripts.sae import load_sae
        from scripts.train_phrase_gpt import load_vocab_top_k_remap
        sae = load_sae(args.sae)
        lookup, tokens = load_vocab_top_k_remap(args.vocab, 8191)
        front_encoder = sae_front_encoder(sae, mode=args.sae_mode, window=args.sae_window, latent_offset=len(tokens), lookup=lookup, index_map=None)
        remap = None  # probes already index-mapped; lookup handles top-8k; latent ids must not be re-remapped
    result = run_sweep(..., front_encoder=front_encoder)   # add the kwarg to the existing call
```

- [ ] **Step 4: Run tests**

Run: `uv run --group dev python -m pytest tests/test_hybrid_cross.py tests/test_eval_hybrid_context_sweep.py -q`
Expected: all pass (existing sweeps unaffected: `front_encoder=None` default)

- [ ] **Step 5: Run full suite and commit**

Run: `uv run --group dev python -m pytest tests -q -m "not slow"`
Expected: all pass

```bash
git add scripts/hybrid_sweep.py scripts/eval_hybrid_context_sweep.py scripts/build_sae_context_shards.py tests/test_hybrid_cross.py
git commit -m "Sweep eval: pluggable SAE front encoder for latent-id contexts

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Runners + Stage 1 execution

**Files:**
- Create: `scripts/run_sae_stage1.sh`
- Create: `scripts/run_sae_posttrain.sh`

**Interfaces:**
- Consumes: all previous tasks' CLIs.
- Produces: Stage-1 artifacts `sae_bags_out/`, `sae_bags_val_out/`, `sae_chain_out/grid_report.json` (+gate verdict), `sae_window_out/grid_report.json`; runner for Stage-2 arms.

- [ ] **Step 1: Write `scripts/run_sae_stage1.sh`**

```bash
#!/bin/bash
# Stage 1: extract bags, train SAE grids (chain + window), record gate verdicts.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== extract train bags (60k stories ~ 2M+ bags) ==="
uv run python -m scripts.extract_sae_bags \
    --records phrase_quote_split_out/phrase_index.jsonl.gz \
    --index-map phrase_quote_split_ils_out/old_to_new.json \
    --vocab phrase_quote_split_ils_out/vocab.json --vocab-top-k 8191 \
    --out-dir sae_bags_out --max-stories 60000

echo "=== extract validation bags ==="
uv run python -m scripts.extract_sae_bags \
    --records phrase_quote_split_out/phrase_index_validation.jsonl.gz \
    --index-map phrase_quote_split_ils_out/old_to_new.json \
    --vocab phrase_quote_split_ils_out/vocab.json --vocab-top-k 8191 \
    --out-dir sae_bags_val_out --max-stories 5000

echo "=== chain-arm grid ==="
uv run python -m scripts.train_sae_grid \
    --train-bags sae_bags_out/chain_bags.npz --val-bags sae_bags_val_out/chain_bags.npz \
    --out-dir sae_chain_out --epochs 3 --device mps

echo "=== window-arm grid ==="
uv run python -m scripts.train_sae_grid \
    --train-bags sae_bags_out/window_bags.npz --val-bags sae_bags_val_out/window_bags.npz \
    --out-dir sae_window_out --epochs 3 --device mps

echo "=== gate verdicts ==="
python3 -c "import json; [print(d, json.load(open(f'{d}/grid_report.json'))['gate_passed'], json.load(open(f'{d}/grid_report.json'))['best']) for d in ('sae_chain_out','sae_window_out')]"
```

- [ ] **Step 2: Write `scripts/run_sae_posttrain.sh`** (guarded; one arm per invocation)

```bash
#!/bin/bash
# Stage 2 arm post-train: ./scripts/run_sae_posttrain.sh <chain|window>
set -euo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?usage: run_sae_posttrain.sh <chain|window>}"
SHARDS="sae_${ARM}_shards/manifest.json"
OUT="phrase_gpt_sae_${ARM}_post_out"
PIDFILE="phrase_gpt_sae_${ARM}.pid"
LOG="sae_${ARM}_post_train.log"
WATCHDOG_LOG="sae_${ARM}_post_watchdog.log"
MAX_RSS_GB="${MAX_RSS_GB:-15}"
LATENT_DIM=$(python3 -c "import json; print(json.load(open('${SHARDS}'))['sae_latent_dim'])")

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Trainer already running; refusing duplicate." >&2; exit 1
fi
if pgrep -f "scripts.train_phrase_gpt" >/dev/null 2>&1; then
    echo "A trainer is already running; wait or kill it first." >&2; exit 1
fi

RESUME_FROM=phrase_gpt_base_v2_out/phrase_gpt.pt
if [ -f "$OUT/phrase_gpt.pt" ]; then RESUME_FROM="$OUT/phrase_gpt.pt"; fi

caffeinate -i uv run python -m scripts.train_phrase_gpt \
    --vocab phrase_gpt_base_v2_out/vocab_top_k.json \
    --shards "$SHARDS" \
    --out-dir "$OUT" \
    --sequence-len 128 --depth 2 --n-embd 128 --n-head 4 \
    --batch-size 16 --lr 1e-3 --epochs 4 \
    --extend-phrase-vocab "$LATENT_DIM" \
    --save-best --checkpoint-every-shards 1 \
    --resume "$RESUME_FROM" \
    2>&1 | tee -a "$LOG" &
TEE_PID=$!
echo "$TEE_PID" > "$PIDFILE"
(
    while kill -0 "$TEE_PID" 2>/dev/null; do
        TRAIN_PID=$(pgrep -f "scripts.train_phrase_gpt" | head -1 || true)
        if [ -n "$TRAIN_PID" ]; then
            RSS_KB=$(ps -o rss= -p "$TRAIN_PID" | tr -d ' ' || echo 0)
            echo "$(date '+%Y-%m-%d %H:%M:%S') pid=$TRAIN_PID rss_mb=$((${RSS_KB:-0} / 1024))" >> "$WATCHDOG_LOG"
            if [ "${RSS_KB:-0}" -gt $((MAX_RSS_GB * 1048576)) ]; then
                echo "WATCHDOG kill" | tee -a "$WATCHDOG_LOG" >&2
                pkill -f "scripts.train_phrase_gpt" || true
            fi
        fi
        sleep 60
    done
    rm -f "$PIDFILE"
) &
echo "SAE ${ARM} post-train started (pid $TEE_PID). No --vocab-top-k: shards are pre-remapped."
wait "$TEE_PID"
```

NOTE: no `--vocab-top-k` (SAE shards already live in top-8k+latent space) and
`--vocab` is the 8,192-row `vocab_top_k.json`, so the trainer's `vocab.size`
(and output head) stays 8,192 while `--extend-phrase-vocab` grows the input
table. No in-training sweep (the standalone sweep with `--sae` covers it).

- [ ] **Step 3: chmod, syntax-check, commit**

```bash
chmod +x scripts/run_sae_stage1.sh scripts/run_sae_posttrain.sh
bash -n scripts/run_sae_stage1.sh && bash -n scripts/run_sae_posttrain.sh
git add scripts/run_sae_stage1.sh scripts/run_sae_posttrain.sh
git commit -m "Add SAE Stage-1 and Stage-2 runner scripts

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Execute Stage 1** (in a visible Terminal, ~half a day)

Run: `osascript -e 'tell application "Terminal" to do script "cd \"/Users/jgrimm/dev/Local AI/nanochat\" && ./scripts/run_sae_stage1.sh 2>&1 | tee sae_stage1.log"'`

- [ ] **Step 5: Record the gate decision**

Read `sae_chain_out/grid_report.json`. If `gate_passed` is false, Stage 2 runs
only the window arm; record the verdict + per-length F1 table in the workbook
("SAE Comparison" sheet) either way, with the reproduction commands.

---

### Task 9: Stage 2 execution + reporting

**Files:**
- Uses Tasks 6-8 CLIs; workbook `outputs/nanochat-phrase-results/nanochat_phrase_results.xlsx`.

- [ ] **Step 1: Build arm shards** (one command per surviving arm)

```bash
uv run python -m scripts.build_sae_context_shards \
    --records phrase_quote_split_out/phrase_index.jsonl.gz \
    --validation-records phrase_quote_split_out/phrase_index_validation.jsonl.gz \
    --index-map phrase_quote_split_ils_out/old_to_new.json \
    --vocab phrase_quote_split_ils_out/vocab.json --vocab-top-k 8191 \
    --sae sae_chain_out/sae_best.pt --mode chain \
    --out-dir sae_chain_shards --limit-examples 30000 --split-seed 42
# window arm: swap --sae sae_window_out/sae_best.pt --mode window --out-dir sae_window_shards
```

- [ ] **Step 2: Post-train each arm** (sequential, guarded)

Run: `./scripts/run_sae_posttrain.sh chain` then, after completion, `./scripts/run_sae_posttrain.sh window`

- [ ] **Step 3: Sweep each arm's checkpoint**

```bash
uv run python -m scripts.eval_hybrid_context_sweep \
    --checkpoint phrase_gpt_sae_chain_post_out/phrase_gpt.pt \
    --records phrase_quote_split_out/phrase_index_validation.jsonl.gz \
    --split validation --max-probes 2000 \
    --index-map phrase_quote_split_ils_out/old_to_new.json \
    --vocab phrase_quote_split_ils_out/vocab.json \
    --sae sae_chain_out/sae_best.pt --sae-mode chain \
    --x-values 0,1,2,4,8,16 --d-values 1,2,4,8,16,32 > sae_chain_sweep.log 2>&1
# window arm: swap checkpoint/sae/mode/log accordingly (--sae-window 4)
```

- [ ] **Step 4: Report**

Add "SAE Comparison" sheet to the workbook (match Carlito styling and the Vocab
Reorder sheet's structure): Stage-1 grid + gate table; Stage-2 table with rows
X=0/1/2/4/8/16 and columns per arm (multihot 27.9% X=0 row from the existing v2
block; SAE-chain; SAE-window) plus each arm's `classic_1hot` regression check;
provenance rows (commit ids, all commands above verbatim). Update memory file
`vocab-reorder-results.md` (or a new `sae-comparison-results` memory) with the
headline numbers and artifact locations.

---

## Self-Review Notes

- Spec coverage: Stage-1 SAE+grid+gate (Tasks 1-4, 8); latent-ids-as-vocab and
  surgery (Tasks 5-6); three-arm protocol + sweep (Tasks 6-9); workbook +
  reproduction commands (Task 9); rate-distortion sweep explicitly deferred.
- Types: `sae_front_encoder(sae, mode, window, latent_offset, lookup, index_map)`
  is consistent between Tasks 6 and 7; `run_grid` grid entries are `(d, k)`
  tuples; `load_bags` returns `(bags, input_dim)` everywhere.
- Known risk flagged inline: optimizer-state shape collision (Task 5 caution)
  and sweep double-remap (Task 7 caution) each carry an explicit fallback.
