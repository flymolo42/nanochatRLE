# Hybrid Context + Sweep Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `hybrid` chain shard mode (randomized compression split: compressed phrase history + recent 1-hot token tail) and a sweep eval that reports top-1/5/10 accuracy and perplexity across recent-tail length X and compressed depth D on one shared probe set.

**Architecture:** All new code is at the data + eval layer. `hybrid` reuses the existing chain builder and target rule; a recent 1-hot token is a length-1 chain, so no model or training-loop change. The eval reuses `_chains_from_token_records`, `collate_phrase_sequences`, and the vocab-top-k remap.

**Tech Stack:** Python 3.10, PyTorch, `unittest`. Run tests with `.venv/bin/python -m unittest`.

## Global Constraints

- Chains are built ONLY from the canonical punctuation token stream (`_canonical_token_stream`).
- `hybrid`: split index `s` chosen uniformly over the story's phrase-boundary token indices, INCLUDING `0` (fully 1-hot) and `N` (fully compressed). Split is seeded reproducibly from `run_seed * 1_000_003 + story_id`.
- Tokens before `s` → phrase chains (`reset_on_clause=True`); tokens from `s` → length-1 chains. Target rule unchanged: `target[t] = first token of the next chain`.
- Eval comparability (hard requirement): every X and D config is scored on the SAME probe set; only the context representation varies. Opener vs interior are separate breakdowns. Vocab (top-8k) held constant.
- Eval metrics per (config, target-type) bucket: top-1, top-5, top-10 accuracy; mean cross-entropy; perplexity `exp(mean CE)`; probe count.
- Eval must be memory-safe: stream predictions, small batches, sort by length, free device cache per batch, cap probes with a flag. (The prior run tied up 40GB; do not regress.)
- No changes to `nanochat/gpt.py`, the loss, or the training loop.
- Work on branch `hybrid-context-sweep`.

---

### Task 1: `hybrid` chain builder + seed threading

**Files:**
- Modify: `scripts/train_phrase_gpt.py` (refactor `_chain_steps`; add hybrid helpers; extend `CHAIN_MODE_BUILDERS` and `examples_from_story_records`)
- Test: `tests/test_phrase_gpt.py`

**Interfaces:**
- Produces:
  - `_chains_from_token_records(token_records, reset_on_clause) -> list[list[int]]`
  - `_steps_from_chains(chains) -> list[tuple[list[int], int]]`
  - `_phrase_boundary_positions(token_records) -> list[int]`
  - `_hybrid_steps_at_split(token_records, split) -> list[tuple[list[int], int]]`
  - `_choose_split(boundaries, seed, story_id) -> int`
  - `_hybrid_steps(records, seed) -> list[tuple[list[int], int]]`
  - `CHAIN_MODE_BUILDERS` gains `"hybrid"`; all builder lambdas take `(records, seed)`.
  - `examples_from_story_records(records, sequence_len, chain_mode="token", seed=0)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_phrase_gpt.py` inside `PhraseGPTTests` (the `_chain_story_records` helper already exists from earlier work: clause 0 = indices [1,3,2], clause 1 = [4,5]):

```python
    def test_phrase_boundary_positions_marks_clause_starts_and_ends(self):
        from scripts.train_phrase_gpt import _canonical_token_stream, _phrase_boundary_positions
        stream = _canonical_token_stream(self._chain_story_records())
        self.assertEqual(_phrase_boundary_positions(stream), [0, 3, 5])

    def test_hybrid_steps_at_split_endpoints_and_middle(self):
        from scripts.train_phrase_gpt import _canonical_token_stream, _hybrid_steps_at_split
        stream = _canonical_token_stream(self._chain_story_records())  # indices 1,3,2,4,5; clauses 0,0,0,1,1

        # split=N (5): fully compressed == phrase mode
        self.assertEqual(_hybrid_steps_at_split(stream, 5), [([1, 3], 2), ([2], 4)])
        # split=0: fully 1-hot, every token predicts the next
        self.assertEqual(_hybrid_steps_at_split(stream, 0), [([1], 3), ([3], 2), ([2], 4), ([4], 5)])
        # split=3: front [1,3,2] compressed -> [[1,3],[2]]; tail [4],[5] 1-hot
        self.assertEqual(_hybrid_steps_at_split(stream, 3), [([1, 3], 2), ([2], 4), ([4], 5)])

    def test_choose_split_is_reproducible_and_in_bounds(self):
        from scripts.train_phrase_gpt import _choose_split
        boundaries = [0, 3, 5]
        a = _choose_split(boundaries, seed=42, story_id=7)
        b = _choose_split(boundaries, seed=42, story_id=7)
        self.assertEqual(a, b)
        self.assertIn(a, boundaries)

    def test_hybrid_mode_dispatch_matches_endpoint_modes(self):
        from scripts.train_phrase_gpt import examples_from_story_records, _canonical_token_stream, _hybrid_steps_at_split, _chunk_steps_into_examples
        records = self._chain_story_records()
        # whatever split the seed picks, the hybrid output must equal _hybrid_steps_at_split at that split
        stream = _canonical_token_stream(records)
        from scripts.train_phrase_gpt import _phrase_boundary_positions, _choose_split
        split = _choose_split(_phrase_boundary_positions(stream), seed=123, story_id=int(stream[0].get("story_id", 0)))
        expected = _chunk_steps_into_examples(_hybrid_steps_at_split(stream, split), sequence_len=10)
        got = examples_from_story_records(records, sequence_len=10, chain_mode="hybrid", seed=123)
        self.assertEqual(
            [(e.input_indices, e.targets) for e in got],
            [(e.input_indices, e.targets) for e in expected],
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_phrase_gpt.PhraseGPTTests.test_hybrid_steps_at_split_endpoints_and_middle -v`
Expected: FAIL with `ImportError: cannot import name '_hybrid_steps_at_split'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/train_phrase_gpt.py`, replace the current `_chain_steps` (lines ~135-152) with the refactor plus the new helpers:

```python
def _chains_from_token_records(token_records, reset_on_clause):
    chains = []
    current = []
    current_clause = None
    for record in token_records:
        if not record.get("indices"):
            continue
        index = int(record["indices"][0])
        clause = int(record.get("phrase_id", 0))
        if current and (index <= current[-1] or (reset_on_clause and clause != current_clause)):
            chains.append(current)
            current = []
        current.append(index)
        current_clause = clause
    if current:
        chains.append(current)
    return chains


def _steps_from_chains(chains):
    return [(chains[position], chains[position + 1][0]) for position in range(len(chains) - 1)]


def _chain_steps(records, reset_on_clause):
    token_records = _canonical_token_stream(records)
    return _steps_from_chains(_chains_from_token_records(token_records, reset_on_clause))


def _phrase_boundary_positions(token_records):
    boundaries = [0]
    previous_clause = None
    for position, record in enumerate(token_records):
        clause = int(record.get("phrase_id", 0))
        if previous_clause is not None and clause != previous_clause:
            boundaries.append(position)
        previous_clause = clause
    boundaries.append(len(token_records))
    return sorted(set(boundaries))


def _hybrid_steps_at_split(token_records, split):
    front = token_records[:split]
    back = token_records[split:]
    chains = _chains_from_token_records(front, reset_on_clause=True)
    # each back token becomes its own length-1 chain (1-hot); extend with one-element lists
    chains.extend([int(record["indices"][0])] for record in back if record.get("indices"))
    return _steps_from_chains(chains)


def _choose_split(boundaries, seed, story_id):
    return random.Random(seed * 1_000_003 + int(story_id)).choice(boundaries)


def _hybrid_steps(records, seed):
    token_records = _canonical_token_stream(records)
    if not token_records:
        return []
    story_id = int(token_records[0].get("story_id", 0))
    boundaries = _phrase_boundary_positions(token_records)
    split = _choose_split(boundaries, seed, story_id)
    return _hybrid_steps_at_split(token_records, split)
```

Then replace `CHAIN_MODE_BUILDERS` and `examples_from_story_records` (lines ~167-177):

```python
CHAIN_MODE_BUILDERS = {
    "token": lambda records, seed: _token_steps(records),
    "phrase": lambda records, seed: _chain_steps(records, reset_on_clause=True),
    "cross-phrase": lambda records, seed: _chain_steps(records, reset_on_clause=False),
    "hybrid": lambda records, seed: _hybrid_steps(records, seed),
}


def examples_from_story_records(records, sequence_len, chain_mode="token", seed=0):
    if chain_mode not in CHAIN_MODE_BUILDERS:
        raise ValueError(f"Unknown chain_mode {chain_mode!r}; expected one of {sorted(CHAIN_MODE_BUILDERS)}")
    return _chunk_steps_into_examples(CHAIN_MODE_BUILDERS[chain_mode](records, seed), sequence_len)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_phrase_gpt -v`
Expected: PASS, including the existing `token`/`phrase`/`cross-phrase` tests (the `_chain_steps` refactor must keep them green).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_phrase_gpt.py tests/test_phrase_gpt.py
git commit -m "Add hybrid chain mode (compressed history + 1-hot recent tail)"
```

---

### Task 2: `--chain-mode hybrid` + `--split-seed` in the shard builder

**Files:**
- Modify: `scripts/build_phrase_gpt_shards.py`
- Test: `tests/test_build_phrase_gpt_shards.py`

**Interfaces:**
- Consumes: `examples_from_story_records(..., chain_mode, seed)` (Task 1).
- Produces: `build_shards_from_records(..., chain_mode="token", split_seed=0)`; manifest keys `chain_mode`, `split_seed`; CLI `--chain-mode hybrid`, `--split-seed`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_build_phrase_gpt_shards.py` inside `BuildPhraseGPTShardsTests`:

```python
    def test_build_shards_hybrid_mode_records_mode_and_seed(self):
        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [1], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [3], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [2], "token_pos": 2},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [4], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [5], "token_pos": 1},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_shards_from_records(
                records=iter(records), out_dir=tmpdir, sequence_len=10,
                examples_per_shard=50, chain_mode="hybrid", split_seed=123,
            )
            saved = json.loads((Path(tmpdir) / "manifest.json").read_text(encoding="utf-8"))
            shard = torch.load(Path(tmpdir) / saved["shards"][0]["file"], map_location="cpu", weights_only=False)

        self.assertEqual(saved["chain_mode"], "hybrid")
        self.assertEqual(saved["split_seed"], 123)
        # deterministic given the seed: matches examples_from_story_records at seed=123
        from scripts.train_phrase_gpt import examples_from_story_records, tensor_shard_to_examples
        expected = examples_from_story_records(records, sequence_len=10, chain_mode="hybrid", seed=123)
        got = tensor_shard_to_examples(shard)
        self.assertEqual(
            [(e.input_indices, e.targets) for e in got],
            [(e.input_indices, e.targets) for e in expected],
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_build_phrase_gpt_shards.BuildPhraseGPTShardsTests.test_build_shards_hybrid_mode_records_mode_and_seed -v`
Expected: FAIL with `TypeError: build_shards_from_records() got an unexpected keyword argument 'split_seed'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/build_phrase_gpt_shards.py`:

Change `_flush_story` to pass a seed:

```python
def _flush_story(current_rows, sequence_len, chain_mode, split_seed):
    if not current_rows:
        return []
    return examples_from_story_records(current_rows, sequence_len=sequence_len, chain_mode=chain_mode, seed=split_seed)
```

Add `split_seed=0` to `build_shards_from_records`'s signature (next to `chain_mode="token"`), pass `split_seed=split_seed` at BOTH `_flush_story` call sites, and add to the manifest dict next to `"chain_mode"`:

```python
        "chain_mode": chain_mode,
        "split_seed": split_seed,
```

Add the CLI arg in `parse_args` (extend the existing `--chain-mode` choices with `hybrid`, add `--split-seed`):

```python
    parser.add_argument("--chain-mode", choices=["token", "phrase", "cross-phrase", "hybrid"], default="token", help="Per-timestep input construction. hybrid = compressed phrase history + a recent 1-hot token tail, split at a random phrase boundary per story.")
    parser.add_argument("--split-seed", type=int, default=0, help="Seed for the hybrid random split point (per-story split = split_seed*1000003 + story_id). Ignored for non-hybrid modes.")
```

Pass it in `main`:

```python
        chain_mode=args.chain_mode,
        split_seed=args.split_seed,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_build_phrase_gpt_shards tests.test_phrase_gpt -v`
Expected: PASS (including the pre-existing default-`token` build test).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_phrase_gpt_shards.py tests/test_build_phrase_gpt_shards.py
git commit -m "Add --chain-mode hybrid + --split-seed to shard builder"
```

---

### Task 3: Sweep eval — probes, context construction, metrics (pure)

**Files:**
- Create: `scripts/eval_hybrid_context_sweep.py`
- Test: `tests/test_eval_hybrid_context_sweep.py`

**Interfaces:**
- Consumes: `_canonical_token_stream`, `_chains_from_token_records` (Task 1) from `scripts.train_phrase_gpt`; `normalize_phrase_records` from `scripts.train_phrase_vectors`.
- Produces:
  - `@dataclass SweepProbe(token_indices, clause_ids, target_pos, is_opener)`
  - `build_sweep_probes(records, min_history=1, max_probes=None) -> list[SweepProbe]`
  - `context_steps_for_probe(probe, x, depth) -> list[list[int]]`
  - `topk_and_ce(logits_row, target, ks=(1,5,10)) -> (dict[int,int], float)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_hybrid_context_sweep.py`:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_eval_hybrid_context_sweep -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.eval_hybrid_context_sweep'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/eval_hybrid_context_sweep.py`:

```python
"""
Sweep eval for hybrid multihot context: next-token top-k accuracy + perplexity as a
function of recent 1-hot tail length X and compressed depth D, on one shared probe set.
"""

import argparse
import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from scripts.train_phrase_gpt import _canonical_token_stream, _chains_from_token_records


@dataclass(frozen=True)
class SweepProbe:
    token_indices: list
    clause_ids: list
    target_pos: int
    is_opener: bool


def _stories(records):
    from scripts.train_phrase_vectors import normalize_phrase_records
    grouped = {}
    for record in normalize_phrase_records(records):
        key = (record["split"], int(record["story_id"]))
        grouped.setdefault(key, []).append(record)
    return grouped


def build_sweep_probes(records, min_history=1, max_probes=None):
    probes = []
    for _key, rows in sorted(_stories(records).items()):
        stream = _canonical_token_stream(rows)
        indices = [int(r["indices"][0]) for r in stream]
        clauses = [int(r.get("phrase_id", 0)) for r in stream]
        for pos in range(len(indices)):
            if pos < min_history:
                continue
            is_opener = pos == 0 or clauses[pos] != clauses[pos - 1]
            probes.append(SweepProbe(indices, clauses, pos, is_opener))
            if max_probes is not None and len(probes) >= max_probes:
                return probes
    return probes


def context_steps_for_probe(probe, x, depth):
    p = probe.target_pos
    tail_start = max(0, p - x)
    front_records = [
        {"indices": [probe.token_indices[i]], "phrase_id": probe.clause_ids[i]}
        for i in range(tail_start)
    ]
    front_chains = _chains_from_token_records(front_records, reset_on_clause=True)
    if depth is not None:
        front_chains = front_chains[-depth:]
    tail = [[probe.token_indices[i]] for i in range(tail_start, p)]
    return front_chains + tail


def topk_and_ce(logits_row, target, ks=(1, 5, 10)):
    top = logits_row.topk(min(max(ks), logits_row.numel())).indices.tolist()
    hits = {k: int(target in top[:k]) for k in ks}
    ce = F.cross_entropy(logits_row.unsqueeze(0), torch.tensor([int(target)], device=logits_row.device)).item()
    return hits, ce
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_eval_hybrid_context_sweep -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_hybrid_context_sweep.py tests/test_eval_hybrid_context_sweep.py
git commit -m "Add hybrid sweep eval: probes, context construction, metrics"
```

---

### Task 4: Sweep eval — model runner, aggregation, CLI

**Files:**
- Modify: `scripts/eval_hybrid_context_sweep.py`
- Test: `tests/test_eval_hybrid_context_sweep.py`

**Interfaces:**
- Consumes: Task 3 functions; `PhraseSequenceExample`, `collate_phrase_sequences`, `choose_device`, `load_vocab_top_k_remap` from `scripts.train_phrase_gpt`; `load_model_from_checkpoint`, `resolve_vocab_remap` from `scripts.eval_phrase_gpt_packed_vs_single`.
- Produces:
  - `predict_probe_logits(model, contexts, batch_size, device) -> torch.Tensor` (one last-position logit row per context, in input order)
  - `run_sweep(model, probes, x_values, d_values, fixed_x_for_depth, remap, batch_size, device) -> dict`
  - `main()`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval_hybrid_context_sweep.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_eval_hybrid_context_sweep.HybridSweepPureTests.test_run_sweep_scores_same_probes_across_configs -v`
Expected: FAIL with `ImportError: cannot import name 'run_sweep'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/eval_hybrid_context_sweep.py`:

```python
from scripts.train_phrase_gpt import PhraseSequenceExample, collate_phrase_sequences, choose_device
from scripts.train_phrase_vectors import iter_records
from scripts.eval_phrase_gpt_packed_vs_single import load_model_from_checkpoint, resolve_vocab_remap


def _remap_steps(steps, remap):
    if remap is None:
        return steps
    return [[int(remap[i]) for i in step] for step in steps]


def predict_probe_logits(model, contexts, batch_size, device):
    """contexts: list[list[list[int]]] (per probe: list of chains). Returns a tensor
    [len(contexts), vocab] of the last-position logits, in input order. Memory-safe:
    sort by length, small batches, free cache per batch."""
    model.eval()
    if not contexts:
        return torch.empty((0, 0))
    order = sorted(range(len(contexts)), key=lambda i: len(contexts[i]))
    rows = [None] * len(contexts)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            steps = [contexts[i] for i in batch_idx]
            seq_len = max(2, max(len(s) for s in steps))
            batch = collate_phrase_sequences(
                [PhraseSequenceExample(input_indices=s, targets=[-1] * len(s)) for s in steps],
                sequence_len=seq_len, dummy_token_id=0, device=device,
            )
            logits = model(batch.idx, phrase_indices=batch.phrase_indices,
                           phrase_offsets=batch.phrase_offsets, phrase_batch_positions=batch.phrase_batch_positions)
            last = torch.tensor([len(s) - 1 for s in steps], device=logits.device)
            picked = logits[torch.arange(len(steps), device=logits.device), last, :].cpu()
            for slot, row in zip(batch_idx, picked):
                rows[slot] = row
            del logits, batch
            if device == "mps":
                torch.mps.empty_cache()
    return torch.stack(rows)


def _aggregate(probes, logits, remap):
    buckets = {name: {"top1": 0, "top5": 0, "top10": 0, "ce": 0.0, "count": 0}
               for name in ("all", "opener", "interior")}
    for probe, row in zip(probes, logits):
        target = int(remap[probe.token_indices[probe.target_pos]]) if remap is not None else probe.token_indices[probe.target_pos]
        hits, ce = topk_and_ce(row, target, ks=(1, 5, 10))
        for name in ("all", "opener" if probe.is_opener else "interior"):
            b = buckets[name]
            b["top1"] += hits[1]; b["top5"] += hits[5]; b["top10"] += hits[10]
            b["ce"] += ce; b["count"] += 1
    out = {}
    for name, b in buckets.items():
        n = max(b["count"], 1)
        mean_ce = b["ce"] / n
        out[name] = {
            "top1": b["top1"] / n, "top5": b["top5"] / n, "top10": b["top10"] / n,
            "mean_ce": mean_ce, "perplexity": math.exp(mean_ce) if b["count"] else float("nan"),
            "count": b["count"],
        }
    return out


def run_sweep(model, probes, x_values, d_values, fixed_x_for_depth, remap, batch_size, device):
    result = {"x_sweep": {}, "d_sweep": {}, "num_probes": len(probes)}
    for x in x_values:
        contexts = [_remap_steps(context_steps_for_probe(p, x=x, depth=None), remap) for p in probes]
        logits = predict_probe_logits(model, contexts, batch_size, device)
        result["x_sweep"][str(x)] = _aggregate(probes, logits, remap)
    for d in d_values:
        contexts = [_remap_steps(context_steps_for_probe(p, x=fixed_x_for_depth, depth=d), remap) for p in probes]
        logits = predict_probe_logits(model, contexts, batch_size, device)
        result["d_sweep"][str(d)] = _aggregate(probes, logits, remap)
    return result


def _parse_int_list(text):
    return [int(v) for v in text.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid context sweep eval (top-k accuracy + perplexity vs X and D).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--vocab", default=None, help="Original vocab.json, if the checkpoint's recorded path is wrong.")
    parser.add_argument("--x-values", default="0,1,2,4,8,16")
    parser.add_argument("--d-values", default="1,2,4,8,16,32", help="Compressed-depth caps; 'none' appended automatically for unbounded.")
    parser.add_argument("--fixed-x-for-depth", type=int, default=0)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--max-probes", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    records = list(iter_records(args.records))
    probes = build_sweep_probes(records, min_history=args.min_history, max_probes=args.max_probes)
    if not probes:
        raise SystemExit("No probes found.")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, _config = load_model_from_checkpoint(checkpoint, device=device)
    remap = resolve_vocab_remap(checkpoint.get("config", {}), vocab_override=args.vocab)
    if remap is not None:
        remap = remap.to("cpu")
    print(f"probes={len(probes)} device={device} remap={'yes' if remap is not None else 'no'}", flush=True)
    d_values = _parse_int_list(args.d_values) + [None]
    result = run_sweep(model, probes, x_values=_parse_int_list(args.x_values), d_values=d_values,
                       fixed_x_for_depth=args.fixed_x_for_depth, remap=remap, batch_size=args.batch_size, device=device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_eval_hybrid_context_sweep -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_hybrid_context_sweep.py tests/test_eval_hybrid_context_sweep.py
git commit -m "Add hybrid sweep eval runner, aggregation, and CLI"
```

---

### Task 5: Full regression + manual subset build / post-train / sweep (verification)

**Files:** none (verification)

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/python -m unittest discover -s tests 2>&1 | grep -E "^(Ran|OK|FAILED)|^(ERROR|FAIL):"`
Expected: one known-unrelated failure only — `test_attention_fallback` (missing `pytest`).

- [ ] **Step 2: Build a small hybrid shard set (FOREGROUND — background detaches on this machine)**

Run (note: the gz stream is slow ~26k examples/min; keep the limit modest and run in the foreground so it isn't killed detached):

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.build_phrase_gpt_shards --chain-mode hybrid --split-seed 42 \
    --vocab phrase_vectors_full_out/vocab.json \
    --records phrase_vectors_full_out/phrase_index.jsonl.gz \
    --out-dir phrase_hybrid_shards_len128 --sequence-len 128 --limit-examples 200000
```
Expected: `manifest.json` with `"chain_mode": "hybrid"`, `"split_seed": 42`.

- [ ] **Step 3: Post-train the top-8k checkpoint on hybrid shards**

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.train_phrase_gpt \
    --resume phrase_gpt_len128_topk8k_out/best_phrase_gpt.pt \
    --shards phrase_hybrid_shards_len128/manifest.json \
    --vocab phrase_vectors_full_out/vocab.json --vocab-top-k 8191 --lr 3e-4 --epochs 2 --save-best
```
Expected: `resuming from ...`, finite decreasing train loss. Watch RSS (batch 32 keeps peak small).

- [ ] **Step 4: Run the sweep eval and report the curves**

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.eval_hybrid_context_sweep \
    --checkpoint phrase_gpt_len128_topk8k_out/best_phrase_gpt.pt \
    --records <a small held-out .jsonl slice> --max-probes 5000
```
Report `accuracy(X)` and `accuracy(D)` (top-1/5/10 + perplexity), opener vs interior. Do NOT commit checkpoints or shard outputs.

---
