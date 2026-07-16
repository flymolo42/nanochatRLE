# k-way Hub-Token Duplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give high-conflict tokens k>2 copies at data-chosen within-clause positions and measure the ordering-compressibility gain on JS/TS code vs no-dup and k=2.

**Architecture:** A planner ranks tokens by conflict mass, collects each candidate's within-clause position histogram, and picks k (= number of separated position modes, or a fixed CLI value) with per-copy target positions. An apply step remaps each occurrence to the nearest-target copy id in a renumbered vocab; copies are first-class tokens, so the existing reorder + chain-measurement run unchanged. A driver compares no-dup / fixed-k2 / data-kway / fixed-kN arms.

**Tech Stack:** numpy, existing nanochat scripts (reorder_phrase_vocab, plan_token_duplicates, code_stream, code_vocab_experiment, measure_chain_lengths), unittest.

**Spec:** `docs/superpowers/specs/2026-07-15-kway-duplication-design.md`

## Global Constraints

- Run tests with `uv run --group dev python -m pytest tests/<file> -q` from `nanochat/`; unittest-style classes.
- Analysis-only: no training, no shard building. Streams are `[(clause_id, token_id)]` after remap.
- All arms (no-dup / fixed-k2 / data-driven k-way / fixed-kN) go through ONE code path; the k=2 baseline is `--fixed-k 2`, NOT the record-based plan_token_duplicates.
- Copies are appended adjacent to the parent in index order; parent keeps the first copy slot; `vocab_size_new = vocab_size_old + sum(k_i - 1)`.
- Within-clause relative position of an occurrence = `position_in_clause / max(clause_len - 1, 1)`, in [0, 1].
- Occurrence -> copy = nearest `target_position`; ties -> lowest new_index.
- Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Key existing interfaces (read-only):**
- `scripts/plan_token_duplicates.conflict_losses(codes, counts, vocab_size) -> (loss ndarray[V], directed ndarray[V])`
- `scripts/plan_token_duplicates.renumber_array(vocab_size, parents_sorted) -> ndarray` (old->new, one gap per parent)
- `scripts/reorder_phrase_vocab.PairCounter(vocab_size, chunk_size).add_stream([(clause,id),...]); .finalize() -> (codes, counts)`
- `scripts/reorder_phrase_vocab.optimize_order(codes, counts, vocab_size, warm_order, max_passes=, ils_restarts=, ils_generations=, ils_seed=, jobs=) -> (order, stats)`; `stats["ascending_after"]`, `stats["total_pairs"]`; `_inverse_permutation(order) -> positions`
- `scripts/code_stream.file_streams(paths, split_identifiers) -> yields (file_id, [(clause, token), ...])`
- `scripts/code_vocab_experiment.census_pass(paths, split_identifiers, ...) -> dict{"vocab":[{token,index,count,avg_position}], ...}`; `_build_lookup(census_vocab, min_count) -> (lookup{token->id}, kept[list])`; `_stream_ids(stream, lookup, unk_id) -> [(clause, id)]`
- `scripts/measure_chain_lengths.chain_length_histogram(stream, mapping, reset_on_clause, histogram=None, max_chain_len=None)`; `summarize(histogram) -> {chains, avg_chain_len, len1_fraction_of_chains, ...}`

---

### Task 1: Position histograms + k selection (`scripts/plan_kway_duplicates.py`)

**Files:**
- Create: `scripts/plan_kway_duplicates.py`
- Test: `tests/test_plan_kway_duplicates.py`

**Interfaces:**
- Consumes: `conflict_losses` (existing).
- Produces: `collect_position_histograms(id_streams, vocab_size, bins=20) -> ndarray[vocab_size, bins]` where `id_streams` is an iterable of `[(clause, token_id), ...]`; `select_k(hist_row, k_max, min_mass_frac=0.15) -> (k, centroids list[float])` (data-driven modality); `fixed_centroids(k) -> list[float]`; `select_candidates(codes, counts, vocab_size, top_n) -> list[int]` (top conflict-mass ids, descending).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plan_kway_duplicates.py
import numpy as np
import unittest

from scripts.plan_kway_duplicates import (
    collect_position_histograms,
    fixed_centroids,
    select_candidates,
    select_k,
)


def _pair_arrays(vocab_size, pair_counts):
    codes = np.array(sorted(l * vocab_size + r for l, r in pair_counts), dtype=np.int64)
    counts = np.array([pair_counts[(int(c) // vocab_size, int(c) % vocab_size)] for c in codes], dtype=np.int64)
    return codes, counts


class PositionHistogramTests(unittest.TestCase):
    def test_relative_positions_binned(self):
        # one clause of 3 tokens (ids 5, 5, 9): positions 0, .5, 1
        streams = [[(0, 5), (0, 5), (0, 9)]]
        hist = collect_position_histograms(streams, vocab_size=10, bins=4)
        # token 5 seen at rel-pos 0.0 (bin 0) and 0.5 (bin 2)
        self.assertEqual(hist[5].tolist(), [1, 0, 1, 0])
        # token 9 at rel-pos 1.0 -> last bin
        self.assertEqual(hist[9][-1], 1)

    def test_singleton_clause_is_position_zero(self):
        hist = collect_position_histograms([[(0, 3)]], vocab_size=5, bins=4)
        self.assertEqual(hist[3][0], 1)


class SelectKTests(unittest.TestCase):
    def test_uniform_distribution_gives_k_max(self):
        row = np.ones(20, dtype=np.int64) * 5
        k, centroids = select_k(row, k_max=6)
        self.assertEqual(k, 6)
        self.assertEqual(len(centroids), 6)
        self.assertTrue(all(0.0 <= c <= 1.0 for c in centroids))

    def test_bimodal_gives_two(self):
        row = np.zeros(20, dtype=np.int64)
        row[1] = 50   # ~0.05
        row[18] = 50  # ~0.9
        k, centroids = select_k(row, k_max=6)
        self.assertEqual(k, 2)
        self.assertLess(centroids[0], 0.3)
        self.assertGreater(centroids[1], 0.7)

    def test_monomodal_gives_one(self):
        row = np.zeros(20, dtype=np.int64)
        row[9] = 100
        k, centroids = select_k(row, k_max=6)
        self.assertEqual(k, 1)

    def test_empty_row_gives_one(self):
        k, centroids = select_k(np.zeros(20, dtype=np.int64), k_max=6)
        self.assertEqual(k, 1)


class FixedAndCandidateTests(unittest.TestCase):
    def test_fixed_centroids_evenly_spaced(self):
        self.assertEqual(fixed_centroids(2), [0.25, 0.75])
        self.assertEqual(fixed_centroids(4), [0.125, 0.375, 0.625, 0.875])

    def test_select_candidates_by_conflict_mass(self):
        vocab_size = 5
        codes, counts = _pair_arrays(vocab_size, {(0, 1): 10, (1, 0): 8, (2, 3): 6, (3, 2): 6})
        # loss: t0=8,t1=8,t2=6,t3=6 -> top-3 = [0,1,2] (stable)
        self.assertEqual(select_candidates(codes, counts, vocab_size, top_n=3), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_plan_kway_duplicates.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'scripts.plan_kway_duplicates'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/plan_kway_duplicates.py
"""
k-way hub-token duplication planner: rank tokens by conflict mass, collect each
candidate's within-clause position distribution, and choose k copies with target
positions (data-driven modality, or a fixed count).
"""

import numpy as np

from scripts.plan_token_duplicates import conflict_losses


def collect_position_histograms(id_streams, vocab_size, bins=20):
    hist = np.zeros((vocab_size, bins), dtype=np.int64)
    for stream in id_streams:
        clause_tokens = []
        current = None
        for clause, token_id in stream:
            if current is not None and clause != current:
                _bin_clause(hist, clause_tokens, bins)
                clause_tokens = []
            current = clause
            clause_tokens.append(int(token_id))
        if clause_tokens:
            _bin_clause(hist, clause_tokens, bins)
    return hist


def _bin_clause(hist, clause_tokens, bins):
    denom = max(len(clause_tokens) - 1, 1)
    for position, token_id in enumerate(clause_tokens):
        rel = position / denom
        bin_index = min(int(rel * bins), bins - 1)
        hist[token_id, bin_index] += 1


def _bin_centroid(hist_row, lo, hi, bins):
    weights = hist_row[lo:hi].astype(np.float64)
    if weights.sum() == 0:
        return (lo + hi) / 2.0 / bins
    centers = (np.arange(lo, hi) + 0.5) / bins
    return float((centers * weights).sum() / weights.sum())


def select_k(hist_row, k_max, min_mass_frac=0.15):
    total = int(hist_row.sum())
    bins = len(hist_row)
    if total == 0:
        return 1, [0.5]
    # split the [0,1] range into equal segments; a segment is a "mode" if it
    # holds >= min_mass_frac of the mass. k = number of qualifying segments among
    # k_max candidate segments, but never more than the count of nonzero regions.
    for k in range(k_max, 1, -1):
        edges = np.linspace(0, bins, k + 1).astype(int)
        seg_mass = [int(hist_row[edges[i]:edges[i + 1]].sum()) for i in range(k)]
        if all(m >= min_mass_frac * total for m in seg_mass):
            centroids = [_bin_centroid(hist_row, edges[i], edges[i + 1], bins) for i in range(k)]
            return k, centroids
    return 1, [_bin_centroid(hist_row, 0, bins, bins)]


def fixed_centroids(k):
    return [round((i + 0.5) / k, 6) for i in range(k)]


def select_candidates(codes, counts, vocab_size, top_n):
    loss, _ = conflict_losses(codes, counts, vocab_size)
    ranked = np.argsort(-loss, kind="stable")
    return [int(t) for t in ranked if loss[t] > 0][:top_n]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_plan_kway_duplicates.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/plan_kway_duplicates.py tests/test_plan_kway_duplicates.py
git commit -m "Add k-way duplication planner: position histograms + k selection

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Build plan + apply to streams (`scripts/plan_kway_duplicates.py`)

**Files:**
- Modify: `scripts/plan_kway_duplicates.py`
- Test: `tests/test_plan_kway_duplicates.py`

**Interfaces:**
- Consumes: `select_k`, `fixed_centroids`, `renumber_array` (existing).
- Produces: `build_plan(candidates, histograms, vocab_size, k_max=6, fixed_k=None) -> dict` with `vocab_size_old`, `vocab_size_new`, `copies_added`, `parents: {old_index: {"base_new_index": int, "copies": [{"new_index": int, "target": float}]}}` (JSON keys are str at write time; in-memory int keys ok); `apply_kway(stream, plan) -> [(clause, new_id)]` remapping each occurrence by nearest target using its within-clause relative position, non-parents via renumber.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_plan_kway_duplicates.py`:

```python
from scripts.plan_kway_duplicates import apply_kway, build_plan


class BuildAndApplyTests(unittest.TestCase):
    def test_build_plan_fixed_k_renumbers_with_gaps(self):
        hist = np.zeros((4, 20), dtype=np.int64)
        plan = build_plan([1], hist, vocab_size=4, fixed_k=3)
        self.assertEqual(plan["vocab_size_old"], 4)
        self.assertEqual(plan["vocab_size_new"], 6)  # +2 extra copies for parent 1
        parent = plan["parents"][1]
        self.assertEqual(parent["base_new_index"], 1)          # parent keeps first slot
        self.assertEqual([c["new_index"] for c in parent["copies"]], [1, 2, 3])
        self.assertEqual([round(c["target"], 3) for c in parent["copies"]], [0.167, 0.5, 0.833])
        # non-parent id 2 renumbers to 2+2=4 (two extra copies inserted before it)
        stream = [(0, 3), (0, 2)]
        self.assertEqual([nid for _, nid in apply_kway(stream, plan)], [5, 4])

    def test_apply_assigns_nearest_target(self):
        # parent 0 with copies at targets ~0.167/0.5/0.833; a 3-token clause of
        # token 0 at rel-pos 0, .5, 1 -> nearest copies
        hist = np.zeros((1, 20), dtype=np.int64)
        plan = build_plan([0], hist, vocab_size=1, fixed_k=3)
        stream = [(0, 0), (0, 0), (0, 0)]
        got = [nid for _, nid in apply_kway(stream, plan)]
        # targets in new-index order: copy0=idx0(0.167), copy1=idx1(0.5), copy2=idx2(0.833)
        self.assertEqual(got, [0, 1, 2])

    def test_data_driven_k_uses_selected_centroids(self):
        hist = np.zeros((2, 20), dtype=np.int64)
        hist[1, 1] = 50
        hist[1, 18] = 50   # bimodal -> k=2
        plan = build_plan([1], hist, vocab_size=2, k_max=6)
        self.assertEqual(len(plan["parents"][1]["copies"]), 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_plan_kway_duplicates.py -q`
Expected: ImportError (`apply_kway`, `build_plan` undefined)

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/plan_kway_duplicates.py`:

```python
from scripts.plan_token_duplicates import renumber_array


def build_plan(candidates, histograms, vocab_size, k_max=6, fixed_k=None):
    ks = {}
    centroids = {}
    for old_index in candidates:
        if fixed_k is not None:
            k, cents = fixed_k, fixed_centroids(fixed_k)
        else:
            k, cents = select_k(histograms[old_index], k_max)
        if k >= 2:
            ks[old_index] = k
            centroids[old_index] = cents
    parents_sorted = sorted(ks)
    # each surviving parent adds (k-1) extra slots; extras are inserted directly
    # after the parent's renumbered index, in index order
    base_renumber = _kway_renumber(vocab_size, parents_sorted, ks)
    parents = {}
    for old_index in parents_sorted:
        base = int(base_renumber[old_index])
        k = ks[old_index]
        copies = [{"new_index": base + offset, "target": round(centroids[old_index][offset], 6)}
                  for offset in range(k)]
        parents[old_index] = {"base_new_index": base, "copies": copies}
    return {
        "format": "kway_duplicates_plan_v1",
        "vocab_size_old": vocab_size,
        "vocab_size_new": vocab_size + sum(ks[p] - 1 for p in parents_sorted),
        "copies_added": sum(ks[p] - 1 for p in parents_sorted),
        "k_max": k_max,
        "fixed_k": fixed_k,
        "parents": parents,
    }


def _kway_renumber(vocab_size, parents_sorted, ks):
    # new base index of old id t = t + sum over parents p<t of (k_p - 1)
    extras = np.zeros(vocab_size, dtype=np.int64)
    for parent in parents_sorted:
        extras[parent + 1:] += ks[parent] - 1
    return np.arange(vocab_size, dtype=np.int64) + extras


def apply_kway(stream, plan):
    parents = plan["parents"]
    vocab_size = plan["vocab_size_old"]
    ks = {p: len(info["copies"]) for p, info in parents.items()}
    renumber = _kway_renumber(vocab_size, sorted(parents), ks)
    out = []
    clause_tokens = []
    current = None

    def flush():
        denom = max(len(clause_tokens) - 1, 1)
        for position, (clause, token_id) in enumerate(clause_tokens_full):
            if token_id in parents:
                rel = position / denom
                copies = parents[token_id]["copies"]
                best = min(copies, key=lambda c: (abs(c["target"] - rel), c["new_index"]))
                out.append((clause, best["new_index"]))
            else:
                out.append((clause, int(renumber[token_id])))

    clause_tokens_full = []
    for clause, token_id in stream:
        if current is not None and clause != current:
            clause_tokens = clause_tokens_full
            flush()
            clause_tokens_full = []
        current = clause
        clause_tokens_full.append((clause, int(token_id)))
    if clause_tokens_full:
        clause_tokens = clause_tokens_full
        flush()
    return out
```

NOTE for implementer: `parents` keys are ints in-memory (from build_plan) but
become strings if the plan is round-tripped through JSON. `apply_kway` must
handle both — normalize keys to int at the top of `apply_kway`
(`parents = {int(k): v for k, v in plan["parents"].items()}`). Add that line;
the tests above pass an in-memory (int-keyed) plan, so also keep it working for
int keys.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_plan_kway_duplicates.py -q`
Expected: 11 passed

- [ ] **Step 5: Run full suite and commit**

Run: `uv run --group dev python -m pytest tests -q -m "not slow"`
Expected: all pass

```bash
git add scripts/plan_kway_duplicates.py tests/test_plan_kway_duplicates.py
git commit -m "k-way duplication: build_plan + nearest-target apply

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Experiment driver (`scripts/run_kway_experiment.py`)

**Files:**
- Create: `scripts/run_kway_experiment.py`
- Test: `tests/test_run_kway_experiment.py`

**Interfaces:**
- Consumes: `census_pass`, `_build_lookup`, `_stream_ids` (code_vocab_experiment); `file_streams` (code_stream); `PairCounter`, `optimize_order`, `_inverse_permutation` (reorder_phrase_vocab); `collect_position_histograms`, `select_candidates`, `build_plan`, `apply_kway`; `chain_length_histogram`, `summarize` (measure_chain_lengths).
- Produces: `run_arm(name, train_id_streams_fn, eval_id_streams_fn, vocab_size, plan, out_dir, max_passes, ils_restarts, ils_generations, jobs, max_chain_len) -> dict` with `arm`, `extra_slots`, `ascending_fraction`, `validation_chains` (from summarize, histogram popped); a `build_arms(...)` that constructs the no-dup/fixed-k2/data-kway/fixed-kN plans and runs each; CLI.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_kway_experiment.py
import numpy as np
import unittest

from scripts.run_kway_experiment import run_arm
from scripts.plan_kway_duplicates import build_plan


class RunArmTests(unittest.TestCase):
    def test_no_dup_arm_reports_chains(self):
        # 6 identical 3-token clauses; ids in [0,3)
        streams = [[(i, 0), (i, 1), (i, 2)] for i in range(6)]
        plan = build_plan([], np.zeros((3, 20), dtype=np.int64), vocab_size=3, fixed_k=None)  # empty -> no copies
        result = run_arm("nodup", lambda: iter(streams), lambda: iter(streams[:2]),
                         vocab_size=3, plan=plan, out_dir=None, max_passes=5,
                         ils_restarts=1, ils_generations=1, jobs=1, max_chain_len=9)
        self.assertEqual(result["arm"], "nodup")
        self.assertEqual(result["extra_slots"], 0)
        self.assertGreater(result["validation_chains"]["chains"], 0)
        self.assertIn("ascending_fraction", result)

    def test_kway_arm_expands_vocab(self):
        streams = [[(i, 0), (i, 1), (i, 0)] for i in range(8)]
        hist = np.zeros((2, 20), dtype=np.int64)
        hist[0, 1] = 20
        hist[0, 18] = 20  # bimodal -> k=2 for token 0
        plan = build_plan([0], hist, vocab_size=2, k_max=6)
        result = run_arm("kway", lambda: iter(streams), lambda: iter(streams[:2]),
                         vocab_size=2, plan=plan, out_dir=None, max_passes=5,
                         ils_restarts=1, ils_generations=1, jobs=1, max_chain_len=9)
        self.assertEqual(result["extra_slots"], 1)  # token 0 -> 2 copies = +1 slot


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_run_kway_experiment.py -q`
Expected: collection ERROR, module not found

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/run_kway_experiment.py
"""
Compare no-dup / fixed-k2 / data-driven k-way / fixed-kN duplication on JS/TS
code: for each arm, remap streams through its plan, count pairs, reorder+ILS,
and measure chains on held-out eval files. Analysis-only.

Example:
python -m scripts.run_kway_experiment \
    --train ../stack_smol_jsts_raw/data/javascript/data.json \
            ../stack_smol_jsts_raw/data/typescript/data.json \
    --out-dir kway_out --min-count 25 --top-n 300 --k-max 6 --fixed-k-arm 4
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.code_stream import file_streams
from scripts.code_vocab_experiment import _build_lookup, _stream_ids, census_pass
from scripts.measure_chain_lengths import chain_length_histogram, summarize
from scripts.plan_kway_duplicates import apply_kway, build_plan, collect_position_histograms, select_candidates
from scripts.reorder_phrase_vocab import PairCounter, _inverse_permutation, optimize_order


def run_arm(name, train_id_streams_fn, eval_id_streams_fn, vocab_size, plan, out_dir,
            max_passes, ils_restarts, ils_generations, jobs, max_chain_len):
    new_vocab_size = plan["vocab_size_new"]
    counter = PairCounter(vocab_size=new_vocab_size, chunk_size=8_000_000)
    for stream in train_id_streams_fn():
        counter.add_stream(apply_kway(stream, plan))
    codes, counts = counter.finalize()
    order, stats = optimize_order(codes, counts, new_vocab_size, np.arange(new_vocab_size, dtype=np.int64),
                                  max_passes=max_passes, ils_restarts=ils_restarts,
                                  ils_generations=ils_generations, ils_seed=1, jobs=jobs)
    positions = _inverse_permutation(order)
    histogram = {}
    for stream in eval_id_streams_fn():
        chain_length_histogram(apply_kway(stream, plan), positions, reset_on_clause=False,
                               histogram=histogram, max_chain_len=max_chain_len)
    chains = summarize(histogram)
    chains.pop("histogram", None)
    if out_dir is not None:
        with open(Path(out_dir) / f"positions_{name}.json", "w", encoding="utf-8") as file:
            json.dump(positions.tolist(), file)
    return {
        "arm": name,
        "extra_slots": plan["vocab_size_new"] - plan["vocab_size_old"],
        "ascending_fraction": round(stats["ascending_after"] / stats["total_pairs"], 4) if stats["total_pairs"] else 0.0,
        "validation_chains": chains,
    }


def _empty_plan(vocab_size):
    return build_plan([], np.zeros((vocab_size, 1), dtype=np.int64), vocab_size=vocab_size, fixed_k=None)


def parse_args():
    parser = argparse.ArgumentParser(description="k-way duplication comparison on JS/TS code.")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--fixed-k-arm", type=int, default=4)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--max-passes", type=int, default=200)
    parser.add_argument("--ils-restarts", type=int, default=8)
    parser.add_argument("--ils-generations", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-chain-len", type=int, default=9)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    import random
    args = parse_args()
    started = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    # deterministic train/eval file split
    records = []
    for path in args.train:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            records += [line for line in (l.strip() for l in handle) if line]
    random.Random(args.seed).shuffle(records)
    cut = max(1, round(len(records) * args.eval_fraction))
    train_path = Path(args.out_dir) / "_train.jsonl"
    eval_path = Path(args.out_dir) / "_eval.jsonl"
    train_path.write_text("\n".join(records[cut:]) + "\n", encoding="utf-8")
    eval_path.write_text("\n".join(records[:cut]) + "\n", encoding="utf-8")

    census = census_pass([train_path], split_identifiers=False)
    lookup, kept = _build_lookup(census["vocab"], args.min_count)
    vocab_size = len(kept)
    unk = vocab_size - 1

    def train_id_streams():
        for _, stream in file_streams([train_path], split_identifiers=False):
            yield _stream_ids(stream, lookup, unk)

    def eval_id_streams():
        for _, stream in file_streams([eval_path], split_identifiers=False):
            yield _stream_ids(stream, lookup, unk)

    # conflict-mass candidates + position histograms (one pass each)
    counter = PairCounter(vocab_size=vocab_size, chunk_size=8_000_000)
    for stream in train_id_streams():
        counter.add_stream(stream)
    codes, counts = counter.finalize()
    candidates = select_candidates(codes, counts, vocab_size, args.top_n)
    histograms = collect_position_histograms(train_id_streams(), vocab_size, bins=args.bins)

    plans = {
        "nodup": _empty_plan(vocab_size),
        "k2": build_plan(candidates, histograms, vocab_size, fixed_k=2),
        "kway_data": build_plan(candidates, histograms, vocab_size, k_max=args.k_max),
        f"kfixed{args.fixed_k_arm}": build_plan(candidates, histograms, vocab_size, fixed_k=args.fixed_k_arm),
    }
    report = {"format": "kway_experiment_v1", "vocab_size": vocab_size, "candidates": len(candidates),
              "train_files": len(records) - cut, "eval_files": cut, "arms": {}}
    for name, plan in plans.items():
        print(f"=== arm {name} (extra slots {plan['vocab_size_new'] - plan['vocab_size_old']}) ===", flush=True)
        result = run_arm(name, train_id_streams, eval_id_streams, vocab_size, plan, args.out_dir,
                         args.max_passes, args.ils_restarts, args.ils_generations, args.jobs, args.max_chain_len)
        report["arms"][name] = result
        print(json.dumps(result, indent=2), flush=True)
    report["elapsed_seconds"] = round(time.time() - started, 1)
    with open(Path(args.out_dir) / "kway_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_run_kway_experiment.py -q`
Expected: 2 passed

- [ ] **Step 5: Run full suite and commit**

Run: `uv run --group dev python -m pytest tests -q -m "not slow"`
Expected: all pass

```bash
git add scripts/run_kway_experiment.py tests/test_run_kway_experiment.py
git commit -m "Add k-way duplication comparison driver (4 arms, JS/TS)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Execute + report

**Files:** uses Task 3 CLI; workbook `outputs/nanochat-phrase-results/nanochat_phrase_results.xlsx`.

- [ ] **Step 1 (controller): launch in a visible Terminal**

Run: `osascript -e 'tell application "Terminal" to do script "cd \"/Users/jgrimm/dev/Local AI/nanochat\" && caffeinate -i uv run python -m scripts.run_kway_experiment --train \"../stack_smol_jsts_raw/data/javascript/data.json\" \"../stack_smol_jsts_raw/data/typescript/data.json\" --out-dir kway_out --min-count 25 --top-n 300 --k-max 6 --fixed-k-arm 4 2>&1 | tee kway_experiment.log"'`

- [ ] **Step 2 (controller): record results**

Read `kway_out/kway_report.json`. Add a "k-way Duplication" block to the workbook's Code sheet (or a new sheet): rows nodup / k2 / kway_data / kfixed4, columns extra-slots, ascending %, mean chain length, len-1 %, and derived ascending-gain-per-100-slots. Note whether data-driven k-way beats fixed-k2 per slot. Update the sae/vocab memory with the headline.

## Self-Review Notes

- Spec coverage: candidate ranking + position histograms + data/fixed k (Task 1); plan + nearest-target apply + renumber (Task 2); 4-arm comparison incl. per-slot metric (Tasks 3-4). Analysis-only, code corpus, cap-9 measurement — all present.
- Type consistency: `build_plan(candidates, histograms, vocab_size, k_max=6, fixed_k=None)`, `apply_kway(stream, plan)`, `run_arm(...)` signatures identical across Tasks 2-3; plan dict keys (`vocab_size_old/new`, `parents[old]["copies"][i]["new_index"/"target"]`) consistent.
- Flagged inline: JSON str-vs-int key normalization in `apply_kway`; the empty-candidate plan path (`_empty_plan`) for the no-dup arm.
