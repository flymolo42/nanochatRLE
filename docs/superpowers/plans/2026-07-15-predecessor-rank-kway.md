# Predecessor-Rank k-Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Choose k-way duplication copies from a token's predecessor-rank distribution (not within-clause position), and compare it against the position-based signal on code and mixed corpora.

**Architecture:** A new histogram keyed on each occurrence's predecessor's rank in a fixed reference order feeds the existing `select_k`/`build_plan`. A new `apply_kway_predrank` assigns occurrences by predecessor rank. New comparison arms in both experiment drivers use the no-dup order as the reference. Analysis-only.

**Tech Stack:** numpy, existing nanochat scripts (plan_kway_duplicates, run_kway_experiment, run_mixed_corpus_experiment, reorder_phrase_vocab, measure_chain_lengths), unittest.

**Spec:** `docs/superpowers/specs/2026-07-15-predecessor-rank-kway-design.md`

## Global Constraints

- Run tests with `uv run --group dev python -m pytest tests/<file> -q` from `nanochat/`; unittest-style classes.
- Analysis-only. Streams are `[(clause_id, token_id)]`.
- Reference order for predecessor rank is FIXED at the no-dup order for the run; assignment is deterministic (no chicken-and-egg).
- Predecessor = previous token in the STREAM (cross-clause; these experiments use `reset_on_clause=False`). First token in a stream has no predecessor -> parent's base copy (`copies[0]`, lowest new_index).
- Predecessor rank normalized to [0,1] as `reference_positions[pred_id] / vocab_size_old` to match copy `target` scale.
- `apply_kway_predrank` must normalize str/int plan parent keys (same as `apply_kway`).
- Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Key existing interfaces (read-only):**
- `scripts/plan_kway_duplicates.build_plan(candidates, histograms, vocab_size, k_max=6, fixed_k=None) -> plan`; `apply_kway(stream, plan)`; `_kway_renumber(vocab_size, parents_sorted, ks)`; `select_candidates`, `collect_position_histograms`.
- `scripts/run_kway_experiment.run_arm(name, train_id_streams_fn, eval_id_streams_fn, vocab_size, plan, out_dir, max_passes, ils_restarts, ils_generations, jobs, max_chain_len) -> dict`; uses `apply_kway` internally; `main()` computes `codes,counts` then `candidates`/`histograms`/`plans` and loops arms.
- `scripts/run_mixed_corpus_experiment.run_experiment(...)`: builds `mixed_pos` (no-dup mixed order over union vocab) and per-domain arms.
- `scripts/reorder_phrase_vocab.optimize_order(...) -> (order, stats)`; `_inverse_permutation(order) -> positions`.

---

### Task 1: Predecessor-rank histogram + apply (`scripts/plan_kway_duplicates.py`)

**Files:**
- Modify: `scripts/plan_kway_duplicates.py`
- Test: `tests/test_plan_kway_duplicates.py`

**Interfaces:**
- Consumes: `_kway_renumber` (existing).
- Produces: `collect_predecessor_rank_histograms(id_streams, reference_positions, vocab_size, bins=20) -> ndarray[vocab_size, bins]` — for each token, a histogram of its predecessors' normalized reference ranks; `apply_kway_predrank(stream, plan, reference_positions) -> [(clause, new_id)]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_plan_kway_duplicates.py`:

```python
from scripts.plan_kway_duplicates import apply_kway_predrank, collect_predecessor_rank_histograms


class PredecessorRankTests(unittest.TestCase):
    def test_histogram_bins_predecessor_ranks(self):
        # reference_positions over vocab_size=4: token 3 is rank 3 (-> ~0.75), token 0 rank 0
        reference = np.array([0, 1, 2, 3])
        # stream ids: 9? no; use ids < 4. token 2 preceded by token 3 (rank 3 -> bin 3 of 4)
        streams = [[(0, 3), (0, 2)]]
        hist = collect_predecessor_rank_histograms(streams, reference, vocab_size=4, bins=4)
        # token 2's predecessor is token 3 -> ref rank 3/4=0.75 -> bin 3
        self.assertEqual(hist[2].tolist(), [0, 0, 0, 1])
        # token 3 has no predecessor -> contributes nothing
        self.assertEqual(hist[3].sum(), 0)

    def test_apply_predrank_assigns_by_predecessor_rank(self):
        # parent token 0 gets 2 copies (fixed_k=2, targets 0.25/0.75); vocab_size 4
        hist = np.zeros((4, 20), dtype=np.int64)
        plan = build_plan([0], hist, vocab_size=4, fixed_k=2)
        reference = np.array([0, 1, 2, 3])  # rank(id)=id
        # stream: [3, 0, 1, 0] -> occurrence of 0 at pos1 has predecessor 3 (rank .75 -> late copy),
        # occurrence of 0 at pos3 has predecessor 1 (rank .25 -> early copy)
        stream = [(0, 3), (0, 0), (0, 1), (0, 0)]
        out = [nid for _, nid in apply_kway_predrank(stream, plan, reference)]
        base = plan["parents"][0]["base_new_index"]
        copies = [c["new_index"] for c in plan["parents"][0]["copies"]]  # [base, base+1] targets .25/.75
        # renumber: parent 0 -> copies; non-parents 1,3 shift by +1
        self.assertEqual(out[1], copies[1])  # pred rank .75 -> nearest .75 -> second copy
        self.assertEqual(out[3], copies[0])  # pred rank .25 -> nearest .25 -> first copy

    def test_apply_predrank_first_token_uses_base_copy(self):
        hist = np.zeros((4, 20), dtype=np.int64)
        plan = build_plan([0], hist, vocab_size=4, fixed_k=2)
        reference = np.array([0, 1, 2, 3])
        stream = [(0, 0), (0, 1)]  # token 0 is first -> no predecessor -> base copy
        out = [nid for _, nid in apply_kway_predrank(stream, plan, reference)]
        self.assertEqual(out[0], plan["parents"][0]["base_new_index"])

    def test_apply_predrank_normalizes_json_keys(self):
        import json as _json
        hist = np.zeros((4, 20), dtype=np.int64)
        plan = _json.loads(_json.dumps(build_plan([0], hist, vocab_size=4, fixed_k=2)))
        reference = np.array([0, 1, 2, 3])
        out = apply_kway_predrank([(0, 3), (0, 0)], plan, reference)
        self.assertEqual(len(out), 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_plan_kway_duplicates.py -q`
Expected: ImportError (`apply_kway_predrank`, `collect_predecessor_rank_histograms` undefined)

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/plan_kway_duplicates.py`:

```python
def collect_predecessor_rank_histograms(id_streams, reference_positions, vocab_size, bins=20):
    """For each token, histogram of its predecessors' normalized reference ranks
    (reference_positions[pred] / vocab_size in [0,1]). The predecessor is the
    previous token in the stream; first-in-stream tokens contribute nothing."""
    hist = np.zeros((vocab_size, bins), dtype=np.int64)
    for stream in id_streams:
        previous = None
        for _clause, token_id in stream:
            token_id = int(token_id)
            if previous is not None:
                rel = reference_positions[previous] / vocab_size
                bin_index = min(int(rel * bins), bins - 1)
                hist[token_id, bin_index] += 1
            previous = token_id
    return hist


def apply_kway_predrank(stream, plan, reference_positions):
    """Assign each occurrence of a duplicated token to the copy whose target is
    nearest its predecessor's normalized reference rank; first-in-stream parent
    occurrences use the base copy. Non-parents go through the renumber map."""
    parents = {int(k): v for k, v in plan["parents"].items()}
    vocab_size = plan["vocab_size_old"]
    ks = {p: len(info["copies"]) for p, info in parents.items()}
    renumber = _kway_renumber(vocab_size, sorted(parents), ks)
    out = []
    previous = None
    for clause, token_id in stream:
        token_id = int(token_id)
        if token_id in parents:
            copies = parents[token_id]["copies"]
            if previous is None:
                best = copies[0]
            else:
                rel = reference_positions[previous] / vocab_size
                best = min(copies, key=lambda c: (abs(c["target"] - rel), c["new_index"]))
            out.append((clause, best["new_index"]))
        else:
            out.append((clause, int(renumber[token_id])))
        previous = token_id
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_plan_kway_duplicates.py -q`
Expected: all pass (existing + 4 new)

- [ ] **Step 5: Run full suite and commit**

Run: `uv run --group dev python -m pytest tests -q -m "not slow"`
Expected: all pass

```bash
git add scripts/plan_kway_duplicates.py tests/test_plan_kway_duplicates.py
git commit -m "k-way: predecessor-rank histograms + apply

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: kway_predrank arm in code experiment (`scripts/run_kway_experiment.py`)

**Files:**
- Modify: `scripts/run_kway_experiment.py`
- Test: `tests/test_run_kway_experiment.py`

**Interfaces:**
- Consumes: `collect_predecessor_rank_histograms`, `apply_kway_predrank` (Task 1); `run_arm` (existing).
- Produces: `run_arm(..., reference_positions=None)` — when given, remaps via `apply_kway_predrank(stream, plan, reference_positions)` instead of `apply_kway`; `main()` computes the no-dup order once as the reference and adds a `kway_predrank` arm.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_kway_experiment.py`:

```python
import numpy as np
from scripts.run_kway_experiment import run_arm
from scripts.plan_kway_duplicates import build_plan


class PredrankArmTests(unittest.TestCase):
    def test_run_arm_uses_predrank_apply_when_reference_given(self):
        streams = [[(i, 0), (i, 1), (i, 0)] for i in range(8)]
        hist = np.zeros((2, 20), dtype=np.int64)
        hist[0, 1] = 20
        hist[0, 18] = 20
        plan = build_plan([0], hist, vocab_size=2, k_max=6)
        reference = np.array([0, 1])
        result = run_arm("kway_predrank", lambda: iter(streams), lambda: iter(streams[:2]),
                         vocab_size=2, plan=plan, out_dir=None, max_passes=5,
                         ils_restarts=1, ils_generations=1, jobs=1, max_chain_len=9,
                         reference_positions=reference)
        self.assertEqual(result["arm"], "kway_predrank")
        self.assertGreater(result["validation_chains"]["chains"], 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_run_kway_experiment.py -q`
Expected: FAIL — `run_arm() got an unexpected keyword argument 'reference_positions'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/run_kway_experiment.py`:

1. Update imports:
```python
from scripts.plan_kway_duplicates import (apply_kway, apply_kway_predrank, build_plan,
                                          collect_position_histograms, collect_predecessor_rank_histograms,
                                          select_candidates)
```

2. `run_arm` signature and both apply sites:
```python
def run_arm(name, train_id_streams_fn, eval_id_streams_fn, vocab_size, plan, out_dir,
            max_passes, ils_restarts, ils_generations, jobs, max_chain_len, reference_positions=None):
    def _apply(stream):
        return apply_kway_predrank(stream, plan, reference_positions) if reference_positions is not None else apply_kway(stream, plan)
    new_vocab_size = plan["vocab_size_new"]
    counter = PairCounter(vocab_size=new_vocab_size, chunk_size=8_000_000)
    for stream in train_id_streams_fn():
        counter.add_stream(_apply(stream))
    codes, counts = counter.finalize()
    order, stats = optimize_order(codes, counts, new_vocab_size, np.arange(new_vocab_size, dtype=np.int64),
                                  max_passes=max_passes, ils_restarts=ils_restarts,
                                  ils_generations=ils_generations, ils_seed=1, jobs=jobs)
    positions = _inverse_permutation(order)
    histogram = {}
    for stream in eval_id_streams_fn():
        chain_length_histogram(_apply(stream), positions, reset_on_clause=False,
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
```

3. In `main()`, after `candidates`/`histograms` are computed, add the no-dup reference order and the predrank plan/arm. Replace the `plans` dict build and arm loop with:
```python
    nodup_order, _ = optimize_order(codes, counts, vocab_size, np.arange(vocab_size, dtype=np.int64),
                                    max_passes=args.max_passes, ils_restarts=args.ils_restarts,
                                    ils_generations=args.ils_generations, ils_seed=1, jobs=args.jobs)
    reference_positions = _inverse_permutation(nodup_order)
    predrank_hist = collect_predecessor_rank_histograms(train_id_streams(), reference_positions, vocab_size, bins=args.bins)

    plans = {
        "nodup": (_empty_plan(vocab_size), None),
        "k2": (build_plan(candidates, histograms, vocab_size, fixed_k=2), None),
        "kway_data": (build_plan(candidates, histograms, vocab_size, k_max=args.k_max), None),
        f"kfixed{args.fixed_k_arm}": (build_plan(candidates, histograms, vocab_size, fixed_k=args.fixed_k_arm), None),
        "kway_predrank": (build_plan(candidates, predrank_hist, vocab_size, k_max=args.k_max), reference_positions),
    }
    report = {"format": "kway_experiment_v1", "vocab_size": vocab_size, "candidates": len(candidates),
              "train_files": len(records) - cut, "eval_files": cut, "arms": {}}
    for name, (plan, reference) in plans.items():
        print(f"=== arm {name} (extra slots {plan['vocab_size_new'] - plan['vocab_size_old']}) ===", flush=True)
        result = run_arm(name, train_id_streams, eval_id_streams, vocab_size, plan, args.out_dir,
                         args.max_passes, args.ils_restarts, args.ils_generations, args.jobs, args.max_chain_len,
                         reference_positions=reference)
        report["arms"][name] = result
        print(json.dumps(result, indent=2), flush=True)
```
(`import numpy as np`, `_inverse_permutation`, `optimize_order` are already imported.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_run_kway_experiment.py -q`
Expected: all pass (existing + 1 new)

- [ ] **Step 5: Run full suite and commit**

Run: `uv run --group dev python -m pytest tests -q -m "not slow"`
Expected: all pass

```bash
git add scripts/run_kway_experiment.py tests/test_run_kway_experiment.py
git commit -m "Code k-way: add predecessor-rank arm (reference = no-dup order)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: mixed+kway_predrank arm (`scripts/run_mixed_corpus_experiment.py`)

**Files:**
- Modify: `scripts/run_mixed_corpus_experiment.py`
- Test: `tests/test_run_mixed_corpus_experiment.py`

**Interfaces:**
- Consumes: `collect_predecessor_rank_histograms`, `apply_kway_predrank` (Task 1); `measure_domain` (existing, `plan=` path only handles `apply_kway`).
- Produces: `measure_domain(..., plan=None, reference_positions=None)` — when `reference_positions` given with a plan, uses `apply_kway_predrank`; `run_experiment` adds `mixed_kway_predrank` per domain built from predecessor-rank histograms with reference = `mixed_pos`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_mixed_corpus_experiment.py`:

```python
class PredrankMixedTests(unittest.TestCase):
    def test_report_has_mixed_kway_predrank_per_domain(self):
        report = run_experiment(_prose, _prose, _code, _code, vocab_size=3,
                                top_n=3, k_max=4, max_passes=5, ils_restarts=1,
                                ils_generations=1, jobs=1, max_chain_len=9)
        for domain in ("prose", "code"):
            self.assertIn("mixed_kway_predrank", report[domain])
            self.assertGreater(report[domain]["mixed_kway_predrank"]["chains"], 0)
```

(`_prose`/`_code` fixtures already exist at the top of the file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_run_mixed_corpus_experiment.py -q`
Expected: FAIL — `KeyError: 'mixed_kway_predrank'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/run_mixed_corpus_experiment.py`:

1. Imports:
```python
from scripts.plan_kway_duplicates import (apply_kway, apply_kway_predrank, build_plan,
                                          collect_position_histograms, collect_predecessor_rank_histograms,
                                          select_candidates)
```

2. `measure_domain` gains a reference:
```python
def measure_domain(id_streams_fn, positions, max_chain_len, plan=None, reference_positions=None):
    histogram = {}
    for stream in id_streams_fn():
        if plan is None:
            remapped = stream
        elif reference_positions is not None:
            remapped = apply_kway_predrank(stream, plan, reference_positions)
        else:
            remapped = apply_kway(stream, plan)
        chain_length_histogram(remapped, positions, reset_on_clause=False,
                               histogram=histogram, max_chain_len=max_chain_len)
    result = summarize(histogram)
    result.pop("histogram", None)
    return result
```

3. In `run_experiment`, after `mixed_pos` and the existing k-way (`plan`, `kway_pos`) are built, add the predrank plan + order. Insert right before the `report = {...}` construction:
```python
    # predecessor-rank k-way arm: reference = the no-dup mixed order (mixed_pos)
    predrank_hist = collect_predecessor_rank_histograms(both_train_fn(), mixed_pos, vocab_size)
    predrank_plan = build_plan(candidates, predrank_hist, vocab_size, k_max=k_max)
    pr_new_vocab = predrank_plan["vocab_size_new"]
    pr_counter = PairCounter(vocab_size=pr_new_vocab, chunk_size=8_000_000)
    for stream in both_train_fn():
        pr_counter.add_stream(apply_kway_predrank(stream, predrank_plan, mixed_pos))
    prcodes, prcounts = pr_counter.finalize()
    prorder, _prstats = optimize_order(prcodes, prcounts, pr_new_vocab, np.arange(pr_new_vocab, dtype=np.int64),
                                       ils_seed=1, **kw)
    predrank_pos = _inverse_permutation(prorder)
```

4. Add the predrank measurement to each domain's arms (in the per-domain loop):
```python
    for domain, bespoke_pos, eval_fn in (("prose", prose_pos, prose_eval_fn), ("code", code_pos, code_eval_fn)):
        report[domain] = {
            "bespoke": measure_domain(eval_fn, bespoke_pos, max_chain_len),
            "mixed": measure_domain(eval_fn, mixed_pos, max_chain_len),
            "mixed_kway": measure_domain(eval_fn, kway_pos, max_chain_len, plan=plan),
            "mixed_kway_predrank": measure_domain(eval_fn, predrank_pos, max_chain_len,
                                                  plan=predrank_plan, reference_positions=mixed_pos),
        }
```

5. Add predrank extra-slots to the report (next to `kway_extra_slots`):
```python
        "kway_predrank_extra_slots": pr_new_vocab - vocab_size,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_run_mixed_corpus_experiment.py -q`
Expected: all pass

- [ ] **Step 5: Run full suite and commit**

Run: `uv run --group dev python -m pytest tests -q -m "not slow"`
Expected: all pass

```bash
git add scripts/run_mixed_corpus_experiment.py tests/test_run_mixed_corpus_experiment.py
git commit -m "Mixed k-way: add predecessor-rank arm per domain (reference = mixed no-dup order)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Chained unattended execution + report

**Files:** uses Tasks 2-3 CLIs; workbook.

- [ ] **Step 1 (controller): launch both experiments chained in a Terminal**

Run: `osascript -e 'tell application "Terminal" to do script "cd \"/Users/jgrimm/dev/Local AI/nanochat\" && caffeinate -i sh -c '\''uv run python -m scripts.run_kway_experiment --train ../stack_smol_jsts_raw/data/javascript/data.json ../stack_smol_jsts_raw/data/typescript/data.json --out-dir kway_predrank_out --min-count 25 --top-n 300 --k-max 6 --fixed-k-arm 4 && uv run python -m scripts.run_mixed_corpus_experiment --prose ../pg19/train --prose-eval ../pg19/validation --code ../stack_smol_jsts_raw/data/javascript/data.json ../stack_smol_jsts_raw/data/typescript/data.json --out-dir mixed_predrank_out --min-count 25 --top-n 300 --k-max 6 --balance-pairs'\'' 2>&1 | tee predrank_chained.log; echo CHAINED-DONE"'`

- [ ] **Step 2 (controller): record results**

Read `kway_predrank_out/kway_report.json` and `mixed_predrank_out/mixed_report.json`. In the workbook Code sheet: add `kway_predrank` to the code k-way table and `mixed_kway_predrank` to the mixed table; compare against `kway_data`/`mixed_kway` (position-based) in ascending %, mean chain length, per-slot gain. Note whether predecessor-rank duplicates the punctuation hubs (from the report's `duplication` block on the mixed run) that position-based skipped. Update memory.

## Self-Review Notes

- Spec coverage: predecessor-rank histogram + apply (Task 1); code arm (Task 2); mixed arm (Task 3); chained execution + report incl. duplication-breakdown comparison (Task 4). Reference fixed at no-dup order, str/int key normalization, first-token-no-predecessor — all present.
- Type consistency: `collect_predecessor_rank_histograms(id_streams, reference_positions, vocab_size, bins=20)` and `apply_kway_predrank(stream, plan, reference_positions)` identical across Tasks 1-3; `run_arm(..., reference_positions=None)` and `measure_domain(..., plan=None, reference_positions=None)` consistent with their tests.
- Flagged inline: `measure_domain`/`run_arm` route to `apply_kway_predrank` ONLY when `reference_positions` is set; position arms pass none, so they are byte-unchanged.
