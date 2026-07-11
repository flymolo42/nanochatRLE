# In-Training Sweep Eval + Library Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the hybrid sweep eval into a shared library, add bootstrap confidence intervals, and run the sweep on a held-out probe set during post-training so the epoch/step trajectory lands in one `metrics.json` without saving intermediate weights.

**Architecture:** Move the eval logic to `scripts/hybrid_sweep.py`; `eval_hybrid_context_sweep.py` becomes a thin CLI re-exporting it. `train_phrase_gpt.py` imports the library *lazily* (only when `--sweep-eval-records` is set) to avoid an import cycle, builds probes once, and appends each sweep result to `metrics["sweep_trajectory"]`.

**Tech Stack:** Python 3.10, PyTorch, `unittest`. Run tests with `.venv/bin/python -m unittest`.

## Global Constraints

- The library imports `train_phrase_gpt` primitives at module load; `train_phrase_gpt` imports the library ONLY via a lazy import inside the hook. Non-sweep training runs pay zero cost.
- `build_sweep_probes` must STREAM: accept a record iterator, group story-by-story (records are contiguous by story), normalize per-story (not the whole file), filter to `split`, and stop at `max_probes`. Never materialize the full file. `main()` passes `iter_records(path)` (an iterator), not `list(...)`.
- Bootstrap CIs: seeded `torch.Generator` (no wall-clock / global RNG); `bootstrap=0` omits CI keys; perplexity CI = `exp` of the mean-CE CI bounds.
- Comparability preserved: one probe set scored across all configs; opener/interior partition `all`.
- In-training sweep: build probes ONCE before the epoch loop; run on the in-memory model on the training device with the run's own `vocab_remap`; save `was_training = model.training` and restore `model.train(was_training)` after.
- Trajectory entry shape: `{"epoch": E, "shard": S_or_null, "sweep": <run_sweep result>}`; `shard` is null for end-of-epoch, else the within-epoch 1-based shard index. Restored on `--resume` from `metrics["sweep_trajectory"]`.
- No change to the model, loss, training math, or hybrid shard format.
- Work on branch `in-training-sweep-eval`.

---

### Task 1: Extract library + streaming `build_sweep_probes`

**Files:**
- Create: `scripts/hybrid_sweep.py`
- Modify: `scripts/eval_hybrid_context_sweep.py` (becomes thin CLI + re-export)
- Test: `tests/test_eval_hybrid_context_sweep.py`

**Interfaces:**
- Produces (in `scripts.hybrid_sweep`): `SweepProbe`, `_iter_story_raw_rows`, `build_sweep_probes(records, min_history=1, max_probes=None, split=None)`, `context_steps_for_probe`, `topk_and_ce`, `_remap_steps`, `predict_probe_logits`, `_aggregate`, `run_sweep`, `_parse_int_list`.
- `scripts.eval_hybrid_context_sweep` re-exports all the above and keeps `parse_args`/`main`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval_hybrid_context_sweep.py` (the `_story` helper already exists in that file):

```python
    def test_build_sweep_probes_streams_filters_split_and_stops_early(self):
        from scripts.hybrid_sweep import build_sweep_probes

        def records():
            for r in _story(0, [(0, [1, 2, 3])]):   # split "train"
                yield r
            for r in _story(1, [(0, [4, 5, 6])]):
                r = dict(r); r["split"] = "validation"
                yield r
            raise AssertionError("build_sweep_probes read past the story it needed")

        # split filter: only the validation story's probes, and max_probes stops early
        probes = build_sweep_probes(records(), min_history=1, max_probes=1, split="validation")
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0].token_indices, [4, 5, 6])
```

Note: `_story` builds records with `split="train"`; the test rewrites split to `validation` for the second story. The trailing `raise` proves streaming stops once `max_probes` is hit.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_eval_hybrid_context_sweep.HybridSweepPureTests.test_build_sweep_probes_streams_filters_split_and_stops_early -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.hybrid_sweep'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/hybrid_sweep.py`. Move these functions **verbatim** from `scripts/eval_hybrid_context_sweep.py`: `SweepProbe`, `context_steps_for_probe`, `topk_and_ce`, `_remap_steps`, `predict_probe_logits`, `_aggregate`, `run_sweep`, `_parse_int_list`. Keep their imports (`torch`, `torch.nn.functional as F`, `math`, `dataclass`, and the `from scripts.train_phrase_gpt import ...` / `from scripts.eval_phrase_gpt_packed_vs_single import ...` lines). Do NOT move `_stories` (replaced below), `parse_args`, or `main`.

Add the streaming probe builder (replaces the old `_stories` + `build_sweep_probes`):

```python
def _iter_story_raw_rows(records):
    """Group a stream of raw records into (split, story_id) -> rows, streaming.
    Records in phrase_index.jsonl are contiguous by story, so consecutive grouping
    is exact and never materializes the whole file."""
    current_key = None
    current_rows = []
    for record in records:
        key = (record["split"], int(record["story_id"]))
        if current_key is not None and key != current_key:
            yield current_key, current_rows
            current_rows = []
        current_key = key
        current_rows.append(record)
    if current_rows:
        yield current_key, current_rows


def build_sweep_probes(records, min_history=1, max_probes=None, split=None):
    # Clamp to >= 1: a position-0 probe has empty context (predict_probe_logits would
    # index last=-1, scoring the padding row). Never emit position-0 probes.
    from scripts.train_phrase_vectors import normalize_phrase_records
    min_history = max(1, min_history)
    probes = []
    for key, raw_rows in _iter_story_raw_rows(records):
        if split is not None and key[0] != split:
            continue
        stream = _canonical_token_stream(normalize_phrase_records(raw_rows))
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
```

Ensure `scripts/hybrid_sweep.py` imports `_canonical_token_stream` and `_chains_from_token_records` from `scripts.train_phrase_gpt` (needed by `build_sweep_probes` and `context_steps_for_probe`).

Now rewrite `scripts/eval_hybrid_context_sweep.py` to re-export from the library and keep only the CLI:

```python
"""
CLI for the hybrid context sweep eval. Logic lives in scripts.hybrid_sweep.
"""

import argparse
import json

import torch

from scripts.hybrid_sweep import (
    SweepProbe, build_sweep_probes, context_steps_for_probe, topk_and_ce,
    predict_probe_logits, run_sweep, _parse_int_list,
)
from scripts.train_phrase_gpt import choose_device
from scripts.train_phrase_vectors import iter_records
from scripts.eval_phrase_gpt_packed_vs_single import load_model_from_checkpoint, resolve_vocab_remap


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid context sweep eval (top-k accuracy + perplexity vs X and D).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--vocab", default=None, help="Original vocab.json, if the checkpoint's recorded path is wrong.")
    parser.add_argument("--x-values", default="0,1,2,4,8,16")
    parser.add_argument("--d-values", default="1,2,4,8,16,32", help="Compressed-depth caps; unbounded appended automatically.")
    parser.add_argument("--fixed-x-for-depth", type=int, default=0)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--max-probes", type=int, default=20000)
    parser.add_argument("--split", default=None, help="Only score stories from this split (e.g. 'validation').")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for CIs; 0 disables.")
    parser.add_argument("--sweep-seed", type=int, default=0, help="Seed for bootstrap CIs.")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    probes = build_sweep_probes(iter_records(args.records), min_history=args.min_history,
                                max_probes=args.max_probes, split=args.split)
    if not probes:
        raise SystemExit("No probes found.")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, _config = load_model_from_checkpoint(checkpoint, device=device)
    remap = resolve_vocab_remap(checkpoint.get("config", {}), vocab_override=args.vocab)
    if remap is not None:
        remap = remap.to("cpu")
    print(f"probes={len(probes)} device={device} remap={'yes' if remap is not None else 'no'}", flush=True)
    # Compressed contexts longer than the model's sequence_len are truncated by the
    # final-layer attention window, so large D and unbounded D can coincide on long
    # stories — the D-sweep is only meaningful up to ~sequence_len chains of history.
    d_values = _parse_int_list(args.d_values) + [None]
    result = run_sweep(model, probes, x_values=_parse_int_list(args.x_values), d_values=d_values,
                       fixed_x_for_depth=args.fixed_x_for_depth, remap=remap, batch_size=args.batch_size,
                       device=device, bootstrap=args.bootstrap, bootstrap_seed=args.sweep_seed)
    result["split"] = args.split
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

Note: `run_sweep` gains `bootstrap`/`bootstrap_seed` params in Task 2; for Task 1, add them to `run_sweep`'s signature now with defaults `bootstrap=0, bootstrap_seed=0` and ignore them (Task 2 wires the behavior), so this CLI call signature is stable. Concretely, in the moved `run_sweep`, change the signature to `def run_sweep(model, probes, x_values, d_values, fixed_x_for_depth, remap, batch_size, device, bootstrap=0, bootstrap_seed=0):` and pass `bootstrap`/`bootstrap_seed` into each `_aggregate(...)` call (also add those two params to `_aggregate` with defaults, unused until Task 2).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_eval_hybrid_context_sweep -v`
Expected: PASS — all existing tests (they import from `scripts.eval_hybrid_context_sweep`, which now re-exports) plus the new streaming test.

- [ ] **Step 5: Commit**

```bash
git add scripts/hybrid_sweep.py scripts/eval_hybrid_context_sweep.py tests/test_eval_hybrid_context_sweep.py
git commit -m "Extract hybrid sweep eval into scripts/hybrid_sweep.py; stream probe build"
```

---

### Task 2: Bootstrap confidence intervals

**Files:**
- Modify: `scripts/hybrid_sweep.py` (`_aggregate`, `run_sweep`)
- Test: `tests/test_eval_hybrid_context_sweep.py`

**Interfaces:**
- Consumes: `run_sweep`/`_aggregate` (Task 1).
- Produces: per-bucket CI keys `top1_ci`, `top5_ci`, `top10_ci`, `mean_ce_ci`, `perplexity_ci` (each `[low, high]`), emitted when `bootstrap > 0`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_eval_hybrid_context_sweep.py`:

```python
    def test_run_sweep_bootstrap_cis_present_reproducible_and_bracket_point(self):
        import torch
        from nanochat.gpt import GPT, GPTConfig
        import nanochat.flash_attention as fa_module
        from scripts.hybrid_sweep import build_sweep_probes, run_sweep

        fa_module._override_impl = "sdpa"
        fa_module.USE_FA3 = fa_module._resolve_use_fa3()
        torch.manual_seed(0)
        config = GPTConfig(sequence_len=8, vocab_size=8, n_layer=1, n_head=2, n_kv_head=2,
                           n_embd=32, window_pattern="L", phrase_vocab_size=8)
        model = GPT(config, pad_vocab_size_to=1); model.init_weights()
        records = _story(0, [(0, [1, 3, 2]), (1, [4, 5])]) + _story(1, [(0, [1, 2]), (1, [3, 4, 5])])
        probes = build_sweep_probes(records, min_history=1)

        kw = dict(x_values=[0], d_values=[None], fixed_x_for_depth=0, remap=None, batch_size=4, device="cpu")
        a = run_sweep(model, probes, bootstrap=200, bootstrap_seed=7, **kw)
        b = run_sweep(model, probes, bootstrap=200, bootstrap_seed=7, **kw)
        none = run_sweep(model, probes, bootstrap=0, **kw)

        cell = a["x_sweep"]["0"]["all"]
        for key in ("top1_ci", "top5_ci", "top10_ci", "mean_ce_ci", "perplexity_ci"):
            self.assertIn(key, cell)
            self.assertEqual(len(cell[key]), 2)
            self.assertLessEqual(cell[key][0], cell[key][1])
        # CI brackets the point estimate
        self.assertLessEqual(cell["top1_ci"][0], cell["top1"])
        self.assertLessEqual(cell["top1"], cell["top1_ci"][1])
        # reproducible for a fixed seed
        self.assertEqual(a["x_sweep"]["0"]["all"]["top1_ci"], b["x_sweep"]["0"]["all"]["top1_ci"])
        # bootstrap=0 omits CI keys
        self.assertNotIn("top1_ci", none["x_sweep"]["0"]["all"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_eval_hybrid_context_sweep.HybridSweepPureTests.test_run_sweep_bootstrap_cis_present_reproducible_and_bracket_point -v`
Expected: FAIL (KeyError / assertIn on `top1_ci`).

- [ ] **Step 3: Write minimal implementation**

In `scripts/hybrid_sweep.py`, replace `_aggregate` with a per-probe-collecting version plus a bootstrap helper:

```python
def _bootstrap_cis(per_probe, bootstrap, seed):
    import torch
    n = len(per_probe["ce"])
    tensors = {k: torch.tensor(per_probe[k], dtype=torch.float64) for k in ("h1", "h5", "h10", "ce")}
    generator = torch.Generator().manual_seed(int(seed))
    idx = torch.randint(0, n, (bootstrap, n), generator=generator)

    def ci(key):
        samples = tensors[key][idx].mean(dim=1)
        return [torch.quantile(samples, 0.025).item(), torch.quantile(samples, 0.975).item()]

    ce_lo, ce_hi = ci("ce")
    return {
        "top1_ci": ci("h1"), "top5_ci": ci("h5"), "top10_ci": ci("h10"),
        "mean_ce_ci": [ce_lo, ce_hi], "perplexity_ci": [math.exp(ce_lo), math.exp(ce_hi)],
    }


def _aggregate(probes, logits, remap, bootstrap=0, bootstrap_seed=0):
    per = {name: {"h1": [], "h5": [], "h10": [], "ce": []} for name in ("all", "opener", "interior")}
    for probe, row in zip(probes, logits):
        target = int(remap[probe.token_indices[probe.target_pos]]) if remap is not None else probe.token_indices[probe.target_pos]
        hits, ce = topk_and_ce(row, target, ks=(1, 5, 10))
        for name in ("all", "opener" if probe.is_opener else "interior"):
            per[name]["h1"].append(hits[1]); per[name]["h5"].append(hits[5])
            per[name]["h10"].append(hits[10]); per[name]["ce"].append(ce)
    out = {}
    for name, arr in per.items():
        n = len(arr["ce"])
        mean_ce = (sum(arr["ce"]) / n) if n else float("nan")
        cell = {
            "top1": (sum(arr["h1"]) / n) if n else 0.0,
            "top5": (sum(arr["h5"]) / n) if n else 0.0,
            "top10": (sum(arr["h10"]) / n) if n else 0.0,
            "mean_ce": mean_ce,
            "perplexity": math.exp(mean_ce) if n else float("nan"),
            "count": n,
        }
        if bootstrap and n:
            cell.update(_bootstrap_cis(arr, bootstrap, bootstrap_seed))
        out[name] = cell
    return out
```

In `run_sweep`, pass the bootstrap params through to both `_aggregate` calls:

```python
        result["x_sweep"][str(x)] = _aggregate(probes, logits, remap, bootstrap=bootstrap, bootstrap_seed=bootstrap_seed)
```
```python
        result["d_sweep"][str(d)] = _aggregate(probes, logits, remap, bootstrap=bootstrap, bootstrap_seed=bootstrap_seed)
```

(The `run_sweep` signature already has `bootstrap=0, bootstrap_seed=0` from Task 1.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_eval_hybrid_context_sweep -v`
Expected: PASS (new CI test + all prior tests; point-estimate keys unchanged so existing comparability test still holds).

- [ ] **Step 5: Commit**

```bash
git add scripts/hybrid_sweep.py tests/test_eval_hybrid_context_sweep.py
git commit -m "Add bootstrap confidence intervals to hybrid sweep aggregation"
```

---

### Task 3: In-training sweep hook in `train_phrase_gpt.py`

**Files:**
- Modify: `scripts/train_phrase_gpt.py` (`parse_args`, `main`, add `_run_training_sweep`)
- Test: `tests/test_phrase_gpt.py`

**Interfaces:**
- Consumes: `scripts.hybrid_sweep.build_sweep_probes`, `run_sweep`, `_parse_int_list` (lazy import); `iter_records` (already imported in train_phrase_gpt).
- Produces: `_run_training_sweep(model, probes, args, remap, device, epoch, shard, trajectory)`; `metrics["sweep_trajectory"]`; `--sweep-*` CLI flags.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_phrase_gpt.py` inside `PhraseGPTTests`. It reuses `_write_shard_fixture`/`_main_argv` (already in the file) and writes a tiny records file for probes:

```python
    def test_run_training_sweep_appends_entry_and_restores_train_mode(self):
        from scripts.train_phrase_gpt import _run_training_sweep
        from scripts.hybrid_sweep import build_sweep_probes
        import argparse

        _force_sdpa()
        model = _tiny_phrase_gpt()
        model.train(True)
        records = [
            {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [1], "token_pos": 0},
            {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [2], "token_pos": 1},
            {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [3], "token_pos": 2},
        ]
        probes = build_sweep_probes(records, min_history=1, split="validation")
        args = argparse.Namespace(sweep_x_values="0", sweep_d_values="1", sweep_batch_size=4,
                                  sweep_bootstrap=0, sweep_seed=0, sweep_eval_split="validation")
        trajectory = []
        _run_training_sweep(model, probes, args, None, "cpu", epoch=3, shard=None, trajectory=trajectory)

        self.assertEqual(len(trajectory), 1)
        self.assertEqual(trajectory[0]["epoch"], 3)
        self.assertIsNone(trajectory[0]["shard"])
        self.assertIn("x_sweep", trajectory[0]["sweep"])
        self.assertTrue(model.training)  # restored

    def test_main_records_sweep_trajectory_per_epoch(self):
        from scripts.train_phrase_gpt import main

        _force_sdpa()
        train = [
            PhraseSequenceExample(input_indices=[[0], [1], [2]], targets=[1, 2, 3]),
            PhraseSequenceExample(input_indices=[[3], [4], [5]], targets=[4, 5, 1]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            vocab, manifest = _write_shard_fixture(tmp, [train])
            sweep_records = tmp / "val_records.jsonl"
            rows = [
                {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [1], "token_pos": 0},
                {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [2], "token_pos": 1},
                {"split": "validation", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [3], "token_pos": 2},
            ]
            sweep_records.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            out_dir = str(tmp / "out")
            extra = ["--epochs", "2", "--sweep-eval-records", str(sweep_records),
                     "--sweep-eval-split", "validation", "--sweep-x-values", "0",
                     "--sweep-d-values", "1", "--sweep-max-probes", "5", "--sweep-bootstrap", "0"]
            with mock.patch("sys.argv", _main_argv(vocab, manifest, out_dir, extra=extra)):
                main()
            metrics = json.loads((Path(out_dir) / "metrics.json").read_text(encoding="utf-8"))

        traj = metrics["sweep_trajectory"]
        self.assertEqual([e["epoch"] for e in traj], [1, 2])
        self.assertTrue(all(e["shard"] is None for e in traj))
        self.assertTrue(all("x_sweep" in e["sweep"] for e in traj))
        counts = {e["sweep"]["num_probes"] for e in traj}
        self.assertEqual(len(counts), 1)  # same probe set every epoch
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_phrase_gpt.PhraseGPTTests.test_main_records_sweep_trajectory_per_epoch -v`
Expected: FAIL with `unrecognized arguments: --sweep-eval-records`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/train_phrase_gpt.py`:

Add the flags in `parse_args` (near the other args):

```python
    parser.add_argument("--sweep-eval-records", default=None, help="Held-out records for in-training sweep eval (enables it).")
    parser.add_argument("--sweep-eval-split", default=None, help="Restrict sweep probes to this split, e.g. 'validation'.")
    parser.add_argument("--sweep-eval-every-epochs", type=int, default=1)
    parser.add_argument("--sweep-eval-every-shards", type=int, default=0)
    parser.add_argument("--sweep-x-values", default="0,1,2,4,8,16")
    parser.add_argument("--sweep-d-values", default="1,2,4,8,16,32")
    parser.add_argument("--sweep-max-probes", type=int, default=2000)
    parser.add_argument("--sweep-batch-size", type=int, default=32)
    parser.add_argument("--sweep-bootstrap", type=int, default=1000)
    parser.add_argument("--sweep-seed", type=int, default=0)
```

Add the helper near the top-level functions (e.g. above `def main():`):

```python
def _run_training_sweep(model, probes, args, remap, device, epoch, shard, trajectory):
    from scripts.hybrid_sweep import run_sweep, _parse_int_list  # lazy: avoids import cycle
    was_training = model.training
    result = run_sweep(
        model, probes,
        x_values=_parse_int_list(args.sweep_x_values),
        d_values=_parse_int_list(args.sweep_d_values) + [None],
        fixed_x_for_depth=0, remap=remap,
        batch_size=args.sweep_batch_size, device=device,
        bootstrap=args.sweep_bootstrap, bootstrap_seed=args.sweep_seed,
    )
    result["split"] = args.sweep_eval_split
    model.train(was_training)
    trajectory.append({"epoch": epoch, "shard": shard, "sweep": result})
    return result
```

Add `"sweep_trajectory": []` to the `metrics` dict initializer (next to `"epochs": []`).

In the `if args.resume:` block, restore it (next to the `metrics["epochs"] = ...` line):

```python
        metrics["sweep_trajectory"] = list(saved_metrics.get("sweep_trajectory", []))
```

Before the epoch loop (`for epoch in range(...)`), build probes once:

```python
    sweep_probes = None
    if args.sweep_eval_records:
        from scripts.hybrid_sweep import build_sweep_probes  # lazy
        sweep_probes = build_sweep_probes(iter_records(args.sweep_eval_records),
                                          min_history=1, max_probes=args.sweep_max_probes,
                                          split=args.sweep_eval_split)
        print(f"sweep eval: {len(sweep_probes)} probes from {args.sweep_eval_records} (split={args.sweep_eval_split})", flush=True)
```

Extend the per-shard callback: rename `save_rolling_checkpoint` to `on_shard_end` and add the per-shard sweep, then pass `on_shard_end=on_shard_end` to `run_epoch_on_shards`:

```python
        def on_shard_end(shard_idx, rows, epoch=epoch):
            if args.checkpoint_every_shards > 0 and shard_idx % args.checkpoint_every_shards == 0:
                save_checkpoint(args.out_dir, model, checkpoint_config, metrics, optimizer=optimizer, extra={
                    "epoch": epoch - 1,
                    "epochs_without_improvement": early_state.epochs_without_improvement,
                    "shard_progress": {"epoch": epoch, "shards_completed": shard_idx, "metric_rows": rows},
                })
            if sweep_probes is not None and args.sweep_eval_every_shards > 0 and shard_idx % args.sweep_eval_every_shards == 0:
                _run_training_sweep(model, sweep_probes, args, vocab_remap, device, epoch, shard_idx, metrics["sweep_trajectory"])
```

(Update the `on_shard_end=save_rolling_checkpoint` argument in the `run_epoch_on_shards(...)` train call to `on_shard_end=on_shard_end`.)

After the per-epoch `row` is appended (`metrics["epochs"].append(row)`) and before the end-of-epoch `save_checkpoint(...)`, add the per-epoch sweep:

```python
        if sweep_probes is not None and epoch % max(1, args.sweep_eval_every_epochs) == 0:
            _run_training_sweep(model, sweep_probes, args, vocab_remap, device, epoch, None, metrics["sweep_trajectory"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_phrase_gpt -v`
Expected: PASS (both new tests + all existing; a no-`--sweep-eval-records` run leaves `sweep_probes=None` so behavior is unchanged).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_phrase_gpt.py tests/test_phrase_gpt.py
git commit -m "Run hybrid sweep eval during training; record per-epoch/shard trajectory"
```

---

### Task 4: Full regression + validation extract + end-to-end trajectory (verification)

**Files:** none (verification)

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/python -m unittest discover -s tests 2>&1 | grep -E "^(Ran|OK|FAILED)|^(ERROR|FAIL):"`
Expected: one known-unrelated failure only — `test_attention_fallback` (missing `pytest`).

- [ ] **Step 2: Extract the validation split once (small, fast, genuinely held-out)**

Run:
```bash
cd '/Users/jgrimm/dev/Local AI/nanochat'
gunzip -c phrase_vectors_full_out/phrase_index.jsonl.gz | grep '"split": "validation"' | gzip > phrase_vectors_full_out/phrase_index_validation.jsonl.gz
```
(This streams the full file once; it is slow but one-time. Do it in the FOREGROUND — background processes detach unreliably on this machine.)

- [ ] **Step 3: Post-train with in-training sweep on the validation set (FOREGROUND, small)**

Reuse the existing 30k hybrid shards if present, else rebuild a small set. Then:
```bash
PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.train_phrase_gpt \
    --resume phrase_gpt_len128_topk8k_out/best_phrase_gpt.pt \
    --shards phrase_hybrid_shards_len128/manifest.json \
    --vocab phrase_vectors_full_out/vocab.json --vocab-top-k 8191 --sequence-len 128 \
    --lr 3e-4 --epochs 10 --out-dir phrase_gpt_hybrid_traj_out \
    --sweep-eval-records phrase_vectors_full_out/phrase_index_validation.jsonl.gz \
    --sweep-eval-split validation --sweep-max-probes 2000 --sweep-x-values 0,2,8 --sweep-d-values 1,4,16
```
Expected: per-epoch `sweep eval: N probes ...` and a growing `metrics["sweep_trajectory"]`.

- [ ] **Step 4: Report the trajectory**

Read `phrase_gpt_hybrid_traj_out/metrics.json`, summarize `sweep_trajectory`: X=0 top-1 and perplexity (with CIs) per epoch, showing the compressed-context skill improving across epochs. Do NOT commit checkpoints, shards, or the validation extract (large artifacts).

---
