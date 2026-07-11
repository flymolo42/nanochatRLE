# In-Training Sweep Eval + Library Extraction — Design

Date: 2026-07-11

## Goal

Record the hybrid context sweep (top-k accuracy + perplexity vs recent-tail X and
compressed depth D) *during* post-training, so the epoch-by-epoch trajectory is captured
in one `metrics.json` without saving intermediate checkpoints. Extract the sweep-eval
logic into a shared library so the in-training hook and the standalone CLI run identical
code. Add bootstrap confidence intervals for publication-ready error bars.

## Background

The sweep eval currently lives entirely in `scripts/eval_hybrid_context_sweep.py`
(pure probe/context/metric functions + a model runner + a CLI). Post-training resumes
the top-8k checkpoint on hybrid shards. We want the sweep to run at intervals through
post-training on a held-out probe set, producing a trajectory for a paper figure
("how results change through epochs"). See memory `multihot-generation-scheme`.

## Components

### 1. Library extraction — `scripts/hybrid_sweep.py`

Move the reusable eval logic out of `eval_hybrid_context_sweep.py` into a new
`scripts/hybrid_sweep.py`:

- `SweepProbe`, `_stories`, `build_sweep_probes`, `context_steps_for_probe`,
  `topk_and_ce`, `_remap_steps`, `predict_probe_logits`, `_aggregate`, `run_sweep`.

`scripts/eval_hybrid_context_sweep.py` keeps only the CLI (`parse_args`, `main`) and
re-exports the public names (`from scripts.hybrid_sweep import (...)`) so the existing
`tests/test_eval_hybrid_context_sweep.py` imports and the CLI stay valid unchanged. One
implementation, two callers → guaranteed consistency.

The library imports `train_phrase_gpt` primitives at module load (as the eval does
today). The training hook imports the library **lazily** (inside the hook, only when the
sweep is enabled) to avoid a module-load import cycle; non-sweep runs pay zero cost.

### 2. Bootstrap confidence intervals in `_aggregate` / `run_sweep`

`_aggregate` currently accumulates sums. Change it to collect per-probe values
(hit@1, hit@5, hit@10, cross-entropy, is_opener), then:

- point estimates as now (top1/top5/top10, mean_ce, perplexity, count), plus
- **bootstrap CIs**: resample the per-probe values with replacement `B` times (seeded),
  compute each metric per resample, take the 2.5/97.5 percentiles. Perplexity CI =
  `exp` of the mean-CE CI bounds.

Output per (config, target-type) bucket gains parallel keys: `top1_ci`, `top5_ci`,
`top10_ci`, `mean_ce_ci`, `perplexity_ci`, each `[low, high]`. `run_sweep` gains
`bootstrap=1000, bootstrap_seed=0`; `bootstrap=0` disables CIs (keys omitted). RNG is a
seeded `torch.Generator` for reproducibility (no wall-clock/global-random dependence).

### 3. In-training sweep hook in `scripts/train_phrase_gpt.py`

New optional flags (all no-ops unless `--sweep-eval-records` is set):

- `--sweep-eval-records PATH` — held-out records for probes (enables the hook).
- `--sweep-eval-split SPLIT` — restrict probes to this split (e.g. `validation`).
- `--sweep-eval-every-epochs N` (default 1) — run after every Nth completed epoch.
- `--sweep-eval-every-shards K` (default 0 = off) — additionally run every K train
  shards, mid-epoch, via the existing `on_shard_end` callback in `run_epoch_on_shards`.
- `--sweep-x-values` (default `0,1,2,4,8,16`), `--sweep-d-values` (default
  `1,2,4,8,16,32`; unbounded auto-appended), `--sweep-max-probes` (default 2000),
  `--sweep-batch-size` (default 32), `--sweep-bootstrap` (default 1000),
  `--sweep-seed` (default 0).

Behavior:

- Build the probe set **once** at startup (before the epoch loop) from
  `--sweep-eval-records` + split; reuse it at every eval point so the trajectory is
  comparable across epochs.
- At each eval point, run `run_sweep` on the **in-memory model** on the training device,
  using the same `vocab_remap` the run already built. Capture `was_training =
  model.training`, and restore `model.train(was_training)` after (the runner calls
  `model.eval()` internally).
- The per-shard hook composes with the existing rolling-checkpoint `on_shard_end`
  callback (both fire from the same callback; neither replaces the other).

### Data flow: the trajectory

A single list `metrics["sweep_trajectory"]`, each entry:

```json
{"epoch": 8, "shard": null, "sweep": { ...run_sweep result with CIs... }}
```

`shard` is `null` for an end-of-epoch point, or the within-epoch train-shard index
(1-based, as passed to `on_shard_end`) for an intra-epoch point. Order the trajectory by
`(epoch, shard-with-null-last)` to get the x-axis; plotting can derive a global step.
Both per-epoch and per-shard eval points append here. On `--resume`, `metrics` (and thus
the trajectory) carries over, so a resumed run continues the same trajectory. This one
list is the plot data for the epoch/step figure.

## Testing

- Library: existing `tests/test_eval_hybrid_context_sweep.py` continues to pass against
  the re-exported names (comparability, metrics, memory-safe runner).
- Bootstrap CIs: a test asserting each bucket gains `*_ci` `[low, high]` keys with
  `low <= point <= high`, that CIs are reproducible for a fixed seed, and that
  `bootstrap=0` omits them.
- In-training hook: a `main()` test with `--sweep-eval-records` on a tiny shard set +
  tiny model asserts (a) `metrics["sweep_trajectory"]` gets one entry per epoch with a
  `sweep` block, (b) the model is left in training mode afterward, (c) probes are built
  once (same `num_probes` across entries), and (d) a run without the flag adds no
  trajectory and behaves exactly as before.

## Out of scope

- No change to the eval's core metric definitions or the hybrid shard format.
- No plotting/figure code (the trajectory JSON is the deliverable).
- No change to the model, loss, or the training math.
- Multi-token output (unchanged future work).
