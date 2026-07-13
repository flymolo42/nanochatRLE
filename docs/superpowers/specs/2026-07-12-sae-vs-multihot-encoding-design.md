# SAE vs Multihot Encoding Comparison — Design

**Date:** 2026-07-12
**Status:** Approved (brainstorm with jgrimm)
**Question:** What is the value of the hand-built linguistic encoding (position-ordered
vocab + ascending multihot chains) versus a learned sparse-autoencoder code — and does
an SAE work on multihot (set-valued) inputs at all?

## Background

The v2 pipeline compresses story history into multihot chain bags (quote-split + ILS
reordered vocab, cross-clause chaining, ~1.9x fewer timesteps). Post-training the v2
base model for 1 epoch on 30k hybrid-cross examples reached X=0 (fully compressed
context) top-1 = 27.9% / top-5 = 59.0% on 2,000 true-validation probes, against a
classic 1-hot ceiling of 58.8% top-1. All compute is a single MacBook (MPS).

## Experiment shape: staged, three arms

### Stage 1 — does an SAE work on multihot? (reconstruction)

Train a standard top-k sparse autoencoder on sparse binary bags over the top-8k vocab:

- Architecture: linear encoder (8192 -> d), top-k activation, linear decoder (d -> 8192),
  MSE loss, unit-norm decoder columns. No bias games; textbook.
- Grid: d in {512, 1024} x k in {8, 32}. Winner per arm chosen by validation set-recovery
  F1 (same metric as the gate, computed on that arm's bag type).
- Training data: ~2M chain bags (chain arm) and ~2M fixed-4-token-window bags (window arm),
  extracted from train-split canonical streams with the existing top-8k remap
  (same <unk> handling as training). Validation bags from the true-validation side file.

**Gate:** on validation chains of length <= 6, decoded set recovery — take the top-m
decoder outputs where m = true set size — must reach F1 >= 0.9. If the gate fails,
arm 2 (SAE-on-multihot) is dropped and the experiment reports arm 3 alone as
"the accuracy price of not using SAE".

### Stage 2 — downstream value (three post-train arms)

All arms share the exact existing protocol: resume `phrase_gpt_base_v2_out/phrase_gpt.pt`,
1 post-train epoch, 30k examples, split-seed 42, sweep on the same 2,000 true-validation
probes (deterministic probe build), `classic_1hot` always reported as the
no-degradation check.

| Arm | Compressed slot content | Boundaries |
|---|---|---|
| 1. Multihot chains (done) | exact chain bag via EmbeddingBag | linguistic (cross-clause ascending) |
| 2. SAE-coded chains | SAE code of the chain bag | linguistic (same as arm 1) |
| 3. SAE-coded windows | SAE code of the window bag | fixed 4-token windows (~matched slot count; 4 ~= mean chain length) |

Readout: one table — compressed-context top-1/top-5/top-10 per arm vs the 58.8%
classic ceiling, plus each arm's classic_1hot regression check.

## Integration trick: latent ids as vocabulary

A compressed slot is represented as the **set of active latent ids**, offset past the
token vocab (ids 8192 .. 8192+d-1). An SAE code is then just another multihot bag over
a larger input vocabulary, so tensor shards, the EmbeddingBag input path, the trainer,
and the sweep eval all work unchanged.

- Output head stays 8,192: targets are always real tokens (`phrase_vocab_size` is
  independent of `vocab_size` in GPTConfig).
- Model surgery: extend the base checkpoint's phrase-embedding table with d new
  randomly-initialized rows before post-training (`--extend-phrase-vocab` trainer flag).
- V1 limitation (accepted): top-k activations are binarized (magnitudes dropped).
  If an SAE arm looks magnitude-starved, per-sample weights on EmbeddingBag are the
  follow-up.
- 1-hot tail steps in SAE arms remain real token ids (< 8192); only the compressed
  front slots use latent ids. Front/tail split mirrors `_hybrid_steps` (random phrase
  boundary per story, split-seed 42).

## Components

| Unit | Purpose | Test strategy |
|---|---|---|
| `scripts/sae.py` | top-k SAE model, training loop, save/load | synthetic bags with known optimal dictionary; recovery + gate metrics |
| `scripts/extract_sae_bags.py` | stream records -> chain/window bags (npz) | small synthetic records; boundary + remap correctness |
| `scripts/build_sae_context_shards.py` | bags -> SAE codes -> latent-id tensor shards | tiny SAE + records; slot ids offset correctly; tail stays token ids |
| trainer `--extend-phrase-vocab` | embedding-table surgery on resume | resumed checkpoint predicts identically on token-only inputs |
| sweep front-encoder variant | SAE-coded fronts in `hybrid_sweep` | context builder unit tests, same style as `hybrid-cross` |

All development is TDD; scripts follow the repo's existing streaming, guarded-runner,
and workbook conventions. Results land in a new "SAE Comparison" sheet in
`outputs/nanochat-phrase-results/nanochat_phrase_results.xlsx` with commit ids and
reproduction commands, per the Vocab Reorder sheet's pattern.

## Budget

- Stage 1: ~half a day (extraction ~1h streaming; each SAE config minutes on MPS).
- Stage 2: ~2-3h per arm (shard build + 1-epoch post-train + sweep), sequential,
  guarded runners (pidfile + RSS watchdog), one trainer at a time.
- Deferred (explicitly out of scope): rate-distortion sweep over latent width/sparsity
  (accuracy-vs-bits curve); revisit when bigger machines arrive.

## Success criteria

1. Stage 1 reports set-recovery F1 by chain length and (d, k); gate decision recorded.
2. Stage 2 table compares all surviving arms on identical probes, with classic_1hot
   regression checks — answering (a) the value of the linguistic encoding vs a learned
   code at matched slot budget, and (b) the accuracy price of skipping SAE if it fails
   on multihot.
