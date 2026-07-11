# Hybrid Context + Sweep Eval — Design

Date: 2026-07-11

## Goal

Give the multihot-chains post-training a framework to answer two questions with
**directly comparable** metrics:

1. **Does a recent 1-hot token tail help accuracy?** (keep the "last X tokens 1-hot"
   idea only if it earns its keep)
2. **How much does deeper compressed-context post-training help?**

Answer both from **one** post-trained model by randomizing the compression split
during training, then sweeping the context configuration at eval time.

## Background / prior context

- Generation is token-level (single-token output, per-token temperature preserved).
  Compression applies to CONTEXT only: recent tokens 1-hot, older tokens folded into
  phrase multihots. See memory `multihot-generation-scheme`.
- A recent 1-hot token is a length-1 chain through `phrase_wte`; the model already
  handles length-1 chains. So NO model or training-loop change is required — the whole
  framework is data construction + an eval.
- Target rule is unchanged from the existing chain builder:
  `target[t] = first token of the next chain = the next token in the stream`. For a
  length-1 chain the first token is the token itself, so compressed-phrase positions
  and 1-hot-token positions share one uniform target rule.

## The unifying idea: a split point

Represent a story as `[older tokens compressed into phrase chains] ++ [recent tokens
as length-1 chains]`, split at index `s` in the canonical punctuation token stream:

- `s` at the end → everything compressed → today's `phrase` mode (X=0 tail)
- `s` at the start → everything 1-hot → the base model's `token` mode
- `s` in the middle → hybrid: compressed history + a recent 1-hot tail

Randomizing `s` per example makes one dataset (and one post-trained model) span every
configuration; the eval sweep then reads off the curves.

## Components

### 1. `hybrid` chain mode in the shard builder

Add `hybrid` alongside `token`/`phrase`/`cross-phrase` (extends the merged
`--chain-mode` work). Per story, from the canonical punctuation token stream:

- Choose split index `s` uniformly at random over the story's **phrase (clause)
  boundary token indices**, including `0` (fully 1-hot) and `N` (fully compressed) so
  both endpoints are reachable. Seeded from the story id + a run seed, so builds are
  reproducible.
- Tokens before `s`: group into phrase-mode chains (break on out-of-order vocab index
  and at clause boundaries — reuse `_chain_steps(reset_on_clause=True)`).
- Tokens from `s` onward: emit each as its own length-1 chain (1-hot).
- Concatenate the two chain lists; apply the existing target rule and
  `_chunk_steps_into_examples`. No new target logic.

One split per story (seeded). Optional future knob: emit K splits per story for denser
coverage — out of scope for v1.

The manifest records `chain_mode: hybrid` and the split seed.

### 2. Sweep eval (`scripts/eval_hybrid_context_sweep.py`)

A held-out **next-token** eval. Central rule — **comparability**: build ONE shared set
of probe targets, then evaluate every configuration on that identical set, varying only
the context representation. Same targets, same top-8k vocab, same denominator.

Per sampled probe (a target token at story position `p` with sufficient history):

- **X sweep** (recent 1-hot tail): for each `X` in a configured set (default
  `0,1,2,4,8,16`), context = `[tokens[0..p-X-1] compressed into phrase chains] ++
  [tokens[p-X..p-1] as 1-hot]`; predict token `p`.
- **D sweep** (compressed depth): for each `D` in a configured set (default
  `1,2,4,8,16,32` and unbounded), cap the compressed region to the most recent `D`
  phrases before the tail; predict token `p`. Run at a fixed `X` (default `X=0`) so the
  depth axis is isolated from the tail axis.
- Record the probe's **target type**: phrase-opener (first token of its phrase) vs.
  mid-phrase interior.

Reference endpoints, on the same probes: `X=0` (fully compressed) and no-compression
(all-1-hot, the base token regime).

**Metrics per (config, target-type) bucket**, over the shared probe set:

- top-1 accuracy, top-5 accuracy, top-10 accuracy
- mean cross-entropy and perplexity (`exp(mean CE)`)
- probe count (denominator, shown so buckets are never silently unequal)

Memory-safe like the fixed context eval: stream predictions, small batches, sort by
length, free the device cache per batch, cap probe count with a flag. Uses the top-k
vocab remap auto-detected from the checkpoint (as the current eval does).

### 3. Post-training run + guiding curves (verification)

Resume the top-8k checkpoint on randomized-split `hybrid` shards
(`--resume … --vocab-top-k 8191 --lr 3e-4`), then run the sweep eval:

- `accuracy(X)` flat past `X=0` → the 1-hot tail is not needed; a rising curve → keep it
  (and interior-token buckets show where it helps most).
- `accuracy(D)` rising with depth → the deeper post-training is justified.

## Comparability guarantees (explicit, per user requirement)

- All X and D configs are scored on the **same probe targets** — only the context
  representation varies.
- Opener vs. interior are reported as **separate breakdowns**, never compared to each
  other as a single accuracy.
- These next-token sweep numbers are a **distinct metric family** from the existing
  Packed Eval sheet (which scores next-phrase-opener single-vs-packed at a boundary).
  They are recorded separately, not merged into those rows.
- Vocab (top-8k) is held constant across every config.

## Out of scope

- Multi-token / phrase-level output (future work).
- Any change to `nanochat/gpt.py`, the loss, or the training loop.
- Whether to permanently adopt the 1-hot tail — that is the decision the metrics inform,
  not a build-time choice.
- Emitting multiple splits per story (v2 coverage knob).
