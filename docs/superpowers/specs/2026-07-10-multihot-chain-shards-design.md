# Multihot Chain Shards + Post-Training — Design

Date: 2026-07-10

## Goal

Enable post-training the existing phrase-GPT checkpoint on **multihot-per-timestep**
input, so the model learns to consume a summed phrase/chain vector per position
instead of one token per position. This makes the "packed"/multihot path
in-distribution and lets the context eval measure it fairly (it currently scores
multihot mode out-of-distribution, ~6.3%).

Acceptance bar (see project memory `multihot-chains-direction`): multihot on par
with, or a little worse than, token-per-step is a success, because multihot uses
~3× fewer timesteps and a fixed window reaches ~3× deeper into the story.

## Key insight: no trainer changes

Per-phrase and cross-phrase multihots fit the **existing** tensor-shard format
(`phrase_gpt_tensor_shard_v1`) unchanged — each timestep already may carry multiple
phrase indices via `vector_offsets`; today's `token`-mode shards just happen to have
exactly one. The trainer (`scripts/train_phrase_gpt.py`) already consumes that format
and already supports `--resume` and `--vocab-top-k`. Therefore **all new code lives at
the sharding layer**; post-training is the existing training command.

## Components

### 1. Two new example-builders in `scripts/train_phrase_gpt.py`

Both return `list[PhraseSequenceExample]` (same type as `_examples_from_story_records`),
so shard writing and training are untouched. Both operate on one story's records.

- **`phrase` mode** — one timestep per phrase.
  - Timestep t input = phrase t's multihot: reuse the `packed` record's `indices`
    when present; otherwise fall back to the union of the phrase's `single` records
    (same fallback the eval's `build_context_examples` uses). This mirrors the chosen
    "one multihot per phrase (reuse packed)" definition.
  - Target[t] = first token (lowest `token_pos`) of phrase t+1.
  - Phrases with no usable indices are skipped; a target requires a following phrase
    with at least one single record.

- **`cross-phrase` mode** — story-spanning in-order chains.
  - Walk the story's canonical token stream in story order (reuse
    `_canonical_token_stream`, which orders by absolute token position).
  - Accumulate token indices into the current chain while the **vocab index strictly
    increases**. When a token's index ≤ the previous token's index, close the current
    chain (one timestep multihot) and start a new chain beginning with that token.
    Chains may be shorter than a phrase or span phrase boundaries.
  - Rationale: vocab index encodes average phrase position; strictly-increasing index
    is a heuristic that makes longer in-order runs likely. This is the "out-of-order
    token breaks the vector and a new one starts" rule (see memory
    `phrase-vocab-index-order`), applied across phrases.
  - Target[t] = first token of chain t+1 (i.e. the token that broke chain t).

Sequences are chunked into `sequence_len` windows exactly as `_examples_from_story_records`
already does (reuse the same chunking helper).

### 2. `--chain-mode {token,phrase,cross-phrase}` on `scripts/build_phrase_gpt_shards.py`

- Default `token` = current behavior; fully backward-compatible (existing shards and
  callers unaffected).
- The flag selects which builder `_flush_story` invokes.
- The written `manifest.json` records `"chain_mode"` so shard sets are self-documenting.
- Existing `--limit-examples` provides the subset-first run.

### 3. Post-training run (no new code)

Subset first (~500k examples, ~1–2 h on the M3):

```
python -m scripts.build_phrase_gpt_shards --chain-mode phrase \
    --vocab phrase_vectors_full_out/vocab.json \
    --records phrase_vectors_full_out/phrase_index.jsonl.gz \
    --out-dir phrase_chain_shards_len128 --sequence-len 128 --limit-examples 500000

python -m scripts.train_phrase_gpt \
    --resume phrase_gpt_len128_topk8k_out/best_phrase_gpt.pt \
    --shards phrase_chain_shards_len128/manifest.json \
    --vocab phrase_vectors_full_out/vocab.json \
    --vocab-top-k 8191 --lr 3e-4 --epochs 2 --save-best
```

`--resume` loads the plateaued weights + optimizer; `--vocab-top-k 8191` remaps the
multihot indices into the model's output space (resume enforces the top-k match).
Then re-run `scripts/eval_phrase_gpt_packed_vs_single.py --context-phrases 8` — its
packed mode now matches training distribution.

Full-corpus reshard is deferred, likely to the incoming Ryzen box.

## Testing (TDD)

Unit tests in `tests/test_phrase_gpt.py` (builders) and
`tests/test_build_phrase_gpt_shards.py` (flag/manifest):

- `phrase` mode on a hand-built story: packed reuse when present; single-record
  fallback when no packed; target = next phrase's first token; correct skipping of
  index-less phrases and last-phrase (no target).
- `cross-phrase` mode: a story where an out-of-order token splits one phrase's tokens
  into two chains, and an in-order run merges tokens across a phrase boundary into one
  chain; verify chain multihots and targets.
- Both modes respect `sequence_len` chunking (window boundaries).
- `--chain-mode` end-to-end: build tiny shards in each mode, assert `manifest.json`
  records `chain_mode`, and that round-tripped examples (`tensor_shard_to_examples`)
  match the builder output. `token` default reproduces current behavior.

## Out of scope

- No head/loss change (single-token softmax retained).
- No multi-label / next-multihot target.
- No full-corpus reshard in this work.
- No changes to `train_phrase_gpt.py` training loop (only new builder functions added).
