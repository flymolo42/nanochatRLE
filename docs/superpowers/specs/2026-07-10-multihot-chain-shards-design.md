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

## Data reality: "phrase" is 8 overlapping label types

`extract_phrase_occurrences` (`scripts/phrase_vectors.py`) emits, per sentence, a
flat list of labeled spans mixing two kinds:

- **`punctuation`** — linear clause segmentation: every token, split at punctuation.
  These clauses are a disjoint left-to-right partition of the sentence. This is the
  ONLY label the current token model trained on (`_canonical_token_stream` filters to
  `label == "punctuation"`).
- **`subject`/`object`/`verb`/`relation`/`root_clause`/`noun_chunk`** — overlapping
  syntactic spans over the same tokens.

`phrase_id` enumerates this mixed list, so consecutive `phrase_id`s overlap and are not
a story-ordered progression. Therefore both new modes work strictly from the
**punctuation token stream** (`_canonical_token_stream`), and a clause = one
`punctuation` `phrase_id`.

## Components

### 1. Unified chain-step builder in `scripts/train_phrase_gpt.py`

A single core routine builds "chains" (in-order multihot runs) from the punctuation
token stream; the three modes differ only in where chains break. All builders return
`list[(input_indices, target_index)]` "steps", chunked into `PhraseSequenceExample`s
by one shared helper, so shard writing and training are untouched.

**Core rule — accumulate while strictly increasing, break otherwise.** Walk the
canonical punctuation token stream in story order (`_canonical_token_stream`, ordered
by absolute token position; each record carries its clause `phrase_id`). Keep adding
token vocab indices to the current chain while the **vocab index strictly increases**;
break (close the current chain as one timestep multihot, start a new chain at this
token) when the index does not increase. Rationale: vocab index encodes average phrase
position, so a strictly-increasing run is a coherent in-order unit; a drop signals a
new unit (see memory `phrase-vocab-index-order`). For a monotone clause this yields
exactly one chain — identical to that clause's `packed` record.

**The one knob — `reset_on_clause`:**

- **`phrase` mode** (`reset_on_clause=True`) — a chain also breaks at every
  `punctuation` clause boundary (when `phrase_id` changes). Chains never span two
  clauses. Monotone clause → one multihot (its packed record); clause with an
  out-of-order token → splits into 2+ in-order multihots.
- **`cross-phrase` mode** (`reset_on_clause=False`) — no clause reset; chains run
  through the whole story's punctuation token stream, breaking only on out-of-order
  index, so an in-order run merges across clause boundaries.

**`token` mode** (default, current behavior) — one timestep per token: input =
`[token_index]`, target = next token's index. Preserved exactly by refactoring the
existing `_examples_from_story_records` logic into a `_token_steps` builder.

**Target** at each timestep = first token of the next chain (the token that broke the
current chain). For `token` mode this reduces to "next token", matching today.

Concrete shapes (private helpers; `records` is one story's records):

- `_token_steps(records) -> list[(list[int], int)]`
- `_chain_steps(records, reset_on_clause: bool) -> list[(list[int], int)]`
- `_chunk_steps_into_examples(steps, sequence_len) -> list[PhraseSequenceExample]`
- `examples_from_story_records(records, sequence_len, chain_mode="token")` — public
  dispatch over `{"token", "phrase", "cross-phrase"}`.
- `_examples_from_story_records(records, sequence_len)` retained as a thin alias for
  `chain_mode="token"` (its existing importers in `build_phrase_gpt_shards.py` keep
  working).

### 2. `--chain-mode {token,phrase,cross-phrase}` on `scripts/build_phrase_gpt_shards.py`

- Default `token` = current behavior; fully backward-compatible (existing shards and
  callers unaffected).
- Threaded through `build_shards_from_records` → `_flush_story` →
  `examples_from_story_records(records, sequence_len, chain_mode)`.
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

- `token` mode unchanged: the refactor preserves existing behavior (reuse/extend the
  existing `test_build_phrase_sequence_examples_*` expectations via
  `examples_from_story_records(..., chain_mode="token")`).
- `phrase` mode (`reset_on_clause=True`) on a hand-built story:
  - a monotone clause yields exactly one chain (equal to its token set);
  - a clause containing an out-of-order token splits into two in-order chains;
  - two adjacent clauses whose tokens would be in-order across the boundary still
    break at the boundary (no cross-clause chain);
  - target[t] = first token of chain t+1.
- `cross-phrase` mode (`reset_on_clause=False`): the same in-order run that spans two
  clauses merges into one chain (contrast with `phrase` mode), and an out-of-order
  token still breaks a chain mid-stream.
- Both modes respect `sequence_len` chunking (window boundaries).
- `--chain-mode` end-to-end: build tiny shards in each mode, assert `manifest.json`
  records `chain_mode`, and that round-tripped examples (`tensor_shard_to_examples`)
  match the builder output. `token` default reproduces current behavior.

## Out of scope

- No head/loss change (single-token softmax retained).
- No multi-label / next-multihot target.
- No full-corpus reshard in this work.
- No changes to `train_phrase_gpt.py` training loop (only new builder functions added).

## Follow-up (not this work)

The context eval (`scripts/eval_phrase_gpt_packed_vs_single.py`) currently builds its
packed steps from all-label `phrase_id` grouping. For train/eval alignment with a
`phrase`-mode checkpoint, its packed path should later switch to the same
punctuation-clause chain construction. Deferred; called out so the eval delta is
interpreted with this mismatch in mind until then.
