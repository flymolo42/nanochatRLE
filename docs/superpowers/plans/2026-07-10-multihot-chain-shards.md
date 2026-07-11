# Multihot Chain Shards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `phrase` and `cross-phrase` multihot chain shard-building modes so the existing phrase-GPT checkpoint can be post-trained on one-multihot-per-chain input.

**Architecture:** All new code is at the sharding layer. A single core routine builds in-order "chains" from the punctuation token stream; a `reset_on_clause` flag distinguishes `phrase` (break at clause boundaries) from `cross-phrase` (break only on out-of-order vocab index). `token` mode preserves current behavior. Chains reuse the unchanged tensor-shard format, so the trainer and its `--resume`/`--vocab-top-k` paths need no changes.

**Tech Stack:** Python 3.10, PyTorch, `unittest`. Run tests with `.venv/bin/python -m unittest`.

## Global Constraints

- Both new modes read ONLY the punctuation token stream via `_canonical_token_stream` (`label == "punctuation"` singles, ordered by absolute token position). One clause = one `punctuation` `phrase_id`.
- Chain rule: accumulate token vocab indices while **strictly increasing**; break (close current chain, start new one at this token) when the next index ≤ the current chain's last index.
- Target at each timestep = first token (index 0) of the next chain.
- `token` mode remains the default and must reproduce current behavior exactly (backward compatible).
- No changes to the tensor-shard format (`phrase_gpt_tensor_shard_v1`) or the training loop.
- Work on branch `multihot-chain-shards`.

---

### Task 1: Refactor token-mode into unified step/chunk structure (no behavior change)

**Files:**
- Modify: `scripts/train_phrase_gpt.py` (replace `_examples_from_story_records`, lines ~126-141)
- Test: `tests/test_phrase_gpt.py`

**Interfaces:**
- Produces:
  - `_token_steps(records) -> list[tuple[list[int], int]]`
  - `_chunk_steps_into_examples(steps, sequence_len) -> list[PhraseSequenceExample]`
  - `examples_from_story_records(records, sequence_len, chain_mode="token") -> list[PhraseSequenceExample]`
  - `_examples_from_story_records(records, sequence_len)` retained as alias for `chain_mode="token"`.
  - `CHAIN_MODE_BUILDERS: dict[str, callable]` with key `"token"` (keys `"phrase"`, `"cross-phrase"` added in Tasks 2-3).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_phrase_gpt.py` inside `PhraseGPTTests`:

```python
    def test_examples_from_story_records_token_mode_matches_legacy_helper(self):
        from scripts.train_phrase_gpt import examples_from_story_records, _examples_from_story_records

        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [10], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [11], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [12], "token_pos": 2},
        ]

        via_dispatch = examples_from_story_records(records, sequence_len=2, chain_mode="token")
        via_legacy = _examples_from_story_records(records, sequence_len=2)

        self.assertEqual(
            [(e.input_indices, e.targets) for e in via_dispatch],
            [(e.input_indices, e.targets) for e in via_legacy],
        )
        self.assertEqual([(e.input_indices, e.targets) for e in via_dispatch], [([[10], [11]], [11, 12])])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_phrase_gpt.PhraseGPTTests.test_examples_from_story_records_token_mode_matches_legacy_helper -v`
Expected: FAIL with `ImportError: cannot import name 'examples_from_story_records'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/train_phrase_gpt.py`, replace the existing `_examples_from_story_records` function (currently lines ~126-141) with:

```python
def _token_steps(records):
    token_records = _canonical_token_stream(records)
    return [
        (list(previous["indices"]), int(current["indices"][0]))
        for previous, current in zip(token_records, token_records[1:])
        if previous.get("indices") and current.get("indices")
    ]


def _chunk_steps_into_examples(steps, sequence_len):
    examples = []
    for start in range(0, len(steps), sequence_len):
        chunk = steps[start:start + sequence_len]
        if chunk:
            examples.append(PhraseSequenceExample(
                input_indices=[indices for indices, _ in chunk],
                targets=[target for _, target in chunk],
            ))
    return examples


CHAIN_MODE_BUILDERS = {
    "token": lambda records: _token_steps(records),
}


def examples_from_story_records(records, sequence_len, chain_mode="token"):
    if chain_mode not in CHAIN_MODE_BUILDERS:
        raise ValueError(f"Unknown chain_mode {chain_mode!r}; expected one of {sorted(CHAIN_MODE_BUILDERS)}")
    return _chunk_steps_into_examples(CHAIN_MODE_BUILDERS[chain_mode](records), sequence_len)


def _examples_from_story_records(records, sequence_len):
    return examples_from_story_records(records, sequence_len, chain_mode="token")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_phrase_gpt -v`
Expected: PASS, including the existing `test_build_phrase_sequence_examples_*` tests (they call `_examples_from_story_records` indirectly and must stay green).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_phrase_gpt.py tests/test_phrase_gpt.py
git commit -m "Refactor story-example builder into step/chunk dispatch (token mode)"
```

---

### Task 2: `phrase` mode — chains break on order and clause boundary

**Files:**
- Modify: `scripts/train_phrase_gpt.py` (add `_chain_steps`, register `"phrase"` in `CHAIN_MODE_BUILDERS`)
- Test: `tests/test_phrase_gpt.py`

**Interfaces:**
- Consumes: `_canonical_token_stream`, `_chunk_steps_into_examples`, `examples_from_story_records` (Task 1).
- Produces: `_chain_steps(records, reset_on_clause: bool) -> list[tuple[list[int], int]]`; `CHAIN_MODE_BUILDERS["phrase"]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_phrase_gpt.py` inside `PhraseGPTTests`. The story stream (ordered by `start + token_pos`) is clause 0 = indices [1, 3, 2], clause 1 = indices [4, 5]:

```python
    def _chain_story_records(self):
        return [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [1], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [3], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [2], "token_pos": 2},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [4], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [5], "token_pos": 1},
        ]

    def test_phrase_mode_breaks_on_order_and_clause_boundary(self):
        from scripts.train_phrase_gpt import examples_from_story_records

        examples = examples_from_story_records(self._chain_story_records(), sequence_len=10, chain_mode="phrase")

        # clause 0 -> [1,3] then break on out-of-order 2 -> [2]; clause boundary breaks before [4,5]
        # chains: [1,3], [2], [4,5]; steps target = next chain's first token
        self.assertEqual(
            [(e.input_indices, e.targets) for e in examples],
            [([[1, 3], [2]], [2, 4])],
        )

    def test_phrase_mode_monotone_clause_is_single_chain(self):
        from scripts.train_phrase_gpt import examples_from_story_records

        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [10], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [11], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [12], "token_pos": 2},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [20], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [21], "token_pos": 1},
        ]

        examples = examples_from_story_records(records, sequence_len=10, chain_mode="phrase")

        # monotone clause 0 -> single chain [10,11,12]; target = first token of clause-1 chain
        self.assertEqual([(e.input_indices, e.targets) for e in examples], [([[10, 11, 12]], [20])])

    def test_phrase_mode_respects_sequence_len_chunking(self):
        from scripts.train_phrase_gpt import examples_from_story_records

        examples = examples_from_story_records(self._chain_story_records(), sequence_len=1, chain_mode="phrase")

        self.assertEqual(
            [(e.input_indices, e.targets) for e in examples],
            [([[1, 3]], [2]), ([[2]], [4])],
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_phrase_gpt.PhraseGPTTests.test_phrase_mode_breaks_on_order_and_clause_boundary -v`
Expected: FAIL with `ValueError: Unknown chain_mode 'phrase'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/train_phrase_gpt.py`, add `_chain_steps` (place it right after `_token_steps`):

```python
def _chain_steps(records, reset_on_clause):
    token_records = _canonical_token_stream(records)
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
    return [(chains[position], chains[position + 1][0]) for position in range(len(chains) - 1)]
```

Then register the mode by replacing the `CHAIN_MODE_BUILDERS` dict:

```python
CHAIN_MODE_BUILDERS = {
    "token": lambda records: _token_steps(records),
    "phrase": lambda records: _chain_steps(records, reset_on_clause=True),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_phrase_gpt -v`
Expected: PASS (all, including Task 1 and existing tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/train_phrase_gpt.py tests/test_phrase_gpt.py
git commit -m "Add phrase-mode multihot chains (break on order + clause boundary)"
```

---

### Task 3: `cross-phrase` mode — chains merge across clause boundaries

**Files:**
- Modify: `scripts/train_phrase_gpt.py` (register `"cross-phrase"` in `CHAIN_MODE_BUILDERS`)
- Test: `tests/test_phrase_gpt.py`

**Interfaces:**
- Consumes: `_chain_steps` (Task 2), `_chain_story_records` test helper (Task 2).
- Produces: `CHAIN_MODE_BUILDERS["cross-phrase"]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_phrase_gpt.py` inside `PhraseGPTTests`:

```python
    def test_cross_phrase_mode_merges_in_order_run_across_clauses(self):
        from scripts.train_phrase_gpt import examples_from_story_records

        examples = examples_from_story_records(self._chain_story_records(), sequence_len=10, chain_mode="cross-phrase")

        # stream 1,3,2,4,5: break only on out-of-order (2<=3) -> chains [1,3] and [2,4,5]
        # (2->4->5 merges across the clause boundary because there is no clause reset)
        self.assertEqual(
            [(e.input_indices, e.targets) for e in examples],
            [([[1, 3]], [2])],
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_phrase_gpt.PhraseGPTTests.test_cross_phrase_mode_merges_in_order_run_across_clauses -v`
Expected: FAIL with `ValueError: Unknown chain_mode 'cross-phrase'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/train_phrase_gpt.py`, add the `"cross-phrase"` entry to `CHAIN_MODE_BUILDERS`:

```python
CHAIN_MODE_BUILDERS = {
    "token": lambda records: _token_steps(records),
    "phrase": lambda records: _chain_steps(records, reset_on_clause=True),
    "cross-phrase": lambda records: _chain_steps(records, reset_on_clause=False),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_phrase_gpt -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_phrase_gpt.py tests/test_phrase_gpt.py
git commit -m "Add cross-phrase multihot chains (break only on order)"
```

---

### Task 4: `--chain-mode` flag on the shard builder + manifest record

**Files:**
- Modify: `scripts/build_phrase_gpt_shards.py` (`_flush_story`, `build_shards_from_records`, `parse_args`, `main`)
- Test: `tests/test_build_phrase_gpt_shards.py`

**Interfaces:**
- Consumes: `examples_from_story_records` (Task 1).
- Produces: `build_shards_from_records(..., chain_mode="token")`; manifest key `"chain_mode"`; CLI `--chain-mode`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_build_phrase_gpt_shards.py` inside `BuildPhraseGPTShardsTests`:

```python
    def test_build_shards_phrase_mode_records_mode_and_multihot_examples(self):
        records = [
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [1], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [3], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [2], "token_pos": 2},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [4], "token_pos": 0},
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 3, "end": 5, "record_type": "single", "indices": [5], "token_pos": 1},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_shards_from_records(
                records=iter(records),
                out_dir=tmpdir,
                sequence_len=10,
                examples_per_shard=50,
                chain_mode="phrase",
            )
            saved = json.loads((Path(tmpdir) / "manifest.json").read_text(encoding="utf-8"))
            shard = torch.load(Path(tmpdir) / saved["shards"][0]["file"], map_location="cpu", weights_only=False)

        self.assertEqual(saved["chain_mode"], "phrase")
        example = tensor_shard_to_examples(shard)[0]
        self.assertEqual((example.input_indices, example.targets), ([[1, 3], [2]], [2, 4]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_build_phrase_gpt_shards.BuildPhraseGPTShardsTests.test_build_shards_phrase_mode_records_mode_and_multihot_examples -v`
Expected: FAIL with `TypeError: build_shards_from_records() got an unexpected keyword argument 'chain_mode'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/build_phrase_gpt_shards.py`:

Update the import (top of file) to also import the dispatch function:

```python
from scripts.train_phrase_gpt import (
    examples_from_story_records,
    examples_to_tensor_shard,
)
```

Replace `_flush_story` to accept and use `chain_mode`:

```python
def _flush_story(current_rows, sequence_len, chain_mode):
    if not current_rows:
        return []
    return examples_from_story_records(current_rows, sequence_len=sequence_len, chain_mode=chain_mode)
```

Change `build_shards_from_records` signature to add `chain_mode="token"`:

```python
def build_shards_from_records(records, out_dir, sequence_len, examples_per_shard, records_path="", vocab_path="", progress_every=100000, max_examples=None, chain_mode="token"):
```

Update the two `_flush_story(...)` call sites inside `build_shards_from_records` (currently lines ~85 and ~98) to pass `chain_mode`:

```python
                if not add_examples(current_key[0], _flush_story(current_rows, sequence_len=sequence_len, chain_mode=chain_mode)):
```

and

```python
        add_examples(current_key[0], _flush_story(current_rows, sequence_len=sequence_len, chain_mode=chain_mode))
```

Add `chain_mode` to the manifest dict (next to `"sequence_len"`):

```python
        "chain_mode": chain_mode,
```

Add the CLI argument in `parse_args`:

```python
    parser.add_argument("--chain-mode", choices=["token", "phrase", "cross-phrase"], default="token", help="How to build per-timestep inputs: token (one token/step, default), phrase (one multihot chain per punctuation clause, breaking on out-of-order vocab index), cross-phrase (chains span clause boundaries, breaking only on out-of-order index).")
```

Pass it through in `main`:

```python
        max_examples=args.limit_examples,
        chain_mode=args.chain_mode,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_build_phrase_gpt_shards tests.test_phrase_gpt -v`
Expected: PASS (including the pre-existing `test_build_shards_from_records_writes_tensor_shards_and_manifest`, which uses the default `token` mode)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_phrase_gpt_shards.py tests/test_build_phrase_gpt_shards.py
git commit -m "Add --chain-mode {token,phrase,cross-phrase} to shard builder"
```

---

### Task 5: Full regression + manual subset post-training (verification only)

**Files:** none (verification)

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/python -m unittest discover -s tests -v`
Expected: PASS except the pre-existing `test_attention_fallback` import error (missing `pytest`), which is unrelated to this work.

- [ ] **Step 2: Build a small phrase-mode shard set (subset first)**

Run:
```bash
.venv/bin/python -m scripts.build_phrase_gpt_shards --chain-mode phrase \
    --vocab phrase_vectors_full_out/vocab.json \
    --records phrase_vectors_full_out/phrase_index.jsonl.gz \
    --out-dir phrase_chain_shards_len128 --sequence-len 128 --limit-examples 500000
```
Expected: `manifest.json` written with `"chain_mode": "phrase"`. NOTE: this reads the full gz stream; it can run for a while. Keep an eye on memory but the builder streams stories, so RSS stays bounded.

- [ ] **Step 3: Post-train the existing checkpoint (subset, few epochs)**

Run:
```bash
.venv/bin/python -m scripts.train_phrase_gpt \
    --resume phrase_gpt_len128_topk8k_out/best_phrase_gpt.pt \
    --shards phrase_chain_shards_len128/manifest.json \
    --vocab phrase_vectors_full_out/vocab.json \
    --vocab-top-k 8191 --lr 3e-4 --epochs 2 --save-best
```
Expected: `resuming from ... at epoch 8` (or later), training loss finite and decreasing. This is a long run; launch in the background and watch RSS (stay well under 18GB — batch 32 keeps peak small).

- [ ] **Step 4: Report**

Summarize final train/val loss+accuracy vs the token-trained baseline. Do NOT commit checkpoints or shard outputs (they are large build artifacts).

---
