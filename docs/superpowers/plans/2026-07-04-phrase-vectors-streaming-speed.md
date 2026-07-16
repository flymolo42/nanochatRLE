# Phrase Vectors Streaming Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts.phrase_vectors` faster and safer for full TinyStories runs by adding spaCy multiprocessing and streaming/two-pass output.

**Architecture:** Keep the current phrase extraction and record schema unchanged. Add a streamable phrase-row JSONL cache so pass 1 parses stories and builds vocab stats, then pass 2 emits records without retaining every phrase row and record in memory. Expose `--n-process`, `--disable-components`, `--progress-every`, and `--streaming` CLI options.

**Tech Stack:** Python 3.10, spaCy `nlp.pipe`, unittest, JSONL.

---

### Task 1: spaCy Pipeline Options

**Files:**
- Modify: `scripts/phrase_vectors.py`
- Test: `tests/test_phrase_vectors.py`

- [ ] Add a failing test that `load_spacy_model("model", disable_components=["ner"])` passes `disable=["ner"]` to `spacy.load`.
- [ ] Run `rtk uv run python -m unittest tests.test_phrase_vectors -v`; expected failure is `load_spacy_model()` not accepting `disable_components`.
- [ ] Update `load_spacy_model(model_name, disable_components=None)` to call `spacy.load(model_name, disable=disable_components or [])`.
- [ ] Add CLI option `--disable-components`, default `["ner"]`.
- [ ] Run `rtk uv run python -m unittest tests.test_phrase_vectors -v`; expected pass.

### Task 2: Multiprocess Pipe

**Files:**
- Modify: `scripts/phrase_vectors.py`
- Test: `tests/test_phrase_vectors.py`

- [ ] Add a failing test using a fake `nlp.pipe` that asserts `_consume_batch(..., batch_size=7, n_process=3)` forwards both values.
- [ ] Run `rtk uv run python -m unittest tests.test_phrase_vectors -v`; expected failure is missing `n_process` argument.
- [ ] Thread `n_process` through `collect_phrases`, `_consume_batch`, `build_outputs`, and CLI `--n-process`.
- [ ] Call `nlp.pipe(texts, batch_size=batch_size, n_process=n_process)`.
- [ ] Run phrase-vector tests; expected pass.

### Task 3: Streaming Phrase Row Cache

**Files:**
- Modify: `scripts/phrase_vectors.py`
- Test: `tests/test_phrase_vectors.py`

- [ ] Add tests for `write_phrase_rows_jsonl` and `iter_phrase_rows_jsonl` round-tripping `PhraseOccurrence` rows.
- [ ] Run tests and confirm failure for missing functions.
- [ ] Implement JSONL serialization using dictionaries with `split`, `story_id`, and `phrases`.
- [ ] Add `collect_phrases_streaming` that writes phrase rows as docs are parsed and updates vocab stats directly.
- [ ] Add `build_vocab_from_stats` and make existing `build_vocab` delegate through it.
- [ ] Run tests and confirm pass.

### Task 4: Streaming Build Outputs

**Files:**
- Modify: `scripts/phrase_vectors.py`
- Test: `tests/test_phrase_vectors.py`

- [ ] Add a test that `build_outputs(..., streaming=True)` writes outputs without returning phrase rows in memory.
- [ ] Run tests and confirm failure for missing `streaming`.
- [ ] Implement `build_outputs_streaming`: pass 1 writes phrase rows cache and vocab, pass 2 reads cache and streams `phrase_index.jsonl` and `samples.jsonl`.
- [ ] Add CLI `--streaming` defaulting to true and `--no-streaming` for old in-memory behavior.
- [ ] Store `streaming`, `n_process`, `disable_components`, `progress_every`, elapsed seconds, and phrase row cache path in metadata.
- [ ] Run tests and confirm pass.

### Task 5: Verification

**Files:**
- Test: `tests/test_phrase_vectors.py`

- [ ] Run `rtk uv run python -m unittest tests.test_phrase_vectors tests.test_phrase_gpt tests.test_eval_packed_vs_single tests.test_eval_phrase_gpt_packed_vs_single -v`; expected pass.
- [ ] Run smoke command: `rtk uv run python -m scripts.phrase_vectors --tinystories-dir ../TinyStories --source-format txt --out-dir /private/tmp/phrase_vectors_streaming_smoke --limit 20 --batch-size 16 --n-process 1 --streaming --sample-limit 5`; expected metadata JSON and generated outputs.
- [ ] Run a timing comparison against the old in-memory mode with `--no-streaming` on a small limit only if useful; expected comparable output counts.
