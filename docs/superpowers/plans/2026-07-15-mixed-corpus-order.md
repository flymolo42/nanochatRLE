# Mixed-Corpus Vocabulary Ordering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one vocabulary order over PG-19 prose + JS/TS code and measure whether it compresses each domain as well as a bespoke per-domain order, plus whether k-way duplication of shared hubs recovers any interference.

**Architecture:** Both corpora are tokenized by their own tokenizers into one pooled token-string vocabulary (shared symbols merge). Three orders (prose-only / code-only / mixed) plus a mixed+k-way arm are built in that single union id space; each held-out domain is measured under its bespoke order vs the mixed order. Analysis-only.

**Tech Stack:** numpy, existing nanochat scripts (pg19_stream, code_stream, reorder_phrase_vocab, measure_chain_lengths, plan_kway_duplicates, phrase_vectors.build_vocab_from_stats), unittest.

**Spec:** `docs/superpowers/specs/2026-07-15-mixed-corpus-order-design.md`

## Global Constraints

- Run tests with `uv run --group dev python -m pytest tests/<file> -q` from `nanochat/`; unittest-style classes.
- Analysis-only: no training. Streams are `[(clause_id, token)]` (strings) then `[(clause_id, token_id)]` after union-mapping.
- Union token space: PG-19 via `pg19_stream.tokenize_clauses`, code via `code_stream.tokenize_code(text, split_identifiers=False)` (whole identifiers). Pool token STRINGS; shared strings share one id; a trailing `<unk>` absorbs below-min-count tokens.
- All three orders + the k-way arm are permutations/expansions of the SAME union vocab id space, so per-domain chain measurements are directly comparable.
- Per-domain deterministic train/eval file split (seed 7); every arm reuses the same split.
- Chain measurement: cross-clause (`reset_on_clause=False`), `max_chain_len=9`.
- k-way numbers are a documented LOWER BOUND (crude `select_k`); the k=2-style baseline is not in scope here (this experiment's arms are bespoke / mixed / mixed+kway).
- Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Key existing interfaces (read-only):**
- `scripts/pg19_stream.tokenize_clauses(text) -> [(clause, token), ...]`
- `scripts/code_stream.tokenize_code(text, split_identifiers=False) -> [(clause, token), ...]`
- `scripts/phrase_vectors.build_vocab_from_stats(stats, min_count=1) -> [{token, index, count, avg_position}]` where `stats` is `dict[token -> {"count": int, "position_sum": float}]`
- `scripts/code_vocab_experiment._build_lookup(census_vocab, min_count) -> (lookup{token->id}, kept[list])` and `_stream_ids(stream, lookup, unk_id) -> [(clause, id)]`
- `scripts/reorder_phrase_vocab.PairCounter(vocab_size, chunk_size).add_stream([(clause,id),...]); .finalize() -> (codes, counts)`
- `scripts/reorder_phrase_vocab.optimize_order(codes, counts, vocab_size, warm_order, max_passes=, ils_restarts=, ils_generations=, ils_seed=, jobs=) -> (order, stats)`; `stats["ascending_after"]`, `stats["total_pairs"]`; `_inverse_permutation(order) -> positions`
- `scripts/measure_chain_lengths.chain_length_histogram(stream, mapping, reset_on_clause, histogram=None, max_chain_len=None)`; `summarize(histogram) -> {chains, avg_chain_len, len1_fraction_of_chains, ...}`
- `scripts/plan_kway_duplicates.{collect_position_histograms, select_candidates, build_plan, apply_kway}`

---

### Task 1: Union streams + census (`scripts/mixed_corpus_streams.py`)

**Files:**
- Create: `scripts/mixed_corpus_streams.py`
- Test: `tests/test_mixed_corpus_streams.py`

**Interfaces:**
- Consumes: `pg19_stream.tokenize_clauses`, `code_stream.tokenize_code`, `phrase_vectors.build_vocab_from_stats`.
- Produces: `prose_file_streams(paths) -> yields (path_stem, [(clause, token)])` (reads .txt); `code_file_streams(paths) -> yields (row_index_str, [(clause, token)])` (reads JSONL `content`, whole-identifier); `union_census(tagged_stream_iters) -> census_vocab list` where `tagged_stream_iters` is an iterable of `(domain, [(clause, token)])` — accumulates count + within-clause position and returns `build_vocab_from_stats` output.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mixed_corpus_streams.py
import json
import tempfile
import unittest
from pathlib import Path

from scripts.mixed_corpus_streams import code_file_streams, prose_file_streams, union_census


class FileStreamTests(unittest.TestCase):
    def test_prose_streams_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "b.txt").write_text("A dog runs.", encoding="utf-8")
            streams = list(prose_file_streams([Path(tmp) / "b.txt"]))
        self.assertEqual(streams[0][0], "b")
        self.assertEqual([t for _, t in streams[0][1]], ["a", "dog", "runs", "."])

    def test_code_streams_jsonl_whole_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.json"
            p.write_text(json.dumps({"content": "let userName = 1;"}) + "\n", encoding="utf-8")
            streams = list(code_file_streams([p]))
        self.assertEqual([t for _, t in streams[0][1]], ["let", "username", "=", "1", ";"])


class UnionCensusTests(unittest.TestCase):
    def test_shared_symbol_merges_domain_specific_disjoint(self):
        tagged = [
            ("prose", [(0, "the"), (0, "dog"), (0, ".")]),
            ("code", [(0, "let"), (0, "x"), (0, ".")]),
        ]
        vocab = union_census(tagged)
        tokens = {row["token"]: row["count"] for row in vocab}
        self.assertEqual(tokens["."], 2)   # shared symbol: one entry, count 2
        self.assertEqual(tokens["the"], 1)
        self.assertEqual(tokens["let"], 1)
        self.assertNotIn("<unk>", tokens)  # census is raw; unk added by _build_lookup


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_mixed_corpus_streams.py -q`
Expected: collection ERROR, module not found

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/mixed_corpus_streams.py
"""
Union streams for the mixed-corpus (PG-19 prose + JS/TS code) ordering
experiment. Each corpus keeps its own tokenizer; token strings are pooled into
one vocabulary so shared symbols (punctuation, digits) merge while words and
identifiers stay disjoint.
"""

import json
from collections import defaultdict
from pathlib import Path

from scripts.code_stream import tokenize_code
from scripts.pg19_stream import tokenize_clauses
from scripts.phrase_vectors import build_vocab_from_stats


def prose_file_streams(paths):
    for path in paths:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        yield path.stem, tokenize_clauses(text)


def code_file_streams(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for row_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                content = json.loads(line).get("content")
                if not content:
                    continue
                yield str(row_index), tokenize_code(content, split_identifiers=False)


def union_census(tagged_stream_iters):
    stats = defaultdict(lambda: {"count": 0, "position_sum": 0.0})
    for _domain, stream in tagged_stream_iters:
        clause_tokens = []
        current = None
        for clause, token in stream:
            if current is not None and clause != current:
                _accumulate(stats, clause_tokens)
                clause_tokens = []
            current = clause
            clause_tokens.append(token)
        if clause_tokens:
            _accumulate(stats, clause_tokens)
    return build_vocab_from_stats(stats, min_count=1)


def _accumulate(stats, clause_tokens):
    denominator = max(len(clause_tokens) - 1, 1)
    for position, token in enumerate(clause_tokens):
        entry = stats[token]
        entry["count"] += 1
        entry["position_sum"] += position / denominator
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_mixed_corpus_streams.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/mixed_corpus_streams.py tests/test_mixed_corpus_streams.py
git commit -m "Add union streams + census for mixed-corpus ordering

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Experiment core + driver (`scripts/run_mixed_corpus_experiment.py`)

**Files:**
- Create: `scripts/run_mixed_corpus_experiment.py`
- Test: `tests/test_run_mixed_corpus_experiment.py`

**Interfaces:**
- Consumes: Task 1 functions; `_build_lookup`, `_stream_ids` (code_vocab_experiment); `PairCounter`, `optimize_order`, `_inverse_permutation` (reorder_phrase_vocab); `chain_length_histogram`, `summarize` (measure_chain_lengths); `collect_position_histograms`, `select_candidates`, `build_plan`, `apply_kway` (plan_kway_duplicates).
- Produces: `order_from(id_streams_fn, vocab_size, max_passes, ils_restarts, ils_generations, jobs) -> (positions ndarray, ascending_fraction float)`; `measure_domain(id_streams_fn, positions, max_chain_len, plan=None) -> dict` (summarize output, histogram popped; if plan given, streams are apply_kway'd first and positions must be over the plan's new vocab); `run_experiment(prose_train_fn, prose_eval_fn, code_train_fn, code_eval_fn, vocab_size, top_n, k_max, max_passes, ils_restarts, ils_generations, jobs, max_chain_len) -> report dict`. Each `*_fn` is a zero-arg callable returning a FRESH iterator of union-mapped `[(clause, id)]` streams.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_mixed_corpus_experiment.py
import unittest

from scripts.run_mixed_corpus_experiment import order_from, run_experiment


def _prose():  # ascending-friendly under identity: ids rise within clause
    return iter([[(0, 0), (0, 1), (0, 2)] for _ in range(20)])


def _code():   # different regularity: ids fall
    return iter([[(0, 2), (0, 1), (0, 0)] for _ in range(20)])


class OrderFromTests(unittest.TestCase):
    def test_order_from_returns_positions_and_fraction(self):
        positions, frac = order_from(_prose, vocab_size=3, max_passes=5,
                                     ils_restarts=1, ils_generations=1, jobs=1)
        self.assertEqual(sorted(positions.tolist()), [0, 1, 2])
        self.assertGreaterEqual(frac, 0.0)


class RunExperimentTests(unittest.TestCase):
    def test_report_has_bespoke_mixed_and_kway_per_domain(self):
        report = run_experiment(_prose, _prose, _code, _code, vocab_size=3,
                                top_n=3, k_max=4, max_passes=5, ils_restarts=1,
                                ils_generations=1, jobs=1, max_chain_len=9)
        for domain in ("prose", "code"):
            self.assertIn("bespoke", report[domain])
            self.assertIn("mixed", report[domain])
            self.assertIn("mixed_kway", report[domain])
            self.assertGreater(report[domain]["bespoke"]["chains"], 0)
        self.assertIn("kway_extra_slots", report)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_run_mixed_corpus_experiment.py -q`
Expected: collection ERROR, module not found

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/run_mixed_corpus_experiment.py
"""
Mixed-corpus ordering experiment: build prose-only / code-only / mixed orders in
one union vocab, measure each held-out domain under its bespoke vs the mixed
order, and test whether a mixed + k-way-duplication order recovers interference.
Analysis-only.

Example:
python -m scripts.run_mixed_corpus_experiment \
    --prose ../pg19/train --prose-eval ../pg19/validation \
    --code ../stack_smol_jsts_raw/data/javascript/data.json \
           ../stack_smol_jsts_raw/data/typescript/data.json \
    --out-dir mixed_out --min-count 25 --top-n 300 --k-max 6
"""

import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from scripts.code_vocab_experiment import _build_lookup, _stream_ids
from scripts.measure_chain_lengths import chain_length_histogram, summarize
from scripts.mixed_corpus_streams import code_file_streams, prose_file_streams, union_census
from scripts.plan_kway_duplicates import apply_kway, build_plan, collect_position_histograms, select_candidates
from scripts.reorder_phrase_vocab import PairCounter, _inverse_permutation, optimize_order


def order_from(id_streams_fn, vocab_size, max_passes, ils_restarts, ils_generations, jobs):
    counter = PairCounter(vocab_size=vocab_size, chunk_size=8_000_000)
    for stream in id_streams_fn():
        counter.add_stream(stream)
    codes, counts = counter.finalize()
    order, stats = optimize_order(codes, counts, vocab_size, np.arange(vocab_size, dtype=np.int64),
                                  max_passes=max_passes, ils_restarts=ils_restarts,
                                  ils_generations=ils_generations, ils_seed=1, jobs=jobs)
    frac = round(stats["ascending_after"] / stats["total_pairs"], 4) if stats["total_pairs"] else 0.0
    return _inverse_permutation(order), frac


def measure_domain(id_streams_fn, positions, max_chain_len, plan=None):
    histogram = {}
    for stream in id_streams_fn():
        remapped = apply_kway(stream, plan) if plan is not None else stream
        chain_length_histogram(remapped, positions, reset_on_clause=False,
                               histogram=histogram, max_chain_len=max_chain_len)
    result = summarize(histogram)
    result.pop("histogram", None)
    return result


def _both_fn(prose_fn, code_fn):
    return lambda: itertools.chain(prose_fn(), code_fn())


def run_experiment(prose_train_fn, prose_eval_fn, code_train_fn, code_eval_fn, vocab_size,
                   top_n, k_max, max_passes, ils_restarts, ils_generations, jobs, max_chain_len):
    kw = dict(max_passes=max_passes, ils_restarts=ils_restarts, ils_generations=ils_generations, jobs=jobs)
    prose_pos, prose_asc = order_from(prose_train_fn, vocab_size, **kw)
    code_pos, code_asc = order_from(code_train_fn, vocab_size, **kw)
    both_train_fn = _both_fn(prose_train_fn, code_train_fn)
    mixed_pos, mixed_asc = order_from(both_train_fn, vocab_size, **kw)

    # k-way plan from the union pair counts + union position histograms
    counter = PairCounter(vocab_size=vocab_size, chunk_size=8_000_000)
    for stream in both_train_fn():
        counter.add_stream(stream)
    codes, counts = counter.finalize()
    candidates = select_candidates(codes, counts, vocab_size, top_n)
    histograms = collect_position_histograms(both_train_fn(), vocab_size)
    plan = build_plan(candidates, histograms, vocab_size, k_max=k_max)
    new_vocab_size = plan["vocab_size_new"]
    kway_counter = PairCounter(vocab_size=new_vocab_size, chunk_size=8_000_000)
    for stream in both_train_fn():
        kway_counter.add_stream(apply_kway(stream, plan))
    kcodes, kcounts = kway_counter.finalize()
    korder, kstats = optimize_order(kcodes, kcounts, new_vocab_size, np.arange(new_vocab_size, dtype=np.int64),
                                    ils_seed=1, **kw)
    kway_pos = _inverse_permutation(korder)

    report = {
        "format": "mixed_corpus_experiment_v1",
        "vocab_size": vocab_size,
        "ascending": {"prose": prose_asc, "code": code_asc, "mixed": mixed_asc,
                      "mixed_kway": round(kstats["ascending_after"] / kstats["total_pairs"], 4) if kstats["total_pairs"] else 0.0},
        "kway_extra_slots": new_vocab_size - vocab_size,
    }
    for domain, bespoke_pos, eval_fn in (("prose", prose_pos, prose_eval_fn), ("code", code_pos, code_eval_fn)):
        report[domain] = {
            "bespoke": measure_domain(eval_fn, bespoke_pos, max_chain_len),
            "mixed": measure_domain(eval_fn, mixed_pos, max_chain_len),
            "mixed_kway": measure_domain(eval_fn, kway_pos, max_chain_len, plan=plan),
        }
    return report


def _split(paths, eval_fraction, seed):
    paths = sorted(str(p) for p in paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    cut = max(1, round(len(paths) * eval_fraction))
    return paths[cut:], paths[:cut]


def parse_args():
    parser = argparse.ArgumentParser(description="Mixed-corpus (PG-19 + JS/TS) vocabulary ordering experiment.")
    parser.add_argument("--prose", required=True, help="PG-19 train dir (.txt files).")
    parser.add_argument("--prose-eval", required=True, help="PG-19 eval dir (.txt files).")
    parser.add_argument("--code", nargs="+", required=True, help="JS/TS data.json JSONL file(s).")
    parser.add_argument("--code-eval-fraction", type=float, default=0.02)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--prose-limit", type=int, default=3000, help="Cap PG-19 train books for a fast run.")
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--max-passes", type=int, default=200)
    parser.add_argument("--ils-restarts", type=int, default=8)
    parser.add_argument("--ils-generations", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-chain-len", type=int, default=9)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    prose_all = sorted(str(p) for p in Path(args.prose).glob("*.txt"))[:args.prose_limit]
    prose_train, prose_eval = _split(prose_all, 0.05, args.seed)
    prose_eval_paths = sorted(str(p) for p in Path(args.prose_eval).glob("*.txt"))[:200] or prose_eval

    # code files are rows inside JSONL; split at the row level via a materialized list
    code_records = []
    for path in args.code:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            code_records += [line for line in (l.strip() for l in handle) if line]
    random.Random(args.seed).shuffle(code_records)
    cut = max(1, round(len(code_records) * args.code_eval_fraction))
    code_train_path = Path(args.out_dir) / "_code_train.jsonl"
    code_eval_path = Path(args.out_dir) / "_code_eval.jsonl"
    code_train_path.write_text("\n".join(code_records[cut:]) + "\n", encoding="utf-8")
    code_eval_path.write_text("\n".join(code_records[:cut]) + "\n", encoding="utf-8")

    def prose_tagged_train():
        for _, stream in prose_file_streams(prose_train):
            yield "prose", stream

    def code_tagged_train():
        for _, stream in code_file_streams([code_train_path]):
            yield "code", stream

    census_vocab = union_census(itertools.chain(prose_tagged_train(), code_tagged_train()))
    lookup, kept = _build_lookup(census_vocab, args.min_count)
    vocab_size = len(kept)
    unk = vocab_size - 1
    print(f"union vocab (min_count={args.min_count}): {vocab_size:,}", flush=True)

    def prose_train_fn():
        for _, stream in prose_file_streams(prose_train):
            yield _stream_ids(stream, lookup, unk)

    def prose_eval_fn():
        for _, stream in prose_file_streams(prose_eval_paths):
            yield _stream_ids(stream, lookup, unk)

    def code_train_fn():
        for _, stream in code_file_streams([code_train_path]):
            yield _stream_ids(stream, lookup, unk)

    def code_eval_fn():
        for _, stream in code_file_streams([code_eval_path]):
            yield _stream_ids(stream, lookup, unk)

    report = run_experiment(prose_train_fn, prose_eval_fn, code_train_fn, code_eval_fn, vocab_size,
                            args.top_n, args.k_max, args.max_passes, args.ils_restarts,
                            args.ils_generations, args.jobs, args.max_chain_len)
    report["prose_train_books"] = len(prose_train)
    report["code_train_files"] = len(code_records) - cut
    report["elapsed_seconds"] = round(time.time() - started, 1)
    with open(Path(args.out_dir) / "mixed_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_run_mixed_corpus_experiment.py -q`
Expected: 2 passed

- [ ] **Step 5: Run full suite and commit**

Run: `uv run --group dev python -m pytest tests -q -m "not slow"`
Expected: all pass

```bash
git add scripts/run_mixed_corpus_experiment.py tests/test_run_mixed_corpus_experiment.py
git commit -m "Add mixed-corpus ordering driver (bespoke/mixed/mixed+kway per domain)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Execute + report

**Files:** uses Task 2 CLI; workbook `outputs/nanochat-phrase-results/nanochat_phrase_results.xlsx`.

- [ ] **Step 1 (controller): launch in a visible Terminal**

Run: `osascript -e 'tell application "Terminal" to do script "cd \"/Users/jgrimm/dev/Local AI/nanochat\" && caffeinate -i uv run python -m scripts.run_mixed_corpus_experiment --prose ../pg19/train --prose-eval ../pg19/validation --code \"../stack_smol_jsts_raw/data/javascript/data.json\" \"../stack_smol_jsts_raw/data/typescript/data.json\" --out-dir mixed_out --min-count 25 --top-n 300 --k-max 6 2>&1 | tee mixed_corpus_experiment.log"'`

- [ ] **Step 2 (controller): record results**

Read `mixed_out/mixed_report.json`. Add a "Mixed corpus" block to the Code sheet (or a new sheet): for each domain {prose, code}, mean chain length + ascending % under bespoke / mixed / mixed+kway; interference cost = bespoke - mixed; k-way recovery = mixed_kway - mixed; extra slots. Note the shared-hub interference prediction and whether k-way recovered it. Update memory with the headline.

## Self-Review Notes

- Spec coverage: union token space + census (Task 1); three orders + per-domain bespoke-vs-mixed measurement + k-way arm (Task 2); execution + workbook (Task 3). Analysis-only, cap-9 cross-clause, single union id space — all present.
- Type consistency: `order_from(id_streams_fn, vocab_size, max_passes, ils_restarts, ils_generations, jobs)`, `measure_domain(id_streams_fn, positions, max_chain_len, plan=None)`, `run_experiment(...)` consistent between Task 2 definition and test; `*_fn` are fresh-iterator factories throughout (matches the kway run_arm pattern).
- Flagged inline: the k-way arm's `measure_domain` MUST pass `plan=plan` so eval streams are apply_kway'd and scored under `kway_pos` (new vocab space); the bespoke/mixed calls pass no plan (original vocab space). Both positions are over the correct id space by construction.
