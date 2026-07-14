# Hub-Token Duplication Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reserve ~100 vocab slots for duplicate copies of high-conflict hub tokens and measure the chain-compression payoff (no training).

**Architecture:** A planner selects the top-conflict tokens from saved pair counts and computes a renumbered vocab space with each copy adjacent to its parent; a streaming records transform (mirroring `split_quote_token`) assigns each occurrence to early/late copy by context class; the existing reorder and chain-measurement tools then run unchanged on the transformed data.

**Tech Stack:** numpy, existing nanochat scripts (`reorder_phrase_vocab`, `split_quote_token` patterns, `measure_chain_lengths`), unittest.

**Spec:** `docs/superpowers/specs/2026-07-13-hub-token-duplication-design.md`

## Global Constraints

- Run tests with `uv run --group dev python -m pytest tests/<file> -q` from `nanochat/`; unittest-style test classes.
- Streaming discipline: one story in memory at a time; the transform must rewrite ALL record representations of the same absolute token position consistently (same guarantee as `scripts/split_quote_token.py`).
- Assignment rule (spec, verbatim): occurrence of duplicated token t goes to t_late iff its predecessor in the canonical stream has OLD-index greater than t's OLD-index; otherwise t_early. First-in-story occurrences go to t_early.
- Selection (spec): loss(t) = sum over v of min(W[t][v], W[v][t]) from `phrase_quote_split_ils_out/pair_counts.npz`; exclude tokens with conflict share < 0.05; take top `--num-duplicates` (default 100).
- Copies keep the parent's surface token in records' `tokens` lists; the vocab's late-copy row gets token string parent+"~dup" (append further "~" on collision).
- Multi-element records keep positional alignment (do NOT sort indices — matches `split_quote_token`, not `remap_record_indices`).
- Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Data inputs that exist: `phrase_quote_split_ils_out/pair_counts.npz` (codes/counts/vocab_size=46327), `phrase_quote_split_ils_out/vocab.json`, `phrase_quote_split_out/phrase_index.jsonl.gz`.

**Key existing interfaces:**
- `scripts/split_quote_token.py`: `_element_positions(record)` yields `(slot, absolute_position)`; story-grouping generator pattern with `stats` dict.
- `scripts/train_phrase_vectors.iter_records(path)`; `scripts/reorder_phrase_vocab.story_index_streams`.
- `np.load(pair_counts.npz)` keys: `codes` (sorted int64 u*V+v), `counts`, `vocab_size`, `stories`.

---

### Task 1: Duplicate planner (`scripts/plan_token_duplicates.py`)

**Files:**
- Create: `scripts/plan_token_duplicates.py`
- Test: `tests/test_plan_token_duplicates.py`

**Interfaces:**
- Produces: `conflict_losses(codes, counts, vocab_size) -> (loss ndarray[V], directed ndarray[V])`; `select_duplicates(codes, counts, vocab_size, num_duplicates=100, min_share=0.05) -> sorted list[int]`; `renumber_array(vocab_size, parents_sorted) -> ndarray[V]` (old->new for non-late ids; parent maps to its early slot); `build_plan(pair_counts_path, vocab_path, num_duplicates, min_share) -> dict` and CLI writing `duplicates_plan.json` with fields `format="duplicates_plan_v1"`, `vocab_size_old`, `vocab_size_new`, `parents: [{old_index, token, early_new_index, late_new_index, conflict_mass, conflict_share}]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plan_token_duplicates.py
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.plan_token_duplicates import (
    build_plan,
    conflict_losses,
    renumber_array,
    select_duplicates,
)


def _pair_arrays(vocab_size, pair_counts):
    codes = np.array(sorted(left * vocab_size + right for left, right in pair_counts), dtype=np.int64)
    counts = np.array([pair_counts[(int(c) // vocab_size, int(c) % vocab_size)] for c in codes], dtype=np.int64)
    return codes, counts


class ConflictLossTests(unittest.TestCase):
    def test_loss_is_min_mass_per_conflicted_pair(self):
        vocab_size = 4
        codes, counts = _pair_arrays(vocab_size, {(0, 1): 5, (1, 0): 3, (2, 3): 7})
        loss, directed = conflict_losses(codes, counts, vocab_size)
        # pair {0,1} conflicted with min 3 -> both endpoints get 3; pair (2,3) one-directional -> 0
        self.assertEqual(loss.tolist(), [3.0, 3.0, 0.0, 0.0])
        # directed mass: token 0: out 5 + in 3 = 8; token 2: out 7; token 3: in 7
        self.assertEqual(directed.tolist(), [8.0, 8.0, 7.0, 7.0])

    def test_select_excludes_low_share_and_ranks_by_loss(self):
        vocab_size = 6
        # token 0<->1 heavy conflict; token 2->3 huge one-way plus tiny conflict (share < 5%)
        codes, counts = _pair_arrays(vocab_size, {
            (0, 1): 10, (1, 0): 8,
            (2, 3): 1000, (3, 2): 4,
            (4, 5): 6, (5, 4): 6,
        })
        parents = select_duplicates(codes, counts, vocab_size, num_duplicates=3, min_share=0.05)
        # loss: t0=8, t1=8, t2=4(share 4/1004<0.05 excluded), t3 likewise, t4=6, t5=6
        self.assertEqual(parents, [0, 1, 4])  # top-3 by loss among eligible, sorted


class RenumberTests(unittest.TestCase):
    def test_renumber_inserts_copy_slots_after_parents(self):
        new = renumber_array(6, [1, 3])
        # old:      0  1  2  3  4  5
        # parents:     ^     ^
        # new ids:  0  1  3  4  6  7   (late copies occupy 2 and 5)
        self.assertEqual(new.tolist(), [0, 1, 3, 4, 6, 7])

    def test_renumber_no_parents_is_identity(self):
        self.assertEqual(renumber_array(4, []).tolist(), [0, 1, 2, 3])


class BuildPlanTests(unittest.TestCase):
    def test_build_plan_end_to_end(self):
        vocab_size = 6
        codes, counts = _pair_arrays(vocab_size, {(0, 1): 10, (1, 0): 8, (4, 5): 6, (5, 4): 6})
        vocab = [{"token": f"tok{i}", "index": i, "count": 100} for i in range(vocab_size)]
        with tempfile.TemporaryDirectory() as tmpdir:
            pc_path = Path(tmpdir) / "pair_counts.npz"
            np.savez(pc_path, codes=codes, counts=counts, vocab_size=np.int64(vocab_size), stories=np.int64(1))
            vocab_path = Path(tmpdir) / "vocab.json"
            vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
            plan = build_plan(pc_path, vocab_path, num_duplicates=2, min_share=0.05)
        self.assertEqual(plan["format"], "duplicates_plan_v1")
        self.assertEqual(plan["vocab_size_old"], 6)
        self.assertEqual(plan["vocab_size_new"], 8)
        self.assertEqual([p["old_index"] for p in plan["parents"]], [0, 1])
        first = plan["parents"][0]
        self.assertEqual(first["token"], "tok0")
        self.assertEqual(first["early_new_index"], 0)
        self.assertEqual(first["late_new_index"], 1)
        second = plan["parents"][1]
        self.assertEqual(second["early_new_index"], 2)
        self.assertEqual(second["late_new_index"], 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_plan_token_duplicates.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'scripts.plan_token_duplicates'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/plan_token_duplicates.py
"""
Select hub tokens for duplication and plan the renumbered vocab space.

loss(t) = sum over partners v of min(W[t][v], W[v][t]) — the irreducible
conflict mass t participates in under any single-position vocab. The top-N
eligible tokens get a second ("late") copy inserted immediately after their
parent in index order.

Example:
python -m scripts.plan_token_duplicates \
    --pair-counts phrase_quote_split_ils_out/pair_counts.npz \
    --vocab phrase_quote_split_ils_out/vocab.json \
    --out duplication_out/duplicates_plan.json
"""

import argparse
import json
import os

import numpy as np


def conflict_losses(codes, counts, vocab_size):
    left = codes // vocab_size
    right = codes % vocab_size
    reverse_codes = right * vocab_size + left
    slots = np.minimum(np.searchsorted(codes, reverse_codes), codes.size - 1)
    has_reverse = codes[slots] == reverse_codes
    reverse_counts = np.where(has_reverse, counts[slots], 0)
    mins = np.minimum(counts, reverse_counts)
    loss = np.bincount(left, weights=mins.astype(np.float64), minlength=vocab_size)
    directed = (
        np.bincount(left, weights=counts.astype(np.float64), minlength=vocab_size)
        + np.bincount(right, weights=counts.astype(np.float64), minlength=vocab_size)
    )
    return loss, directed


def select_duplicates(codes, counts, vocab_size, num_duplicates=100, min_share=0.05):
    loss, directed = conflict_losses(codes, counts, vocab_size)
    share = loss / np.maximum(directed, 1.0)
    ranked = np.argsort(-loss, kind="stable")
    parents = [int(t) for t in ranked if loss[t] > 0 and share[t] >= min_share][:num_duplicates]
    return sorted(parents)


def renumber_array(vocab_size, parents_sorted):
    parents = np.asarray(sorted(parents_sorted), dtype=np.int64)
    old = np.arange(vocab_size, dtype=np.int64)
    return old + np.searchsorted(parents, old, side="left")


def build_plan(pair_counts_path, vocab_path, num_duplicates=100, min_share=0.05):
    saved = np.load(pair_counts_path)
    codes = saved["codes"]
    counts = saved["counts"]
    vocab_size = int(saved["vocab_size"])
    with open(vocab_path, "r", encoding="utf-8") as file:
        tokens = {int(row["index"]): row["token"] for row in json.load(file)}
    parents = select_duplicates(codes, counts, vocab_size, num_duplicates=num_duplicates, min_share=min_share)
    loss, directed = conflict_losses(codes, counts, vocab_size)
    share = loss / np.maximum(directed, 1.0)
    renumber = renumber_array(vocab_size, parents)
    return {
        "format": "duplicates_plan_v1",
        "pair_counts": str(pair_counts_path),
        "vocab": str(vocab_path),
        "vocab_size_old": vocab_size,
        "vocab_size_new": vocab_size + len(parents),
        "num_duplicates": len(parents),
        "min_share": min_share,
        "parents": [
            {
                "old_index": p,
                "token": tokens.get(p, ""),
                "early_new_index": int(renumber[p]),
                "late_new_index": int(renumber[p]) + 1,
                "conflict_mass": float(loss[p]),
                "conflict_share": round(float(share[p]), 4),
            }
            for p in parents
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Plan hub-token duplicates from saved pair counts.")
    parser.add_argument("--pair-counts", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--num-duplicates", type=int, default=100)
    parser.add_argument("--min-share", type=float, default=0.05)
    parser.add_argument("--out", required=True, help="Path for duplicates_plan.json")
    return parser.parse_args()


def main():
    args = parse_args()
    plan = build_plan(args.pair_counts, args.vocab, num_duplicates=args.num_duplicates, min_share=args.min_share)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(plan, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps({"num_duplicates": plan["num_duplicates"], "vocab_size_new": plan["vocab_size_new"],
                      "top5": [(p["token"], p["conflict_mass"]) for p in plan["parents"][:5]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_plan_token_duplicates.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/plan_token_duplicates.py tests/test_plan_token_duplicates.py
git commit -m "Add hub-token duplicate planner (conflict-mass selection + renumbering)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Records transform (`scripts/duplicate_hub_tokens.py`)

**Files:**
- Create: `scripts/duplicate_hub_tokens.py`
- Test: `tests/test_duplicate_hub_tokens.py`

**Interfaces:**
- Consumes: `renumber_array` (Task 1); `_element_positions` semantics from `scripts/split_quote_token.py` (copy the 8-line helper rather than importing a private).
- Produces: `transform_story(story_records, renumber, parent_set, late_of, stats) -> list[record]`; `duplicate_records(records, plan, stats=None, progress_every=0)` generator; `run_transform(records_path, vocab_path, plan_path, out_dir, compresslevel=4, progress_every=100000) -> report dict`; CLI. Output dir contains `phrase_index.jsonl.gz` (renumbered records), `vocab.json` (vocab_size_new rows: parents keep token at early index with early-count, late rows token parent+"~dup" with late-count, others renumbered with counts unchanged), `copy_map.json` (late_new_index -> early_new_index as JSON dict of str->int), `transform_report.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_duplicate_hub_tokens.py
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.duplicate_hub_tokens import duplicate_records, run_transform
from scripts.plan_token_duplicates import renumber_array
from scripts.train_phrase_vectors import iter_records


def _single(story_id, phrase_id, index, token_pos, start=0, split="train", label="punctuation"):
    return {
        "split": split, "story_id": story_id, "phrase_id": phrase_id, "label": label,
        "start": start, "end": start + 5, "record_type": "single",
        "indices": [index], "tokens": [f"tok{index}"], "token_pos": token_pos,
    }


def _plan(vocab_size, parents):
    renumber = renumber_array(vocab_size, parents)
    return {
        "format": "duplicates_plan_v1",
        "vocab_size_old": vocab_size,
        "vocab_size_new": vocab_size + len(parents),
        "num_duplicates": len(parents),
        "parents": [
            {"old_index": p, "token": f"tok{p}", "early_new_index": int(renumber[p]),
             "late_new_index": int(renumber[p]) + 1, "conflict_mass": 1.0, "conflict_share": 0.5}
            for p in sorted(parents)
        ],
    }


class DuplicateRecordsTests(unittest.TestCase):
    def test_assignment_by_predecessor_old_index(self):
        # parent token 2; stream old-indices: 5, 2, 1, 2  ->  first 2 has pred 5>2 => late; second 2 has pred 1<2 => early
        records = [
            _single(0, 0, 5, token_pos=0),
            _single(0, 0, 2, token_pos=1),
            _single(0, 0, 1, token_pos=2),
            _single(0, 0, 2, token_pos=3),
        ]
        plan = _plan(6, [2])
        out = list(duplicate_records(iter(records), plan))
        # renumber with parent 2: old->new = [0,1,2,4,5,6]; late copy id = 3
        self.assertEqual([r["indices"][0] for r in out], [6, 3, 1, 2])

    def test_first_in_story_is_early(self):
        records = [_single(0, 0, 2, token_pos=0), _single(0, 0, 4, token_pos=1)]
        plan = _plan(6, [2])
        out = list(duplicate_records(iter(records), plan))
        self.assertEqual(out[0]["indices"], [2])  # early copy keeps parent slot

    def test_all_representations_of_a_position_rewritten_consistently(self):
        records = [
            _single(0, 0, 5, token_pos=0),
            _single(0, 0, 2, token_pos=1),
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "root_clause", "start": 0,
             "end": 2, "record_type": "single", "indices": [2], "tokens": ["tok2"], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0,
             "end": 2, "record_type": "packed", "indices": [5, 2], "tokens": ["tok5", "tok2"]},
        ]
        plan = _plan(6, [2])
        out = list(duplicate_records(iter(records), plan))
        # canonical: pos0=5, pos1=2 (pred 5>2 => late id 3); every representation of pos1 gets 3
        self.assertEqual(out[1]["indices"], [3])
        self.assertEqual(out[2]["indices"], [3])
        self.assertEqual(out[3]["indices"], [6, 3])  # packed keeps positional alignment, no sort
        self.assertEqual(out[3]["tokens"], ["tok5", "tok2"])  # surface tokens unchanged

    def test_parity_resets_per_story_and_stats(self):
        records = [
            _single(0, 0, 5, token_pos=0), _single(0, 0, 2, token_pos=1),
            _single(1, 0, 2, token_pos=0),
        ]
        plan = _plan(6, [2])
        stats = {"stories": 0, "early": 0, "late": 0}
        out = list(duplicate_records(iter(records), plan, stats=stats))
        self.assertEqual(out[2]["indices"], [2])  # first-in-story -> early
        self.assertEqual(stats, {"stories": 2, "early": 1, "late": 1})


class RunTransformTests(unittest.TestCase):
    def test_end_to_end_writes_records_vocab_copy_map(self):
        vocab = [{"token": f"tok{i}", "index": i, "count": 10 + i, "avg_position": 0.5} for i in range(6)]
        records = [
            _single(0, 0, 5, token_pos=0), _single(0, 0, 2, token_pos=1),
            _single(1, 0, 2, token_pos=0),
        ]
        plan = _plan(6, [2])
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
            (tmp / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            with gzip.open(tmp / "records.jsonl.gz", "wt", encoding="utf-8") as file:
                for record in records:
                    file.write(json.dumps(record) + "\n")
            report = run_transform(tmp / "records.jsonl.gz", tmp / "vocab.json", tmp / "plan.json", tmp / "out")
            new_vocab = json.loads((tmp / "out" / "vocab.json").read_text(encoding="utf-8"))
            copy_map = json.loads((tmp / "out" / "copy_map.json").read_text(encoding="utf-8"))
            new_records = list(iter_records(tmp / "out" / "phrase_index.jsonl.gz"))

        self.assertEqual(len(new_vocab), 7)
        by_index = {row["index"]: row for row in new_vocab}
        self.assertEqual(by_index[2]["token"], "tok2")
        self.assertEqual(by_index[2]["count"], 1)   # one early occurrence
        self.assertEqual(by_index[3]["token"], "tok2~dup")
        self.assertEqual(by_index[3]["count"], 1)   # one late occurrence
        self.assertEqual(by_index[4]["token"], "tok3")  # renumbered non-parent, count unchanged
        self.assertEqual(by_index[4]["count"], 13)
        self.assertEqual(copy_map, {"3": 2})
        self.assertEqual([r["indices"][0] for r in new_records], [6, 3, 2])
        self.assertEqual(report["stories"], 2)
        self.assertEqual(report["late_occurrences"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --group dev python -m pytest tests/test_duplicate_hub_tokens.py -q`
Expected: collection ERROR, module not found

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/duplicate_hub_tokens.py
"""
Rewrite phrase records so high-conflict hub tokens split into early/late copies.

Assignment (per spec): an occurrence goes to the LATE copy iff its predecessor
in the story's canonical stream has an OLD index greater than the token's own
OLD index (it would break a chain under the current order); first-in-story and
ascending-context occurrences keep the parent (early) slot. All record
representations of the same absolute token position are rewritten consistently,
and every index is renumbered into the extended vocab space.

Example:
python -m scripts.duplicate_hub_tokens \
    --records phrase_quote_split_out/phrase_index.jsonl.gz \
    --vocab phrase_quote_split_ils_out/vocab.json \
    --plan duplication_out/duplicates_plan.json \
    --out-dir phrase_dup100_out
"""

import argparse
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.plan_token_duplicates import renumber_array
from scripts.train_phrase_vectors import iter_records


def _element_positions(record):
    start = int(record.get("start", 0))
    if record.get("record_type") == "single":
        yield 0, start + int(record.get("token_pos", 0))
        return
    for slot in range(len(record.get("indices", []))):
        yield slot, start + slot


def transform_story(story_records, renumber, parent_set, late_of, stats):
    positions = {}
    for record in story_records:
        if record.get("record_type") != "single" or record.get("label") != "punctuation" or not record.get("indices"):
            continue
        position = int(record.get("start", 0)) + int(record.get("token_pos", 0))
        positions.setdefault(position, int(record["indices"][0]))

    late_positions = set()
    previous = None
    for position in sorted(positions):
        old = positions[position]
        if old in parent_set:
            if previous is not None and previous > old:
                late_positions.add(position)
                stats["late"] += 1
            else:
                stats["early"] += 1
        previous = old

    out = []
    for record in story_records:
        indices = record.get("indices", [])
        if not indices:
            out.append(record)
            continue
        new_indices = list(indices)
        for slot, position in _element_positions(record):
            old = int(indices[slot])
            if old in parent_set and position in late_positions:
                new_indices[slot] = late_of[old]
            else:
                new_indices[slot] = int(renumber[old])
        new_record = dict(record)
        new_record["indices"] = new_indices
        out.append(new_record)
    return out


def duplicate_records(records, plan, stats=None, progress_every=0):
    stats = stats if stats is not None else {"stories": 0, "early": 0, "late": 0}
    parents = [p["old_index"] for p in plan["parents"]]
    parent_set = set(parents)
    late_of = {p["old_index"]: p["late_new_index"] for p in plan["parents"]}
    renumber = renumber_array(plan["vocab_size_old"], parents)

    current_key = None
    story_records = []

    def flush():
        stats["stories"] += 1
        if progress_every > 0 and stats["stories"] % progress_every == 0:
            print(f"duplicated {stats['stories']} stories (early={stats['early']} late={stats['late']})", flush=True)
        yield from transform_story(story_records, renumber, parent_set, late_of, stats)

    for record in records:
        key = (record["split"], int(record["story_id"]))
        if current_key is not None and key != current_key:
            yield from flush()
            story_records = []
        current_key = key
        story_records.append(record)
    if story_records:
        yield from flush()


def _extended_vocab(vocab_rows, plan, early_counts, late_counts):
    parents = {p["old_index"]: p for p in plan["parents"]}
    existing_tokens = {row["token"] for row in vocab_rows}
    renumber = renumber_array(plan["vocab_size_old"], sorted(parents))
    new_rows = []
    for row in sorted(vocab_rows, key=lambda r: int(r["index"])):
        old = int(row["index"])
        base = dict(row)
        base["index"] = int(renumber[old])
        if old in parents:
            base["count"] = early_counts.get(old, 0)
            new_rows.append(base)
            dup_token = row["token"] + "~dup"
            while dup_token in existing_tokens:
                dup_token += "~"
            existing_tokens.add(dup_token)
            new_rows.append({
                "token": dup_token,
                "index": parents[old]["late_new_index"],
                "count": late_counts.get(old, 0),
                "avg_position": row.get("avg_position", 0.0),
            })
        else:
            new_rows.append(base)
    return sorted(new_rows, key=lambda r: r["index"])


def run_transform(records_path, vocab_path, plan_path, out_dir, compresslevel=4, progress_every=100000):
    started_at = time.time()
    with open(plan_path, "r", encoding="utf-8") as file:
        plan = json.load(file)
    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab_rows = json.load(file)
    if len(vocab_rows) != plan["vocab_size_old"]:
        raise SystemExit(f"vocab size {len(vocab_rows)} != plan vocab_size_old {plan['vocab_size_old']}")

    os.makedirs(out_dir, exist_ok=True)
    out_dir = Path(out_dir)
    parent_set = {p["old_index"] for p in plan["parents"]}
    late_of = {p["old_index"]: p["late_new_index"] for p in plan["parents"]}
    early_of = {p["old_index"]: p["early_new_index"] for p in plan["parents"]}

    # per-parent occurrence tallies for vocab counts
    early_counts = {}
    late_counts = {}

    class TallyStats(dict):
        pass

    stats = {"stories": 0, "early": 0, "late": 0}

    # wrap transform to tally per-parent counts from the canonical assignment:
    # simplest correct accounting — count singles (canonical positions) per new id.
    def tallying(records_iter):
        for record in records_iter:
            if record.get("record_type") == "single" and record.get("label") == "punctuation" and record.get("indices"):
                new_id = int(record["indices"][0])
                for old, late in late_of.items():
                    if new_id == late:
                        late_counts[old] = late_counts.get(old, 0) + 1
                        break
                else:
                    for old, early in early_of.items():
                        if new_id == early:
                            early_counts[old] = early_counts.get(old, 0) + 1
                            break
            yield record

    out_records = out_dir / "phrase_index.jsonl.gz"
    with gzip.open(out_records, "wt", encoding="utf-8", compresslevel=compresslevel) as file:
        for record in tallying(duplicate_records(iter_records(records_path), plan, stats=stats, progress_every=progress_every)):
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")

    new_vocab = _extended_vocab(vocab_rows, plan, early_counts, late_counts)
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as file:
        json.dump(new_vocab, file, ensure_ascii=False)
        file.write("\n")
    copy_map = {str(p["late_new_index"]): p["early_new_index"] for p in plan["parents"]}
    with open(out_dir / "copy_map.json", "w", encoding="utf-8") as file:
        json.dump(copy_map, file)
        file.write("\n")

    report = {
        "format": "hub_duplication_report_v1",
        "records": str(records_path),
        "plan": str(plan_path),
        "vocab_size_new": plan["vocab_size_new"],
        "stories": stats["stories"],
        "early_occurrences": stats["early"],
        "late_occurrences": stats["late"],
        "elapsed_seconds": time.time() - started_at,
    }
    with open(out_dir / "transform_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Apply hub-token duplication to phrase records.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--compresslevel", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_transform(args.records, args.vocab, args.plan, args.out_dir,
                           compresslevel=args.compresslevel, progress_every=args.progress_every)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

NOTE for the implementer: the per-record `late_of`/`early_of` reverse lookups in
`tallying` are O(P) per single record (P=100). At 1.6B records that is too slow.
Replace the two inner loops with precomputed dicts before committing:
`late_to_old = {p["late_new_index"]: p["old_index"] for p in plan["parents"]}` and
`early_to_old = {p["early_new_index"]: p["old_index"] for p in plan["parents"]}`,
then `old = late_to_old.get(new_id)` / `early_to_old.get(new_id)` — O(1) per record.
The test asserts only the resulting counts, so both implementations pass; ship the O(1) one.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --group dev python -m pytest tests/test_duplicate_hub_tokens.py -q`
Expected: 5 passed

- [ ] **Step 5: Run full suite and commit**

Run: `uv run --group dev python -m pytest tests -q -m "not slow"`
Expected: all pass (189+)

```bash
git add scripts/duplicate_hub_tokens.py tests/test_duplicate_hub_tokens.py
git commit -m "Add hub-token duplication records transform

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Stage A runner + execution

**Files:**
- Create: `scripts/run_duplication_stage_a.sh`

- [ ] **Step 1: Write the runner**

```bash
#!/bin/bash
# Stage A: plan duplicates -> transform records -> reorder -> measure chains.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. plan duplicates (minutes) ==="
uv run python -m scripts.plan_token_duplicates \
    --pair-counts phrase_quote_split_ils_out/pair_counts.npz \
    --vocab phrase_quote_split_ils_out/vocab.json \
    --num-duplicates 100 \
    --out phrase_dup100_out/duplicates_plan.json

echo "=== 2. transform records (~3h) ==="
uv run python -m scripts.duplicate_hub_tokens \
    --records phrase_quote_split_out/phrase_index.jsonl.gz \
    --vocab phrase_quote_split_ils_out/vocab.json \
    --plan phrase_dup100_out/duplicates_plan.json \
    --out-dir phrase_dup100_out

echo "=== 3. reorder + ILS (~1.5h) ==="
uv run python -m scripts.reorder_phrase_vocab \
    --vocab phrase_dup100_out/vocab.json \
    --records phrase_dup100_out/phrase_index.jsonl.gz \
    --out-dir phrase_dup100_ils_out \
    --max-passes 200 --ils-restarts 8 --ils-generations 8 --ils-seed 1 --jobs 8 \
    --progress-every 200000

echo "=== 4. measure chains (cross-clause, 90k-story sample) ==="
uv run python -m scripts.measure_chain_lengths \
    --records phrase_dup100_out/phrase_index.jsonl.gz \
    --map dup100=phrase_dup100_ils_out/old_to_new.json \
    --no-clause-reset --max-records 70000000

echo "=== STAGE A DONE — baseline: 5,069,958 chains / 8.1% len-1; gate: >= 8% chain reduction ==="
```

- [ ] **Step 2: chmod, syntax-check, commit**

```bash
chmod +x scripts/run_duplication_stage_a.sh && bash -n scripts/run_duplication_stage_a.sh
git add scripts/run_duplication_stage_a.sh
git commit -m "Add Stage A runner for hub-token duplication

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 3 (controller executes): launch in Terminal**

Run: `osascript -e 'tell application "Terminal" to do script "cd \"/Users/jgrimm/dev/Local AI/nanochat\" && ./scripts/run_duplication_stage_a.sh 2>&1 | tee duplication_stage_a.log"'`

- [ ] **Step 4 (controller): record results + gate verdict**

Extend the workbook Vocab Reorder ladder with the duplication row (chains,
mean length, len-1 share, ascending fraction, new ceiling) and the gate
verdict (>= 8% chain reduction vs 5,069,958). Update the memory file. If the
gate passes, Stage B gets its own plan.

## Self-Review Notes

- Spec coverage: selection metric + exclusion (Task 1), renumbering adjacent
  to parents (Task 1), context-class assignment + cross-representation
  consistency + positional alignment (Task 2), pipeline & gate (Task 3).
- Type consistency: `renumber_array(vocab_size, parents_sorted)` used
  identically in Tasks 1-2; plan JSON fields match between builder and
  transform consumer.
- Known inefficiency flagged inline (tallying reverse lookup) with the exact
  required replacement.
