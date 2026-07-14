"""
Rewrite phrase records so high-conflict hub tokens split into early/late copies.

Assignment (per spec): an occurrence goes to the LATE copy iff its predecessor
in the story's canonical stream has a CURRENT-ORDER position (i.e. its ILS
position, from phrase_quote_split_ils_out/old_to_new.json) greater than the
token's own current-order position (it would break a chain under the current
order); first-in-story and ascending-context occurrences keep the parent
(early) slot. Records and the pair-counts plan live in ORIGINAL (pre-ILS)
vocab id space throughout; only the assignment comparison itself looks up
ILS positions. All record representations of the same absolute token position
are rewritten consistently, and every index is renumbered into the extended
vocab space.

Example:
python -m scripts.duplicate_hub_tokens \
    --records phrase_quote_split_out/phrase_index.jsonl.gz \
    --vocab phrase_quote_split_out/vocab.json \
    --plan duplication_out/duplicates_plan.json \
    --ils-map phrase_quote_split_ils_out/old_to_new.json \
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


def transform_story(story_records, renumber, parent_set, late_of, ils_positions, stats):
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
            if previous is not None and ils_positions[previous] > ils_positions[old]:
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


def duplicate_records(records, plan, ils_positions, stats=None, progress_every=0):
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
        yield from transform_story(story_records, renumber, parent_set, late_of, ils_positions, stats)

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


def run_transform(records_path, vocab_path, plan_path, ils_map_path, out_dir, compresslevel=4, progress_every=100000):
    started_at = time.time()
    with open(plan_path, "r", encoding="utf-8") as file:
        plan = json.load(file)
    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab_rows = json.load(file)
    if len(vocab_rows) != plan["vocab_size_old"]:
        raise SystemExit(f"vocab size {len(vocab_rows)} != plan vocab_size_old {plan['vocab_size_old']}")
    # Guard against feeding a vocab from a different index space than the plan
    # (e.g. the ILS-reordered vocab, whose "index" is a post-reorder position):
    # every plan parent's token must match the vocab row at its old_index.
    token_by_index = {int(row["index"]): row["token"] for row in vocab_rows}
    for parent in plan["parents"]:
        old = int(parent["old_index"])
        if token_by_index.get(old) != parent["token"]:
            raise SystemExit(
                "index-space mismatch between plan and vocab: plan parent old_index "
                f"{old} has token {parent['token']!r} but vocab row {old} is "
                f"{token_by_index.get(old)!r} — the plan and --vocab must both use "
                "ORIGINAL (pre-ILS) indices"
            )
    with open(ils_map_path, "r", encoding="utf-8") as file:
        ils_positions = json.load(file)
    if len(ils_positions) != plan["vocab_size_old"]:
        raise SystemExit(f"ils map size {len(ils_positions)} != plan vocab_size_old {plan['vocab_size_old']}")

    os.makedirs(out_dir, exist_ok=True)
    out_dir = Path(out_dir)
    late_of = {p["old_index"]: p["late_new_index"] for p in plan["parents"]}
    early_of = {p["old_index"]: p["early_new_index"] for p in plan["parents"]}

    # per-parent occurrence tallies for vocab counts. Precompute O(1) reverse
    # lookups (new-index -> old parent index) instead of scanning the parent
    # list per record: this runs over up to 1.6B records in production.
    late_to_old = {p["late_new_index"]: p["old_index"] for p in plan["parents"]}
    early_to_old = {p["early_new_index"]: p["old_index"] for p in plan["parents"]}

    early_counts = {}
    late_counts = {}

    stats = {"stories": 0, "early": 0, "late": 0}

    def tallying(records_iter):
        for record in records_iter:
            if record.get("record_type") == "single" and record.get("label") == "punctuation" and record.get("indices"):
                new_id = int(record["indices"][0])
                old = late_to_old.get(new_id)
                if old is not None:
                    late_counts[old] = late_counts.get(old, 0) + 1
                else:
                    old = early_to_old.get(new_id)
                    if old is not None:
                        early_counts[old] = early_counts.get(old, 0) + 1
            yield record

    out_records = out_dir / "phrase_index.jsonl.gz"
    with gzip.open(out_records, "wt", encoding="utf-8", compresslevel=compresslevel) as file:
        for record in tallying(duplicate_records(
            iter_records(records_path), plan, ils_positions, stats=stats, progress_every=progress_every
        )):
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
    parser.add_argument("--ils-map", required=True,
                         help="old_to_new.json (list: original id -> current/ILS-order position)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--compresslevel", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_transform(args.records, args.vocab, args.plan, args.ils_map, args.out_dir,
                           compresslevel=args.compresslevel, progress_every=args.progress_every)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
