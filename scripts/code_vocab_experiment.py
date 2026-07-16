"""
JS/TS vocab-ordering transfer experiment (the-stack-smol). For each tokenization
variant (whole identifiers vs split identifiers): census, SCC census, ILS order
built from a book sample, and chain measurement on held-out files. Mirrors
scripts/pg19_order_experiment.py but sources streams from scripts/code_stream.

Example:
python -m scripts.code_vocab_experiment \
    --train ../stack_smol_jsts_raw/data/javascript/data.json \
            ../stack_smol_jsts_raw/data/typescript/data.json \
    --out-dir code_vocab_out --min-count 25 --sample-fraction 1.0 --eval-fraction 0.02
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.code_stream import file_streams
from scripts.measure_chain_lengths import chain_length_histogram, summarize
from scripts.phrase_vectors import build_vocab_from_stats
from scripts.reorder_phrase_vocab import (
    PairCounter,
    _inverse_permutation,
    _net_edges,
    _strongly_connected_components,
    optimize_order,
)

UNK = "<unk>"


def census_pass(paths, split_identifiers, min_counts=(1, 5, 25), caps=(8192, 32768)):
    stats = defaultdict(lambda: {"count": 0, "position_sum": 0.0})
    tokens = 0
    clauses = 0
    for _, stream in file_streams(paths, split_identifiers=split_identifiers):
        clause_tokens = []
        current = None
        for clause, token in stream:
            if current is not None and clause != current:
                _accumulate(stats, clause_tokens)
                clauses += 1
                clause_tokens = []
            current = clause
            clause_tokens.append(token)
            tokens += 1
        if clause_tokens:
            _accumulate(stats, clause_tokens)
            clauses += 1
    counts = sorted((entry["count"] for entry in stats.values()), reverse=True)
    vocab = build_vocab_from_stats(stats, min_count=1)
    return {
        "vocab": vocab,
        "counts": {row["token"]: row["count"] for row in vocab},
        "tokens": tokens,
        "clauses": clauses,
        "vocab_size": len(vocab),
        "vocab_size_by_min_count": {str(t): sum(1 for c in counts if c >= t) for t in min_counts},
        "coverage_by_cap": {str(cap): round(sum(counts[:cap]) / tokens, 6) if tokens else 0.0 for cap in caps},
    }


def _accumulate(stats, clause_tokens):
    denominator = max(len(clause_tokens) - 1, 1)
    for position, token in enumerate(clause_tokens):
        entry = stats[token]
        entry["count"] += 1
        entry["position_sum"] += position / denominator


def _build_lookup(census_vocab, min_count):
    kept_rows = [row for row in sorted(census_vocab, key=lambda r: int(r["index"])) if int(row["count"]) >= min_count]
    kept = [row["token"] for row in kept_rows] + [UNK]
    lookup = {token: position for position, token in enumerate(kept[:-1])}
    return lookup, kept


def _stream_ids(stream, lookup, unk_id):
    return [(clause, lookup.get(token, unk_id)) for clause, token in stream]


def _scc_stats(codes, counts, vocab_size):
    heads, tails, nets = _net_edges(codes, counts, vocab_size)
    component, num_components = _strongly_connected_components(heads, tails, vocab_size)
    sizes = np.bincount(component, minlength=num_components)
    active = np.unique(np.concatenate([heads, tails])).size if heads.size else 0
    total = int(counts.sum())
    left, right = codes // vocab_size, codes % vocab_size
    reverse = right * vocab_size + left
    slots = np.minimum(np.searchsorted(codes, reverse), max(codes.size - 1, 0))
    has_reverse = codes.size > 0 and (codes[slots] == reverse)
    reverse_counts = np.where(has_reverse, counts[slots], 0) if codes.size else np.zeros(0, dtype=np.int64)
    irreducible = int(np.minimum(counts, reverse_counts).sum()) // 2 if codes.size else 0
    return {
        "largest_scc": int(sizes.max()) if num_components else 0,
        "largest_scc_frac_of_active": round(float(sizes.max()) / active, 4) if active else 0.0,
        "active_tokens": int(active),
        "num_components": int(num_components),
        "total_pairs": total,
        "ceiling_fraction": round((total - irreducible) / total, 4) if total else 0.0,
    }


def run_variant(name, train_paths, eval_paths, out_dir, split_identifiers, min_count, sample_fraction,
                max_passes, ils_restarts, ils_generations, jobs, max_chain_len, sample_seed=7):
    os.makedirs(out_dir, exist_ok=True)
    census = census_pass(train_paths, split_identifiers)
    lookup, kept = _build_lookup(census["vocab"], min_count)
    vocab_size = len(kept)
    unk_id = vocab_size - 1

    counter = PairCounter(vocab_size=vocab_size, chunk_size=8_000_000)
    streams = list(file_streams(train_paths, split_identifiers=split_identifiers))
    rng = random.Random(sample_seed)
    sampled = streams if sample_fraction >= 1.0 else rng.sample(streams, max(1, round(len(streams) * sample_fraction)))
    for _, stream in sampled:
        counter.add_stream(_stream_ids(stream, lookup, unk_id))
    codes, counts = counter.finalize()

    scc = _scc_stats(codes, counts, vocab_size)
    order, stats = optimize_order(codes, counts, vocab_size, np.arange(vocab_size, dtype=np.int64),
                                  max_passes=max_passes, ils_restarts=ils_restarts,
                                  ils_generations=ils_generations, ils_seed=1, jobs=jobs)
    positions = _inverse_permutation(order)
    with open(Path(out_dir) / f"old_to_new_{name}.json", "w", encoding="utf-8") as file:
        json.dump(positions.tolist(), file)

    histogram = {}
    for _, stream in file_streams(eval_paths, split_identifiers=split_identifiers):
        chain_length_histogram(_stream_ids(stream, lookup, unk_id), positions, reset_on_clause=False,
                               histogram=histogram, max_chain_len=max_chain_len)
    return {
        "variant": name,
        "split_identifiers": split_identifiers,
        "census": {k: census[k] for k in ("tokens", "clauses", "vocab_size", "vocab_size_by_min_count", "coverage_by_cap")},
        "vocab_size_used": vocab_size,
        "scc": scc,
        "order": {"ascending_fraction": round(stats["ascending_after"] / stats["total_pairs"], 4) if stats["total_pairs"] else 0.0},
        "validation_chains": summarize(histogram),
    }


def _split_eval(train_paths, eval_fraction, seed):
    """Hold out a deterministic fraction of files (by content) for eval."""
    all_records = []
    for path in train_paths:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    all_records.append(line)
    rng = random.Random(seed)
    rng.shuffle(all_records)
    cut = max(1, round(len(all_records) * eval_fraction))
    return all_records[cut:], all_records[:cut]


def _write_split(records, path):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="JS/TS vocab-ordering transfer experiment (both tokenizations).")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--max-passes", type=int, default=200)
    parser.add_argument("--ils-restarts", type=int, default=8)
    parser.add_argument("--ils-generations", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-chain-len", type=int, default=9)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    started_at = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    train_records, eval_records = _split_eval(args.train, args.eval_fraction, args.seed)
    train_path = Path(args.out_dir) / "_train.jsonl"
    eval_path = Path(args.out_dir) / "_eval.jsonl"
    _write_split(train_records, train_path)
    _write_split(eval_records, eval_path)
    print(f"files: {len(train_records)} train, {len(eval_records)} eval", flush=True)

    report = {"format": "code_vocab_experiment_v1", "train_files": len(train_records),
              "eval_files": len(eval_records), "min_count": args.min_count, "variants": {}}
    for name, split_ids in (("whole", False), ("split", True)):
        print(f"=== variant: {name} ===", flush=True)
        result = run_variant(name, [train_path], [eval_path], args.out_dir, split_identifiers=split_ids,
                             min_count=args.min_count, sample_fraction=args.sample_fraction,
                             max_passes=args.max_passes, ils_restarts=args.ils_restarts,
                             ils_generations=args.ils_generations, jobs=args.jobs, max_chain_len=args.max_chain_len)
        result["validation_chains"].pop("histogram", None)
        report["variants"][name] = result
        print(json.dumps({k: result[k] for k in ("census", "scc", "order", "validation_chains")}, indent=2), flush=True)
    report["elapsed_seconds"] = round(time.time() - started_at, 1)
    with open(Path(args.out_dir) / "experiment_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
