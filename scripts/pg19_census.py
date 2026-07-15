"""
PG-19 vocab census: one streaming pass over the book files collecting token
counts and within-clause position stats (the reorder warm start), reporting
vocab sizes at min-count thresholds and corpus coverage at top-k caps.

Example:
python -m scripts.pg19_census --data-dir ../pg19/train --out-dir pg19_census_out
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from scripts.pg19_stream import book_streams
from scripts.phrase_vectors import build_vocab_from_stats


def run_census(data_dir, out_dir, min_counts=(1, 5, 25), caps=(8192, 32768, 131072), progress_every=0):
    started_at = time.time()
    stats = defaultdict(lambda: {"count": 0, "position_sum": 0.0})
    books = 0
    tokens = 0
    clauses = 0
    paths = sorted(Path(data_dir).glob("*.txt"))
    for book_id, stream in book_streams(paths):
        books += 1
        clause_tokens = []
        current_clause = None
        for clause, token in stream:
            if current_clause is not None and clause != current_clause:
                _accumulate(stats, clause_tokens)
                clauses += 1
                clause_tokens = []
            current_clause = clause
            clause_tokens.append(token)
            tokens += 1
        if clause_tokens:
            _accumulate(stats, clause_tokens)
            clauses += 1
        if progress_every and books % progress_every == 0:
            print(f"census {books}/{len(paths)} books; tokens={tokens:,} types={len(stats):,}", flush=True)

    counts = sorted((entry["count"] for entry in stats.values()), reverse=True)
    report = {
        "format": "pg19_census_v1",
        "data_dir": str(data_dir),
        "books": books,
        "tokens": tokens,
        "clauses": clauses,
        "mean_clause_len": round(tokens / clauses, 3) if clauses else 0.0,
        "vocab_size_by_min_count": {
            str(threshold): sum(1 for count in counts if count >= threshold) for threshold in min_counts
        },
        "coverage_by_cap": {
            str(cap): round(sum(counts[:cap]) / tokens, 6) if tokens else 0.0 for cap in caps
        },
        "elapsed_seconds": round(time.time() - started_at, 1),
    }
    os.makedirs(out_dir, exist_ok=True)
    vocab = build_vocab_from_stats(stats, min_count=1)
    with open(Path(out_dir) / "vocab.json", "w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False)
        file.write("\n")
    with open(Path(out_dir) / "census_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return report


def _accumulate(stats, clause_tokens):
    denominator = max(len(clause_tokens) - 1, 1)
    for position, token in enumerate(clause_tokens):
        entry = stats[token]
        entry["count"] += 1
        entry["position_sum"] += position / denominator


def parse_args():
    parser = argparse.ArgumentParser(description="PG-19 vocab census + warm-start vocab.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-counts", default="1,5,25")
    parser.add_argument("--caps", default="8192,32768,131072")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_census(
        args.data_dir, args.out_dir,
        min_counts=tuple(int(v) for v in args.min_counts.split(",")),
        caps=tuple(int(v) for v in args.caps.split(",")),
        progress_every=args.progress_every,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
