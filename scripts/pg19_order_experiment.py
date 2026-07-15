"""
PG-19 ordering experiment: pair counts + SCC census + ILS orders built from
(a) the full train split and (b) a 10% book sample, both evaluated on the
held-out validation books. Answers: largest SCC at natural-vocab scale, and
how much ordering quality a 10x-cheaper sample costs.

Example:
python -m scripts.pg19_order_experiment \
    --data-dir ../pg19/train --eval-dir ../pg19/validation \
    --census-vocab pg19_census_out/vocab.json \
    --out-dir pg19_order_out --min-count 25 --sample-fraction 0.1
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from scripts.measure_chain_lengths import chain_length_histogram, summarize
from scripts.pg19_stream import book_streams
from scripts.reorder_phrase_vocab import (
    PairCounter,
    _net_edges,
    _strongly_connected_components,
    ascending_pair_count,
    _inverse_permutation,
    optimize_order,
)

UNK_TOKEN = "<unk>"


def build_token_lookup(census_vocab, min_count):
    kept_rows = [row for row in sorted(census_vocab, key=lambda r: int(r["index"])) if int(row["count"]) >= min_count]
    kept = [row["token"] for row in kept_rows] + [UNK_TOKEN]
    lookup = {token: position for position, token in enumerate(kept[:-1])}
    return lookup, kept


def stream_to_ids(stream, lookup, unk_id):
    return [(clause, lookup.get(token, unk_id)) for clause, token in stream]


def sample_books(paths, fraction, seed):
    paths = sorted(str(p) for p in paths)
    count = max(1, round(len(paths) * fraction))
    return sorted(random.Random(seed).sample(paths, count))


def count_pairs(paths, lookup, unk_id, vocab_size, progress_every=1000, label=""):
    counter = PairCounter(vocab_size=vocab_size, chunk_size=8_000_000)
    books = 0
    for _, stream in book_streams(paths):
        counter.add_stream(stream_to_ids(stream, lookup, unk_id))
        books += 1
        if progress_every and books % progress_every == 0:
            print(f"[{label}] pairs from {books}/{len(paths)} books", flush=True)
    return counter.finalize()


def scc_stats(codes, counts, vocab_size):
    heads, tails, nets = _net_edges(codes, counts, vocab_size)
    component, num_components = _strongly_connected_components(heads, tails, vocab_size)
    sizes = np.bincount(component, minlength=num_components)
    active = np.unique(np.concatenate([heads, tails])).size if heads.size else 0
    total = int(counts.sum())
    left = codes // vocab_size
    right = codes % vocab_size
    reverse = right * vocab_size + left
    slots = np.minimum(np.searchsorted(codes, reverse), max(codes.size - 1, 0))
    has_reverse = codes.size > 0 and (codes[slots] == reverse)
    reverse_counts = np.where(has_reverse, counts[slots], 0) if codes.size else np.zeros(0, dtype=np.int64)
    irreducible = int(np.minimum(counts, reverse_counts).sum()) // 2 if codes.size else 0
    return {
        "net_edges": int(nets.size),
        "num_components": int(num_components),
        "largest_scc": int(sizes.max()) if num_components else 0,
        "largest_scc_frac_of_active": round(float(sizes.max()) / active, 4) if active else 0.0,
        "active_tokens": int(active),
        "total_pairs": total,
        "ceiling_fraction": round((total - irreducible) / total, 4) if total else 0.0,
    }


def measure_orders(eval_paths, lookup, unk_id, orders, max_chain_len=None):
    histograms = {name: {} for name in orders}
    for _, stream in book_streams(eval_paths):
        ids = stream_to_ids(stream, lookup, unk_id)
        for name, positions in orders.items():
            chain_length_histogram(ids, positions, reset_on_clause=False,
                                   histogram=histograms[name], max_chain_len=max_chain_len)
    return {name: summarize(histogram) for name, histogram in histograms.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="PG-19 SCC census + full-vs-sample ordering experiment.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--census-vocab", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--sample-fraction", type=float, default=0.1)
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--max-passes", type=int, default=200)
    parser.add_argument("--ils-restarts", type=int, default=8)
    parser.add_argument("--ils-generations", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-chain-len", type=int, default=9)
    return parser.parse_args()


def main():
    args = parse_args()
    started_at = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.census_vocab, "r", encoding="utf-8") as file:
        census_vocab = json.load(file)
    lookup, kept = build_token_lookup(census_vocab, args.min_count)
    vocab_size = len(kept)
    unk_id = vocab_size - 1
    print(f"vocab (min_count={args.min_count}): {vocab_size:,} incl {UNK_TOKEN}", flush=True)

    train_paths = sorted(Path(args.data_dir).glob("*.txt"))
    eval_paths = sorted(Path(args.eval_dir).glob("*.txt"))
    sample_paths = sample_books(train_paths, args.sample_fraction, args.sample_seed)

    report = {"format": "pg19_order_experiment_v1", "vocab_size": vocab_size, "min_count": args.min_count,
              "train_books": len(train_paths), "sample_books": len(sample_paths), "eval_books": len(eval_paths),
              "max_chain_len": args.max_chain_len}
    warm_order = np.arange(vocab_size, dtype=np.int64)
    orders_for_eval = {"warm_census": np.arange(vocab_size, dtype=np.int64).tolist()}

    for label, paths in (("full", train_paths), ("sample10", sample_paths)):
        codes, counts = count_pairs(paths, lookup, unk_id, vocab_size, label=label)
        np.savez(Path(args.out_dir) / f"pair_counts_{label}.npz", codes=codes, counts=counts,
                 vocab_size=np.int64(vocab_size), stories=np.int64(len(paths)))
        report[f"scc_{label}"] = scc_stats(codes, counts, vocab_size)
        print(f"[{label}] scc: {json.dumps(report[f'scc_{label}'])}", flush=True)
        order, stats = optimize_order(codes, counts, vocab_size, warm_order, max_passes=args.max_passes,
                                      ils_restarts=args.ils_restarts, ils_generations=args.ils_generations,
                                      ils_seed=1, jobs=args.jobs)
        positions = _inverse_permutation(order)
        with open(Path(args.out_dir) / f"old_to_new_{label}.json", "w", encoding="utf-8") as file:
            json.dump(positions.tolist(), file)
        report[f"order_{label}"] = {
            "ascending_fraction": round(stats["ascending_after"] / stats["total_pairs"], 4) if stats["total_pairs"] else 0.0,
            "ils_improvement_pairs": stats.get("ils", {}).get("improvement_pairs"),
        }
        orders_for_eval[label] = positions.tolist()
        print(f"[{label}] order: {json.dumps(report[f'order_{label}'])}", flush=True)

    report["validation_chains"] = measure_orders(eval_paths, lookup, unk_id, orders_for_eval,
                                                 max_chain_len=args.max_chain_len)
    for name, summary in report["validation_chains"].items():
        summary.pop("histogram", None)
    report["elapsed_seconds"] = round(time.time() - started_at, 1)
    with open(Path(args.out_dir) / "experiment_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
