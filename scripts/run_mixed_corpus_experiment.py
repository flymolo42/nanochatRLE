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
    total_pairs = int(stats["total_pairs"])
    frac = round(stats["ascending_after"] / total_pairs, 4) if total_pairs else 0.0
    return _inverse_permutation(order), frac, total_pairs


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


def every_nth(base_fn, keep_every):
    """Deterministic subsample: yield every keep_every-th stream from base_fn.
    Returns a fresh-iterator factory, so it re-yields the same subset per call."""
    def factory():
        for index, item in enumerate(base_fn()):
            if index % keep_every == 0:
                yield item
    return factory


def run_experiment(prose_train_fn, prose_eval_fn, code_train_fn, code_eval_fn, vocab_size,
                   top_n, k_max, max_passes, ils_restarts, ils_generations, jobs, max_chain_len,
                   balance_pairs=False):
    kw = dict(max_passes=max_passes, ils_restarts=ils_restarts, ils_generations=ils_generations, jobs=jobs)
    # bespoke orders always use the FULL per-corpus streams
    prose_pos, prose_asc, prose_pairs = order_from(prose_train_fn, vocab_size, **kw)
    code_pos, code_asc, code_pairs = order_from(code_train_fn, vocab_size, **kw)
    # for the mixed/kway arms, optionally thin the majority corpus toward pair parity
    mixed_prose_fn = prose_train_fn
    balance = None
    if balance_pairs and code_pairs > 0:
        keep_every = max(1, round(prose_pairs / code_pairs))
        mixed_prose_fn = every_nth(prose_train_fn, keep_every)
        balance = {"prose_keep_every": keep_every, "unbalanced_prose_pairs": prose_pairs, "code_pairs": code_pairs}
    both_train_fn = _both_fn(mixed_prose_fn, code_train_fn)
    mixed_pos, mixed_asc, mixed_pairs = order_from(both_train_fn, vocab_size, **kw)

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
        "pairs": {"prose": prose_pairs, "code": code_pairs, "mixed": mixed_pairs, "mixed_kway": int(kstats["total_pairs"])},
        "kway_extra_slots": new_vocab_size - vocab_size,
        "duplication": [
            {"old_index": int(old_index), "k": len(info["copies"]),
             "targets": [round(c["target"], 3) for c in info["copies"]]}
            for old_index, info in plan["parents"].items()
        ],
    }
    if balance is not None:
        report["balance"] = balance
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
    parser.add_argument("--balance-pairs", action="store_true", help="Thin the majority corpus toward pair parity with the minority for the mixed/kway arms (controls the size confound).")
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
                            args.ils_generations, args.jobs, args.max_chain_len,
                            balance_pairs=args.balance_pairs)
    # attach the token string for each duplicated id so the report self-documents
    for entry in report.get("duplication", []):
        entry["token"] = kept[entry["old_index"]]
    report["duplication"].sort(key=lambda e: -e["k"])
    report["prose_train_books"] = len(prose_train)
    report["code_train_files"] = len(code_records) - cut
    report["elapsed_seconds"] = round(time.time() - started, 1)
    with open(Path(args.out_dir) / "mixed_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
