# scripts/run_kway_experiment.py
"""
Compare no-dup / fixed-k2 / data-driven k-way / fixed-kN duplication on JS/TS
code: for each arm, remap streams through its plan, count pairs, reorder+ILS,
and measure chains on held-out eval files. Analysis-only.

Example:
python -m scripts.run_kway_experiment \
    --train ../stack_smol_jsts_raw/data/javascript/data.json \
            ../stack_smol_jsts_raw/data/typescript/data.json \
    --out-dir kway_out --min-count 25 --top-n 300 --k-max 6 --fixed-k-arm 4
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.code_stream import file_streams
from scripts.code_vocab_experiment import _build_lookup, _stream_ids, census_pass
from scripts.measure_chain_lengths import chain_length_histogram, summarize
from scripts.plan_kway_duplicates import (apply_kway, apply_kway_predrank, build_plan,
                                          collect_position_histograms, collect_predecessor_rank_histograms,
                                          select_candidates)
from scripts.reorder_phrase_vocab import PairCounter, _inverse_permutation, optimize_order


def run_arm(name, train_id_streams_fn, eval_id_streams_fn, vocab_size, plan, out_dir,
            max_passes, ils_restarts, ils_generations, jobs, max_chain_len, reference_positions=None):
    def _apply(stream):
        return apply_kway_predrank(stream, plan, reference_positions) if reference_positions is not None else apply_kway(stream, plan)
    new_vocab_size = plan["vocab_size_new"]
    counter = PairCounter(vocab_size=new_vocab_size, chunk_size=8_000_000)
    for stream in train_id_streams_fn():
        counter.add_stream(_apply(stream))
    codes, counts = counter.finalize()
    order, stats = optimize_order(codes, counts, new_vocab_size, np.arange(new_vocab_size, dtype=np.int64),
                                  max_passes=max_passes, ils_restarts=ils_restarts,
                                  ils_generations=ils_generations, ils_seed=1, jobs=jobs)
    positions = _inverse_permutation(order)
    histogram = {}
    for stream in eval_id_streams_fn():
        chain_length_histogram(_apply(stream), positions, reset_on_clause=False,
                               histogram=histogram, max_chain_len=max_chain_len)
    chains = summarize(histogram)
    chains.pop("histogram", None)
    if out_dir is not None:
        with open(Path(out_dir) / f"positions_{name}.json", "w", encoding="utf-8") as file:
            json.dump(positions.tolist(), file)
    return {
        "arm": name,
        "extra_slots": plan["vocab_size_new"] - plan["vocab_size_old"],
        "ascending_fraction": round(stats["ascending_after"] / stats["total_pairs"], 4) if stats["total_pairs"] else 0.0,
        "validation_chains": chains,
    }


def _empty_plan(vocab_size):
    return build_plan([], np.zeros((vocab_size, 1), dtype=np.int64), vocab_size=vocab_size, fixed_k=None)


def parse_args():
    parser = argparse.ArgumentParser(description="k-way duplication comparison on JS/TS code.")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--fixed-k-arm", type=int, default=4)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--max-passes", type=int, default=200)
    parser.add_argument("--ils-restarts", type=int, default=8)
    parser.add_argument("--ils-generations", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-chain-len", type=int, default=9)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    import random
    args = parse_args()
    started = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    # deterministic train/eval file split
    records = []
    for path in args.train:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            records += [line for line in (l.strip() for l in handle) if line]
    random.Random(args.seed).shuffle(records)
    cut = max(1, round(len(records) * args.eval_fraction))
    train_path = Path(args.out_dir) / "_train.jsonl"
    eval_path = Path(args.out_dir) / "_eval.jsonl"
    train_path.write_text("\n".join(records[cut:]) + "\n", encoding="utf-8")
    eval_path.write_text("\n".join(records[:cut]) + "\n", encoding="utf-8")

    census = census_pass([train_path], split_identifiers=False)
    lookup, kept = _build_lookup(census["vocab"], args.min_count)
    vocab_size = len(kept)
    unk = vocab_size - 1

    def train_id_streams():
        for _, stream in file_streams([train_path], split_identifiers=False):
            yield _stream_ids(stream, lookup, unk)

    def eval_id_streams():
        for _, stream in file_streams([eval_path], split_identifiers=False):
            yield _stream_ids(stream, lookup, unk)

    # conflict-mass candidates + position histograms (one pass each)
    counter = PairCounter(vocab_size=vocab_size, chunk_size=8_000_000)
    for stream in train_id_streams():
        counter.add_stream(stream)
    codes, counts = counter.finalize()
    candidates = select_candidates(codes, counts, vocab_size, args.top_n)
    histograms = collect_position_histograms(train_id_streams(), vocab_size, bins=args.bins)

    nodup_order, _ = optimize_order(codes, counts, vocab_size, np.arange(vocab_size, dtype=np.int64),
                                    max_passes=args.max_passes, ils_restarts=args.ils_restarts,
                                    ils_generations=args.ils_generations, ils_seed=1, jobs=args.jobs)
    reference_positions = _inverse_permutation(nodup_order)
    predrank_hist = collect_predecessor_rank_histograms(train_id_streams(), reference_positions, vocab_size, bins=args.bins)

    plans = {
        "nodup": (_empty_plan(vocab_size), None),
        "k2": (build_plan(candidates, histograms, vocab_size, fixed_k=2), None),
        "kway_data": (build_plan(candidates, histograms, vocab_size, k_max=args.k_max), None),
        f"kfixed{args.fixed_k_arm}": (build_plan(candidates, histograms, vocab_size, fixed_k=args.fixed_k_arm), None),
        "kway_predrank": (build_plan(candidates, predrank_hist, vocab_size, k_max=args.k_max), reference_positions),
    }
    report = {"format": "kway_experiment_v1", "vocab_size": vocab_size, "candidates": len(candidates),
              "train_files": len(records) - cut, "eval_files": cut, "arms": {}}
    for name, (plan, reference) in plans.items():
        print(f"=== arm {name} (extra slots {plan['vocab_size_new'] - plan['vocab_size_old']}) ===", flush=True)
        result = run_arm(name, train_id_streams, eval_id_streams, vocab_size, plan, args.out_dir,
                         args.max_passes, args.ils_restarts, args.ils_generations, args.jobs, args.max_chain_len,
                         reference_positions=reference)
        report["arms"][name] = result
        print(json.dumps(result, indent=2), flush=True)
    report["elapsed_seconds"] = round(time.time() - started, 1)
    with open(Path(args.out_dir) / "kway_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
