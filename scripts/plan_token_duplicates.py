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
