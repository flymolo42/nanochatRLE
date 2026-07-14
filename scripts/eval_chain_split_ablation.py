"""
Paired ablation: for the same probes, score the model with the last compressed
chain (a) intact, (b) split at its midpoint into two slots, (c) exploded into
1-hot singletons. Within-probe deltas isolate the representation effect from
text difficulty — the causal version of accuracy-by-chain-length.

Chains are ascending-index runs, and the top-k remap is monotone, so splitting
the slot list positionally splits the chain in story order.

Example:
python -m scripts.eval_chain_split_ablation \
    --checkpoint phrase_gpt_hybrid_cross_post_out/phrase_gpt.pt \
    --records phrase_quote_split_out/phrase_index_validation.jsonl.gz \
    --split validation --max-probes 20000 \
    --index-map phrase_quote_split_ils_out/old_to_new.json \
    --vocab phrase_quote_split_ils_out/vocab.json
"""

import argparse
import json

import torch
import torch.nn.functional as F

from scripts.eval_accuracy_by_chain_length import bucket_label
from scripts.eval_phrase_gpt_packed_vs_single import load_model_from_checkpoint, resolve_vocab_remap
from scripts.hybrid_sweep import _probe_contexts, build_sweep_probes, predict_probe_logits
from scripts.train_phrase_gpt import choose_device
from scripts.train_phrase_vectors import iter_records


def split_last_slot(context):
    last = context[-1]
    half = (len(last) + 1) // 2
    return context[:-1] + [last[:half], last[half:]]


def explode_last_slot(context):
    last = context[-1]
    return context[:-1] + [[token] for token in last]


def paired_deltas(records):
    buckets = {}
    for record in records:
        for label in (bucket_label(record["length"]), "all"):
            bucket = buckets.setdefault(label, {"count": 0, "d_top1": 0.0, "d_ce": 0.0})
            bucket["count"] += 1
            bucket["d_top1"] += record["intact_top1"] - record["variant_top1"]
            bucket["d_ce"] += record["intact_ce"] - record["variant_ce"]
    return {
        label: {
            "count": bucket["count"],
            "delta_top1": round(bucket["d_top1"] / bucket["count"], 4),
            "delta_ce": round(bucket["d_ce"] / bucket["count"], 4),
        }
        for label, bucket in buckets.items()
    }


def _score(logits, targets):
    top1 = []
    ces = []
    for row, target in enumerate(targets):
        row_logits = logits[row]
        top1.append(int(row_logits.argmax().item() == target))
        ces.append(F.cross_entropy(row_logits.unsqueeze(0), torch.tensor([int(target)])).item())
    return top1, ces


def parse_args():
    parser = argparse.ArgumentParser(description="Paired intact-vs-split-vs-exploded last-chain ablation at X=0.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--index-map", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-probes", type=int, default=20000)
    parser.add_argument("--min-last-len", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    index_map = None
    if args.index_map:
        with open(args.index_map, "r", encoding="utf-8") as file:
            index_map = json.load(file)
    probes = build_sweep_probes(iter_records(args.records), min_history=1,
                                max_probes=args.max_probes, split=args.split, index_map=index_map)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, _config = load_model_from_checkpoint(checkpoint, device=device)
    remap = resolve_vocab_remap(checkpoint.get("config", {}), vocab_override=args.vocab)
    if remap is not None:
        remap = remap.to("cpu")
    contexts = _probe_contexts(probes, x=0, depth=None, remap=remap, reset_on_clause=False)

    keep = [i for i, context in enumerate(contexts) if len(context[-1]) >= args.min_last_len]
    kept_contexts = [contexts[i] for i in keep]
    targets = [
        int(remap[probes[i].token_indices[probes[i].target_pos]]) if remap is not None
        else probes[i].token_indices[probes[i].target_pos]
        for i in keep
    ]
    lengths = [len(context[-1]) for context in kept_contexts]
    print(f"probes with last-chain >= {args.min_last_len}: {len(keep)} of {len(probes)}", flush=True)

    results = {}
    intact_logits = predict_probe_logits(model, kept_contexts, args.batch_size, device)
    intact_top1, intact_ce = _score(intact_logits, targets)
    for name, variant_fn in (("split_half", split_last_slot), ("exploded_1hot", explode_last_slot)):
        variant_contexts = [variant_fn(context) for context in kept_contexts]
        variant_logits = predict_probe_logits(model, variant_contexts, args.batch_size, device)
        variant_top1, variant_ce = _score(variant_logits, targets)
        records = [
            {"length": lengths[i], "intact_top1": intact_top1[i], "variant_top1": variant_top1[i],
             "intact_ce": intact_ce[i], "variant_ce": variant_ce[i]}
            for i in range(len(keep))
        ]
        buckets = paired_deltas(records)
        ordered = {label: buckets[label] for label in sorted(buckets, key=lambda l: (l == "all", l == "10+", int(l.rstrip("+")) if l not in ("all",) else 0))}
        results[name] = ordered

    print(json.dumps({"probes": len(keep), "x": 0, "reset_on_clause": False,
                      "positive_delta_means_intact_better": True, "variants": results}, indent=2))


if __name__ == "__main__":
    main()
