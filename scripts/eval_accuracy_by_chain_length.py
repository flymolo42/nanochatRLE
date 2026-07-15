"""
Next-token accuracy bucketed by the length of the compressed slot immediately
preceding the target (X=0, cross-clause contexts). Answers: can the model
actually decode long multihot chains, or are 10-15-token chains "too good"?

Example:
python -m scripts.eval_accuracy_by_chain_length \
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

from scripts.eval_phrase_gpt_packed_vs_single import load_model_from_checkpoint, resolve_vocab_remap
from scripts.hybrid_sweep import _probe_contexts, build_sweep_probes, predict_probe_logits
from scripts.train_phrase_gpt import choose_device
from scripts.train_phrase_vectors import iter_records


def bucket_label(length):
    return str(length) if length < 10 else "10+"


def bucket_by_last_slot(contexts, logits, targets, ks=(1, 5, 10)):
    accumulators = {}
    for row, (context, target) in enumerate(zip(contexts, targets)):
        label = bucket_label(len(context[-1]))
        bucket = accumulators.setdefault(label, {"count": 0, "hits": {k: 0 for k in ks}, "ce": 0.0})
        bucket["count"] += 1
        row_logits = logits[row]
        top = row_logits.topk(min(max(ks), row_logits.numel())).indices.tolist()
        for k in ks:
            bucket["hits"][k] += int(target in top[:k])
        bucket["ce"] += F.cross_entropy(row_logits.unsqueeze(0), torch.tensor([int(target)])).item()
    out = {}
    for label, bucket in accumulators.items():
        count = bucket["count"]
        out[label] = {
            "count": count,
            "top1": round(bucket["hits"][1] / count, 4),
            "top5": round(bucket["hits"][5] / count, 4),
            "top10": round(bucket["hits"][10] / count, 4),
            "mean_ce": round(bucket["ce"] / count, 4),
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Accuracy bucketed by preceding-chain length at X=0.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--index-map", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-probes", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-chain-len", type=int, default=None)
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
    if not probes:
        raise SystemExit("No probes found.")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, _config = load_model_from_checkpoint(checkpoint, device=device)
    remap = resolve_vocab_remap(checkpoint.get("config", {}), vocab_override=args.vocab)
    if remap is not None:
        remap = remap.to("cpu")
    contexts = _probe_contexts(probes, x=0, depth=None, remap=remap, reset_on_clause=False, max_chain_len=args.max_chain_len)
    logits = predict_probe_logits(model, contexts, args.batch_size, device)
    targets = [
        int(remap[probe.token_indices[probe.target_pos]]) if remap is not None
        else probe.token_indices[probe.target_pos]
        for probe in probes
    ]
    buckets = bucket_by_last_slot(contexts, logits, targets)
    ordered = {label: buckets[label] for label in sorted(buckets, key=lambda l: (l == "10+", int(l.rstrip("+"))))}
    print(json.dumps({"probes": len(probes), "x": 0, "reset_on_clause": False, "by_last_chain_len": ordered}, indent=2))


if __name__ == "__main__":
    main()
