"""
Sweep eval for hybrid multihot context: next-token top-k accuracy + perplexity as a
function of recent 1-hot tail length X and compressed depth D, on one shared probe set.
"""

import argparse
import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from scripts.train_phrase_gpt import _canonical_token_stream, _chains_from_token_records


@dataclass(frozen=True)
class SweepProbe:
    token_indices: list
    clause_ids: list
    target_pos: int
    is_opener: bool


def _stories(records):
    from scripts.train_phrase_vectors import normalize_phrase_records
    grouped = {}
    for record in normalize_phrase_records(records):
        key = (record["split"], int(record["story_id"]))
        grouped.setdefault(key, []).append(record)
    return grouped


def build_sweep_probes(records, min_history=1, max_probes=None, split=None):
    # Clamp to at least 1: a probe at position 0 has an empty context, which
    # predict_probe_logits would index at last=-1 (silently scoring the padding
    # row and polluting every bucket). Position-0 probes must never be emitted.
    min_history = max(1, min_history)
    probes = []
    for key, rows in sorted(_stories(records).items()):
        if split is not None and key[0] != split:
            continue
        stream = _canonical_token_stream(rows)
        indices = [int(r["indices"][0]) for r in stream]
        clauses = [int(r.get("phrase_id", 0)) for r in stream]
        for pos in range(len(indices)):
            if pos < min_history:
                continue
            is_opener = pos == 0 or clauses[pos] != clauses[pos - 1]
            probes.append(SweepProbe(indices, clauses, pos, is_opener))
            if max_probes is not None and len(probes) >= max_probes:
                return probes
    return probes


def context_steps_for_probe(probe, x, depth):
    p = probe.target_pos
    tail_start = max(0, p - x)
    front_records = [
        {"indices": [probe.token_indices[i]], "phrase_id": probe.clause_ids[i]}
        for i in range(tail_start)
    ]
    front_chains = _chains_from_token_records(front_records, reset_on_clause=True)
    if depth is not None:
        front_chains = front_chains[-depth:]
    tail = [[probe.token_indices[i]] for i in range(tail_start, p)]
    return front_chains + tail


def topk_and_ce(logits_row, target, ks=(1, 5, 10)):
    top = logits_row.topk(min(max(ks), logits_row.numel())).indices.tolist()
    hits = {k: int(target in top[:k]) for k in ks}
    ce = F.cross_entropy(logits_row.unsqueeze(0), torch.tensor([int(target)], device=logits_row.device)).item()
    return hits, ce


from scripts.train_phrase_gpt import PhraseSequenceExample, collate_phrase_sequences, choose_device
from scripts.train_phrase_vectors import iter_records
from scripts.eval_phrase_gpt_packed_vs_single import load_model_from_checkpoint, resolve_vocab_remap


def _remap_steps(steps, remap):
    if remap is None:
        return steps
    return [[int(remap[i]) for i in step] for step in steps]


def predict_probe_logits(model, contexts, batch_size, device):
    """contexts: list[list[list[int]]] (per probe: list of chains). Returns a tensor
    [len(contexts), vocab] of the last-position logits, in input order. Memory-safe:
    sort by length, small batches, free cache per batch."""
    model.eval()
    if not contexts:
        return torch.empty((0, 0))
    order = sorted(range(len(contexts)), key=lambda i: len(contexts[i]))
    rows = [None] * len(contexts)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            steps = [contexts[i] for i in batch_idx]
            seq_len = max(2, max(len(s) for s in steps))
            batch = collate_phrase_sequences(
                [PhraseSequenceExample(input_indices=s, targets=[-1] * len(s)) for s in steps],
                sequence_len=seq_len, dummy_token_id=0, device=device,
            )
            logits = model(batch.idx, phrase_indices=batch.phrase_indices,
                           phrase_offsets=batch.phrase_offsets, phrase_batch_positions=batch.phrase_batch_positions)
            last = torch.tensor([len(s) - 1 for s in steps], device=logits.device)
            picked = logits[torch.arange(len(steps), device=logits.device), last, :].cpu()
            for slot, row in zip(batch_idx, picked):
                rows[slot] = row
            del logits, batch
            if device == "mps":
                torch.mps.empty_cache()
    return torch.stack(rows)


def _aggregate(probes, logits, remap):
    buckets = {name: {"top1": 0, "top5": 0, "top10": 0, "ce": 0.0, "count": 0}
               for name in ("all", "opener", "interior")}
    for probe, row in zip(probes, logits):
        target = int(remap[probe.token_indices[probe.target_pos]]) if remap is not None else probe.token_indices[probe.target_pos]
        hits, ce = topk_and_ce(row, target, ks=(1, 5, 10))
        for name in ("all", "opener" if probe.is_opener else "interior"):
            b = buckets[name]
            b["top1"] += hits[1]; b["top5"] += hits[5]; b["top10"] += hits[10]
            b["ce"] += ce; b["count"] += 1
    out = {}
    for name, b in buckets.items():
        n = max(b["count"], 1)
        mean_ce = b["ce"] / n
        out[name] = {
            "top1": b["top1"] / n, "top5": b["top5"] / n, "top10": b["top10"] / n,
            "mean_ce": mean_ce, "perplexity": math.exp(mean_ce) if b["count"] else float("nan"),
            "count": b["count"],
        }
    return out


def run_sweep(model, probes, x_values, d_values, fixed_x_for_depth, remap, batch_size, device):
    result = {"x_sweep": {}, "d_sweep": {}, "num_probes": len(probes)}
    for x in x_values:
        contexts = [_remap_steps(context_steps_for_probe(p, x=x, depth=None), remap) for p in probes]
        logits = predict_probe_logits(model, contexts, batch_size, device)
        result["x_sweep"][str(x)] = _aggregate(probes, logits, remap)
    for d in d_values:
        contexts = [_remap_steps(context_steps_for_probe(p, x=fixed_x_for_depth, depth=d), remap) for p in probes]
        logits = predict_probe_logits(model, contexts, batch_size, device)
        result["d_sweep"][str(d)] = _aggregate(probes, logits, remap)
    return result


def _parse_int_list(text):
    return [int(v) for v in text.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid context sweep eval (top-k accuracy + perplexity vs X and D).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--vocab", default=None, help="Original vocab.json, if the checkpoint's recorded path is wrong.")
    parser.add_argument("--x-values", default="0,1,2,4,8,16")
    parser.add_argument("--d-values", default="1,2,4,8,16,32", help="Compressed-depth caps; 'none' appended automatically for unbounded.")
    parser.add_argument("--fixed-x-for-depth", type=int, default=0)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--max-probes", type=int, default=20000)
    parser.add_argument("--split", default=None,
                         help="Only score stories from this split (e.g. 'validation'). "
                              "Default: all splits in --records (pass a held-out-only file, "
                              "or set this, to avoid scoring on training data).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    records = list(iter_records(args.records))
    probes = build_sweep_probes(records, min_history=args.min_history, max_probes=args.max_probes,
                                 split=args.split)
    if not probes:
        raise SystemExit("No probes found.")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, _config = load_model_from_checkpoint(checkpoint, device=device)
    remap = resolve_vocab_remap(checkpoint.get("config", {}), vocab_override=args.vocab)
    if remap is not None:
        remap = remap.to("cpu")
    print(f"probes={len(probes)} device={device} remap={'yes' if remap is not None else 'no'}", flush=True)
    # Note: compressed contexts longer than the model's sequence_len are truncated
    # by the final-layer attention window, so very large D and unbounded D (None)
    # can look identical on long stories — the D-sweep is only meaningful up to
    # ~sequence_len chains of history.
    d_values = _parse_int_list(args.d_values) + [None]
    result = run_sweep(model, probes, x_values=_parse_int_list(args.x_values), d_values=d_values,
                       fixed_x_for_depth=args.fixed_x_for_depth, remap=remap, batch_size=args.batch_size, device=device)
    result["split"] = args.split
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
