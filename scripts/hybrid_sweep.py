"""
Sweep eval logic for hybrid multihot context: next-token top-k accuracy + perplexity
as a function of recent 1-hot tail length X and compressed depth D, on one shared
probe set. This module is the library; scripts/eval_hybrid_context_sweep.py is the
thin CLI wrapper.
"""

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


def _iter_story_raw_rows(records):
    """Group a stream of raw records into (split, story_id) -> rows, streaming.
    Records in phrase_index.jsonl are contiguous by story, so consecutive grouping
    is exact and never materializes the whole file."""
    current_key = None
    current_rows = []
    for record in records:
        key = (record["split"], int(record["story_id"]))
        if current_key is not None and key != current_key:
            yield current_key, current_rows
            current_rows = []
        current_key = key
        current_rows.append(record)
    if current_rows:
        yield current_key, current_rows


def build_sweep_probes(records, min_history=1, max_probes=None, split=None, index_map=None):
    # Clamp to >= 1: a position-0 probe has empty context (predict_probe_logits would
    # index last=-1, scoring the padding row). Never emit position-0 probes.
    from scripts.train_phrase_vectors import normalize_phrase_records
    min_history = max(1, min_history)
    probes = []
    for key, raw_rows in _iter_story_raw_rows(records):
        if split is not None and key[0] != split:
            continue
        stream = _canonical_token_stream(normalize_phrase_records(raw_rows))
        indices = [int(r["indices"][0]) for r in stream]
        if index_map is not None:
            indices = [int(index_map[index]) for index in indices]
        clauses = [int(r.get("phrase_id", 0)) for r in stream]
        for pos in range(len(indices)):
            if pos < min_history:
                continue
            is_opener = pos == 0 or clauses[pos] != clauses[pos - 1]
            probes.append(SweepProbe(indices, clauses, pos, is_opener))
            if max_probes is not None and len(probes) >= max_probes:
                return probes
    return probes


def context_steps_for_probe(probe, x, depth, reset_on_clause=True):
    p = probe.target_pos
    tail_start = max(0, p - x)
    front_records = [
        {"indices": [probe.token_indices[i]], "phrase_id": probe.clause_ids[i]}
        for i in range(tail_start)
    ]
    front_chains = _chains_from_token_records(front_records, reset_on_clause=reset_on_clause)
    if depth is not None:
        front_chains = front_chains[-depth:]
    tail = [[probe.token_indices[i]] for i in range(tail_start, p)]
    return front_chains + tail


def topk_and_ce(logits_row, target, ks=(1, 5, 10)):
    top = logits_row.topk(min(max(ks), logits_row.numel())).indices.tolist()
    hits = {k: int(target in top[:k]) for k in ks}
    ce = F.cross_entropy(logits_row.unsqueeze(0), torch.tensor([int(target)], device=logits_row.device)).item()
    return hits, ce


from scripts.train_phrase_gpt import PhraseSequenceExample, collate_phrase_sequences


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


def _bootstrap_cis(per_probe, bootstrap, seed):
    n = len(per_probe["ce"])
    tensors = {k: torch.tensor(per_probe[k], dtype=torch.float64) for k in ("h1", "h5", "h10", "ce")}
    generator = torch.Generator().manual_seed(int(seed))
    idx = torch.randint(0, n, (bootstrap, n), generator=generator)

    def ci(key):
        samples = tensors[key][idx].mean(dim=1)
        return [torch.quantile(samples, 0.025).item(), torch.quantile(samples, 0.975).item()]

    ce_lo, ce_hi = ci("ce")
    return {
        "top1_ci": ci("h1"), "top5_ci": ci("h5"), "top10_ci": ci("h10"),
        "mean_ce_ci": [ce_lo, ce_hi], "perplexity_ci": [math.exp(ce_lo), math.exp(ce_hi)],
    }


def _aggregate(probes, logits, remap, bootstrap=0, bootstrap_seed=0):
    per = {name: {"h1": [], "h5": [], "h10": [], "ce": []} for name in ("all", "opener", "interior")}
    for probe, row in zip(probes, logits):
        target = int(remap[probe.token_indices[probe.target_pos]]) if remap is not None else probe.token_indices[probe.target_pos]
        hits, ce = topk_and_ce(row, target, ks=(1, 5, 10))
        for name in ("all", "opener" if probe.is_opener else "interior"):
            per[name]["h1"].append(hits[1]); per[name]["h5"].append(hits[5])
            per[name]["h10"].append(hits[10]); per[name]["ce"].append(ce)
    out = {}
    for name, arr in per.items():
        n = len(arr["ce"])
        mean_ce = (sum(arr["ce"]) / n) if n else float("nan")
        cell = {
            "top1": (sum(arr["h1"]) / n) if n else 0.0,
            "top5": (sum(arr["h5"]) / n) if n else 0.0,
            "top10": (sum(arr["h10"]) / n) if n else 0.0,
            "mean_ce": mean_ce,
            "perplexity": math.exp(mean_ce) if n else float("nan"),
            "count": n,
        }
        if bootstrap and n:
            cell.update(_bootstrap_cis(arr, bootstrap, bootstrap_seed))
        out[name] = cell
    return out


def classic_context_steps(probe):
    """Full 1-hot history: every context token as its own singleton step
    (the classic uncompressed baseline)."""
    return [[int(index)] for index in probe.token_indices[:probe.target_pos]]


def _probe_contexts(probes, x, depth, remap, reset_on_clause=True, front_encoder=None):
    contexts = []
    for p in probes:
        if front_encoder is None:
            steps = context_steps_for_probe(p, x=x, depth=depth, reset_on_clause=reset_on_clause)
            contexts.append(_remap_steps(steps, remap))
        else:
            # front_encoder output is SAE latent ids (>= latent_offset) that live
            # outside the vocab-sized `remap` lookup's range; the tail is already
            # resolved through front_encoder.tail_lookup (the top-8k remap). Do
            # NOT run either through _remap_steps -- `remap` here is only used
            # (unconditionally, by the caller) for the classic_1hot baseline and
            # for scoring the target token in _aggregate, both of which stay in
            # original-token-id space.
            tail_start = max(0, p.target_pos - x)
            front = front_encoder(p.token_indices[:tail_start], p.clause_ids[:tail_start])
            if depth is not None:
                front = front[-depth:]
            tail = [[int(front_encoder.tail_lookup[p.token_indices[i]])] for i in range(tail_start, p.target_pos)]
            contexts.append(front + tail)
    return contexts


def run_sweep(model, probes, x_values, d_values, fixed_x_for_depth, remap, batch_size, device,
              bootstrap=0, bootstrap_seed=0, reset_on_clause=True, front_encoder=None):
    result = {"x_sweep": {}, "d_sweep": {}, "num_probes": len(probes), "reset_on_clause": reset_on_clause}
    for x in x_values:
        contexts = _probe_contexts(probes, x=x, depth=None, remap=remap, reset_on_clause=reset_on_clause, front_encoder=front_encoder)
        logits = predict_probe_logits(model, contexts, batch_size, device)
        result["x_sweep"][str(x)] = _aggregate(probes, logits, remap, bootstrap=bootstrap, bootstrap_seed=bootstrap_seed)
    for d in d_values:
        contexts = _probe_contexts(probes, x=fixed_x_for_depth, depth=d, remap=remap, reset_on_clause=reset_on_clause, front_encoder=front_encoder)
        logits = predict_probe_logits(model, contexts, batch_size, device)
        result["d_sweep"][str(d)] = _aggregate(probes, logits, remap, bootstrap=bootstrap, bootstrap_seed=bootstrap_seed)
    contexts = [_remap_steps(classic_context_steps(p), remap) for p in probes]
    logits = predict_probe_logits(model, contexts, batch_size, device)
    result["classic_1hot"] = _aggregate(probes, logits, remap, bootstrap=bootstrap, bootstrap_seed=bootstrap_seed)
    return result


def _parse_int_list(text):
    return [int(v) for v in text.split(",") if v.strip()]
