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


def build_sweep_probes(records, min_history=1, max_probes=None):
    probes = []
    for _key, rows in sorted(_stories(records).items()):
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
