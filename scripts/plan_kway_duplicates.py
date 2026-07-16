"""
k-way hub-token duplication planner: rank tokens by conflict mass, collect each
candidate's within-clause position distribution, and choose k copies with target
positions (data-driven modality, or a fixed count).
"""

import numpy as np

from scripts.plan_token_duplicates import conflict_losses


def collect_position_histograms(id_streams, vocab_size, bins=20):
    hist = np.zeros((vocab_size, bins), dtype=np.int64)
    for stream in id_streams:
        clause_tokens = []
        current = None
        for clause, token_id in stream:
            if current is not None and clause != current:
                _bin_clause(hist, clause_tokens, bins)
                clause_tokens = []
            current = clause
            clause_tokens.append(int(token_id))
        if clause_tokens:
            _bin_clause(hist, clause_tokens, bins)
    return hist


def _bin_clause(hist, clause_tokens, bins):
    denom = max(len(clause_tokens) - 1, 1)
    for position, token_id in enumerate(clause_tokens):
        rel = position / denom
        bin_index = min(int(rel * bins), bins - 1)
        hist[token_id, bin_index] += 1


def _bin_centroid(hist_row, lo, hi, bins):
    weights = hist_row[lo:hi].astype(np.float64)
    if weights.sum() == 0:
        return (lo + hi) / 2.0 / bins
    centers = (np.arange(lo, hi) + 0.5) / bins
    return float((centers * weights).sum() / weights.sum())


def select_k(hist_row, k_max, min_mass_frac=0.15):
    total = int(hist_row.sum())
    bins = len(hist_row)
    if total == 0:
        return 1, [0.5]
    # split the [0,1] range into equal segments; a segment is a "mode" if it
    # holds >= min_mass_frac of the mass. k = number of qualifying segments among
    # k_max candidate segments, but never more than the count of nonzero regions.
    for k in range(k_max, 1, -1):
        edges = np.linspace(0, bins, k + 1).astype(int)
        seg_mass = [int(hist_row[edges[i]:edges[i + 1]].sum()) for i in range(k)]
        if all(m >= min_mass_frac * total for m in seg_mass):
            centroids = [_bin_centroid(hist_row, edges[i], edges[i + 1], bins) for i in range(k)]
            return k, centroids
    return 1, [_bin_centroid(hist_row, 0, bins, bins)]


def fixed_centroids(k):
    return [round((i + 0.5) / k, 6) for i in range(k)]


def select_candidates(codes, counts, vocab_size, top_n):
    loss, _ = conflict_losses(codes, counts, vocab_size)
    ranked = np.argsort(-loss, kind="stable")
    return [int(t) for t in ranked if loss[t] > 0][:top_n]
