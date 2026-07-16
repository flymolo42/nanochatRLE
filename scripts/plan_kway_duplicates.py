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


from scripts.plan_token_duplicates import renumber_array


def build_plan(candidates, histograms, vocab_size, k_max=6, fixed_k=None):
    ks = {}
    centroids = {}
    for old_index in candidates:
        if fixed_k is not None:
            k, cents = fixed_k, fixed_centroids(fixed_k)
        else:
            k, cents = select_k(histograms[old_index], k_max)
        if k >= 2:
            ks[old_index] = k
            centroids[old_index] = cents
    parents_sorted = sorted(ks)
    # each surviving parent adds (k-1) extra slots; extras are inserted directly
    # after the parent's renumbered index, in index order
    base_renumber = _kway_renumber(vocab_size, parents_sorted, ks)
    parents = {}
    for old_index in parents_sorted:
        base = int(base_renumber[old_index])
        k = ks[old_index]
        copies = [{"new_index": base + offset, "target": round(centroids[old_index][offset], 6)}
                  for offset in range(k)]
        parents[old_index] = {"base_new_index": base, "copies": copies}
    return {
        "format": "kway_duplicates_plan_v1",
        "vocab_size_old": vocab_size,
        "vocab_size_new": vocab_size + sum(ks[p] - 1 for p in parents_sorted),
        "copies_added": sum(ks[p] - 1 for p in parents_sorted),
        "k_max": k_max,
        "fixed_k": fixed_k,
        "parents": parents,
    }


def _kway_renumber(vocab_size, parents_sorted, ks):
    # new base index of old id t = t + sum over parents p<t of (k_p - 1)
    extras = np.zeros(vocab_size, dtype=np.int64)
    for parent in parents_sorted:
        extras[parent + 1:] += ks[parent] - 1
    return np.arange(vocab_size, dtype=np.int64) + extras


def apply_kway(stream, plan):
    parents = {int(k): v for k, v in plan["parents"].items()}
    vocab_size = plan["vocab_size_old"]
    ks = {p: len(info["copies"]) for p, info in parents.items()}
    renumber = _kway_renumber(vocab_size, sorted(parents), ks)
    out = []
    clause_tokens = []
    current = None

    def flush():
        denom = max(len(clause_tokens) - 1, 1)
        for position, (clause, token_id) in enumerate(clause_tokens_full):
            if token_id in parents:
                rel = position / denom
                copies = parents[token_id]["copies"]
                best = min(copies, key=lambda c: (abs(c["target"] - rel), c["new_index"]))
                out.append((clause, best["new_index"]))
            else:
                out.append((clause, int(renumber[token_id])))

    clause_tokens_full = []
    for clause, token_id in stream:
        if current is not None and clause != current:
            clause_tokens = clause_tokens_full
            flush()
            clause_tokens_full = []
        current = clause
        clause_tokens_full.append((clause, int(token_id)))
    if clause_tokens_full:
        clause_tokens = clause_tokens_full
        flush()
    return out
