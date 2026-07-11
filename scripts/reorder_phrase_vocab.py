"""
Reorder the phrase vocab so adjacent tokens in the canonical story stream map to
ascending indices as often as possible (longer strictly-ascending chains, fewer
timesteps in chain/hybrid modes).

The objective is the weighted linear ordering problem on the token-adjacency
digraph: maximize sum of W[u][v] over pairs ordered u before v, where W counts
how often token u immediately precedes token v within a punctuation clause.

Example:
python -m scripts.reorder_phrase_vocab \
    --vocab phrase_vectors_full_out/vocab.json \
    --records phrase_vectors_full_out/phrase_index.jsonl.gz \
    --out-dir phrase_vocab_reordered
"""

import argparse
import heapq
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from scripts.train_phrase_vectors import _legacy_record_to_typed_records, iter_records


def story_index_streams(records):
    """Yield (split, story_id, [(clause_id, vocab_index), ...]) per story.

    Mirrors train_phrase_gpt._canonical_token_stream: only record_type=="single"
    with label=="punctuation" contribute, deduped by absolute token position
    (first record wins). Holds one story in memory at a time.
    """
    current_key = None
    positions = {}
    for raw_record in records:
        typed_records = [raw_record] if "record_type" in raw_record else _legacy_record_to_typed_records(raw_record)
        for record in typed_records:
            key = (record["split"], int(record["story_id"]))
            if current_key is not None and key != current_key:
                yield current_key[0], current_key[1], [positions[pos] for pos in sorted(positions)]
                positions = {}
            current_key = key
            if record.get("record_type") != "single" or record.get("label") != "punctuation" or not record.get("indices"):
                continue
            absolute_pos = int(record.get("start", 0)) + int(record.get("token_pos", 0))
            positions.setdefault(absolute_pos, (int(record["phrase_id"]), int(record["indices"][0])))
    if current_key is not None:
        yield current_key[0], current_key[1], [positions[pos] for pos in sorted(positions)]


class PairCounter:
    """Memory-bounded adjacency-pair counter.

    Pair (u, v) is encoded as int64 u * vocab_size + v and buffered in a
    fixed-size chunk; full chunks are folded into a sorted unique
    (codes, counts) pair via sort + reduceat, so peak memory is
    chunk_size + distinct pairs, independent of corpus length.
    """

    def __init__(self, vocab_size, chunk_size=4_000_000, include_clause_crossings=False):
        self.vocab_size = int(vocab_size)
        self.chunk_size = int(chunk_size)
        self.include_clause_crossings = include_clause_crossings
        self._chunk = np.empty(self.chunk_size, dtype=np.int64)
        self._chunk_len = 0
        self._codes = np.empty(0, dtype=np.int64)
        self._counts = np.empty(0, dtype=np.int64)

    def add_stream(self, stream):
        if len(stream) < 2:
            return
        rows = np.asarray(stream, dtype=np.int64)
        clauses = rows[:, 0]
        indices = rows[:, 1]
        left = indices[:-1]
        right = indices[1:]
        mask = left != right
        if not self.include_clause_crossings:
            mask &= clauses[:-1] == clauses[1:]
        self._append(left[mask] * self.vocab_size + right[mask])

    def _append(self, codes):
        offset = 0
        while offset < codes.size:
            take = min(self.chunk_size - self._chunk_len, codes.size - offset)
            self._chunk[self._chunk_len:self._chunk_len + take] = codes[offset:offset + take]
            self._chunk_len += take
            offset += take
            if self._chunk_len == self.chunk_size:
                self._fold_chunk()

    def _fold_chunk(self):
        if self._chunk_len == 0:
            return
        chunk_codes, chunk_counts = np.unique(self._chunk[:self._chunk_len], return_counts=True)
        chunk_counts = chunk_counts.astype(np.int64)
        self._chunk_len = 0
        if self._codes.size == 0:
            self._codes = chunk_codes
            self._counts = chunk_counts
            return
        # the accumulator is already sorted: merge instead of re-sorting everything
        slots = np.searchsorted(self._codes, chunk_codes)
        clipped = np.minimum(slots, self._codes.size - 1)
        matched = self._codes[clipped] == chunk_codes
        self._counts[clipped[matched]] += chunk_counts[matched]
        if not matched.all():
            fresh = ~matched
            self._codes = np.insert(self._codes, slots[fresh], chunk_codes[fresh])
            self._counts = np.insert(self._counts, slots[fresh], chunk_counts[fresh])

    def finalize(self):
        self._fold_chunk()
        return self._codes, self._counts


def _inverse_permutation(order):
    positions = np.empty(order.size, dtype=np.int64)
    positions[order] = np.arange(order.size)
    return positions


def ascending_pair_count(codes, counts, positions, vocab_size):
    left = codes // vocab_size
    right = codes % vocab_size
    return int(counts[positions[left] < positions[right]].sum())


def _net_edges(codes, counts, vocab_size, slice_size=8_000_000):
    """Keep, for each conflicting token pair, only the heavier direction with its
    net weight W[u][v] - W[v][u]; ties and self-consistent pairs drop out.
    Processed in slices to bound temporary allocations."""
    kept_left = []
    kept_right = []
    kept_net = []
    for start in range(0, codes.size, slice_size):
        stop = min(start + slice_size, codes.size)
        left = codes[start:stop] // vocab_size
        right = codes[start:stop] % vocab_size
        reverse_codes = right * vocab_size + left
        reverse_slots = np.minimum(np.searchsorted(codes, reverse_codes), codes.size - 1)
        has_reverse = codes[reverse_slots] == reverse_codes
        reverse_counts = np.where(has_reverse, counts[reverse_slots], 0)
        net = counts[start:stop] - reverse_counts
        keep = net > 0
        kept_left.append(left[keep])
        kept_right.append(right[keep])
        kept_net.append(net[keep])
    if not kept_left:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    return np.concatenate(kept_left), np.concatenate(kept_right), np.concatenate(kept_net)


def _csr(heads, values_by_edge, num_nodes):
    order = np.argsort(heads, kind="stable")
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(heads, minlength=num_nodes))
    return indptr, [values[order] for values in values_by_edge]


def _strongly_connected_components(heads, tails, num_nodes):
    """Iterative Tarjan; returns (component_id per node, component count)."""
    indptr, (targets,) = _csr(heads, (tails,), num_nodes)
    index = np.full(num_nodes, -1, dtype=np.int64)
    lowlink = np.zeros(num_nodes, dtype=np.int64)
    on_stack = np.zeros(num_nodes, dtype=bool)
    component = np.full(num_nodes, -1, dtype=np.int64)
    next_index = 0
    num_components = 0
    member_stack = []
    for root in range(num_nodes):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            node, edge_offset = work.pop()
            if edge_offset == 0:
                index[node] = lowlink[node] = next_index
                next_index += 1
                member_stack.append(node)
                on_stack[node] = True
            descended = False
            for slot in range(indptr[node] + edge_offset, indptr[node + 1]):
                target = targets[slot]
                if index[target] == -1:
                    work.append((node, slot - indptr[node] + 1))
                    work.append((target, 0))
                    descended = True
                    break
                if on_stack[target]:
                    lowlink[node] = min(lowlink[node], index[target])
            if descended:
                continue
            if lowlink[node] == index[node]:
                while True:
                    member = member_stack.pop()
                    on_stack[member] = False
                    component[member] = num_components
                    if member == node:
                        break
                num_components += 1
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
    return component, num_components


def _condensation_order(component, num_components, heads, tails, warm_positions):
    """Topologically order components (Kahn), breaking ties toward the smallest
    mean warm position so the global early-to-late gradient is preserved.
    Tokens stay contiguous per component, in warm order within it."""
    component_heads = component[heads]
    component_tails = component[tails]
    cross = component_heads != component_tails
    if cross.any():
        edge_codes = np.unique(component_heads[cross] * num_components + component_tails[cross])
        component_heads = (edge_codes // num_components).astype(np.int64)
        component_tails = (edge_codes % num_components).astype(np.int64)
    else:
        component_heads = np.empty(0, dtype=np.int64)
        component_tails = np.empty(0, dtype=np.int64)

    position_sums = np.bincount(component, weights=warm_positions, minlength=num_components)
    member_counts = np.bincount(component, minlength=num_components)
    priorities = position_sums / np.maximum(member_counts, 1)

    indegree = np.bincount(component_tails, minlength=num_components)
    indptr, (successors,) = _csr(component_heads, (component_tails,), num_components)
    ready = [(priorities[comp], comp) for comp in range(num_components) if indegree[comp] == 0]
    heapq.heapify(ready)
    topo_rank = np.empty(num_components, dtype=np.int64)
    rank = 0
    while ready:
        _, comp = heapq.heappop(ready)
        topo_rank[comp] = rank
        rank += 1
        for slot in range(indptr[comp], indptr[comp + 1]):
            successor = successors[slot]
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, (priorities[successor], successor))
    if rank != num_components:
        raise RuntimeError("condensation graph is not acyclic; SCC computation is broken")
    return np.lexsort((warm_positions, topo_rank[component])).astype(np.int64)


def _insertion_candidates(indptr, neighbors, gains, positions, tokens):
    """Best single-token reinsertion per token against a positions snapshot.

    Returns (improvements, desired_keys) for `tokens`; improvement is the exact
    objective delta if only that token moved."""
    improvements = np.zeros(tokens.size, dtype=np.int64)
    desired_keys = np.zeros(tokens.size, dtype=np.float64)
    for slot, token in enumerate(tokens):
        start, stop = indptr[token], indptr[token + 1]
        neighbor_positions = positions[neighbors[start:stop]]
        sorted_slots = np.argsort(neighbor_positions)
        sorted_positions = neighbor_positions[sorted_slots]
        # prefix[k] = sum of gains of neighbors placed before token if token sits at slot k
        prefix = np.zeros(sorted_slots.size + 1, dtype=np.int64)
        np.cumsum(gains[start:stop][sorted_slots], out=prefix[1:])
        current_slot = int(np.searchsorted(sorted_positions, positions[token]))
        best_slot = int(np.argmin(prefix))
        improvements[slot] = 2 * (prefix[current_slot] - prefix[best_slot])
        if best_slot == 0:
            desired_keys[slot] = sorted_positions[0] - 0.5
        elif best_slot == sorted_slots.size:
            desired_keys[slot] = sorted_positions[-1] + 0.5
        else:
            desired_keys[slot] = (sorted_positions[best_slot - 1] + sorted_positions[best_slot]) / 2.0
    return improvements, desired_keys


def _build_incidence(heads, tails, nets, vocab_size):
    """Signed-gain incidence CSR: for each token, its net-edge neighbors and
    the gain of ordering the token before each neighbor."""
    incidence_heads = np.concatenate([heads, tails])
    incidence_neighbors = np.concatenate([tails, heads])
    incidence_gains = np.concatenate([nets, -nets])
    indptr, (neighbors, gains) = _csr(incidence_heads, (incidence_neighbors, incidence_gains), vocab_size)
    movable = np.flatnonzero(indptr[1:] > indptr[:-1])
    return indptr, neighbors, gains, movable


def _batch_insertion(codes, counts, vocab_size, order, incidence, max_passes):
    """Batch insertion local search: every pass proposes best-reinsertion moves
    from one positions snapshot, applies a fraction of them (highest gain
    first), and keeps the result only if the true objective improved."""
    indptr, neighbors, gains, movable = incidence

    positions = _inverse_permutation(order)
    best_score = ascending_pair_count(codes, counts, positions, vocab_size)
    move_fraction = 1.0
    passes = 0
    while passes < max_passes and movable.size:
        improvements, desired_keys = _insertion_candidates(indptr, neighbors, gains, positions, movable)
        improving = improvements > 0
        if not improving.any():
            break
        candidates = movable[improving]
        candidate_keys = desired_keys[improving]
        candidate_gains = improvements[improving]
        while passes < max_passes:
            passes += 1
            take = max(1, int(round(move_fraction * candidates.size)))
            chosen = np.argsort(-candidate_gains, kind="stable")[:take]
            keys = positions.astype(np.float64)
            keys[candidates[chosen]] = candidate_keys[chosen]
            trial_order = np.argsort(keys, kind="stable").astype(np.int64)
            trial_positions = _inverse_permutation(trial_order)
            trial_score = ascending_pair_count(codes, counts, trial_positions, vocab_size)
            if trial_score > best_score:
                order = trial_order
                positions = trial_positions
                best_score = trial_score
                move_fraction = min(1.0, move_fraction * 2.0)
                break
            move_fraction /= 2.0
            if move_fraction * candidates.size < 1.0:
                return order, best_score, passes
    return order, best_score, passes


def perturb_order(order, rng, num_moves=8, max_block=256):
    """ILS kick: relocate (and sometimes reverse) a few random blocks."""
    order = order.copy()
    for _ in range(num_moves):
        length = int(rng.integers(2, min(max_block, order.size - 1) + 1))
        start = int(rng.integers(0, order.size - length + 1))
        block = order[start:start + length].copy()
        if rng.random() < 0.5:
            block = block[::-1]
        rest = np.concatenate([order[:start], order[start + length:]])
        destination = int(rng.integers(0, rest.size + 1))
        order = np.concatenate([rest[:destination], block, rest[destination:]])
    return order


_ILS_STATE = None


def _ils_worker(seed):
    codes, counts, vocab_size, order, incidence, max_passes, num_moves, max_block = _ILS_STATE
    rng = np.random.default_rng(seed)
    start_order = perturb_order(order, rng, num_moves=num_moves, max_block=max_block)
    new_order, score, _ = _batch_insertion(codes, counts, vocab_size, start_order, incidence, max_passes)
    return int(score), int(seed), new_order


def _run_ils_generation(seeds, jobs):
    if jobs <= 1 or len(seeds) <= 1:
        results = [_ils_worker(seed) for seed in seeds]
    else:
        try:
            context = multiprocessing.get_context("fork")
        except ValueError:
            results = [_ils_worker(seed) for seed in seeds]
        else:
            # fork shares _ILS_STATE (the large read-only arrays) copy-on-write
            with context.Pool(processes=min(jobs, len(seeds))) as pool:
                results = pool.map(_ils_worker, seeds)
    # deterministic winner regardless of completion order
    results.sort(key=lambda result: (-result[0], result[1]))
    return results[0]


def _iterated_local_search(codes, counts, vocab_size, order, best_score, incidence, max_passes, restarts, generations, seed, jobs, num_moves, max_block):
    global _ILS_STATE
    base_score = best_score
    generation_trace = []
    for generation in range(generations):
        _ILS_STATE = (codes, counts, vocab_size, order, incidence, max_passes, num_moves, max_block)
        seeds = [seed * 1000003 + generation * 1009 + restart for restart in range(restarts)]
        candidate_score, candidate_seed, candidate_order = _run_ils_generation(seeds, jobs)
        adopted = candidate_score > best_score
        if adopted:
            order = candidate_order
            best_score = candidate_score
        generation_trace.append({
            "generation": generation,
            "best_candidate": int(candidate_score),
            "candidate_seed": int(candidate_seed),
            "incumbent": int(best_score),
            "adopted": adopted,
        })
        print(f"ils generation {generation}: candidate={candidate_score} incumbent={best_score} adopted={adopted}", flush=True)
    _ILS_STATE = None
    metrics = {
        "restarts": restarts,
        "generations_run": generations,
        "jobs": jobs,
        "base_ascending": int(base_score),
        "ascending_after_ils": int(best_score),
        "improvement_pairs": int(best_score - base_score),
        "generations": generation_trace,
    }
    return order, best_score, metrics


def optimize_order(codes, counts, vocab_size, warm_order, max_passes=40, ils_restarts=0, ils_generations=0, ils_seed=0, jobs=1, ils_moves=8, ils_max_block=256):
    """Compute a vocab order maximizing ascending adjacent pairs.

    Never returns an order worse than warm_order: the SCC condensation step is
    provably non-losing, insertion passes only keep strict improvements, and
    ILS only adopts candidates that beat the incumbent."""
    warm_order = np.asarray(warm_order, dtype=np.int64)
    warm_positions = _inverse_permutation(warm_order)
    total_pairs = int(counts.sum())
    ascending_before = ascending_pair_count(codes, counts, warm_positions, vocab_size)

    heads, tails, nets = _net_edges(codes, counts, vocab_size)
    component, num_components = _strongly_connected_components(heads, tails, vocab_size)
    order = _condensation_order(component, num_components, heads, tails, warm_positions)
    incidence = _build_incidence(heads, tails, nets, vocab_size)
    order, best_score, passes = _batch_insertion(codes, counts, vocab_size, order, incidence, max_passes)

    ils_metrics = None
    if ils_restarts > 0 and ils_generations > 0:
        order, best_score, ils_metrics = _iterated_local_search(
            codes, counts, vocab_size, order, best_score, incidence, max_passes,
            restarts=ils_restarts, generations=ils_generations, seed=ils_seed,
            jobs=jobs, num_moves=ils_moves, max_block=ils_max_block,
        )

    component_sizes = np.bincount(component, minlength=num_components)
    stats = {
        "total_pairs": total_pairs,
        "ascending_before": ascending_before,
        "ascending_after": best_score,
        "num_components": int(num_components),
        "largest_component": int(component_sizes.max()) if num_components else 0,
        "num_net_edges": int(nets.size),
        "passes": passes,
    }
    if ils_metrics is not None:
        stats["ils"] = ils_metrics
    return order, stats


def reorder_vocab_rows(rows, order):
    """Re-index vocab rows to the new order; returns (new_rows, old_to_new)."""
    old_to_new = [0] * len(rows)
    for new_index, old_index in enumerate(order.tolist()):
        old_to_new[old_index] = new_index
    rows_by_old_index = {int(row["index"]): row for row in rows}
    new_rows = []
    for new_index, old_index in enumerate(order.tolist()):
        row = dict(rows_by_old_index[int(old_index)])
        row["index"] = new_index
        new_rows.append(row)
    return new_rows, old_to_new


def run_reorder(vocab_path, records_path, out_dir, chunk_size=4_000_000, include_clause_crossings=False, max_passes=40, progress_every=100000, pair_counts_path=None, ils_restarts=0, ils_generations=0, ils_seed=0, jobs=1):
    started_at = time.time()
    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab_rows = json.load(file)
    vocab_size = len(vocab_rows)
    # record indices are the token ids; the existing contiguous index order is the warm start
    warm_order = np.arange(vocab_size, dtype=np.int64)

    if pair_counts_path is not None:
        saved = np.load(pair_counts_path)
        if int(saved["vocab_size"]) != vocab_size:
            raise ValueError(f"pair counts vocab size {int(saved['vocab_size'])} does not match vocab size {vocab_size}")
        codes = saved["codes"]
        counts = saved["counts"]
        stories = int(saved["stories"])
    else:
        counter = PairCounter(vocab_size=vocab_size, chunk_size=chunk_size, include_clause_crossings=include_clause_crossings)
        stories = 0
        for _, _, stream in story_index_streams(iter_records(records_path)):
            counter.add_stream(stream)
            stories += 1
            if progress_every > 0 and stories % progress_every == 0:
                print(f"counted pairs for {stories} stories", flush=True)
        codes, counts = counter.finalize()

    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        Path(out_dir) / "pair_counts.npz",
        codes=codes,
        counts=counts,
        vocab_size=np.int64(vocab_size),
        stories=np.int64(stories),
        include_clause_crossings=np.bool_(include_clause_crossings),
    )

    order, stats = optimize_order(
        codes, counts, vocab_size, warm_order, max_passes=max_passes,
        ils_restarts=ils_restarts, ils_generations=ils_generations, ils_seed=ils_seed, jobs=jobs,
    )
    new_rows, old_to_new = reorder_vocab_rows(vocab_rows, order)

    report = dict(stats)
    report.update({
        "format": "phrase_vocab_reorder_report_v1",
        "vocab": str(vocab_path),
        "records": str(records_path) if records_path is not None else str(pair_counts_path),
        "vocab_size": vocab_size,
        "stories": stories,
        "distinct_pairs": int(codes.size),
        "include_clause_crossings": include_clause_crossings,
        "ascending_fraction_before": stats["ascending_before"] / stats["total_pairs"] if stats["total_pairs"] else 0.0,
        "ascending_fraction_after": stats["ascending_after"] / stats["total_pairs"] if stats["total_pairs"] else 0.0,
        "elapsed_seconds": time.time() - started_at,
    })

    out_dir = Path(out_dir)
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as file:
        json.dump(new_rows, file, ensure_ascii=False)
        file.write("\n")
    with open(out_dir / "old_to_new.json", "w", encoding="utf-8") as file:
        json.dump(old_to_new, file)
        file.write("\n")
    with open(out_dir / "reorder_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Reorder the phrase vocab to maximize ascending adjacent index pairs.")
    parser.add_argument("--vocab", required=True, help="Path to existing vocab.json (warm start).")
    parser.add_argument("--records", default=None, help="Path to phrase_index.jsonl or .jsonl.gz. Not needed with --pair-counts.")
    parser.add_argument("--out-dir", required=True, help="Output directory for vocab.json, old_to_new.json, reorder_report.json, pair_counts.npz.")
    parser.add_argument("--chunk-size", type=int, default=4_000_000, help="Pair buffer entries before folding (8 bytes each).")
    parser.add_argument("--include-clause-crossings", action="store_true", help="Also count adjacent pairs across clause boundaries.")
    parser.add_argument("--max-passes", type=int, default=40)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--pair-counts", default=None, help="Reuse a pair_counts.npz from a previous run instead of streaming records.")
    parser.add_argument("--ils-restarts", type=int, default=0, help="Perturbed local searches per ILS generation (0 disables ILS).")
    parser.add_argument("--ils-generations", type=int, default=0)
    parser.add_argument("--ils-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers per ILS generation (fork-based).")
    args = parser.parse_args()
    if args.records is None and args.pair_counts is None:
        parser.error("either --records or --pair-counts is required")
    return args


def main():
    args = parse_args()
    report = run_reorder(
        vocab_path=args.vocab,
        records_path=args.records,
        out_dir=args.out_dir,
        chunk_size=args.chunk_size,
        include_clause_crossings=args.include_clause_crossings,
        max_passes=args.max_passes,
        progress_every=args.progress_every,
        pair_counts_path=args.pair_counts,
        ils_restarts=args.ils_restarts,
        ils_generations=args.ils_generations,
        ils_seed=args.ils_seed,
        jobs=args.jobs,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
