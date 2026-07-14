# Hub-Token Duplication (Stage A: chain payoff) — Design

**Date:** 2026-07-13
**Status:** Draft for review
**Question:** Does reserving ~100 top-8k vocab slots for duplicate copies of
high-conflict hub tokens materially reduce chain count (timesteps)? Stage A
measures the chain-level payoff only; training (Stage B) happens only if the
gate passes.

## Background

Current best encoding (quote-split records + ILS order + cross-clause chaining):
5.07M chains on the 89,852-story sample, mean chain length 3.67, len-1 chains
8.1%. The ordering ceiling analysis showed 4.07% of pair mass is irreducibly
conflicted for any single-position vocab; ~85% of that mass involves top-1k
tokens. Duplication gives those tokens two positions so different occurrence
contexts can each ascend. Coverage cost of the budget: ranks 8,092-8,191 are
0.0054% of the corpus — negligible.

## Core design decisions

1. **Copies are first-class ILS vocab tokens.** The transform appends ~100 copy
   rows (vocab 46,327 -> ~46,427), inserting each copy adjacent to its parent in
   index order (all indices renumbered; the transform rewrites records through
   the renumber map anyway). Copy counts (split from parents) guarantee both
   copies land in any top-8k cut, naturally displacing the 100 tail tokens.
2. **Occurrence-to-copy assignment is context-class based, not greedy-ascent.**
   For a duplicated token t with copies t_early / t_late: an occurrence goes to
   t_late iff its predecessor in the canonical stream has a current-order
   position greater than t's own position (i.e., this occurrence would break a
   chain today); otherwise t_early. First-in-story occurrences go to t_early.
   Rationale: this split is stable under the subsequent reorder (which will
   re-place both copies optimally), avoiding the chicken-and-egg between copy
   choice and final positions. Greedy-ascent assignment against pre-reorder
   positions would bake in an order we're about to change.
3. **Selection metric:** per-token irreducible conflict mass
   loss(t) = sum over partners v of min(W[t][v], W[v][t]), computed from the
   saved `phrase_quote_split_ils_out/pair_counts.npz` (within-clause pairs —
   acceptable proxy; the reorder recounts on transformed records regardless).
   Take the top `--num-duplicates` (default 100) tokens, excluding tokens whose
   canonical stream frequency is so dominated by one side that a second copy
   is pointless (min-mass share < 5% of the token's total pair mass).
4. **Decodability preserved:** each copy has one fixed position (monotone
   constraint holds per copy); copies map to one surface token, so rendering
   is unchanged. Repeated-token breaks (t,t) are unaffected (copies don't help
   the equal case — known-negligible, 0.02% of chains).

## Pipeline (Stage A)

| Step | Tool | Cost |
|---|---|---|
| 1. Plan duplicates | new `scripts/plan_token_duplicates.py`: pair_counts.npz -> duplicates plan JSON (token, parent index, copy insert position, assignment threshold) | minutes |
| 2. Transform records | new `scripts/duplicate_hub_tokens.py` (mirrors split_quote_token: streaming, per-story, consistent across record representations by absolute position): rewrites indices through renumber map + copy assignment; emits records gz, extended vocab.json, copy_map.json (copy id -> canonical id), renumber old_to_new | ~3h (20.7GB stream+write) |
| 3. Reorder | existing `scripts.reorder_phrase_vocab` on transformed records + vocab, `--max-passes 200 --ils-restarts 8 --ils-generations 8 --jobs 8` | ~1.5h |
| 4. Measure | existing `scripts.measure_chain_lengths` (cross-clause), same 89,852-story sample | ~15 min |

All heavy steps run in a visible Terminal, per convention.

## Gate for Stage B

Proceed to training iff chain count on the sample drops >= 8% vs the 5.07M
baseline (equivalently: timestep reduction improves from 1.94x to >= ~2.1x).
Report regardless: chain count, mean length, len-1 share, ascending fraction
vs the (new, duplication-aware) ceiling.

## Stage B preview (out of scope here, costed for planning)

Token shards (~4h) + base train (~15h) + post-train/sweeps (~2h) ≈ 21h. Two
small code changes ride along: shard targets map copies -> canonical via
copy_map.json (output head shrinks to canonical tokens; input keeps copies)
and the sweep applies the same merge. Watch-item: each duplicated token's
embedding signal splits across copies; verify via the classic_1hot check.

## Success criteria

1. Duplicates plan is reproducible (seeded, deterministic from pair counts)
   and documented (per-token conflict mass table in the plan JSON).
2. Transformed records verified: per-story canonical streams contain only
   valid ids; copy occurrences match the assignment rule on sampled stories;
   non-duplicated tokens byte-identical through the renumber map.
3. Ladder table extended with the duplication row (chains, mean len, len-1,
   ascending fraction) in the workbook, gate verdict recorded.
