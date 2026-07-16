# k-way Hub-Token Duplication — Design

**Date:** 2026-07-15
**Status:** Approved (brainstorm with jgrimm)
**Question:** Does giving high-conflict tokens k>2 copies at data-chosen positions
raise ordering compressibility, especially on code where conflict is concentrated
in positionally-uniform punctuation and the ordering ceiling is only ~86%?

## Background

Hub-token duplication with k=2 (early/late) moved TinyStories chains only -2.6%
(below its 8% gate): the conflict there was diffuse (~1000 mid-frequency words)
and the ceiling already high (95.9%). Code is the opposite regime, measured on
the-stack-smol JS/TS (whole-identifier tokenization, 19,879-type vocab): 14.2%
irreducible conflict (vs TinyStories' 4.1%), of which the top 10 tokens hold
60% and the top 30 hold 73% — dominated by `,` `.` `(` `:` `=` `[` `)`. These
are positionally *omnipresent*, not bimodal, so a 2-copy split under-serves them.
k-way duplication gives each such token as many ranks as its position
distribution actually needs.

Scope: analysis-only (no training), like the PG-19 and code vocab experiments.

## Approach

### Candidate selection
Rank vocab tokens by irreducible conflict mass `loss(t) = sum_v min(W[t][v],
W[v][t])` (existing `plan_token_duplicates.conflict_losses`). Take the top
`--top-n` (default 300) as candidates; per-token k does the real filtering, so
N is generous.

### k per token — two modes, one code path
- **Data-driven (default):** for each candidate, collect its within-clause
  relative positions (position / max(clause_len - 1, 1), in [0, 1]) across all
  occurrences into a fixed histogram (e.g. 20 bins). Choose k = number of
  well-separated modes via 1-D clustering with a separation test, capped at
  `--k-max` (default 6). Monomodal -> k=1 (no copy, token dropped from the
  plan). Each copy's **target position** = its cluster centroid.
- **Fixed (`--fixed-k N`):** every top-`--top-n` token gets exactly N copies at
  evenly spaced target positions (N even splits of [0,1]). Crude comparison arm.

### Plan format (extends duplicates_plan_v1 -> kway_duplicates_plan_v1)
Per parent: `old_index`, `token`, `copies: [{new_index, target_position}]`
(length k, k>=2 for tokens that survive selection), `conflict_mass`. Renumbering
inserts all k-1 extra copies adjacent to the parent in index order (the parent
keeps the first copy slot). vocab_size_new = vocab_size_old + sum(k_i - 1).

### Occurrence -> copy assignment
Generalizes early/late: each occurrence of a duplicated token goes to the copy
whose `target_position` is nearest to the occurrence's own within-clause
relative position. Ties -> lowest-index copy. Copies are first-class vocab
tokens, so the existing reorder + chain-measurement run unchanged on the result.

### Measurement
Build the k-way duplicated vocab from a pair/position census, re-run reorder+ILS,
measure chains on held-out eval files. Report table rows: no-dup / k=2 / k-way
(data-driven) / k-way (fixed), columns: extra slots spent, achieved ascending %,
mean chain length, len-1 %, conflict-mass-removed-per-slot. The k=2 row reuses
existing `plan_token_duplicates` selection for an honest baseline.

## Components
- `scripts/plan_kway_duplicates.py` — new: position-histogram collection,
  cluster-based k selection (+ fixed-k mode), plan JSON writer. Unit-tested on
  synthetic position distributions with known modality.
- Generalized duplication assignment — extend the k=2 assignment in
  `scripts/duplicate_hub_tokens.py` (or a code-corpus analog) to nearest-of-k by
  target position; the transform stays a streaming per-clause rewrite.
- Reuse `reorder_phrase_vocab` (order) and `measure_chain_lengths` (eval),
  plus the code_stream / code_vocab census helpers to source JS/TS streams.
- A runner script driving no-dup / k=2 / k-way and writing the comparison report.

## Success criteria
1. k-selection is deterministic and correct on synthetic position distributions
   (uniform -> k=k_max, bimodal -> 2, monomodal -> 1).
2. Comparison table on held-out JS/TS eval quantifies k-way vs k=2 vs no-dup in
   achieved ascending %, mean chain length, and per-slot conflict reduction.
3. Result recorded in the workbook with reproduction commands.
