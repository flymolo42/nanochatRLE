# Predecessor-Rank k-Selection for k-way Duplication — Design

**Date:** 2026-07-15
**Status:** Approved (brainstorm with jgrimm)
**Question:** Does choosing k (and copy targets) from a token's PREDECESSOR-rank
distribution — rather than its within-clause position — correctly target the
high-conflict punctuation hubs that position-based select_k skips, and how much
does it improve k-way duplication on code and mixed corpora?

## Background / motivation

Inspecting the balanced mixed-corpus k-way plan revealed that position-based
`select_k` MIS-TARGETS: the highest-conflict punctuation (`.` `,` `(` `)` `{`
`}` `=` `;` `"`) got k=1 (NO copies), while position-agnostic content words got
k=4-5. Root cause: the tokenizer breaks clauses AFTER punctuation, so a
clause-terminal token is always the last token in its clause -> within-clause
relative position is a fixed spike at ~1.0 -> genuinely monomodal -> k=1, for
any modality test. Within-clause position is structurally blind to why `.` is
high-conflict: the conflict comes from the huge variety of tokens that PRECEDE
it, not from varying its own position.

k-way STILL beat bespoke on code despite this mis-targeting, so a correct signal
should widen the gap. The correct signal is the predecessor's rank — the direct
k-way generalization of the original 2-way rule (which keyed early/late on the
predecessor's index).

Scope: analysis-only.

## The predecessor-rank signal
Fix a reference order = the no-dup order for that run (already computed). For
each occurrence of a candidate token t in the training id-streams, its
predecessor is the previous token in the stream; record the predecessor's
normalized reference rank `reference_positions[pred] / vocab_size` in [0, 1].
Histogram these per token (same shape as position histograms). Non-degenerate
for `.`: its predecessors span all ranks -> broad distribution -> high k.

## k and assignment
Reuse `select_k` and `build_plan` unchanged — only the histogram fed in changes;
the copies' `target` now denotes a predecessor-rank band. New
`apply_kway_predrank(stream, plan, reference_positions)`: each occurrence goes
to the copy whose target is nearest its predecessor's reference rank (first
token in a stream has no predecessor -> use the parent's base copy; ties ->
lowest new_index). The reference order is FIXED at the no-dup order for both
pair-counting and eval, so assignment is deterministic (no chicken-and-egg);
the final order is then optimized over the assigned streams.

## Comparison (both corpora)
Add a `kway_predrank` arm beside the existing position-based `kway_data`:
- **code** (`run_kway_experiment`): arms nodup / k2 / kway_data (position) /
  kfixed4 / **kway_predrank** (new). Reference = nodup order.
- **mixed** (`run_mixed_corpus_experiment`): the existing mixed+kway arm plus a
  mixed+kway_predrank arm per domain. Reference = mixed no-dup order.
Same candidates, same ILS budget; only the k-signal differs. Report both against
no-dup: ascending %, mean chain length, per-slot gain, AND the duplication
breakdown (token + k) so position-skips-`.` vs predrank-duplicates-`.` is visible.

## Components
- `scripts/plan_kway_duplicates.py`: `collect_predecessor_rank_histograms(
  id_streams, reference_positions, vocab_size, bins=20) -> ndarray[vocab,bins]`;
  `apply_kway_predrank(stream, plan, reference_positions) -> [(clause, new_id)]`.
  Reuse `select_k`, `build_plan`, `_kway_renumber`.
- `scripts/run_kway_experiment.py`: add the `kway_predrank` arm (reference =
  nodup positions; the driver already builds a nodup order).
- `scripts/run_mixed_corpus_experiment.py`: add mixed+kway_predrank per domain
  (reference = mixed no-dup positions).
- A runner that chains the code and mixed experiments so both complete
  unattended.

## Success criteria
1. `apply_kway_predrank` assigns by predecessor reference-rank (unit-tested,
   incl. first-token-no-predecessor and str/int plan keys).
2. Duplication breakdown shows predecessor-rank duplicating the punctuation hubs
   position-based skipped.
3. Code + mixed comparison tables quantify kway_predrank vs kway_data (position)
   in ascending %, mean chain length, per-slot gain. Recorded in the workbook.
