# Mixed-Corpus Vocabulary Ordering — Design

**Date:** 2026-07-15
**Status:** Approved (brainstorm with jgrimm)
**Question:** Does one vocabulary order over PG-19 prose + JS/TS code (the union)
compress each domain nearly as well as a bespoke per-domain order, or does
cross-domain conflict on shared tokens degrade it — and does k-way duplication
of the shared hubs recover the loss?

## Background

Real LLMs train on mixed code+prose. Our framework predicts mixing hurts
modestly and in a concentrated way: the two vocabularies barely overlap (English
words vs identifiers/operators), so interference is confined to SHARED tokens —
punctuation (`.` `,` `(` `)`), numbers, and `<str>` — which are exactly code's
highest-conflict hubs. Cross-domain conflict is additive (a token early-in-clause
in prose but late in code adds irreducible conflict neither corpus had alone), so
the union's ordering ceiling should be <= min(prose ceiling 91.8%, code ceiling
85.8%). k-way duplication can give a shared hub separate per-domain-position
copies, so the mixed setting is the ideal case for it.

Scope: analysis-only (no training), like the PG-19 and code experiments.

## Union token space
Each corpus is tokenized by its own tokenizer (`pg19_stream.tokenize_clauses`
for PG-19, `code_stream.tokenize_code` whole-identifier for JS/TS). Token
STRINGS are pooled: shared strings (`.`, `,`, digits, etc.) become one vocab
entry; words and identifiers stay disjoint. One census over both corpora builds
one vocab (min-count cut + trailing `<unk>`). All orders and measurements live in
this single union id space, so they are directly comparable.

## Orders (all permutations of the union vocab)
- **prose-order:** pair-count PG-19 streams (union-mapped) -> reorder+ILS -> positions_prose.
- **code-order:** pair-count JS/TS streams (union-mapped) -> positions_code.
- **mixed-order:** pair-count BOTH -> positions_mixed.

## Measurement (held-out files, cross-clause, cap 9)
- Held-out PG-19 chains under positions_prose (bespoke) vs positions_mixed.
- Held-out JS/TS chains under positions_code (bespoke) vs positions_mixed.
- Interference cost per domain = bespoke mean chain length - mixed.

## k-way recovery arm
Build a k-way duplication plan on the UNION pair counts (candidates = top
conflict-mass union tokens; data-driven k from union position histograms), apply,
reorder -> positions_mixed_kway. Measure held-out PG-19 and JS/TS chains under it.
Recovery per domain = (mixed+kway) - mixed. Report extra slots spent.

## Components
- `scripts/mixed_corpus_streams.py` — new: `tagged_streams(pg19_paths,
  code_paths)` yielding `(domain, [(clause, token), ...])` with domain in
  {"prose", "code"}; a `union_census` collecting counts + position stats across
  both; deterministic per-domain train/eval file split.
- `scripts/run_mixed_corpus_experiment.py` — new: build union lookup; produce the
  3 orders + k-way arm; measure each held-out domain under the relevant orders;
  write comparison report. Reuses census/`_build_lookup`/PairCounter/
  optimize_order/measure_chain_lengths and plan_kway_duplicates.
- Reuse existing pg19_stream, code_stream, reorder_phrase_vocab,
  measure_chain_lengths, plan_kway_duplicates unchanged.

## Success criteria
1. Union lookup maps both tokenizers' streams into one id space; shared symbols
   share an id, domain-specific tokens don't collide (unit-tested on a mixed toy).
2. Report table: for each domain {prose, code}, mean chain length + ascending %
   under bespoke / mixed / mixed+kway, plus interference cost and k-way recovery.
3. Result recorded in the workbook with reproduction commands. Interpretation
   caveat carried forward: k-way numbers are a lower bound (crude select_k).
