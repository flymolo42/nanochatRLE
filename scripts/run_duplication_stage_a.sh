#!/bin/bash
# Stage A: plan duplicates -> transform records -> reorder -> measure chains.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. plan duplicates (minutes) ==="
uv run python -m scripts.plan_token_duplicates \
    --pair-counts phrase_quote_split_ils_out/pair_counts.npz \
    --vocab phrase_quote_split_ils_out/vocab.json \
    --num-duplicates 100 \
    --out phrase_dup100_out/duplicates_plan.json

echo "=== 2. transform records (~3h) ==="
uv run python -m scripts.duplicate_hub_tokens \
    --records phrase_quote_split_out/phrase_index.jsonl.gz \
    --vocab phrase_quote_split_ils_out/vocab.json \
    --plan phrase_dup100_out/duplicates_plan.json \
    --out-dir phrase_dup100_out

echo "=== 3. reorder + ILS (~1.5h) ==="
uv run python -m scripts.reorder_phrase_vocab \
    --vocab phrase_dup100_out/vocab.json \
    --records phrase_dup100_out/phrase_index.jsonl.gz \
    --out-dir phrase_dup100_ils_out \
    --max-passes 200 --ils-restarts 8 --ils-generations 8 --ils-seed 1 --jobs 8 \
    --progress-every 200000

echo "=== 4. measure chains (cross-clause, 90k-story sample) ==="
uv run python -m scripts.measure_chain_lengths \
    --records phrase_dup100_out/phrase_index.jsonl.gz \
    --map dup100=phrase_dup100_ils_out/old_to_new.json \
    --no-clause-reset --max-records 70000000

echo "=== STAGE A DONE — baseline: 5,069,958 chains / 8.1% len-1; gate: >= 8% chain reduction ==="
