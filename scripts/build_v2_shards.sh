#!/bin/bash
# Build the two shard sets for the new-vocab (quote-split + ILS) pipeline:
#   A. 30k-example hybrid-cross post-train set (minutes)
#   B. full-corpus token-mode base-training shards (~3-4h streaming)
set -euo pipefail
cd "$(dirname "$0")/.."

VOCAB=phrase_quote_split_ils_out/vocab.json
RECORDS=phrase_quote_split_out/phrase_index.jsonl.gz
MAP=phrase_quote_split_ils_out/old_to_new.json

echo "=== Stage A: 30k hybrid-cross post-train set -> phrase_hybrid_cross_shards_len128_posttrain ==="
uv run python -m scripts.build_phrase_gpt_shards \
    --vocab "$VOCAB" --records "$RECORDS" --index-map "$MAP" \
    --out-dir phrase_hybrid_cross_shards_len128_posttrain \
    --sequence-len 128 --examples-per-shard 30000 --limit-examples 30000 \
    --chain-mode hybrid-cross --split-seed 42 --progress-every 200000

echo
echo "=== Stage B: full-corpus token-mode base shards -> phrase_gpt_shards_len128_v2 (~3-4h) ==="
uv run python -m scripts.build_phrase_gpt_shards \
    --vocab "$VOCAB" --records "$RECORDS" --index-map "$MAP" \
    --out-dir phrase_gpt_shards_len128_v2 \
    --sequence-len 128 --examples-per-shard 50000 \
    --chain-mode token --progress-every 200000

echo
echo "=== ALL SHARDS BUILT ==="
echo "Next step: scripts/run_base_training.sh (guarded: pidfile + RSS watchdog)"
