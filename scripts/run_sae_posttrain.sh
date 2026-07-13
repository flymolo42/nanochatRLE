#!/bin/bash
# Stage 2 arm post-train: ./scripts/run_sae_posttrain.sh <chain|window>
set -euo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?usage: run_sae_posttrain.sh <chain|window>}"
SHARDS="sae_${ARM}_shards/manifest.json"
OUT="phrase_gpt_sae_${ARM}_post_out"
PIDFILE="phrase_gpt_sae_${ARM}.pid"
LOG="sae_${ARM}_post_train.log"
WATCHDOG_LOG="sae_${ARM}_post_watchdog.log"
MAX_RSS_GB="${MAX_RSS_GB:-15}"
LATENT_DIM=$(python3 -c "import json; print(json.load(open('${SHARDS}'))['sae_latent_dim'])")

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Trainer already running; refusing duplicate." >&2; exit 1
fi
if pgrep -f "scripts.train_phrase_gpt" >/dev/null 2>&1; then
    echo "A trainer is already running; wait or kill it first." >&2; exit 1
fi

RESUME_FROM=phrase_gpt_base_v2_out/phrase_gpt.pt
if [ -f "$OUT/phrase_gpt.pt" ]; then RESUME_FROM="$OUT/phrase_gpt.pt"; fi

caffeinate -i uv run python -m scripts.train_phrase_gpt \
    --vocab phrase_gpt_base_v2_out/vocab_top_k.json \
    --shards "$SHARDS" \
    --out-dir "$OUT" \
    --sequence-len 128 --depth 2 --n-embd 128 --n-head 4 \
    --batch-size 16 --lr 1e-3 --epochs 4 \
    --extend-phrase-vocab "$LATENT_DIM" \
    --save-best --checkpoint-every-shards 1 \
    --resume "$RESUME_FROM" \
    2>&1 | tee -a "$LOG" &
TEE_PID=$!
echo "$TEE_PID" > "$PIDFILE"
(
    while kill -0 "$TEE_PID" 2>/dev/null; do
        TRAIN_PID=$(pgrep -f "scripts.train_phrase_gpt" | head -1 || true)
        if [ -n "$TRAIN_PID" ]; then
            RSS_KB=$(ps -o rss= -p "$TRAIN_PID" | tr -d ' ' || echo 0)
            echo "$(date '+%Y-%m-%d %H:%M:%S') pid=$TRAIN_PID rss_mb=$((${RSS_KB:-0} / 1024))" >> "$WATCHDOG_LOG"
            if [ "${RSS_KB:-0}" -gt $((MAX_RSS_GB * 1048576)) ]; then
                echo "WATCHDOG kill" | tee -a "$WATCHDOG_LOG" >&2
                pkill -f "scripts.train_phrase_gpt" || true
            fi
        fi
        sleep 60
    done
    rm -f "$PIDFILE"
) &
echo "SAE ${ARM} post-train started (pid $TEE_PID). No --vocab-top-k: shards are pre-remapped."
wait "$TEE_PID"
