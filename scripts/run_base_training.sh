#!/bin/bash
# Guarded base-model training on the v2 (new-vocab) token shards.
#   - refuses to start if a trainer is already running (pidfile + pgrep)
#   - auto-resumes from the rolling checkpoint if one exists
#   - watchdog kills the trainer if its RSS exceeds MAX_RSS_GB (default 15)
set -euo pipefail
cd "$(dirname "$0")/.."

PIDFILE=phrase_gpt_base_v2.pid
LOG=base_train_v2.log
WATCHDOG_LOG=base_train_v2_watchdog.log
MAX_RSS_GB="${MAX_RSS_GB:-15}"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Trainer already running (pid $(cat "$PIDFILE")); refusing to start a duplicate." >&2
    exit 1
fi
if pgrep -f "scripts.train_phrase_gpt" >/dev/null 2>&1; then
    echo "Stray train_phrase_gpt process found; kill it first:" >&2
    pgrep -fl "scripts.train_phrase_gpt" >&2
    exit 1
fi

RESUME_ARGS=()
if [ -f phrase_gpt_base_v2_out/phrase_gpt.pt ]; then
    echo "Rolling checkpoint found; resuming."
    RESUME_ARGS=(--resume phrase_gpt_base_v2_out/phrase_gpt.pt)
fi

caffeinate -i uv run python -m scripts.train_phrase_gpt \
    --vocab phrase_quote_split_ils_out/vocab.json \
    --shards phrase_gpt_shards_len128_v2/manifest.json \
    --out-dir phrase_gpt_base_v2_out \
    --sequence-len 128 --depth 2 --n-embd 128 --n-head 4 \
    --batch-size 16 --lr 1e-3 --epochs 3 --vocab-top-k 8191 \
    --save-best --patience 1 --checkpoint-every-shards 2 \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} 2>&1 | tee -a "$LOG" &
TEE_PID=$!
echo "$TEE_PID" > "$PIDFILE"

(
    while kill -0 "$TEE_PID" 2>/dev/null; do
        TRAIN_PID=$(pgrep -f "scripts.train_phrase_gpt" | head -1 || true)
        if [ -n "$TRAIN_PID" ]; then
            RSS_KB=$(ps -o rss= -p "$TRAIN_PID" | tr -d ' ' || echo 0)
            echo "$(date '+%Y-%m-%d %H:%M:%S') pid=$TRAIN_PID rss_gb=$(awk "BEGIN{printf \"%.1f\", $RSS_KB/1048576}")" >> "$WATCHDOG_LOG"
            if [ "${RSS_KB:-0}" -gt $((MAX_RSS_GB * 1048576)) ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') WATCHDOG: rss ${RSS_KB}KB > ${MAX_RSS_GB}GB, killing trainer" | tee -a "$WATCHDOG_LOG" >&2
                pkill -f "scripts.train_phrase_gpt" || true
            fi
        fi
        sleep 60
    done
    rm -f "$PIDFILE"
) &

echo "Training started (pipeline pid $TEE_PID, watchdog cap ${MAX_RSS_GB}GB)."
echo "Log: $LOG | RSS trace: $WATCHDOG_LOG"
echo "If it dies, re-run this script: it resumes from the rolling checkpoint."
wait "$TEE_PID"
