#!/bin/bash
# +1-epoch hybrid-cross post-train on the 300k set (data-scaling test), from the
# v2 BASE checkpoint (not the overfit extended-post checkpoint). Same guards.
set -euo pipefail
cd "$(dirname "$0")/.."

PIDFILE=phrase_gpt_post300k.pid
LOG=post_train_300k.log
WATCHDOG_LOG=post_train_300k_watchdog.log
MAX_RSS_GB="${MAX_RSS_GB:-15}"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Trainer already running (pid $(cat "$PIDFILE")); refusing to start a duplicate." >&2
    exit 1
fi
if pgrep -f "scripts.train_phrase_gpt" >/dev/null 2>&1; then
    echo "A trainer is already running; wait for it or kill it first:" >&2
    pgrep -fl "scripts.train_phrase_gpt" >&2 || true
    exit 1
fi

RESUME_FROM=phrase_gpt_base_v2_out/phrase_gpt.pt
if [ -f phrase_gpt_hybrid_cross_post300k_out/phrase_gpt.pt ]; then
    echo "300k post-train rolling checkpoint found; resuming from it."
    RESUME_FROM=phrase_gpt_hybrid_cross_post300k_out/phrase_gpt.pt
fi

caffeinate -i uv run python -m scripts.train_phrase_gpt \
    --vocab phrase_quote_split_ils_out/vocab.json \
    --shards phrase_hybrid_cross_shards_300k/manifest.json \
    --out-dir phrase_gpt_hybrid_cross_post300k_out \
    --sequence-len 128 --depth 2 --n-embd 128 --n-head 4 \
    --batch-size 16 --lr 1e-3 --epochs 4 --vocab-top-k 8191 \
    --checkpoint-every-shards 2 \
    --resume "$RESUME_FROM" \
    2>&1 | tee -a "$LOG" &
TEE_PID=$!
echo "$TEE_PID" > "$PIDFILE"

(
    while kill -0 "$TEE_PID" 2>/dev/null; do
        TRAIN_PID=""
        for p in $(pgrep -f "scripts.train_phrase_gpt" || true); do
            case "$(ps -o comm= -p "$p" 2>/dev/null | sed 's|.*/||')" in [Pp]ython*) TRAIN_PID="$p" ;; esac
        done
        if [ -n "$TRAIN_PID" ]; then
            RSS_KB=$(ps -o rss= -p "$TRAIN_PID" | tr -d ' ' || echo 0)
            echo "$(date '+%Y-%m-%d %H:%M:%S') pid=$TRAIN_PID rss_mb=$((${RSS_KB:-0} / 1024))" >> "$WATCHDOG_LOG"
            if [ "${RSS_KB:-0}" -gt $((MAX_RSS_GB * 1048576)) ]; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') WATCHDOG: rss ${RSS_KB}KB > ${MAX_RSS_GB}GB, killing trainer" | tee -a "$WATCHDOG_LOG" >&2
                pkill -f "scripts.train_phrase_gpt" || true
            fi
        fi
        sleep 60
    done
    rm -f "$PIDFILE"
) &

echo "300k post-train started (pipeline pid $TEE_PID; base has 3 epochs, --epochs 4 = exactly one epoch on 300k)."
wait "$TEE_PID"
