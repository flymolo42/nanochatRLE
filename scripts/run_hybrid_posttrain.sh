#!/bin/bash
# +1-epoch hybrid-cross post-train from the v2 base checkpoint, with in-training
# sweep eval on the true-validation probes. Same guards as run_base_training.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

PIDFILE=phrase_gpt_post_v2.pid
LOG=post_train_v2.log
WATCHDOG_LOG=post_train_v2_watchdog.log
MAX_RSS_GB="${MAX_RSS_GB:-15}"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Trainer already running (pid $(cat "$PIDFILE")); refusing to start a duplicate." >&2
    exit 1
fi
if pgrep -f "scripts.train_phrase_gpt" >/dev/null 2>&1 || pgrep -f "scripts.eval_hybrid_context_sweep" >/dev/null 2>&1; then
    echo "A trainer or sweep eval is still running; wait for it or kill it first:" >&2
    pgrep -fl "scripts.train_phrase_gpt|scripts.eval_hybrid_context_sweep" >&2 || true
    exit 1
fi

RESUME_FROM=phrase_gpt_base_v2_out/phrase_gpt.pt
if [ -f phrase_gpt_hybrid_cross_post_out/phrase_gpt.pt ]; then
    echo "Post-train rolling checkpoint found; resuming from it."
    RESUME_FROM=phrase_gpt_hybrid_cross_post_out/phrase_gpt.pt
fi

caffeinate -i uv run python -m scripts.train_phrase_gpt \
    --vocab phrase_quote_split_ils_out/vocab.json \
    --shards phrase_hybrid_cross_shards_len128_posttrain/manifest.json \
    --out-dir phrase_gpt_hybrid_cross_post_out \
    --sequence-len 128 --depth 2 --n-embd 128 --n-head 4 \
    --batch-size 16 --lr 1e-3 --epochs 4 --vocab-top-k 8191 \
    --save-best --checkpoint-every-shards 1 \
    --resume "$RESUME_FROM" \
    --sweep-eval-records phrase_quote_split_out/phrase_index_validation.jsonl.gz \
    --sweep-eval-split validation \
    --sweep-index-map phrase_quote_split_ils_out/old_to_new.json \
    --sweep-cross-clause --sweep-max-probes 2000 \
    2>&1 | tee -a "$LOG" &
TEE_PID=$!
echo "$TEE_PID" > "$PIDFILE"

(
    while kill -0 "$TEE_PID" 2>/dev/null; do
        TRAIN_PID=$(pgrep -f "python.*scripts.train_phrase_gpt" | head -1 || true)
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

echo "Post-train started (pipeline pid $TEE_PID, watchdog cap ${MAX_RSS_GB}GB)."
echo "Base checkpoint has 3 completed epochs; --epochs 4 runs exactly one epoch on the hybrid-cross set."
echo "Log: $LOG | RSS trace: $WATCHDOG_LOG"
wait "$TEE_PID"
