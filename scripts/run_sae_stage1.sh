#!/bin/bash
# Stage 1: extract bags, train SAE grids (chain + window), record gate verdicts.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== extract train bags (60k stories ~ 2M+ bags) ==="
uv run python -m scripts.extract_sae_bags \
    --records phrase_quote_split_out/phrase_index.jsonl.gz \
    --index-map phrase_quote_split_ils_out/old_to_new.json \
    --vocab phrase_quote_split_ils_out/vocab.json --vocab-top-k 8191 \
    --out-dir sae_bags_out --max-stories 60000

echo "=== extract validation bags ==="
uv run python -m scripts.extract_sae_bags \
    --records phrase_quote_split_out/phrase_index_validation.jsonl.gz \
    --index-map phrase_quote_split_ils_out/old_to_new.json \
    --vocab phrase_quote_split_ils_out/vocab.json --vocab-top-k 8191 \
    --out-dir sae_bags_val_out --max-stories 5000

echo "=== chain-arm grid ==="
uv run python -m scripts.train_sae_grid \
    --train-bags sae_bags_out/chain_bags.npz --val-bags sae_bags_val_out/chain_bags.npz \
    --out-dir sae_chain_out --epochs 3 --device mps

echo "=== window-arm grid ==="
uv run python -m scripts.train_sae_grid \
    --train-bags sae_bags_out/window_bags.npz --val-bags sae_bags_val_out/window_bags.npz \
    --out-dir sae_window_out --epochs 3 --device mps

echo "=== gate verdicts ==="
python3 -c "import json; [print(d, json.load(open(f'{d}/grid_report.json'))['gate_passed'], json.load(open(f'{d}/grid_report.json'))['best']) for d in ('sae_chain_out','sae_window_out')]"
