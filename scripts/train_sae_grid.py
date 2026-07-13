"""
Stage-1 SAE grid: train each (d, k) config, score validation set-recovery,
save the best SAE and a gate verdict.

Example:
python -m scripts.train_sae_grid \
    --train-bags sae_bags_out/chain_bags.npz \
    --val-bags sae_bags_val_out/chain_bags.npz \
    --out-dir sae_chain_out --epochs 3 --device mps
"""

import argparse
import json
import os

from scripts.extract_sae_bags import load_bags
from scripts.sae import save_sae, set_recovery, train_sae

DEFAULT_GRID = [(512, 8), (512, 32), (1024, 8), (1024, 32)]


def _gate_f1(sae, val_bags, input_dim, gate_max_len, device):
    eligible = [bag for bag in val_bags if len(bag) <= gate_max_len]
    if not eligible:
        return 0.0
    return set_recovery(sae, eligible, input_dim, device=device)["f1"]


def run_grid(train_bags, val_bags, input_dim, grid, epochs, batch_size, device, out_dir, gate_f1=0.9, gate_max_len=6, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    configs = []
    best = None
    best_sae = None
    for latent_dim, k in grid:
        sae = train_sae(train_bags, input_dim, latent_dim, k, epochs=epochs, batch_size=batch_size, seed=seed, device=device, progress_every=1)
        metrics = set_recovery(sae, val_bags, input_dim, device=device)
        gate_value = _gate_f1(sae, val_bags, input_dim, gate_max_len, device)
        name = f"d{latent_dim}_k{k}"
        configs.append({"name": name, "d": latent_dim, "k": k, "gate_f1": gate_value, **metrics})
        print(f"grid {name}: f1={metrics['f1']} gate_f1={gate_value}", flush=True)
        if best is None or gate_value > best["gate_f1"]:
            best = configs[-1]
            best_sae = sae
    save_sae(best_sae, os.path.join(out_dir, "sae_best.pt"))
    report = {
        "format": "sae_grid_report_v1",
        "configs": configs,
        "best": best["name"],
        "gate_threshold": gate_f1,
        "gate_max_len": gate_max_len,
        "gate_passed": best["gate_f1"] >= gate_f1,
        "input_dim": input_dim,
        "train_bags": len(train_bags),
        "val_bags": len(val_bags),
    }
    with open(os.path.join(out_dir, "grid_report.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Stage-1 SAE grid and record the gate verdict.")
    parser.add_argument("--train-bags", required=True)
    parser.add_argument("--val-bags", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gate-f1", type=float, default=0.9)
    parser.add_argument("--gate-max-len", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    train_bags, input_dim = load_bags(args.train_bags)
    val_bags, val_dim = load_bags(args.val_bags)
    if val_dim != input_dim:
        raise SystemExit(f"train input_dim {input_dim} != val input_dim {val_dim}")
    report = run_grid(
        train_bags, val_bags, input_dim, DEFAULT_GRID,
        epochs=args.epochs, batch_size=args.batch_size, device=args.device,
        out_dir=args.out_dir, gate_f1=args.gate_f1, gate_max_len=args.gate_max_len, seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
