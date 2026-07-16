"""One-off: extend the SAE scaling curve — d2048/d4096 at k=32 on the existing
chain bags, same gate metric. Reuses the tested run_grid machinery."""
import json

from scripts.extract_sae_bags import load_bags
from scripts.train_sae_grid import run_grid

train_bags, input_dim = load_bags("sae_bags_out/chain_bags.npz")
val_bags, val_dim = load_bags("sae_bags_val_out/chain_bags.npz")
assert input_dim == val_dim

report = run_grid(
    train_bags, val_bags, input_dim,
    grid=[(2048, 32), (4096, 32)],
    epochs=3, batch_size=4096, device="mps",
    out_dir="sae_chain_scale_out", gate_f1=0.9, gate_max_len=6, seed=0,
)
print(json.dumps({c["name"]: {"f1": c["f1"], "gate_f1": c["gate_f1"], "exact": c["exact_set_rate"]}
                  for c in report["configs"]} | {"gate_passed": report["gate_passed"]}, indent=2))
