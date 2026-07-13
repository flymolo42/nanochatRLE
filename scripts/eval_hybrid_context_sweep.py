"""
CLI for the hybrid context sweep eval. Logic lives in scripts.hybrid_sweep.
"""

import argparse
import json

import torch

from scripts.hybrid_sweep import (
    SweepProbe, build_sweep_probes, context_steps_for_probe, topk_and_ce,
    predict_probe_logits, run_sweep, _parse_int_list,
)
from scripts.train_phrase_gpt import choose_device
from scripts.train_phrase_vectors import iter_records
from scripts.eval_phrase_gpt_packed_vs_single import load_model_from_checkpoint, resolve_vocab_remap


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid context sweep eval (top-k accuracy + perplexity vs X and D).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--vocab", default=None, help="Original vocab.json, if the checkpoint's recorded path is wrong.")
    parser.add_argument("--x-values", default="0,1,2,4,8,16")
    parser.add_argument("--d-values", default="1,2,4,8,16,32", help="Compressed-depth caps; unbounded appended automatically.")
    parser.add_argument("--fixed-x-for-depth", type=int, default=0)
    parser.add_argument("--min-history", type=int, default=1)
    parser.add_argument("--max-probes", type=int, default=20000)
    parser.add_argument("--split", default=None, help="Only score stories from this split (e.g. 'validation').")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for CIs; 0 disables.")
    parser.add_argument("--sweep-seed", type=int, default=0, help="Seed for bootstrap CIs.")
    parser.add_argument("--device", default="")
    parser.add_argument("--index-map", default=None, help="old_to_new.json applied to record indices (use when the checkpoint was trained on reordered shards).")
    parser.add_argument("--cross-clause", action="store_true", help="Build probe history with cross-clause chains (match hybrid-cross training).")
    parser.add_argument("--sae", default=None, help="sae_best.pt; enables SAE front encoding.")
    parser.add_argument("--sae-mode", choices=["chain", "window"], default="chain")
    parser.add_argument("--sae-window", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    index_map = None
    if args.index_map:
        with open(args.index_map, "r", encoding="utf-8") as file:
            index_map = json.load(file)
    probes = build_sweep_probes(iter_records(args.records), min_history=args.min_history,
                                max_probes=args.max_probes, split=args.split, index_map=index_map)
    if not probes:
        raise SystemExit("No probes found.")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, _config = load_model_from_checkpoint(checkpoint, device=device)
    remap = resolve_vocab_remap(checkpoint.get("config", {}), vocab_override=args.vocab)
    if remap is not None:
        remap = remap.to("cpu")
    front_encoder = None
    if args.sae:
        from scripts.build_sae_context_shards import sae_front_encoder
        from scripts.sae import load_sae
        from scripts.train_phrase_gpt import load_vocab_top_k_remap
        sae = load_sae(args.sae)
        lookup, tokens = load_vocab_top_k_remap(args.vocab, 8191)
        front_encoder = sae_front_encoder(sae, mode=args.sae_mode, window=args.sae_window, latent_offset=len(tokens), lookup=lookup, index_map=None)
        # Keep `remap` (the vocab-sized ILS->top-8k lookup) even for --sae runs:
        # it still scores the TARGET token in _aggregate and drives the
        # classic_1hot baseline, neither of which touch SAE latent ids.
        # _probe_contexts's front_encoder branch never re-remaps the front/tail
        # steps it builds, so `remap` here cannot double-remap or corrupt them.
    print(f"probes={len(probes)} device={device} remap={'yes' if remap is not None else 'no'}", flush=True)
    # Compressed contexts longer than the model's sequence_len are truncated by the
    # final-layer attention window, so large D and unbounded D can coincide on long
    # stories — the D-sweep is only meaningful up to ~sequence_len chains of history.
    d_values = _parse_int_list(args.d_values) + [None]
    result = run_sweep(model, probes, x_values=_parse_int_list(args.x_values), d_values=d_values,
                       fixed_x_for_depth=args.fixed_x_for_depth, remap=remap, batch_size=args.batch_size,
                       device=device, bootstrap=args.bootstrap, bootstrap_seed=args.sweep_seed,
                       reset_on_clause=not args.cross_clause, front_encoder=front_encoder)
    result["split"] = args.split
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
