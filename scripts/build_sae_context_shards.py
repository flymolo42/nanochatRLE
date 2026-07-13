"""
Build tensor shards whose compressed history slots are SAE codes (active latent
ids offset past the token vocab) and whose recent tail is 1-hot token ids.
Front/tail split mirrors train_phrase_gpt._hybrid_steps (random phrase boundary
per story, same split-seed arithmetic).

Example (chain arm):
python -m scripts.build_sae_context_shards \
    --records phrase_quote_split_out/phrase_index.jsonl.gz \
    --validation-records phrase_quote_split_out/phrase_index_validation.jsonl.gz \
    --index-map phrase_quote_split_ils_out/old_to_new.json \
    --vocab phrase_quote_split_ils_out/vocab.json --vocab-top-k 8191 \
    --sae sae_chain_out/sae_best.pt --mode chain \
    --out-dir sae_chain_shards --limit-examples 30000 --split-seed 42
"""

import argparse
import itertools
import json
import os
import random
import time

import torch

from scripts.build_phrase_gpt_shards import _write_shard
from scripts.extract_sae_bags import chain_bags_from_stream, remap_bags, window_bags_from_stream
from scripts.reorder_phrase_vocab import story_index_streams
from scripts.sae import bags_to_dense, load_sae
from scripts.train_phrase_gpt import PhraseSequenceExample, load_vocab_top_k_remap
from scripts.train_phrase_vectors import iter_records


def _clause_boundaries(stream):
    boundaries = [0]
    previous = None
    for position, (clause, _) in enumerate(stream):
        if previous is not None and clause != previous:
            boundaries.append(position)
        previous = clause
    boundaries.append(len(stream))
    return sorted(set(boundaries))


def _encode_front(front_stream, index_map, lookup, sae, mode, window, latent_offset):
    if not front_stream:
        return []
    if mode == "chain":
        bags = chain_bags_from_stream(front_stream, index_map)
    else:
        bags = window_bags_from_stream(front_stream, index_map, window=window)
    bags = remap_bags(bags, lookup)
    dense = bags_to_dense(bags, sae.input_dim)
    with torch.no_grad():
        _, indices = sae.encode(dense)
    return [sorted(latent_offset + latent for latent in set(row.tolist())) for row in indices]


def sae_front_encoder(sae, mode, window, latent_offset, lookup, index_map):
    """Returns callable(front_tokens, front_clauses) -> list of latent-id slots,
    for use as hybrid_sweep run_sweep(front_encoder=...)."""
    def encode(front_tokens, front_clauses):
        stream = list(zip(front_clauses, front_tokens))
        return _encode_front(stream, index_map, lookup, sae, mode, window, latent_offset)
    encode.tail_lookup = lookup
    return encode


def sae_steps_for_story(stream, index_map, lookup, sae, mode, window, split_seed, story_id, latent_offset, force_split=None):
    if not stream:
        return []
    if force_split is not None:
        split = force_split
    else:
        boundaries = _clause_boundaries(stream)
        split = random.Random(split_seed * 1_000_003 + int(story_id)).choice(boundaries)
    front = stream[:split]
    back = stream[split:]
    tail_tokens = [int(lookup[int(index_map[idx]) if index_map is not None else int(idx)]) for _, idx in back]
    slots = _encode_front(front, index_map, lookup, sae, mode, window, latent_offset)
    slots.extend([token] for token in tail_tokens)
    return _steps(slots, tail_tokens, len(front))


def _steps(slots, tail_tokens, front_len):
    # target of each step is the FIRST token of the next slot; only tail slots
    # (1-hot) can be targets, and every slot after the front is a tail slot.
    steps = []
    num_front = len(slots) - len(tail_tokens)
    for position in range(len(slots) - 1):
        next_slot_tail_index = position + 1 - num_front
        if next_slot_tail_index < 0:
            continue  # next slot is still a compressed front slot: no token target
        steps.append((slots[position], tail_tokens[next_slot_tail_index]))
    return steps


def build(records, out_dir, sae, lookup, index_map, mode, window, sequence_len, examples_per_shard, split_seed, latent_offset, max_examples=None, progress_every=10000, validation_records=None):
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    shards = []
    shard_counts = {}
    pending = {}
    examples_seen = 0

    def flush(split, force=False):
        rows = pending.get(split, [])
        while rows and (len(rows) >= examples_per_shard or (force and rows)):
            batch, rows = rows[:examples_per_shard], rows[examples_per_shard:]
            index = shard_counts.get(split, 0)
            shards.append(_write_shard(out_dir, index, batch, sequence_len, split=split))
            shard_counts[split] = index + 1
        pending[split] = rows

    def consume(stream_iter, limit):
        nonlocal examples_seen
        stories = 0
        for split, story_id, stream in stream_iter:
            stories += 1
            steps = sae_steps_for_story(stream, index_map, lookup, sae, mode, window, split_seed, story_id, latent_offset)
            for start in range(0, len(steps), sequence_len):
                chunk = steps[start:start + sequence_len]
                if not chunk:
                    continue
                pending.setdefault(split, []).append(PhraseSequenceExample(
                    input_indices=[ids for ids, _ in chunk],
                    targets=[target for _, target in chunk],
                ))
                examples_seen += 1
                if limit is not None and examples_seen >= limit:
                    flush(split)
                    return
            flush(split)
            if progress_every and stories % progress_every == 0:
                print(f"sae-sharded {stories} stories; examples={examples_seen}", flush=True)

    consume(story_index_streams(records), max_examples)
    if validation_records is not None:
        consume(story_index_streams(validation_records), None)
    for split in list(pending):
        flush(split, force=True)
    manifest = {
        "format": "phrase_gpt_tensor_shard_manifest_v1",
        "chain_mode": f"sae-{mode}",
        "sequence_len": sequence_len,
        "examples_per_shard": examples_per_shard,
        "split_seed": split_seed,
        "latent_offset": latent_offset,
        "sae_latent_dim": sae.latent_dim,
        "sae_k": sae.k,
        "num_shards": len(shards),
        "num_examples": sum(s["num_examples"] for s in shards),
        "splits": {
            split: {
                "num_shards": sum(1 for s in shards if s["split"] == split),
                "num_examples": sum(s["num_examples"] for s in shards if s["split"] == split),
            }
            for split in sorted({s["split"] for s in shards})
        },
        "num_tokens": sum(s["num_tokens"] for s in shards),
        "elapsed_seconds": time.time() - started,
        "shards": shards,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Build SAE-coded context shards.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--validation-records", default=None)
    parser.add_argument("--index-map", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--vocab-top-k", type=int, default=8191)
    parser.add_argument("--sae", required=True)
    parser.add_argument("--mode", choices=["chain", "window"], required=True)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sequence-len", type=int, default=128)
    parser.add_argument("--examples-per-shard", type=int, default=30000)
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.index_map, "r", encoding="utf-8") as file:
        index_map = json.load(file)
    lookup, tokens = load_vocab_top_k_remap(args.vocab, args.vocab_top_k)
    sae = load_sae(args.sae)
    if sae.input_dim != len(tokens):
        raise SystemExit(f"SAE input_dim {sae.input_dim} != top-k vocab size {len(tokens)}")
    manifest = build(
        records=iter_records(args.records),
        out_dir=args.out_dir,
        sae=sae, lookup=lookup, index_map=index_map,
        mode=args.mode, window=args.window,
        sequence_len=args.sequence_len, examples_per_shard=args.examples_per_shard,
        split_seed=args.split_seed, latent_offset=len(tokens),
        max_examples=args.limit_examples, progress_every=args.progress_every,
        validation_records=iter_records(args.validation_records) if args.validation_records else None,
    )
    print(json.dumps({k: manifest[k] for k in manifest if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
