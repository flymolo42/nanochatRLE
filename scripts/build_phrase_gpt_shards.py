"""
Build tensor shards for phrase GPT training from phrase_index JSONL records.

Example:
python -m scripts.build_phrase_gpt_shards \
    --vocab phrase_vectors_full_out/vocab.json \
    --records phrase_vectors_full_out/phrase_index.jsonl.gz \
    --out-dir phrase_gpt_shards_len128 \
    --sequence-len 128 \
    --examples-per-shard 50000
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from scripts.train_phrase_gpt import (
    examples_from_story_records,
    examples_to_tensor_shard,
)
from scripts.train_phrase_vectors import _legacy_record_to_typed_records, iter_records, load_vocab


def _flush_story(current_rows, sequence_len, chain_mode):
    if not current_rows:
        return []
    return examples_from_story_records(current_rows, sequence_len=sequence_len, chain_mode=chain_mode)


def _write_shard(out_dir, shard_index, examples, sequence_len, split):
    filename = f"{split}_shard_{shard_index:05d}.pt"
    path = Path(out_dir) / filename
    shard = examples_to_tensor_shard(examples, sequence_len=sequence_len)
    shard["split"] = split
    torch.save(shard, path)
    return {
        "file": filename,
        "split": split,
        "num_examples": int(shard["num_examples"]),
        "num_tokens": int((shard["targets"] != -1).sum().item()),
        "num_phrase_indices": int(shard["phrase_indices"].numel()),
        "bytes": path.stat().st_size,
    }


def build_shards_from_records(records, out_dir, sequence_len, examples_per_shard, records_path="", vocab_path="", progress_every=100000, max_examples=None, chain_mode="token"):
    os.makedirs(out_dir, exist_ok=True)
    started_at = time.time()
    current_key = None
    current_rows = []
    pending_examples_by_split = {}
    shard_counts_by_split = {}
    shards = []
    records_seen = 0
    stories_seen = 0
    examples_seen = 0

    def add_examples(split, new_examples):
        nonlocal shards, examples_seen
        pending_examples = pending_examples_by_split.setdefault(split, [])
        for example in new_examples:
            if max_examples is not None and examples_seen >= max_examples:
                return False
            pending_examples.append(example)
            examples_seen += 1
            if len(pending_examples) >= examples_per_shard:
                shard_index = shard_counts_by_split.get(split, 0)
                shards.append(_write_shard(out_dir, shard_index, pending_examples, sequence_len, split=split))
                shard_counts_by_split[split] = shard_index + 1
                pending_examples_by_split[split] = []
                pending_examples = pending_examples_by_split[split]
        return True

    for raw_record in records:
        typed_records = [raw_record] if "record_type" in raw_record else _legacy_record_to_typed_records(raw_record)
        records_seen += 1
        for record in typed_records:
            key = (record["split"], int(record["story_id"]))
            if current_key is not None and key != current_key:
                stories_seen += 1
                if not add_examples(current_key[0], _flush_story(current_rows, sequence_len=sequence_len, chain_mode=chain_mode)):
                    current_rows = []
                    break
                if progress_every > 0 and stories_seen % progress_every == 0:
                    print(f"sharded {stories_seen} stories; examples={examples_seen} records={records_seen} shards={len(shards)}", flush=True)
                current_rows = []
            current_key = key
            current_rows.append(record)
        if max_examples is not None and examples_seen >= max_examples:
            break

    if current_rows and (max_examples is None or examples_seen < max_examples):
        stories_seen += 1
        add_examples(current_key[0], _flush_story(current_rows, sequence_len=sequence_len, chain_mode=chain_mode))

    for split, pending_examples in list(pending_examples_by_split.items()):
        if pending_examples:
            shard_index = shard_counts_by_split.get(split, 0)
            shards.append(_write_shard(out_dir, shard_index, pending_examples, sequence_len, split=split))
            shard_counts_by_split[split] = shard_index + 1

    manifest = {
        "format": "phrase_gpt_tensor_shard_manifest_v1",
        "records": records_path,
        "vocab": vocab_path,
        "sequence_len": sequence_len,
        "chain_mode": chain_mode,
        "examples_per_shard": examples_per_shard,
        "num_shards": len(shards),
        "num_examples": sum(shard["num_examples"] for shard in shards),
        "splits": {
            split: {
                "num_shards": sum(1 for shard in shards if shard["split"] == split),
                "num_examples": sum(shard["num_examples"] for shard in shards if shard["split"] == split),
            }
            for split in sorted({shard["split"] for shard in shards})
        },
        "num_tokens": sum(shard["num_tokens"] for shard in shards),
        "num_phrase_indices": sum(shard["num_phrase_indices"] for shard in shards),
        "records_seen": records_seen,
        "stories_seen": stories_seen,
        "elapsed_seconds": time.time() - started_at,
        "shards": shards,
    }
    with open(Path(out_dir) / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Build phrase GPT tensor shards from phrase_index records.")
    parser.add_argument("--vocab", required=True, help="Path to phrase vocab.json.")
    parser.add_argument("--records", required=True, help="Path to phrase_index.jsonl or .jsonl.gz.")
    parser.add_argument("--out-dir", required=True, help="Output directory for .pt shards and manifest.json.")
    parser.add_argument("--sequence-len", type=int, default=128)
    parser.add_argument("--examples-per-shard", type=int, default=50000)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--chain-mode", choices=["token", "phrase", "cross-phrase"], default="token", help="How to build per-timestep inputs: token (one token/step, default), phrase (one multihot chain per punctuation clause, breaking on out-of-order vocab index), cross-phrase (chains span clause boundaries, breaking only on out-of-order index).")
    return parser.parse_args()


def main():
    args = parse_args()
    vocab = load_vocab(args.vocab)
    print(f"loaded vocab size {vocab.size}", flush=True)
    manifest = build_shards_from_records(
        records=iter_records(args.records),
        out_dir=args.out_dir,
        sequence_len=args.sequence_len,
        examples_per_shard=args.examples_per_shard,
        records_path=args.records,
        vocab_path=args.vocab,
        progress_every=args.progress_every,
        max_examples=args.limit_examples,
        chain_mode=args.chain_mode,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
