"""
Extract SAE training bags from phrase records.

Chain bags: cross-clause strictly-ascending chains over the index-mapped vocab
(the same segmentation the hybrid-cross shards use). Window bags: fixed-length
token windows. Both are then remapped to the top-8k space and stored CSR-style.

Example:
python -m scripts.extract_sae_bags \
    --records phrase_quote_split_out/phrase_index.jsonl.gz \
    --index-map phrase_quote_split_ils_out/old_to_new.json \
    --vocab phrase_quote_split_ils_out/vocab.json --vocab-top-k 8191 \
    --out-dir sae_bags_out --max-stories 60000
"""

import argparse
import itertools
import json

import numpy as np

from scripts.reorder_phrase_vocab import story_index_streams
from scripts.train_phrase_gpt import load_vocab_top_k_remap
from scripts.train_phrase_vectors import iter_records


def _mapped_indices(stream, index_map):
    return [int(index_map[index]) if index_map is not None else int(index) for _, index in stream]


def chain_bags_from_stream(stream, index_map):
    indices = _mapped_indices(stream, index_map)
    bags = []
    current = []
    for index in indices:
        if current and index <= current[-1]:
            bags.append(current)
            current = []
        current.append(index)
    if current:
        bags.append(current)
    return bags


def window_bags_from_stream(stream, index_map, window=4):
    indices = _mapped_indices(stream, index_map)
    return [indices[start:start + window] for start in range(0, len(indices), window)]


def remap_bags(bags, lookup):
    return [sorted({int(lookup[index]) for index in bag}) for bag in bags]


def save_bags(bags, path, input_dim):
    offsets = np.zeros(len(bags) + 1, dtype=np.int64)
    for slot, bag in enumerate(bags):
        offsets[slot + 1] = offsets[slot] + len(bag)
    indices = np.fromiter((index for bag in bags for index in bag), dtype=np.int32, count=int(offsets[-1]))
    np.savez(path, indices=indices, offsets=offsets, input_dim=np.int64(input_dim))


def load_bags(path):
    saved = np.load(path)
    offsets = saved["offsets"]
    indices = saved["indices"]
    bags = [indices[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]
    return bags, int(saved["input_dim"])


def parse_args():
    parser = argparse.ArgumentParser(description="Extract SAE training bags (chains + windows).")
    parser.add_argument("--records", required=True)
    parser.add_argument("--index-map", required=True)
    parser.add_argument("--vocab", required=True, help="Original (ILS) vocab.json for the top-k remap.")
    parser.add_argument("--vocab-top-k", type=int, default=8191)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-stories", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def main():
    import os
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.index_map, "r", encoding="utf-8") as file:
        index_map = json.load(file)
    lookup, tokens = load_vocab_top_k_remap(args.vocab, args.vocab_top_k)
    input_dim = len(tokens)
    chain_bags = []
    window_bags = []
    stories = 0
    streams = story_index_streams(iter_records(args.records))
    if args.max_stories is not None:
        streams = itertools.islice(streams, args.max_stories)
    for _, _, stream in streams:
        stories += 1
        chain_bags.extend(remap_bags(chain_bags_from_stream(stream, index_map), lookup))
        window_bags.extend(remap_bags(window_bags_from_stream(stream, index_map, window=args.window), lookup))
        if args.progress_every and stories % args.progress_every == 0:
            print(f"extracted {stories} stories; chains={len(chain_bags)} windows={len(window_bags)}", flush=True)
    save_bags(chain_bags, f"{args.out_dir}/chain_bags.npz", input_dim)
    save_bags(window_bags, f"{args.out_dir}/window_bags.npz", input_dim)
    report = {
        "stories": stories,
        "chain_bags": len(chain_bags),
        "window_bags": len(window_bags),
        "input_dim": input_dim,
        "window": args.window,
        "records": args.records,
    }
    with open(f"{args.out_dir}/extract_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
