"""
Measure chain-length distributions (chains == timesteps in chain/hybrid modes)
for one or more vocab orders over a records file.

Chains are maximal strictly-ascending index runs of the canonical token stream;
with --no-clause-reset they may continue across clause boundaries.

Example:
python -m scripts.measure_chain_lengths \
    --records phrase_quote_split_out/phrase_index.jsonl.gz \
    --map current=phrase_quote_split_ils_out/old_to_new.json \
    --max-records 70000000
"""

import argparse
import itertools
import json

from scripts.reorder_phrase_vocab import story_index_streams
from scripts.train_phrase_vectors import iter_records


def chain_length_histogram(stream, mapping, reset_on_clause, histogram=None):
    histogram = histogram if histogram is not None else {}
    run = 0
    prev_clause = prev_index = None
    for clause, raw_index in stream:
        index = int(mapping[raw_index]) if mapping is not None else raw_index
        if run and (index <= prev_index or (reset_on_clause and clause != prev_clause)):
            histogram[run] = histogram.get(run, 0) + 1
            run = 0
        run += 1
        prev_clause, prev_index = clause, index
    if run:
        histogram[run] = histogram.get(run, 0) + 1
    return histogram


def summarize(histogram):
    chains = sum(histogram.values())
    tokens = sum(length * count for length, count in histogram.items())
    return {
        "chains": chains,
        "tokens": tokens,
        "avg_chain_len": round(tokens / chains, 3) if chains else 0.0,
        "len1_chains": histogram.get(1, 0),
        "len1_fraction_of_chains": round(histogram.get(1, 0) / chains, 4) if chains else 0.0,
        "max_len": max(histogram) if histogram else 0,
        "histogram": {str(length): histogram[length] for length in sorted(histogram)},
    }


def measure(records, mappings, reset_on_clause, max_records=None):
    if max_records is not None:
        records = itertools.islice(records, max_records)
    histograms = {name: {} for name in mappings}
    stories = 0
    for _, _, stream in story_index_streams(records):
        stories += 1
        for name, mapping in mappings.items():
            chain_length_histogram(stream, mapping, reset_on_clause, histogram=histograms[name])
    result = {"stories": stories, "reset_on_clause": reset_on_clause}
    for name, histogram in histograms.items():
        result[name] = summarize(histogram)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Measure chain-length distributions for vocab orders.")
    parser.add_argument("--records", required=True, help="Path to phrase_index.jsonl or .jsonl.gz.")
    parser.add_argument("--map", action="append", default=[], metavar="NAME=PATH",
                        help="Named old_to_new.json map; repeatable. Omit to measure record indices as-is.")
    parser.add_argument("--no-clause-reset", action="store_true", help="Let chains continue across clause boundaries.")
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    mappings = {}
    for spec in args.map:
        name, _, path = spec.partition("=")
        with open(path, "r", encoding="utf-8") as file:
            mappings[name] = json.load(file)
    if not mappings:
        mappings = {"identity": None}
    result = measure(
        iter_records(args.records),
        mappings,
        reset_on_clause=not args.no_clause_reset,
        max_records=args.max_records,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
