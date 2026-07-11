"""
Split the ambiguous quote-mark token into open/close variants.

Open and close quotes share one surface glyph, so a single vocab index has to
sit both at clause starts and clause ends -- no ordering can make it ascend.
Quotes alternate within a story, so this transform assigns parity by absolute
token position per story (1st quote = open, 2nd = close, ...), rewriting every
record representation of the same position consistently. The close variant is
appended to the vocab as a new final index; downstream detokenization should
render it as a plain double quote.

Example:
python -m scripts.split_quote_token \
    --vocab phrase_vectors_full_out/vocab.json \
    --records phrase_vectors_full_out/phrase_index.jsonl.gz \
    --out-dir phrase_quote_split_out
"""

import argparse
import gzip
import json
import os
import time
from pathlib import Path

from scripts.train_phrase_vectors import iter_records

DEFAULT_QUOTE_TOKEN = '"'
DEFAULT_CLOSE_TOKEN = '"_close'


def _element_positions(record):
    """Yield (slot, absolute_token_position) per element of the record."""
    start = int(record.get("start", 0))
    if record.get("record_type") == "single":
        yield 0, start + int(record.get("token_pos", 0))
        return
    for slot in range(len(record.get("indices", []))):
        yield slot, start + slot


def _rewrite_story(story_records, quote_index, close_index, close_token, stats):
    positions = set()
    for record in story_records:
        indices = record.get("indices", [])
        for slot, position in _element_positions(record):
            if indices[slot] == quote_index:
                positions.add(position)
    if not positions:
        return story_records
    is_close = {position: occurrence % 2 == 1 for occurrence, position in enumerate(sorted(positions))}
    stats["quote_occurrences"] += len(positions)
    stats["close_occurrences"] += sum(is_close.values())

    rewritten = []
    for record in story_records:
        indices = record.get("indices", [])
        changed_slots = [
            slot
            for slot, position in _element_positions(record)
            if indices[slot] == quote_index and is_close[position]
        ]
        if not changed_slots:
            rewritten.append(record)
            continue
        new_record = dict(record)
        new_indices = list(indices)
        new_tokens = list(record.get("tokens", []))
        for slot in changed_slots:
            new_indices[slot] = close_index
            if slot < len(new_tokens):
                new_tokens[slot] = close_token
        new_record["indices"] = new_indices
        if new_tokens:
            new_record["tokens"] = new_tokens
        rewritten.append(new_record)
    return rewritten


def split_quote_records(records, quote_index, close_index, close_token, stats=None, progress_every=0):
    """Stream records, rewriting alternate quote occurrences to close_index.

    Holds one story in memory at a time; stats (if given) accumulates
    stories / quote_occurrences / close_occurrences.
    """
    stats = stats if stats is not None else {"stories": 0, "quote_occurrences": 0, "close_occurrences": 0}
    current_key = None
    story_records = []

    def flush():
        stats["stories"] += 1
        if progress_every > 0 and stats["stories"] % progress_every == 0:
            print(f"quote-split {stats['stories']} stories", flush=True)
        yield from _rewrite_story(story_records, quote_index, close_index, close_token, stats)

    for record in records:
        key = (record["split"], int(record["story_id"]))
        if current_key is not None and key != current_key:
            yield from flush()
            story_records = []
        current_key = key
        story_records.append(record)
    if story_records:
        yield from flush()


def run_split(vocab_path, records_path, out_dir, quote_token=DEFAULT_QUOTE_TOKEN, close_token=DEFAULT_CLOSE_TOKEN, compresslevel=4, progress_every=0):
    started_at = time.time()
    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab_rows = json.load(file)
    if any(row["token"] == close_token for row in vocab_rows):
        raise ValueError(f"close token {close_token!r} already exists in the vocab")
    quote_rows = [row for row in vocab_rows if row["token"] == quote_token]
    if len(quote_rows) != 1:
        raise ValueError(f"expected exactly one vocab row for {quote_token!r}; found {len(quote_rows)}")
    quote_index = int(quote_rows[0]["index"])
    close_index = len(vocab_rows)

    os.makedirs(out_dir, exist_ok=True)
    out_dir = Path(out_dir)
    stats = {"stories": 0, "quote_occurrences": 0, "close_occurrences": 0}
    out_records_path = out_dir / "phrase_index.jsonl.gz"
    with gzip.open(out_records_path, "wt", encoding="utf-8", compresslevel=compresslevel) as file:
        for record in split_quote_records(
            iter_records(records_path),
            quote_index=quote_index,
            close_index=close_index,
            close_token=close_token,
            stats=stats,
            progress_every=progress_every,
        ):
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")

    new_vocab = [dict(row) for row in vocab_rows]
    open_count = stats["quote_occurrences"] - stats["close_occurrences"]
    new_vocab[quote_index]["count"] = open_count
    new_vocab.append({
        "token": close_token,
        "index": close_index,
        "count": stats["close_occurrences"],
        "avg_position": new_vocab[quote_index].get("avg_position", 0.0),
    })
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as file:
        json.dump(new_vocab, file, ensure_ascii=False)
        file.write("\n")

    report = {
        "format": "quote_split_report_v1",
        "vocab": str(vocab_path),
        "records": str(records_path),
        "quote_token": quote_token,
        "close_token": close_token,
        "quote_index": quote_index,
        "close_index": close_index,
        "vocab_size": len(new_vocab),
        "stories": stats["stories"],
        "quote_occurrences": stats["quote_occurrences"],
        "close_occurrences": stats["close_occurrences"],
        "elapsed_seconds": time.time() - started_at,
    }
    with open(out_dir / "split_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Split the quote token into open/close variants.")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out-dir", required=True, help="Output directory for phrase_index.jsonl.gz, vocab.json, split_report.json.")
    parser.add_argument("--quote-token", default=DEFAULT_QUOTE_TOKEN)
    parser.add_argument("--close-token", default=DEFAULT_CLOSE_TOKEN)
    parser.add_argument("--compresslevel", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_split(
        vocab_path=args.vocab,
        records_path=args.records,
        out_dir=args.out_dir,
        quote_token=args.quote_token,
        close_token=args.close_token,
        compresslevel=args.compresslevel,
        progress_every=args.progress_every,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
