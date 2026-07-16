"""
Compare one-token prediction from single-token input vs packed multihot input.

Example:
python -m scripts.eval_packed_vs_single \
    --checkpoint phrase_token_model_out/phrase_token_predictor.pt \
    --vocab phrase_vectors_out/vocab.json \
    --records phrase_vectors_out/phrase_index.jsonl
"""

import argparse
import json
from dataclasses import dataclass

import torch

from scripts.train_phrase_vectors import (
    PhraseTokenPredictor,
    collate_phrase_examples,
    iter_records,
    load_vocab,
    choose_device,
    normalize_phrase_records,
)


@dataclass(frozen=True)
class PairedExample:
    single_indices: list[int]
    packed_indices: list[int]
    target_index: int


def _record_sort_key(record):
    return (
        record["split"],
        int(record["story_id"]),
        int(record["phrase_id"]),
        int(record.get("start", 0)),
        int(record.get("end", 0)),
        int(record.get("token_pos", -1)),
        record.get("record_type", ""),
    )


def _group_records(records):
    grouped = {}
    for record in records:
        key = (record["split"], int(record["story_id"]), int(record["phrase_id"]))
        grouped.setdefault(key, []).append(record)
    for key in grouped:
        grouped[key].sort(key=_record_sort_key)
    return grouped


def build_paired_examples(records):
    grouped = _group_records(normalize_phrase_records(records))
    pairs = []
    ordered_keys = sorted(grouped.keys())
    for split, story_id, phrase_id in ordered_keys:
        current = grouped[(split, story_id, phrase_id)]
        next_key = (split, story_id, phrase_id + 1)
        if next_key not in grouped:
            continue
        singles = [record for record in current if record.get("record_type") == "single"]
        packed = [record for record in current if record.get("record_type") == "packed"]
        next_singles = [record for record in grouped[next_key] if record.get("record_type") == "single"]
        if not singles or not packed or not next_singles:
            continue
        single_input = max(singles, key=lambda record: int(record.get("token_pos", 0)))
        packed_input = packed[0]
        target = min(next_singles, key=lambda record: int(record.get("token_pos", 0)))
        pairs.append(PairedExample(
            single_indices=list(single_input["indices"]),
            packed_indices=list(packed_input["indices"]),
            target_index=int(target["indices"][0]),
        ))
    return pairs


def _examples_from_pairs(pairs, mode):
    from scripts.train_phrase_vectors import PhraseTokenExample

    examples = []
    for idx, pair in enumerate(pairs):
        input_indices = pair.single_indices if mode == "single" else pair.packed_indices
        examples.append(PhraseTokenExample(
            split="eval",
            story_id=idx,
            input_phrase_id=0,
            target_phrase_id=1,
            input_indices=input_indices,
            target_index=pair.target_index,
        ))
    return examples


def predict_logits(model, pairs, mode, batch_size=512, device="cpu"):
    model.eval()
    outputs = []
    examples = _examples_from_pairs(pairs, mode=mode)
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = collate_phrase_examples(examples[start:start + batch_size], device=device)
            outputs.append(model(batch.flat_indices, batch.offsets).cpu())
    if not outputs:
        return torch.empty((0, 0))
    return torch.cat(outputs, dim=0)


def compute_pair_metrics(pairs, single_logits, packed_logits):
    n = len(pairs)
    if n == 0:
        return {
            "paired_examples": 0,
            "seq_accuracy": 0.0,
            "multihot_accuracy": 0.0,
            "same_prediction_rate": 0.0,
            "seq_wins": 0.0,
            "multihot_wins": 0.0,
            "both_correct": 0.0,
            "both_wrong": 0.0,
        }

    targets = torch.tensor([pair.target_index for pair in pairs], dtype=torch.long)
    single_pred = single_logits.argmax(dim=1)
    packed_pred = packed_logits.argmax(dim=1)

    seq_correct = single_pred == targets
    multihot_correct = packed_pred == targets
    same_prediction = single_pred == packed_pred

    def rate(mask):
        return mask.sum().item() / n

    return {
        "paired_examples": n,
        "seq_accuracy": rate(seq_correct),
        "multihot_accuracy": rate(multihot_correct),
        "same_prediction_rate": rate(same_prediction),
        "seq_wins": rate(seq_correct & ~multihot_correct),
        "multihot_wins": rate(multihot_correct & ~seq_correct),
        "both_correct": rate(seq_correct & multihot_correct),
        "both_wrong": rate(~seq_correct & ~multihot_correct),
    }


def load_model(checkpoint_path, vocab_size, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_size = checkpoint["config"]["hidden_size"]
    model = PhraseTokenPredictor(vocab_size=vocab_size, hidden_size=hidden_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate packed-vs-single next-token metrics.")
    parser.add_argument("--checkpoint", required=True, help="Path to phrase_token_predictor.pt.")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json.")
    parser.add_argument("--records", required=True, help="Path to phrase_index.jsonl.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="", help="cpu|cuda. Empty chooses cuda, then cpu.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max paired examples to evaluate.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    vocab = load_vocab(args.vocab)
    records = list(iter_records(args.records))
    pairs = build_paired_examples(records)
    if args.limit is not None:
        pairs = pairs[:args.limit]
    if not pairs:
        raise SystemExit(
            "No paired single/packed examples found. Regenerate phrase_index.jsonl with the latest "
            "scripts.phrase_vectors so records include record_type='single' and record_type='packed'."
        )
    model = load_model(args.checkpoint, vocab_size=vocab.size, device=device)
    single_logits = predict_logits(model, pairs, mode="single", batch_size=args.batch_size, device=device)
    packed_logits = predict_logits(model, pairs, mode="packed", batch_size=args.batch_size, device=device)
    print(json.dumps(compute_pair_metrics(pairs, single_logits, packed_logits), indent=2))


if __name__ == "__main__":
    main()
