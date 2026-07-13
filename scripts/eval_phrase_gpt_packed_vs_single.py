"""
Compare GPT predictions from singleton phrase input vs packed phrase input.

Example:
python -m scripts.eval_phrase_gpt_packed_vs_single \
    --checkpoint phrase_gpt_out/phrase_gpt.pt \
    --records phrase_vectors_out/phrase_index.jsonl
"""

import argparse
import json
import os
from dataclasses import dataclass, fields

import torch

from nanochat.gpt import GPT, GPTConfig
from scripts.build_phrase_gpt_shards import remap_record_indices
from scripts.eval_packed_vs_single import PairedExample, _group_records, build_paired_examples, compute_pair_metrics
from scripts.train_phrase_gpt import PhraseSequenceExample, collate_phrase_sequences, choose_device, load_vocab_top_k_remap
from scripts.train_phrase_vectors import iter_records, normalize_phrase_records


@dataclass(frozen=True)
class ContextExample:
    single_steps: list[list[int]]
    packed_steps: list[list[int]]
    target_index: int


def apply_index_map(records, index_map):
    return [remap_record_indices(record, index_map) for record in records]


def build_context_examples(records, context_phrases):
    """Build longer-context eval examples: a window of `context_phrases` consecutive
    phrases from one story, with the first token of the following phrase as target.
    single_steps has one timestep per token; packed_steps one multihot step per phrase."""
    grouped = _group_records(normalize_phrase_records(records))
    by_story = {}
    for split, story_id, phrase_id in grouped:
        by_story.setdefault((split, story_id), set()).add(phrase_id)

    def phrase_parts(key):
        rows = grouped[key]
        singles = [row for row in rows if row.get("record_type") == "single" and row.get("indices")]
        packed = [row for row in rows if row.get("record_type") == "packed" and row.get("indices")]
        singles.sort(key=lambda row: int(row.get("token_pos", 0)))
        if not packed and singles:
            # packed records are only emitted for multi-token phrases; the multihot of a
            # phrase is by definition the union of its single-token records
            packed = [{"indices": [index for row in singles for index in row["indices"]]}]
        return singles, packed

    examples = []
    for story_key in sorted(by_story):
        split, story_id = story_key
        phrase_ids = sorted(by_story[story_key])
        for last in phrase_ids:
            window = list(range(last - context_phrases + 1, last + 1))
            if window[0] < 0 or any(pid not in by_story[story_key] for pid in window + [last + 1]):
                continue
            single_steps = []
            packed_steps = []
            valid = True
            for pid in window:
                singles, packed = phrase_parts((split, story_id, pid))
                if not singles or not packed:
                    valid = False
                    break
                single_steps.extend([list(row["indices"]) for row in singles])
                packed_steps.append(list(packed[0]["indices"]))
            target_singles, _ = phrase_parts((split, story_id, last + 1))
            if not valid or not target_singles:
                continue
            examples.append(ContextExample(
                single_steps=single_steps,
                packed_steps=packed_steps,
                target_index=int(target_singles[0]["indices"][0]),
            ))
    return examples


def remap_context_examples(examples, lookup):
    def remap_steps(steps):
        return [[int(lookup[index]) for index in step] for step in steps]
    return [
        ContextExample(
            single_steps=remap_steps(example.single_steps),
            packed_steps=remap_steps(example.packed_steps),
            target_index=int(lookup[example.target_index]),
        )
        for example in examples
    ]


def predict_context_predictions(model, examples, mode, batch_size=32, dummy_token_id=0, device="cpu"):
    """Return a LongTensor (N,) of next-token argmax predictions at each example's last
    position, in the original example order. Streams one small prediction per example
    instead of materializing full-vocab logits, and sorts by length so each batch pads
    only to its own longest sequence (bounded memory even with tens of thousands of
    long-context examples)."""
    model.eval()
    if not examples:
        return torch.empty((0,), dtype=torch.long)
    steps_by_example = [example.single_steps if mode == "single" else example.packed_steps for example in examples]
    order = sorted(range(len(examples)), key=lambda i: len(steps_by_example[i]))
    predictions = torch.empty(len(examples), dtype=torch.long)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            batch_order = order[start:start + batch_size]
            steps = [steps_by_example[i] for i in batch_order]
            sequence_len = max(2, max(len(step_list) for step_list in steps))
            batch = collate_phrase_sequences(
                [PhraseSequenceExample(input_indices=step_list, targets=[-1] * len(step_list)) for step_list in steps],
                sequence_len=sequence_len,
                dummy_token_id=dummy_token_id,
                device=device,
            )
            logits = model(
                batch.idx,
                phrase_indices=batch.phrase_indices,
                phrase_offsets=batch.phrase_offsets,
                phrase_batch_positions=batch.phrase_batch_positions,
            )
            last_positions = torch.tensor([len(step_list) - 1 for step_list in steps], device=logits.device)
            batch_preds = logits[torch.arange(len(steps), device=logits.device), last_positions, :].argmax(dim=1).cpu()
            for slot, prediction in zip(batch_order, batch_preds.tolist()):
                predictions[slot] = prediction
            del logits, batch
            if device == "mps":
                torch.mps.empty_cache()
    return predictions


def compute_context_metrics(examples, single_preds, packed_preds):
    n = len(examples)
    if n == 0:
        return {"paired_examples": 0, "seq_accuracy": 0.0, "multihot_accuracy": 0.0,
                "same_prediction_rate": 0.0, "seq_wins": 0.0, "multihot_wins": 0.0,
                "both_correct": 0.0, "both_wrong": 0.0}
    targets = torch.tensor([example.target_index for example in examples], dtype=torch.long)
    seq_correct = single_preds == targets
    multihot_correct = packed_preds == targets
    same = single_preds == packed_preds

    def rate(mask):
        return mask.sum().item() / n

    return {
        "paired_examples": n,
        "seq_accuracy": rate(seq_correct),
        "multihot_accuracy": rate(multihot_correct),
        "same_prediction_rate": rate(same),
        "seq_wins": rate(seq_correct & ~multihot_correct),
        "multihot_wins": rate(multihot_correct & ~seq_correct),
        "both_correct": rate(seq_correct & multihot_correct),
        "both_wrong": rate(~seq_correct & ~multihot_correct),
    }


def remap_paired_examples(pairs, lookup):
    return [
        PairedExample(
            single_indices=[int(lookup[index]) for index in pair.single_indices],
            packed_indices=[int(lookup[index]) for index in pair.packed_indices],
            target_index=int(lookup[pair.target_index]),
        )
        for pair in pairs
    ]


def resolve_vocab_remap(checkpoint_config, vocab_override=None, top_k_override=None):
    top_k = top_k_override if top_k_override is not None else checkpoint_config.get("vocab_top_k")
    if not top_k:
        return None
    vocab_path = vocab_override or checkpoint_config.get("vocab")
    if not vocab_path:
        raise SystemExit("Checkpoint was trained with --vocab-top-k but records no vocab path; pass --vocab.")
    if not os.path.exists(vocab_path):
        raise SystemExit(f"Original vocab file {vocab_path!r} not found (needed to rebuild the top-k remap); pass --vocab with the correct path.")
    lookup, _tokens = load_vocab_top_k_remap(vocab_path, top_k)
    return lookup


def examples_from_pairs(pairs, mode):
    examples = []
    for pair in pairs:
        input_indices = pair.single_indices if mode == "single" else pair.packed_indices
        examples.append(PhraseSequenceExample(
            input_indices=[input_indices],
            targets=[pair.target_index],
        ))
    return examples


def predict_logits(model, pairs, mode, batch_size=512, sequence_len=1, dummy_token_id=0, device="cpu"):
    model.eval()
    outputs = []
    examples = examples_from_pairs(pairs, mode=mode)
    sequence_len = max(2, sequence_len)
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = collate_phrase_sequences(
                examples[start:start + batch_size],
                sequence_len=sequence_len,
                dummy_token_id=dummy_token_id,
                device=device,
            )
            logits = model(
                batch.idx,
                phrase_indices=batch.phrase_indices,
                phrase_offsets=batch.phrase_offsets,
                phrase_batch_positions=batch.phrase_batch_positions,
            )
            outputs.append(logits[:, 0, :].cpu())
    if not outputs:
        return torch.empty((0, 0))
    return torch.cat(outputs, dim=0)


def load_model_from_checkpoint(checkpoint, device):
    allowed_config_keys = {field.name for field in fields(GPTConfig)}
    config_kwargs = {
        key: value
        for key, value in checkpoint["config"].items()
        if key in allowed_config_keys
    }
    config = GPTConfig(**config_kwargs)
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return load_model_from_checkpoint(checkpoint, device)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate phrase-GPT packed-vs-single next-token metrics.")
    parser.add_argument("--checkpoint", required=True, help="Path to phrase_gpt.pt.")
    parser.add_argument("--records", required=True, help="Path to phrase_index.jsonl.")
    parser.add_argument("--vocab", default=None, help="Path to the ORIGINAL vocab.json used in training. Only needed for top-k checkpoints when the path recorded in the checkpoint is wrong.")
    parser.add_argument("--vocab-top-k", type=int, default=None, help="Override the top-k recorded in the checkpoint. Normally auto-detected.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sequence-len", type=int, default=2, help="Evaluation sequence length; only position 0 is scored.")
    parser.add_argument("--dummy-token-id", type=int, default=0)
    parser.add_argument("--device", default="", help="cpu|cuda. Empty chooses cuda, then cpu.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max paired examples to evaluate.")
    parser.add_argument("--index-map", default=None, help="old_to_new.json from scripts.reorder_phrase_vocab; applied to record indices before pairing (use for reordered-vocab checkpoints).")
    parser.add_argument("--context-phrases", type=int, default=0, help="Also run a longer-context eval: windows of N consecutive phrases (token-per-step vs multihot-phrase-per-step), scored on the next phrase's first token. 0 disables.")
    parser.add_argument("--context-batch-size", type=int, default=32, help="Batch size for the longer-context eval. Kept small because context sequences are long; raise it only on a big-memory device.")
    parser.add_argument("--context-limit", type=int, default=20000, help="Max longer-context examples to score (they can number in the tens of thousands). Use 0 for no cap.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    records = list(iter_records(args.records))
    if args.index_map:
        with open(args.index_map, "r", encoding="utf-8") as file:
            records = apply_index_map(records, json.load(file))
    pairs = build_paired_examples(records)
    if args.limit is not None:
        pairs = pairs[:args.limit]
    if not pairs:
        raise SystemExit("No paired single/packed examples found.")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, _config = load_model_from_checkpoint(checkpoint, device=device)
    remap = resolve_vocab_remap(checkpoint.get("config", {}), vocab_override=args.vocab, top_k_override=args.vocab_top_k)
    if remap is not None:
        kept = int(remap.max().item())
        print(f"checkpoint uses vocab top-k: remapping record indices ({kept} kept + <unk>)", flush=True)
        pairs = remap_paired_examples(pairs, remap)
    single_logits = predict_logits(
        model,
        pairs,
        mode="single",
        batch_size=args.batch_size,
        sequence_len=args.sequence_len,
        dummy_token_id=args.dummy_token_id,
        device=device,
    )
    packed_logits = predict_logits(
        model,
        pairs,
        mode="packed",
        batch_size=args.batch_size,
        sequence_len=args.sequence_len,
        dummy_token_id=args.dummy_token_id,
        device=device,
    )
    results = compute_pair_metrics(pairs, single_logits, packed_logits)

    if args.context_phrases > 0:
        context_examples = build_context_examples(records, context_phrases=args.context_phrases)
        eligible = len(context_examples)
        if args.context_limit and eligible > args.context_limit:
            context_examples = context_examples[:args.context_limit]
        print(f"context eval: {eligible} eligible {args.context_phrases}-phrase windows, scoring {len(context_examples)}", flush=True)
        if remap is not None:
            context_examples = remap_context_examples(context_examples, remap)
        context_single = predict_context_predictions(model, context_examples, mode="single", batch_size=args.context_batch_size, dummy_token_id=args.dummy_token_id, device=device)
        context_packed = predict_context_predictions(model, context_examples, mode="packed", batch_size=args.context_batch_size, dummy_token_id=args.dummy_token_id, device=device)
        results["context"] = {
            "context_phrases": args.context_phrases,
            "eligible_examples": eligible,
            **compute_context_metrics(context_examples, context_single, context_packed),
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
