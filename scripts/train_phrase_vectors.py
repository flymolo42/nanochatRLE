"""
Train a one-token predictor from designed phrase input vectors.

Each input is a phrase represented by multiple active vocab indices with
implicit value 1. The model consumes those indices in one EmbeddingBag pass and
predicts one next token from vocab.json.

Example:
python -m scripts.train_phrase_vectors \
    --vocab phrase_vectors_out/vocab.json \
    --records phrase_vectors_out/phrase_index.jsonl \
    --out-dir phrase_model_out \
    --epochs 3
"""

import argparse
import gzip
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.phrase_vectors import (
    build_sparse_records,
    extract_phrase_occurrences,
    iter_archive_stories,
    load_spacy_model,
)


@dataclass(frozen=True)
class PhraseVocab:
    token_to_index: dict[str, int]

    @property
    def size(self):
        return len(self.token_to_index)


@dataclass(frozen=True)
class PhraseTokenExample:
    split: str
    story_id: int
    input_phrase_id: int
    target_phrase_id: int
    input_indices: list[int]
    target_index: int


@dataclass(frozen=True)
class PhraseBatch:
    flat_indices: torch.Tensor
    offsets: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class EarlyStoppingConfig:
    patience: int = 0
    min_delta: float = 0.0
    target_val_loss: float | None = None
    target_val_accuracy: float | None = None


@dataclass
class EarlyStoppingState:
    best_val_loss: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement: int = 0
    should_stop: bool = False
    stop_reason: str | None = None
    is_best: bool = False

    def update(self, val_metrics, epoch, config):
        self.is_best = False
        val_loss = val_metrics.get("loss")
        val_accuracy = val_metrics.get("accuracy")

        if val_loss is not None and not math.isnan(val_loss):
            improved = self.best_val_loss is None or val_loss < self.best_val_loss - config.min_delta
            if improved:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self.is_best = True
            else:
                self.epochs_without_improvement += 1

        if config.target_val_loss is not None and val_loss is not None and val_loss <= config.target_val_loss:
            self.should_stop = True
            self.stop_reason = "target validation loss reached"
            return
        if config.target_val_accuracy is not None and val_accuracy is not None and val_accuracy >= config.target_val_accuracy:
            self.should_stop = True
            self.stop_reason = "target validation accuracy reached"
            return
        if config.patience > 0 and self.epochs_without_improvement >= config.patience:
            self.should_stop = True
            suffix = "epoch" if config.patience == 1 else "epochs"
            self.stop_reason = f"validation loss did not improve for {config.patience} {suffix}"


class PhraseTokenPredictor(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.encoder = nn.EmbeddingBag(vocab_size, hidden_size, mode="sum")
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, flat_indices, offsets):
        hidden = self.encoder(flat_indices, offsets)
        hidden = self.norm(hidden)
        return self.decoder(hidden)


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as file:
        rows = json.load(file)
    ordered = sorted(rows, key=lambda row: row["index"])
    token_to_index = {row["token"]: row["index"] for row in ordered}
    expected = list(range(len(token_to_index)))
    actual = [row["index"] for row in ordered]
    if actual != expected:
        raise ValueError(f"Vocab indices must be contiguous from 0; got first indices {actual[:10]}")
    return PhraseVocab(token_to_index=token_to_index)


def iter_records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def _legacy_record_to_typed_records(record):
    indices = list(record.get("indices", []))
    tokens = list(record.get("tokens", []))
    output = []
    for token_pos, index in enumerate(indices):
        row = dict(record)
        row["record_type"] = "single"
        row["token_pos"] = token_pos
        row["indices"] = [int(index)]
        if token_pos < len(tokens):
            row["tokens"] = [tokens[token_pos]]
        output.append(row)

    if len(indices) > 1 and all(left < right for left, right in zip(indices, indices[1:])):
        row = dict(record)
        row["record_type"] = "packed"
        row["indices"] = [int(index) for index in indices]
        output.append(row)
    return output


def normalize_phrase_records(records):
    normalized = []
    for record in records:
        if "record_type" in record:
            normalized.append(record)
        else:
            normalized.extend(_legacy_record_to_typed_records(record))
    return normalized


def _record_sort_key(record):
    return (
        record["split"],
        int(record["story_id"]),
        int(record["phrase_id"]),
        int(record.get("start", 0)),
        int(record.get("end", 0)),
    )


def build_training_examples(records):
    sorted_records = sorted(normalize_phrase_records(records), key=_record_sort_key)
    examples = []
    previous = None
    for record in sorted_records:
        if previous is not None and previous["split"] == record["split"] and previous["story_id"] == record["story_id"]:
            for target_index in record["indices"]:
                examples.append(PhraseTokenExample(
                    split=record["split"],
                    story_id=int(record["story_id"]),
                    input_phrase_id=int(previous["phrase_id"]),
                    target_phrase_id=int(record["phrase_id"]),
                    input_indices=list(previous["indices"]),
                    target_index=int(target_index),
                ))
        previous = record
    return examples


def collate_phrase_examples(examples, device="cpu"):
    if not examples:
        raise ValueError("Cannot collate an empty batch")

    flat = []
    offsets = []
    targets = []
    cursor = 0
    for example in examples:
        if not example.input_indices:
            raise ValueError("Input phrase has no active indices")
        offsets.append(cursor)
        flat.extend(example.input_indices)
        cursor += len(example.input_indices)
        targets.append(example.target_index)

    return PhraseBatch(
        flat_indices=torch.tensor(flat, dtype=torch.long, device=device),
        offsets=torch.tensor(offsets, dtype=torch.long, device=device),
        targets=torch.tensor(targets, dtype=torch.long, device=device),
    )


def read_examples_from_records(records_path):
    return build_training_examples(iter_records(records_path))


def read_examples_from_archive(archive_path, vocab, spacy_model, splits, limit=None, lowercase=True, batch_size=64):
    nlp = load_spacy_model(spacy_model)
    vocab_rows = [{"token": token, "index": index} for token, index in vocab.token_to_index.items()]
    records = []

    for split in splits:
        pending_ids = []
        pending_texts = []
        for story_id, text in iter_archive_stories(archive_path, split, limit=limit):
            pending_ids.append(story_id)
            pending_texts.append(text)
            if len(pending_texts) == batch_size:
                _append_archive_records(nlp, split, pending_ids, pending_texts, vocab_rows, records, lowercase)
                pending_ids, pending_texts = [], []
        if pending_texts:
            _append_archive_records(nlp, split, pending_ids, pending_texts, vocab_rows, records, lowercase)

    return build_training_examples(records)


def _append_archive_records(nlp, split, story_ids, texts, vocab_rows, records, lowercase):
    for story_id, doc in zip(story_ids, nlp.pipe(texts)):
        phrases = extract_phrase_occurrences(doc, lowercase=lowercase)
        records.extend(build_sparse_records(split, story_id, phrases, vocab_rows))


def split_train_val_examples(examples, validation_split=0.05, seed=42):
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    if not shuffled:
        return [], []
    val_count = max(1, int(len(shuffled) * validation_split)) if validation_split > 0 and len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def iter_batches(examples, batch_size, shuffle=True, seed=42):
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[index] for index in order[start:start + batch_size]]


def run_epoch(
    model,
    examples,
    optimizer=None,
    batch_size=128,
    device="cpu",
    seed=42,
    progress_every=0,
    progress_label="epoch",
    output_fn=print,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_rows = 0
    total_batches = math.ceil(len(examples) / batch_size) if examples else 0

    with torch.set_grad_enabled(training):
        for batch_idx, batch_examples in enumerate(iter_batches(examples, batch_size=batch_size, shuffle=training, seed=seed), start=1):
            batch = collate_phrase_examples(batch_examples, device=device)
            logits = model(batch.flat_indices, batch.offsets)
            loss = F.cross_entropy(logits, batch.targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(batch_examples)
            total_correct += (logits.argmax(dim=1) == batch.targets).sum().item()
            total_rows += len(batch_examples)
            if progress_every > 0 and (batch_idx % progress_every == 0 or batch_idx == total_batches):
                running_loss = total_loss / total_rows
                running_acc = total_correct / total_rows
                output_fn(
                    f"{progress_label} batch {batch_idx}/{total_batches} "
                    f"examples={total_rows}/{len(examples)} loss={running_loss:.6f} acc={running_acc:.4f}"
                )

    if total_rows == 0:
        return {"loss": float("nan"), "accuracy": 0.0}
    return {
        "loss": total_loss / total_rows,
        "accuracy": total_correct / total_rows,
    }


def save_checkpoint(path, model, vocab, config, metrics, filename="phrase_token_predictor.pt"):
    os.makedirs(path, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab.size,
        "config": config,
        "metrics": metrics,
    }, os.path.join(path, filename))
    with open(os.path.join(path, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a one-token predictor on sparse phrase vectors.")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json from scripts.phrase_vectors.")
    parser.add_argument("--records", default=None, help="Optional phrase_index.jsonl. If omitted, --archive is parsed on the fly.")
    parser.add_argument("--archive", default=None, help="archive.zip to parse when --records is omitted.")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model for archive parsing.")
    parser.add_argument("--splits", nargs="+", default=["train"], choices=["train", "validation"], help="Archive splits to train from.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum stories per split when parsing archive.")
    parser.add_argument("--out-dir", default="phrase_model_out", help="Directory for checkpoint and metrics.")
    parser.add_argument("--hidden-size", type=int, default=256, help="Phrase hidden size.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Maximum number of epochs.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Alias for --epochs that emphasizes early-stopping use.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--validation-split", type=float, default=0.05, help="Fraction of examples held out for validation.")
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many validation epochs without loss improvement. Use 0 to disable.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum validation-loss improvement required to reset patience.")
    parser.add_argument("--target-val-loss", type=float, default=None, help="Stop once validation loss is at or below this value.")
    parser.add_argument("--target-val-accuracy", type=float, default=None, help="Stop once validation accuracy is at or above this value.")
    parser.add_argument("--save-best", action="store_true", help="Also write best_phrase_token_predictor.pt whenever validation loss improves.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N batches. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="", help="cpu|cuda|mps. Empty chooses cuda, then cpu. MPS must be requested explicitly.")
    parser.add_argument("--no-lowercase", action="store_true", help="Keep original case when parsing archive on the fly.")
    return parser.parse_args()


def choose_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    args = parse_args()
    if args.max_epochs is not None:
        args.epochs = args.max_epochs
    vocab = load_vocab(args.vocab)
    if args.records:
        examples = read_examples_from_records(args.records)
    else:
        if not args.archive:
            raise SystemExit("Provide either --records phrase_index.jsonl or --archive archive.zip.")
        examples = read_examples_from_archive(
            archive_path=args.archive,
            vocab=vocab,
            spacy_model=args.spacy_model,
            splits=args.splits,
            limit=args.limit,
            lowercase=not args.no_lowercase,
        )

    if not examples:
        raise SystemExit("No adjacent phrase examples found to train on.")

    train_examples, val_examples = split_train_val_examples(examples, validation_split=args.validation_split, seed=args.seed)
    device = choose_device(args.device)
    model = PhraseTokenPredictor(vocab_size=vocab.size, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metrics = {
        "vocab_size": vocab.size,
        "examples": len(examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "target": "one_next_vocab_token",
        "best_epoch": None,
        "best_val_loss": None,
        "stop_reason": None,
        "epochs": [],
    }
    config = vars(args)
    early_stopping = EarlyStoppingConfig(
        patience=args.patience,
        min_delta=args.min_delta,
        target_val_loss=args.target_val_loss,
        target_val_accuracy=args.target_val_accuracy,
    )
    early_state = EarlyStoppingState()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_examples,
            optimizer=optimizer,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + epoch,
            progress_every=args.progress_every,
            progress_label=f"train epoch {epoch}/{args.epochs}",
        )
        val_metrics = run_epoch(
            model,
            val_examples,
            optimizer=None,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            progress_every=args.progress_every,
            progress_label=f"val epoch {epoch}/{args.epochs}",
        ) if val_examples else {"loss": None, "accuracy": None}
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        metrics["epochs"].append(row)
        if val_examples:
            early_state.update(val_metrics, epoch=epoch, config=early_stopping)
            metrics["best_epoch"] = early_state.best_epoch
            metrics["best_val_loss"] = early_state.best_val_loss
            if early_state.is_best and args.save_best:
                save_checkpoint(args.out_dir, model, vocab, config, metrics, filename="best_phrase_token_predictor.pt")
            if early_state.should_stop:
                metrics["stop_reason"] = early_state.stop_reason
                row["stop_reason"] = early_state.stop_reason
        print(json.dumps(row))
        if early_state.should_stop:
            break

    save_checkpoint(args.out_dir, model, vocab, config, metrics)
    print(json.dumps({"saved": str(Path(args.out_dir) / "phrase_token_predictor.pt"), **metrics}, indent=2))


if __name__ == "__main__":
    main()
