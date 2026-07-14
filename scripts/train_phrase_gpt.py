"""
Train nanochat GPT with dummy token ids and sparse phrase vectors as content.

Example:
python -m scripts.train_phrase_gpt \
    --vocab phrase_vectors_out/vocab.json \
    --records phrase_vectors_out/phrase_index.jsonl \
    --out-dir phrase_gpt_out \
    --depth 2 \
    --sequence-len 64
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from scripts.train_phrase_vectors import iter_records, load_vocab, normalize_phrase_records
from scripts.train_phrase_vectors import _legacy_record_to_typed_records


@dataclass(frozen=True)
class PhraseSequenceExample:
    input_indices: list[list[int]]
    targets: list[int]


@dataclass(frozen=True)
class PhraseSequenceBatch:
    idx: torch.Tensor
    targets: torch.Tensor
    phrase_indices: torch.Tensor
    phrase_offsets: torch.Tensor
    phrase_batch_positions: torch.Tensor


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


def _group_phrase_records(records):
    grouped = {}
    for record in normalize_phrase_records(records):
        key = (record["split"], int(record["story_id"]), int(record["phrase_id"]))
        grouped.setdefault(key, []).append(record)
    for rows in grouped.values():
        rows.sort(key=_record_sort_key)
    return grouped


def _canonical_token_stream(rows):
    canonical = {}
    for record in rows:
        if record.get("record_type") != "single":
            continue
        if record.get("label") != "punctuation":
            continue
        if not record.get("indices"):
            continue
        absolute_token_pos = int(record.get("start", 0)) + int(record.get("token_pos", 0))
        canonical.setdefault(absolute_token_pos, record)
    return [canonical[position] for position in sorted(canonical)]


def _token_steps(records):
    token_records = _canonical_token_stream(records)
    return [
        (list(previous["indices"]), int(current["indices"][0]))
        for previous, current in zip(token_records, token_records[1:])
        if previous.get("indices") and current.get("indices")
    ]


def _cap_chains(chains, max_chain_len):
    """Split chains longer than max_chain_len into balanced contiguous chunks
    (contiguous pieces of an ascending run stay ascending). None = no cap."""
    if max_chain_len is None:
        return chains
    capped = []
    for chain in chains:
        if len(chain) <= max_chain_len:
            capped.append(chain)
            continue
        num_chunks = -(-len(chain) // max_chain_len)
        base, remainder = divmod(len(chain), num_chunks)
        start = 0
        for chunk_index in range(num_chunks):
            size = base + (1 if chunk_index < remainder else 0)
            capped.append(chain[start:start + size])
            start += size
    return capped


def _chains_from_token_records(token_records, reset_on_clause, max_chain_len=None):
    chains = []
    current = []
    current_clause = None
    for record in token_records:
        if not record.get("indices"):
            continue
        index = int(record["indices"][0])
        clause = int(record.get("phrase_id", 0))
        if current and (index <= current[-1] or (reset_on_clause and clause != current_clause)):
            chains.append(current)
            current = []
        current.append(index)
        current_clause = clause
    if current:
        chains.append(current)
    return _cap_chains(chains, max_chain_len)


def _steps_from_chains(chains):
    return [(chains[position], chains[position + 1][0]) for position in range(len(chains) - 1)]


def _chain_steps(records, reset_on_clause, max_chain_len=None):
    token_records = _canonical_token_stream(records)
    return _steps_from_chains(_chains_from_token_records(token_records, reset_on_clause, max_chain_len=max_chain_len))


def _phrase_boundary_positions(token_records):
    boundaries = [0]
    previous_clause = None
    for position, record in enumerate(token_records):
        clause = int(record.get("phrase_id", 0))
        if previous_clause is not None and clause != previous_clause:
            boundaries.append(position)
        previous_clause = clause
    boundaries.append(len(token_records))
    return sorted(set(boundaries))


def _hybrid_steps_at_split(token_records, split, reset_on_clause=True, max_chain_len=None):
    front = token_records[:split]
    back = token_records[split:]
    chains = _chains_from_token_records(front, reset_on_clause=reset_on_clause, max_chain_len=max_chain_len)
    # each back token becomes its own length-1 chain (1-hot); extend with one-element lists
    chains.extend([int(record["indices"][0])] for record in back if record.get("indices"))
    return _steps_from_chains(chains)


def _choose_split(boundaries, seed, story_id):
    return random.Random(seed * 1_000_003 + int(story_id)).choice(boundaries)


def _hybrid_steps(records, seed, reset_on_clause=True, max_chain_len=None):
    token_records = _canonical_token_stream(records)
    if not token_records:
        return []
    story_id = int(token_records[0].get("story_id", 0))
    boundaries = _phrase_boundary_positions(token_records)
    split = _choose_split(boundaries, seed, story_id)
    return _hybrid_steps_at_split(token_records, split, reset_on_clause=reset_on_clause, max_chain_len=max_chain_len)


def _chunk_steps_into_examples(steps, sequence_len):
    examples = []
    for start in range(0, len(steps), sequence_len):
        chunk = steps[start:start + sequence_len]
        if chunk:
            examples.append(PhraseSequenceExample(
                input_indices=[indices for indices, _ in chunk],
                targets=[target for _, target in chunk],
            ))
    return examples


CHAIN_MODE_BUILDERS = {
    "token": lambda records, seed, max_chain_len: _token_steps(records),
    "phrase": lambda records, seed, max_chain_len: _chain_steps(records, reset_on_clause=True, max_chain_len=max_chain_len),
    "cross-phrase": lambda records, seed, max_chain_len: _chain_steps(records, reset_on_clause=False, max_chain_len=max_chain_len),
    "hybrid": lambda records, seed, max_chain_len: _hybrid_steps(records, seed, max_chain_len=max_chain_len),
    "hybrid-cross": lambda records, seed, max_chain_len: _hybrid_steps(records, seed, reset_on_clause=False, max_chain_len=max_chain_len),
}


def examples_from_story_records(records, sequence_len, chain_mode="token", seed=0, max_chain_len=None):
    if chain_mode not in CHAIN_MODE_BUILDERS:
        raise ValueError(f"Unknown chain_mode {chain_mode!r}; expected one of {sorted(CHAIN_MODE_BUILDERS)}")
    return _chunk_steps_into_examples(CHAIN_MODE_BUILDERS[chain_mode](records, seed, max_chain_len), sequence_len)


def _examples_from_story_records(records, sequence_len):
    return examples_from_story_records(records, sequence_len, chain_mode="token")


def build_phrase_sequence_examples(records, sequence_len):
    grouped = _group_phrase_records(records)
    by_story = {}
    for split, story_id, phrase_id in grouped:
        by_story.setdefault((split, story_id), []).append(phrase_id)

    examples = []
    for split_story, phrase_ids in by_story.items():
        split, story_id = split_story
        story_rows = []
        for phrase_id in sorted(set(phrase_ids)):
            rows = grouped[(split, story_id, phrase_id)]
            story_rows.extend(rows)
        examples.extend(_examples_from_story_records(story_rows, sequence_len=sequence_len))
    return examples


def build_phrase_sequence_examples_streaming(records, sequence_len, progress_every=100000, max_examples=None):
    examples = []
    current_key = None
    current_rows = []
    records_seen = 0
    stories_seen = 0

    for raw_record in records:
        typed_records = [raw_record] if "record_type" in raw_record else _legacy_record_to_typed_records(raw_record)
        records_seen += 1
        for record in typed_records:
            key = (record["split"], int(record["story_id"]))
            if current_key is not None and key != current_key:
                examples.extend(_examples_from_story_records(current_rows, sequence_len=sequence_len))
                if max_examples is not None and len(examples) >= max_examples:
                    return examples[:max_examples]
                stories_seen += 1
                if progress_every > 0 and stories_seen % progress_every == 0:
                    print(f"loaded {stories_seen} stories from records; examples={len(examples)} records={records_seen}")
                current_rows = []
            current_key = key
            current_rows.append(record)

    if current_rows:
        examples.extend(_examples_from_story_records(current_rows, sequence_len=sequence_len))
        if max_examples is not None and len(examples) > max_examples:
            examples = examples[:max_examples]
        stories_seen += 1
    print(f"loaded {stories_seen} stories from records; examples={len(examples)} records={records_seen}")
    return examples


def split_examples(examples, validation_split=0.05, seed=42):
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1 or validation_split <= 0:
        return shuffled, []
    val_count = max(1, int(len(shuffled) * validation_split))
    return shuffled[val_count:], shuffled[:val_count]


def collate_phrase_sequences(examples, sequence_len, dummy_token_id=0, device="cpu"):
    batch_size = len(examples)
    idx = torch.full((batch_size, sequence_len), int(dummy_token_id), dtype=torch.long, device=device)
    targets = torch.full((batch_size, sequence_len), -1, dtype=torch.long, device=device)
    flat_indices = []
    offsets = []
    positions = []
    cursor = 0

    for batch_idx, example in enumerate(examples):
        length = min(sequence_len, len(example.targets))
        if length == 0:
            continue
        targets[batch_idx, :length] = torch.tensor(example.targets[:length], dtype=torch.long, device=device)
        for time_idx, indices in enumerate(example.input_indices[:length]):
            if not indices:
                continue
            offsets.append(cursor)
            flat_indices.extend(int(index) for index in indices)
            cursor += len(indices)
            positions.append([batch_idx, time_idx])

    return PhraseSequenceBatch(
        idx=idx,
        targets=targets,
        phrase_indices=torch.tensor(flat_indices, dtype=torch.long, device=device),
        phrase_offsets=torch.tensor(offsets, dtype=torch.long, device=device),
        phrase_batch_positions=torch.tensor(positions, dtype=torch.long, device=device),
    )


def examples_to_tensor_shard(examples, sequence_len):
    targets = torch.full((len(examples), sequence_len), -1, dtype=torch.long)
    lengths = torch.zeros(len(examples), dtype=torch.long)
    vector_offsets = torch.zeros((len(examples), sequence_len + 1), dtype=torch.long)
    flat_indices = []
    cursor = 0

    for example_idx, example in enumerate(examples):
        length = min(sequence_len, len(example.targets))
        lengths[example_idx] = length
        if length:
            targets[example_idx, :length] = torch.tensor(example.targets[:length], dtype=torch.long)
        for time_idx in range(sequence_len):
            vector_offsets[example_idx, time_idx] = cursor
            if time_idx < length:
                indices = example.input_indices[time_idx]
                flat_indices.extend(int(index) for index in indices)
                cursor += len(indices)
        vector_offsets[example_idx, sequence_len] = cursor

    return {
        "format": "phrase_gpt_tensor_shard_v1",
        "sequence_len": int(sequence_len),
        "targets": targets,
        "lengths": lengths,
        "phrase_indices": torch.tensor(flat_indices, dtype=torch.long),
        "vector_offsets": vector_offsets,
        "num_examples": len(examples),
    }


def tensor_shard_to_examples(shard):
    sequence_len = int(shard["sequence_len"])
    targets = shard["targets"]
    lengths = shard["lengths"]
    phrase_indices = shard["phrase_indices"]
    vector_offsets = shard["vector_offsets"]
    examples = []

    for example_idx in range(int(shard["num_examples"])):
        length = int(lengths[example_idx].item())
        input_indices = []
        for time_idx in range(length):
            start = int(vector_offsets[example_idx, time_idx].item())
            end = int(vector_offsets[example_idx, time_idx + 1].item())
            input_indices.append([int(index) for index in phrase_indices[start:end].tolist()])
        examples.append(PhraseSequenceExample(
            input_indices=input_indices,
            targets=[int(target) for target in targets[example_idx, :length].tolist()],
        ))
    return examples


PHRASE_UNK_TOKEN = "<unk>"


def load_vocab_top_k_remap(path, top_k):
    """Return (lookup, tokens): lookup maps old vocab indices to new ones, with
    everything outside the top_k most frequent tokens collapsed to a trailing
    <unk> index. Kept tokens are renumbered in original-index order, so the
    remap is monotone and relative order between kept indices is preserved."""
    with open(path, "r", encoding="utf-8") as file:
        rows = json.load(file)
    ranked = sorted(rows, key=lambda row: (-int(row["count"]), int(row["index"])))
    kept = sorted(ranked[:top_k], key=lambda row: int(row["index"]))
    unk_index = len(kept)
    lookup = torch.full((len(rows),), unk_index, dtype=torch.long)
    for new_index, row in enumerate(kept):
        lookup[int(row["index"])] = new_index
    tokens = [row["token"] for row in kept] + [PHRASE_UNK_TOKEN]
    return lookup, tokens


def extend_phrase_vocab_state(checkpoint, extra_rows, n_embd, seed=0, vocab_size=None):
    """Grow phrase_wte by extra_rows (normal 0, 0.02) so a resumed run can
    accept latent-id inputs. Only touches the model weight tensor, resolved
    by name -- NOT optimizer state. Optimizer moments for phrase_wte must be
    padded separately by param identity (see pad_phrase_optimizer_state),
    because in a real checkpoint wte/lm_head/phrase_wte optimizer moments can
    share phrase_wte's exact old shape (all == vocab_size rows) and a
    shape-based rule cannot tell them apart.

    Accepts both this trainer's on-disk checkpoint layout (top-level
    "model_state_dict") and the plain "model" layout used by tests.

    Idempotent when `vocab_size` is given: if phrase_wte.weight already has
    vocab_size + extra_rows rows, the checkpoint is already extended (e.g. a
    runner restart resuming its own previously-extended checkpoint) and this
    is a no-op. Without `vocab_size`, always extends (legacy behavior, used
    by callers that don't have a stable target row count to compare against).
    """
    if extra_rows <= 0:
        return checkpoint
    model_key = "model" if "model" in checkpoint else "model_state_dict"
    model_state = checkpoint[model_key]
    key = next(k for k in model_state if k.endswith("phrase_wte.weight"))
    old_weight = model_state[key]
    if vocab_size is not None and old_weight.shape[0] >= vocab_size + extra_rows:
        return checkpoint
    generator = torch.Generator().manual_seed(seed)
    new_rows = torch.normal(0.0, 0.02, size=(extra_rows, n_embd), generator=generator, dtype=old_weight.dtype)
    model_state[key] = torch.cat([old_weight, new_rows.to(old_weight.device)], dim=0)
    return checkpoint


def pad_phrase_optimizer_state(checkpoint, model, optimizer, extra_rows, vocab_size=None):
    """Zero-pad the AdamW moment tensors belonging to model.phrase_wte.weight
    in checkpoint's saved optimizer state, resolved by PARAM IDENTITY (not
    shape). Must be called after `model` and `optimizer` are built with the
    already-extended phrase_vocab_size (so model.phrase_wte.weight already has
    the new row count) and before optimizer.load_state_dict(...).

    Shape-based padding is unsafe here: in the real deployment checkpoint,
    wte and lm_head optimizer moments have the exact same shape as
    phrase_wte's (all vocab_size rows), so a shape rule would wrongly pad
    them too -- model.load_state_dict succeeds regardless (it never sees
    optimizer state), optimizer.load_state_dict accepts the corrupted sizes
    silently, and the first optimizer.step() then crashes with a shape
    mismatch (or worse, silently misaligns moments to weights).

    Idempotent: `model`'s phrase_wte row count is fixed by config regardless
    of resume history, so `old_rows` (the row count BEFORE this extension) is
    derived from `vocab_size` when given, else from the model shape as before.
    Either way, a phrase_wte moment tensor that's already at the extended row
    count (vocab_size + extra_rows, e.g. from a runner restart resuming a
    checkpoint this trainer already padded once) won't match `old_rows` and is
    left untouched by the per-tensor shape check below -- no double padding.
    """
    if extra_rows <= 0:
        return checkpoint
    params = [p for group in optimizer.param_groups for p in group["params"]]
    phrase_weight = model.phrase_wte.weight
    param_index = next(i for i, p in enumerate(params) if p is phrase_weight)
    old_rows = vocab_size if vocab_size is not None else phrase_weight.shape[0] - extra_rows
    optimizer_key = "optimizer" if "optimizer" in checkpoint else "optimizer_state_dict"
    optimizer_state = checkpoint.get(optimizer_key, {}).get("state", {})
    entry = optimizer_state.get(param_index)
    if entry is None:
        return checkpoint
    for name, tensor in list(entry.items()):
        if name == "step" or not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            continue
        if tensor.dim() == 0 or tensor.shape[0] != old_rows:
            continue
        pad = torch.zeros((extra_rows,) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        entry[name] = torch.cat([tensor, pad], dim=0)
    return checkpoint


def resolve_resume_vocab_top_k(saved_vocab_top_k, requested_vocab_top_k, extend_phrase_vocab):
    """Reconcile a resumed checkpoint's recorded vocab_top_k with the one
    requested on this run's CLI (args.vocab_top_k). Returns the EFFECTIVE
    vocab_top_k to record in this run's own saved checkpoints.

    SAE post-train resumes a --vocab-top-k-trained base checkpoint WITHOUT
    --vocab-top-k, because its shards are pre-remapped already (no remap
    should be applied again at shard-load time). That means
    saved_vocab_top_k (e.g. 8191) and requested_vocab_top_k (None) will
    legitimately differ whenever --extend-phrase-vocab is used and no
    --vocab-top-k was requested -- in that one case, inherit the checkpoint's
    value as the effective one instead of raising. The caller must NOT feed
    this return value back into the shard-load remap decision (that stays
    keyed off the raw requested_vocab_top_k / args.vocab_top_k, which must
    stay None so shards are loaded unremapped, as they already are).

    Any other mismatch (no --extend-phrase-vocab, or an explicit conflicting
    --vocab-top-k) is a real error and still raises.
    """
    if saved_vocab_top_k == requested_vocab_top_k:
        return requested_vocab_top_k
    if extend_phrase_vocab > 0 and requested_vocab_top_k is None:
        return saved_vocab_top_k
    raise SystemExit(f"Checkpoint was trained with --vocab-top-k {saved_vocab_top_k}, but got --vocab-top-k {requested_vocab_top_k}.")


def remap_tensor_shard(shard, lookup):
    remapped = dict(shard)
    remapped["phrase_indices"] = lookup[shard["phrase_indices"]]
    targets = shard["targets"]
    remapped["targets"] = torch.where(targets >= 0, lookup[targets.clamp(min=0)], targets)
    return remapped


def collate_shard_batch(shard, example_indices, dummy_token_id=0, device="cpu"):
    """Build a PhraseSequenceBatch by slicing shard tensors directly, without the
    per-example Python list round-trip of tensor_shard_to_examples + collate."""
    sequence_len = int(shard["sequence_len"])
    example_indices = torch.as_tensor(example_indices, dtype=torch.long)
    targets = shard["targets"][example_indices]
    vector_offsets = shard["vector_offsets"][example_indices]
    counts = vector_offsets[:, 1:] - vector_offsets[:, :-1]
    active = counts > 0
    positions = active.nonzero()
    starts = vector_offsets[:, :-1][active]
    lengths = counts[active]
    offsets = torch.cumsum(lengths, dim=0) - lengths
    within = torch.arange(int(lengths.sum())) - torch.repeat_interleave(offsets, lengths)
    gather = torch.repeat_interleave(starts, lengths) + within
    idx = torch.full((example_indices.numel(), sequence_len), int(dummy_token_id), dtype=torch.long)
    return PhraseSequenceBatch(
        idx=idx.to(device),
        targets=targets.to(device),
        phrase_indices=shard["phrase_indices"][gather].to(device),
        phrase_offsets=offsets.to(device),
        phrase_batch_positions=positions.to(device),
    )


def iter_shard_batches(shard, batch_size, shuffle=True, seed=42, dummy_token_id=0, device="cpu"):
    num_examples = int(shard["num_examples"])
    if shuffle:
        order = torch.randperm(num_examples, generator=torch.Generator().manual_seed(seed))
    else:
        order = torch.arange(num_examples)
    for start in range(0, num_examples, batch_size):
        yield collate_shard_batch(shard, order[start:start + batch_size], dummy_token_id=dummy_token_id, device=device)


def load_shard_manifest(path):
    with open(path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    manifest["manifest_path"] = os.path.abspath(path)
    manifest["base_dir"] = os.path.dirname(os.path.abspath(path))
    return manifest


def iter_shards(manifest, seed=42, shuffle=True, split=None, remap=None, start_shard=0):
    shards = [
        shard for shard in manifest.get("shards", [])
        if split is None or shard.get("split") == split
    ]
    if shuffle:
        random.Random(seed).shuffle(shards)
    for shard in shards[start_shard:]:
        path = os.path.join(manifest["base_dir"], shard["file"])
        tensor_shard = torch.load(path, map_location="cpu", weights_only=False)
        yield remap_tensor_shard(tensor_shard, remap) if remap is not None else tensor_shard


def iter_shard_example_sets(manifest, seed=42, shuffle=True, split=None):
    for tensor_shard in iter_shards(manifest, seed=seed, shuffle=shuffle, split=split):
        yield tensor_shard_to_examples(tensor_shard)


def iter_batches(examples, batch_size, shuffle=True, seed=42):
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[index] for index in order[start:start + batch_size]]


def choose_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_epoch_batches(model, batches, total_batches, optimizer=None, progress_every=25, label="train"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.set_grad_enabled(training):
        for batch_idx, batch in enumerate(batches, start=1):
            logits = model(
                batch.idx,
                phrase_indices=batch.phrase_indices,
                phrase_offsets=batch.phrase_offsets,
                phrase_batch_positions=batch.phrase_batch_positions,
            )
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch.targets.view(-1), ignore_index=-1)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            valid = batch.targets != -1
            token_count = valid.sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count
            total_correct += ((logits.argmax(dim=-1) == batch.targets) & valid).sum().item()
            if progress_every > 0 and (batch_idx % progress_every == 0 or batch_idx == total_batches):
                print(f"{label} batch {batch_idx}/{total_batches} tokens={total_tokens} loss={total_loss / max(total_tokens, 1):.6f} acc={total_correct / max(total_tokens, 1):.4f}")

    return {
        "loss": total_loss / total_tokens if total_tokens else float("nan"),
        "accuracy": total_correct / total_tokens if total_tokens else 0.0,
        "tokens": total_tokens,
    }


def run_epoch(model, examples, optimizer=None, batch_size=16, sequence_len=64, dummy_token_id=0, device="cpu", seed=42, progress_every=25, label="train"):
    training = optimizer is not None
    total_batches = math.ceil(len(examples) / batch_size) if examples else 0
    batches = (
        collate_phrase_sequences(batch_examples, sequence_len=sequence_len, dummy_token_id=dummy_token_id, device=device)
        for batch_examples in iter_batches(examples, batch_size, shuffle=training, seed=seed)
    )
    return run_epoch_batches(model, batches, total_batches, optimizer=optimizer, progress_every=progress_every, label=label)


def combine_metric_totals(metric_rows):
    total_tokens = sum(row.get("tokens", 0) for row in metric_rows)
    if total_tokens <= 0:
        return {"loss": float("nan"), "accuracy": 0.0, "tokens": 0}
    total_loss = sum(row["loss"] * row.get("tokens", 0) for row in metric_rows)
    total_correct = sum(row["accuracy"] * row.get("tokens", 0) for row in metric_rows)
    return {
        "loss": total_loss / total_tokens,
        "accuracy": total_correct / total_tokens,
        "tokens": total_tokens,
    }


def run_epoch_on_shards(model, manifest, optimizer=None, batch_size=16, sequence_len=64, dummy_token_id=0, device="cpu", seed=42, progress_every=25, label="train", split=None, remap=None, start_shard=0, prior_rows=None, on_shard_end=None):
    rows = list(prior_rows) if prior_rows else []
    training = optimizer is not None
    shard_count = manifest.get("splits", {}).get(split, {}).get("num_shards", manifest.get("num_shards", "?"))
    shard_iter = iter_shards(manifest, seed=seed, shuffle=training, split=split, remap=remap, start_shard=start_shard)
    for shard_idx, shard in enumerate(shard_iter, start=start_shard + 1):
        shard_label = f"{label} shard {shard_idx}/{shard_count}"
        total_batches = math.ceil(int(shard["num_examples"]) / batch_size)
        batches = iter_shard_batches(
            shard,
            batch_size,
            shuffle=training,
            seed=seed + shard_idx,
            dummy_token_id=dummy_token_id,
            device=device,
        )
        rows.append(run_epoch_batches(
            model,
            batches,
            total_batches,
            optimizer=optimizer,
            progress_every=progress_every,
            label=shard_label,
        ))
        if on_shard_end is not None:
            on_shard_end(shard_idx, rows)
    return combine_metric_totals(rows)


def save_checkpoint(out_dir, model, config, metrics, filename="phrase_gpt.pt", optimizer=None, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, os.path.join(out_dir, filename))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train nanochat GPT from sparse phrase vectors with dummy token ids.")
    parser.add_argument("--vocab", required=True, help="Path to phrase vocab.json.")
    parser.add_argument("--records", default=None, help="Path to phrase_index.jsonl.")
    parser.add_argument("--shards", default=None, help="Path to phrase GPT tensor shard manifest.json.")
    parser.add_argument("--out-dir", default="phrase_gpt_out")
    parser.add_argument("--sequence-len", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=None, help="Alias for --epochs that emphasizes early-stopping use.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many validation epochs without loss improvement. Use 0 to disable.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum validation-loss improvement required to reset patience.")
    parser.add_argument("--target-val-loss", type=float, default=None, help="Stop once validation loss is at or below this value.")
    parser.add_argument("--target-val-accuracy", type=float, default=None, help="Stop once validation accuracy is at or above this value.")
    parser.add_argument("--save-best", action="store_true", help="Also write best_phrase_gpt.pt whenever validation loss improves.")
    parser.add_argument("--resume", default=None, help="Path to a phrase_gpt.pt checkpoint to resume training from (restores model, optimizer, metrics history, and mid-epoch shard progress).")
    parser.add_argument("--extend-phrase-vocab", type=int, default=0, help="Extra phrase-embedding rows (SAE latent ids) appended past the token vocab. Requires --resume.")
    parser.add_argument("--checkpoint-every-shards", type=int, default=10, help="In shard mode, write a resumable rolling checkpoint every N train shards. Use 0 to disable.")
    parser.add_argument("--vocab-top-k", type=int, default=None, help="Keep only the K most frequent phrase tokens; remap the rest to a trailing <unk> index at shard-load time. Requires --shards.")
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--dummy-token-id", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--load-progress-every", type=int, default=100000, help="Print record-loading progress every N stories. Use 0 to disable intermediate progress.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--sweep-eval-records", default=None, help="Held-out records for in-training sweep eval (enables it).")
    parser.add_argument("--sweep-eval-split", default=None, help="Restrict sweep probes to this split, e.g. 'validation'.")
    parser.add_argument("--sweep-index-map", default=None, help="old_to_new.json applied to sweep-eval record indices (use when shards were built with --index-map).")
    parser.add_argument("--sweep-cross-clause", action="store_true", help="Build sweep probe history with cross-clause chains (match hybrid-cross shards).")
    parser.add_argument("--sweep-eval-every-epochs", type=int, default=1)
    parser.add_argument("--sweep-eval-every-shards", type=int, default=0)
    parser.add_argument("--sweep-x-values", default="0,1,2,4,8,16")
    parser.add_argument("--sweep-d-values", default="1,2,4,8,16,32")
    parser.add_argument("--sweep-max-probes", type=int, default=2000)
    parser.add_argument("--sweep-batch-size", type=int, default=32)
    parser.add_argument("--sweep-bootstrap", type=int, default=1000)
    parser.add_argument("--sweep-seed", type=int, default=0)
    return parser.parse_args()


def _run_training_sweep(model, probes, args, remap, device, epoch, shard, trajectory):
    from scripts.hybrid_sweep import run_sweep, _parse_int_list  # lazy: avoids import cycle
    was_training = model.training
    try:
        result = run_sweep(
            model, probes,
            x_values=_parse_int_list(args.sweep_x_values),
            d_values=_parse_int_list(args.sweep_d_values) + [None],
            fixed_x_for_depth=0, remap=remap,
            batch_size=args.sweep_batch_size, device=device,
            bootstrap=args.sweep_bootstrap, bootstrap_seed=args.sweep_seed,
            reset_on_clause=not getattr(args, "sweep_cross_clause", False),
        )
        result["split"] = args.sweep_eval_split
        trajectory.append({"epoch": epoch, "shard": shard, "sweep": result})
    finally:
        model.train(was_training)
    return result


def main():
    args = parse_args()
    if args.max_epochs is not None:
        args.epochs = args.max_epochs
    vocab = load_vocab(args.vocab)
    manifest = load_shard_manifest(args.shards) if args.shards else None
    vocab_remap = None
    vocab_size = vocab.size
    if args.vocab_top_k is not None:
        if manifest is None:
            raise SystemExit("--vocab-top-k requires --shards (records mode is not supported).")
        vocab_remap, remapped_tokens = load_vocab_top_k_remap(args.vocab, args.vocab_top_k)
        vocab_size = len(remapped_tokens)
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "vocab_top_k.json"), "w", encoding="utf-8") as file:
            json.dump([{"token": token, "index": index} for index, token in enumerate(remapped_tokens)], file)
            file.write("\n")
        print(f"vocab top-k: {vocab.size} -> {vocab_size} (top {args.vocab_top_k} + {PHRASE_UNK_TOKEN})", flush=True)
    if manifest is None:
        if not args.records:
            raise SystemExit("Either --records or --shards is required.")
        print(f"loading sequence examples from {args.records}", flush=True)
        examples = build_phrase_sequence_examples_streaming(
            iter_records(args.records),
            sequence_len=args.sequence_len,
            progress_every=args.load_progress_every,
            max_examples=args.limit_examples,
        )
        if not examples:
            raise SystemExit("No phrase sequence examples found.")
        train_examples, val_examples = split_examples(examples, validation_split=args.validation_split, seed=args.seed)
        train_example_count = len(train_examples)
        val_example_count = len(val_examples)
        example_count = len(examples)
    else:
        if int(manifest["sequence_len"]) != args.sequence_len:
            raise SystemExit(f"Shard sequence_len {manifest['sequence_len']} does not match --sequence-len {args.sequence_len}")
        train_examples = None
        val_examples = []
        example_count = int(manifest.get("num_examples", 0))
        train_example_count = int(manifest.get("splits", {}).get("train", {}).get("num_examples", example_count))
        val_example_count = int(manifest.get("splits", {}).get("validation", {}).get("num_examples", 0))
    device = choose_device(args.device)
    config = GPTConfig(
        sequence_len=args.sequence_len,
        vocab_size=vocab_size,
        n_layer=args.depth,
        n_head=args.n_head,
        n_kv_head=args.n_head,
        n_embd=args.n_embd,
        window_pattern="L",
        phrase_vocab_size=vocab_size + args.extend_phrase_vocab,
    )
    model = GPT(config).to(device)
    model.init_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metrics = {
        "vocab_size": vocab_size,
        "vocab_top_k": args.vocab_top_k,
        "examples": example_count,
        "train_examples": train_example_count,
        "val_examples": val_example_count,
        "input": "dummy_token_ids_plus_phrase_vectors",
        "target": "next_phrase_vocab_token",
        "shards": args.shards,
        "records": args.records,
        "best_epoch": None,
        "best_val_loss": None,
        "stop_reason": None,
        "epochs": [],
        "sweep_trajectory": [],
    }
    early_stopping = EarlyStoppingConfig(
        patience=args.patience,
        min_delta=args.min_delta,
        target_val_loss=args.target_val_loss,
        target_val_accuracy=args.target_val_accuracy,
    )
    early_state = EarlyStoppingState()
    checkpoint_config = config.__dict__ | vars(args)

    start_epoch = 1
    start_shard = 0
    prior_rows = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.extend_phrase_vocab > 0:
            checkpoint = extend_phrase_vocab_state(checkpoint, args.extend_phrase_vocab, args.n_embd, seed=args.seed, vocab_size=vocab_size)
        saved_config = checkpoint.get("config", {})
        if "vocab_top_k" in saved_config:
            effective_vocab_top_k = resolve_resume_vocab_top_k(saved_config["vocab_top_k"], args.vocab_top_k, args.extend_phrase_vocab)
            if effective_vocab_top_k != args.vocab_top_k:
                # SAE post-train: inherited from the checkpoint for bookkeeping only.
                # args.vocab_top_k itself stays None, so vocab_remap above (built
                # only from args.vocab_top_k) stays None too -- shards are loaded
                # unremapped, as they must be (they're pre-remapped already).
                checkpoint_config["vocab_top_k"] = effective_vocab_top_k
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            if args.extend_phrase_vocab > 0:
                checkpoint = pad_phrase_optimizer_state(checkpoint, model, optimizer, args.extend_phrase_vocab, vocab_size=vocab_size)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        saved_metrics = checkpoint.get("metrics", {})
        metrics["epochs"] = list(saved_metrics.get("epochs", []))
        metrics["sweep_trajectory"] = list(saved_metrics.get("sweep_trajectory", []))
        metrics["best_epoch"] = saved_metrics.get("best_epoch")
        metrics["best_val_loss"] = saved_metrics.get("best_val_loss")
        early_state.best_epoch = metrics["best_epoch"]
        early_state.best_val_loss = metrics["best_val_loss"]
        early_state.epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        shard_progress = checkpoint.get("shard_progress")
        if shard_progress:
            start_epoch = int(shard_progress["epoch"])
            start_shard = int(shard_progress["shards_completed"])
            prior_rows = list(shard_progress["metric_rows"])
            if saved_config.get("seed") is not None and saved_config["seed"] != args.seed:
                print(f"warning: resuming mid-epoch with --seed {args.seed} but checkpoint used seed {saved_config['seed']}; shard order will not match and some data may be repeated or skipped this epoch.", flush=True)
        else:
            start_epoch = int(checkpoint.get("epoch", len(metrics["epochs"]))) + 1
        resume_detail = f", shard {start_shard}" if start_shard else ""
        print(f"resuming from {args.resume} at epoch {start_epoch}{resume_detail}", flush=True)
        if start_epoch > args.epochs:
            print(f"checkpoint already has {start_epoch - 1} completed epochs; nothing to do for --epochs {args.epochs}.", flush=True)

    sweep_probes = None
    if args.sweep_eval_records:
        from scripts.hybrid_sweep import build_sweep_probes  # lazy: avoids import cycle
        sweep_index_map = None
        if args.sweep_index_map:
            with open(args.sweep_index_map, "r", encoding="utf-8") as file:
                sweep_index_map = json.load(file)
        sweep_probes = build_sweep_probes(iter_records(args.sweep_eval_records),
                                          min_history=1, max_probes=args.sweep_max_probes,
                                          split=args.sweep_eval_split, index_map=sweep_index_map)
        if not sweep_probes:
            raise SystemExit("--sweep-eval-records produced 0 probes (check --sweep-eval-split / path)")
        print(f"sweep eval: {len(sweep_probes)} probes from {args.sweep_eval_records} (split={args.sweep_eval_split})", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_shard = start_shard if epoch == start_epoch else 0
        epoch_prior_rows = prior_rows if epoch == start_epoch else None

        def on_shard_end(shard_idx, rows, epoch=epoch):
            if args.checkpoint_every_shards > 0 and shard_idx % args.checkpoint_every_shards == 0:
                save_checkpoint(args.out_dir, model, checkpoint_config, metrics, optimizer=optimizer, extra={
                    "epoch": epoch - 1,
                    "epochs_without_improvement": early_state.epochs_without_improvement,
                    "shard_progress": {"epoch": epoch, "shards_completed": shard_idx, "metric_rows": rows},
                })
            if sweep_probes is not None and args.sweep_eval_every_shards > 0 and shard_idx % args.sweep_eval_every_shards == 0:
                _run_training_sweep(model, sweep_probes, args, vocab_remap, device, epoch, shard_idx, metrics["sweep_trajectory"])

        if manifest is None:
            train_metrics = run_epoch(
                model,
                train_examples,
                optimizer=optimizer,
                batch_size=args.batch_size,
                sequence_len=args.sequence_len,
                dummy_token_id=args.dummy_token_id,
                device=device,
                seed=args.seed + epoch,
                progress_every=args.progress_every,
                label=f"train epoch {epoch}/{args.epochs}",
            )
            val_metrics = run_epoch(
                model,
                val_examples,
                optimizer=None,
                batch_size=args.batch_size,
                sequence_len=args.sequence_len,
                dummy_token_id=args.dummy_token_id,
                device=device,
                seed=args.seed,
                progress_every=0,
                label=f"val epoch {epoch}/{args.epochs}",
            ) if val_examples else {"loss": None, "accuracy": None, "tokens": 0}
        else:
            train_metrics = run_epoch_on_shards(
                model,
                manifest,
                optimizer=optimizer,
                batch_size=args.batch_size,
                sequence_len=args.sequence_len,
                dummy_token_id=args.dummy_token_id,
                device=device,
                seed=args.seed + epoch,
                progress_every=args.progress_every,
                label=f"train epoch {epoch}/{args.epochs}",
                split="train" if manifest.get("splits", {}).get("train") else None,
                remap=vocab_remap,
                start_shard=epoch_start_shard,
                prior_rows=epoch_prior_rows,
                on_shard_end=on_shard_end,
            )
            val_metrics = run_epoch_on_shards(
                model,
                manifest,
                optimizer=None,
                batch_size=args.batch_size,
                sequence_len=args.sequence_len,
                dummy_token_id=args.dummy_token_id,
                device=device,
                seed=args.seed,
                progress_every=0,
                label=f"val epoch {epoch}/{args.epochs}",
                split="validation",
                remap=vocab_remap,
            ) if val_example_count else {"loss": None, "accuracy": None, "tokens": 0}
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        metrics["epochs"].append(row)
        if sweep_probes is not None and epoch % max(1, args.sweep_eval_every_epochs) == 0:
            _run_training_sweep(model, sweep_probes, args, vocab_remap, device, epoch, None, metrics["sweep_trajectory"])
        if val_example_count:
            early_state.update(val_metrics, epoch=epoch, config=early_stopping)
            metrics["best_epoch"] = early_state.best_epoch
            metrics["best_val_loss"] = early_state.best_val_loss
            if early_state.is_best and args.save_best:
                save_checkpoint(args.out_dir, model, checkpoint_config, metrics, filename="best_phrase_gpt.pt")
            if early_state.should_stop:
                metrics["stop_reason"] = early_state.stop_reason
                row["stop_reason"] = early_state.stop_reason
        save_checkpoint(args.out_dir, model, checkpoint_config, metrics, optimizer=optimizer, extra={
            "epoch": epoch,
            "epochs_without_improvement": early_state.epochs_without_improvement,
        })
        print(json.dumps(row))
        if early_state.should_stop:
            break

    completed_epochs = metrics["epochs"][-1]["epoch"] if metrics["epochs"] else 0
    save_checkpoint(args.out_dir, model, checkpoint_config, metrics, optimizer=optimizer, extra={
        "epoch": completed_epochs,
        "epochs_without_improvement": early_state.epochs_without_improvement,
    })
    print(json.dumps({"saved": os.path.join(args.out_dir, "phrase_gpt.pt"), **metrics}, indent=2))


if __name__ == "__main__":
    main()
