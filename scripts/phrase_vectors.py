"""
Build a phrase-position vocabulary and compact sparse phrase records.

Example:
python -m scripts.phrase_vectors \
    --archive ../archive.zip \
    --out-dir phrase_vectors_out \
    --limit 100
"""

import argparse
import csv
import gzip
import io
import json
import os
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass


DEFAULT_PHRASE_PUNCT = {".", "!", "?", ";", ":", ","}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
OBJECT_DEPS = {"dobj", "obj", "iobj", "pobj", "attr", "oprd"}
VERB_DEPS = {"ROOT", "conj", "advcl", "relcl", "xcomp", "ccomp"}
RELATION_DEPS = {"prep", "agent"}
MOJIBAKE_REPLACEMENTS = {
    "â€œ": '"',
    "â€": '"',
    "â€˜": "'",
    "â€™": "'",
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "â€": '"',
    "Â": "",
}
ASCII_PUNCTUATION_REPLACEMENTS = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "–": "-",
    "—": "-",
    "…": "...",
    "\u00a0": " ",
}


@dataclass(frozen=True)
class PhraseOccurrence:
    label: str
    tokens: list[str]
    start: int
    end: int


def open_text(path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def jsonl_output_path(out_dir, stem, compress=True):
    suffix = ".jsonl.gz" if compress else ".jsonl"
    return os.path.join(out_dir, f"{stem}{suffix}")


def existing_jsonl_path(out_dir, stem, compress=True):
    preferred = jsonl_output_path(out_dir, stem, compress=compress)
    alternate = jsonl_output_path(out_dir, stem, compress=not compress)
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(alternate):
        return alternate
    return preferred


def normalize_token_text(token, lowercase=True):
    text = token.text.strip()
    return text.lower() if lowercase else text


def normalize_story_text(text):
    """Repair common TinyStories mojibake and normalize smart punctuation to ASCII."""
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        repaired = text

    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(bad, good)
    for bad, good in ASCII_PUNCTUATION_REPLACEMENTS.items():
        repaired = repaired.replace(bad, good)
    return repaired


def _clean_tokens(tokens, lowercase=True):
    return [
        normalize_token_text(token, lowercase=lowercase)
        for token in tokens
        if not getattr(token, "is_space", False) and normalize_token_text(token, lowercase=lowercase)
    ]


def _add_phrase(phrases, seen, label, tokens, lowercase=True):
    cleaned = _clean_tokens(tokens, lowercase=lowercase)
    if not cleaned:
        return
    start = min(token.i for token in tokens)
    end = max(token.i for token in tokens) + 1
    key = (label, start, end, tuple(cleaned))
    if key in seen:
        return
    seen.add(key)
    phrases.append(PhraseOccurrence(label=label, tokens=cleaned, start=start, end=end))


def _subtree_tokens(token):
    if hasattr(token, "subtree"):
        return list(token.subtree)
    children = [child for child in getattr(token, "children", [])]
    if not children:
        return [token]
    tokens = [token]
    for child in children:
        tokens.extend(_subtree_tokens(child))
    return sorted(set(tokens), key=lambda item: item.i)


def _sentence_tokens(doc):
    try:
        return list(doc.sents)
    except Exception:
        return [doc[:]]


def _punctuation_phrases(sent, phrases, seen, lowercase=True, phrase_punct=DEFAULT_PHRASE_PUNCT):
    current = []
    for token in sent:
        current.append(token)
        if token.text in phrase_punct:
            _add_phrase(phrases, seen, "punctuation", current, lowercase=lowercase)
            _add_phrase(phrases, seen, "punctuation_mark", [token], lowercase=lowercase)
            current = []
    if current:
        _add_phrase(phrases, seen, "punctuation", current, lowercase=lowercase)


def _dependency_role_phrases(sent, phrases, seen, lowercase=True):
    sent_tokens = list(sent)
    sent_start = sent_tokens[0].i if sent_tokens else 0
    sent_end = sent_tokens[-1].i + 1 if sent_tokens else 0

    for token in sent_tokens:
        dep = token.dep_
        if dep in SUBJECT_DEPS:
            _add_phrase(phrases, seen, "subject", _subtree_tokens(token), lowercase=lowercase)
        elif dep in OBJECT_DEPS:
            _add_phrase(phrases, seen, "object", _subtree_tokens(token), lowercase=lowercase)
        elif dep in RELATION_DEPS:
            _add_phrase(phrases, seen, "relation", _subtree_tokens(token), lowercase=lowercase)

        if dep in VERB_DEPS and token.pos_ in {"VERB", "AUX"}:
            role_tokens = [token]
            role_tokens.extend(child for child in getattr(token, "children", []) if child.dep_ in SUBJECT_DEPS | OBJECT_DEPS)
            role_tokens = sorted(role_tokens, key=lambda item: item.i)
            _add_phrase(phrases, seen, "verb", role_tokens, lowercase=lowercase)

    roots = [token for token in sent_tokens if token.dep_ == "ROOT"]
    for root in roots:
        left = max(sent_start, min([root.i] + [child.i for child in getattr(root, "children", [])]))
        right = min(sent_end, max([root.i] + [child.i for child in getattr(root, "children", [])]) + 1)
        _add_phrase(phrases, seen, "root_clause", [token for token in sent_tokens if left <= token.i < right], lowercase=lowercase)


def extract_phrase_occurrences(doc, lowercase=True, phrase_punct=DEFAULT_PHRASE_PUNCT):
    phrases = []
    seen = set()

    for sent in _sentence_tokens(doc):
        _punctuation_phrases(sent, phrases, seen, lowercase=lowercase, phrase_punct=phrase_punct)
        _dependency_role_phrases(sent, phrases, seen, lowercase=lowercase)

    try:
        noun_chunks = list(doc.noun_chunks)
    except Exception:
        noun_chunks = []
    for chunk in noun_chunks:
        _add_phrase(phrases, seen, "noun_chunk", list(chunk), lowercase=lowercase)

    return phrases


def build_vocab(phrases, min_count=1):
    stats = defaultdict(lambda: {"count": 0, "position_sum": 0.0})
    for phrase in phrases:
        update_vocab_stats(stats, phrase)
    return build_vocab_from_stats(stats, min_count=min_count)


def update_vocab_stats(stats, phrase):
    denom = max(len(phrase.tokens) - 1, 1)
    for pos, token in enumerate(phrase.tokens):
        stats[token]["count"] += 1
        stats[token]["position_sum"] += pos / denom


def build_vocab_from_stats(stats, min_count=1):
    vocab = []
    for token, token_stats in stats.items():
        count = token_stats["count"]
        if count < min_count:
            continue
        avg_position = token_stats["position_sum"] / count
        vocab.append({
            "token": token,
            "index": -1,
            "count": count,
            "avg_position": avg_position,
        })

    vocab.sort(key=lambda item: (item["avg_position"], -item["count"], item["token"]))
    for index, item in enumerate(vocab):
        item["index"] = index
    return vocab


def build_sparse_records(split, story_id, phrases, vocab, token_to_index=None):
    token_to_index = token_to_index or {entry["token"]: entry["index"] for entry in vocab}
    records = []
    for phrase_id, phrase in enumerate(phrases):
        indexed_tokens = [(token, token_to_index[token]) for token in phrase.tokens if token in token_to_index]
        for token_pos, (token, index) in enumerate(indexed_tokens):
            records.append({
                "split": split,
                "story_id": story_id,
                "phrase_id": phrase_id,
                "record_type": "single",
                "label": phrase.label,
                "start": phrase.start,
                "end": phrase.end,
                "token_pos": token_pos,
                "tokens": [token],
                "indices": [index],
                "values": "implicit_ones",
            })

        indices = [index for _, index in indexed_tokens]
        if not indices:
            continue
        if any(left >= right for left, right in zip(indices, indices[1:])):
            continue
        if len(indices) <= 1:
            continue
        records.append({
            "split": split,
            "story_id": story_id,
            "phrase_id": phrase_id,
            "record_type": "packed",
            "label": phrase.label,
            "start": phrase.start,
            "end": phrase.end,
            "tokens": phrase.tokens,
            "indices": indices,
            "values": "implicit_ones",
        })
    return records


def iter_archive_stories(archive_path, split, limit=None):
    filename = f"{split}.csv"
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(filename) as raw:
            text_file = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.DictReader(text_file)
            for story_id, row in enumerate(reader):
                if limit is not None and story_id >= limit:
                    break
                text = normalize_story_text(row.get("text", ""))
                if text:
                    yield story_id, text


def _tinystories_filename(split, source_format):
    if source_format == "txt":
        stem = "valid" if split == "validation" else split
        return f"TinyStories-{stem}.txt"
    if source_format == "v2-txt":
        stem = "valid" if split == "validation" else split
        return f"TinyStoriesV2-GPT4-{stem}.txt"
    raise ValueError(f"Unsupported TinyStories source format: {source_format}")


def iter_tinystories_stories(tinystories_dir, split, source_format="txt", limit=None):
    path = os.path.join(tinystories_dir, _tinystories_filename(split, source_format))
    story_parts = []
    story_id = 0
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip() == "<|endoftext|>":
                text = normalize_story_text("\n".join(story_parts).strip())
                if text:
                    yield story_id, text
                    story_id += 1
                    if limit is not None and story_id >= limit:
                        return
                story_parts = []
            else:
                story_parts.append(line.rstrip("\n"))

    text = normalize_story_text("\n".join(story_parts).strip())
    if text and (limit is None or story_id < limit):
        yield story_id, text


def iter_input_stories(archive_path, tinystories_dir, source_format, split, limit=None):
    if tinystories_dir:
        yield from iter_tinystories_stories(tinystories_dir, split, source_format=source_format, limit=limit)
    else:
        yield from iter_archive_stories(archive_path, split, limit=limit)


def load_spacy_model(model_name, disable_components=None):
    try:
        import spacy
    except ImportError as exc:
        raise SystemExit(
            "spaCy is required for archive processing. Install it with `uv add spacy` "
            "or `pip install spacy`, then install an English model such as "
            "`python -m spacy download en_core_web_sm`."
        ) from exc

    try:
        return spacy.load(model_name, disable=disable_components or [])
    except OSError as exc:
        raise SystemExit(
            f"Could not load spaCy model {model_name!r}. Install it with "
            f"`python -m spacy download {model_name}`."
        ) from exc


def collect_phrases(nlp, archive_path, tinystories_dir, source_format, splits, limit=None, lowercase=True, batch_size=64, n_process=1, progress_every=100):
    all_phrase_rows = []
    all_phrases = []
    for split in splits:
        pending_ids = []
        pending_texts = []
        for story_id, text in iter_input_stories(archive_path, tinystories_dir, source_format, split, limit=limit):
            if progress_every > 0 and story_id % progress_every == 0:
                print("Processing", split, "story", story_id)
            pending_ids.append(story_id)
            pending_texts.append(text)
            if len(pending_texts) == batch_size:
                _consume_batch(nlp, split, pending_ids, pending_texts, all_phrase_rows, all_phrases, lowercase, batch_size=batch_size, n_process=n_process)
                pending_ids, pending_texts = [], []
        if pending_texts:
            _consume_batch(nlp, split, pending_ids, pending_texts, all_phrase_rows, all_phrases, lowercase, batch_size=batch_size, n_process=n_process)
    return all_phrase_rows, all_phrases


def _consume_batch(nlp, split, story_ids, texts, all_phrase_rows, all_phrases, lowercase, batch_size=64, n_process=1):
    for story_id, doc in zip(story_ids, nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
        phrases = extract_phrase_occurrences(doc, lowercase=lowercase)
        all_phrase_rows.append((split, story_id, phrases))
        all_phrases.extend(phrases)


def phrase_to_dict(phrase):
    return {
        "label": phrase.label,
        "tokens": phrase.tokens,
        "start": phrase.start,
        "end": phrase.end,
    }


def phrase_from_dict(row):
    return PhraseOccurrence(
        label=row["label"],
        tokens=list(row["tokens"]),
        start=int(row["start"]),
        end=int(row["end"]),
    )


def phrase_row_to_dict(split, story_id, phrases):
    return {
        "split": split,
        "story_id": story_id,
        "phrases": [phrase_to_dict(phrase) for phrase in phrases],
    }


def phrase_row_from_dict(row):
    return (
        row["split"],
        int(row["story_id"]),
        [phrase_from_dict(phrase) for phrase in row.get("phrases", [])],
    )


def write_phrase_rows_jsonl(path, rows):
    with open_text(path, "wt") as file:
        for split, story_id, phrases in rows:
            file.write(json.dumps(phrase_row_to_dict(split, story_id, phrases), ensure_ascii=False, sort_keys=True))
            file.write("\n")


def iter_phrase_rows_jsonl(path):
    with open_text(path, "rt") as file:
        for line in file:
            line = line.strip()
            if line:
                yield phrase_row_from_dict(json.loads(line))


def collect_phrase_row_cache_stats(path):
    stats = defaultdict(lambda: {"count": 0, "position_sum": 0.0})
    phrase_row_count = 0
    phrase_count = 0
    completed = set()
    if not os.path.exists(path):
        return stats, phrase_row_count, phrase_count, completed

    for split, story_id, phrases in iter_phrase_rows_jsonl(path):
        completed.add((split, story_id))
        phrase_row_count += 1
        phrase_count += len(phrases)
        for phrase in phrases:
            update_vocab_stats(stats, phrase)
    return stats, phrase_row_count, phrase_count, completed


def collect_phrases_streaming(nlp, archive_path, tinystories_dir, source_format, splits, phrase_rows_path, limit=None, lowercase=True, batch_size=64, n_process=1, progress_every=100, resume=False):
    stats, phrase_row_count, phrase_count, completed = collect_phrase_row_cache_stats(phrase_rows_path) if resume else (
        defaultdict(lambda: {"count": 0, "position_sum": 0.0}),
        0,
        0,
        set(),
    )
    mode = "at" if resume and os.path.exists(phrase_rows_path) else "wt"
    with open_text(phrase_rows_path, mode) as file:
        for split in splits:
            pending_ids = []
            pending_texts = []
            for story_id, text in iter_input_stories(archive_path, tinystories_dir, source_format, split, limit=limit):
                if (split, story_id) in completed:
                    continue
                if progress_every > 0 and story_id % progress_every == 0:
                    print("Processing", split, "story", story_id)
                pending_ids.append(story_id)
                pending_texts.append(text)
                if len(pending_texts) == batch_size:
                    phrase_row_count, phrase_count = _consume_streaming_batch(
                        nlp, split, pending_ids, pending_texts, file, stats,
                        phrase_row_count, phrase_count, lowercase, batch_size, n_process,
                    )
                    pending_ids, pending_texts = [], []
            if pending_texts:
                phrase_row_count, phrase_count = _consume_streaming_batch(
                    nlp, split, pending_ids, pending_texts, file, stats,
                    phrase_row_count, phrase_count, lowercase, batch_size, n_process,
                )
    return stats, phrase_row_count, phrase_count


def _consume_streaming_batch(nlp, split, story_ids, texts, file, stats, phrase_row_count, phrase_count, lowercase, batch_size, n_process):
    for story_id, doc in zip(story_ids, nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
        phrases = extract_phrase_occurrences(doc, lowercase=lowercase)
        for phrase in phrases:
            update_vocab_stats(stats, phrase)
        file.write(json.dumps(phrase_row_to_dict(split, story_id, phrases), ensure_ascii=False, sort_keys=True))
        file.write("\n")
        phrase_row_count += 1
        phrase_count += len(phrases)
    return phrase_row_count, phrase_count


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(path, rows):
    with open_text(path, "wt") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def build_records_streaming(phrase_rows_path, vocab, phrase_index_path, samples_path, sample_limit=100):
    phrase_records_count = 0
    sample_count = 0
    token_to_index = {entry["token"]: entry["index"] for entry in vocab}
    with open_text(phrase_index_path, "wt") as records_file, open_text(samples_path, "wt") as samples_file:
        for split, story_id, phrases in iter_phrase_rows_jsonl(phrase_rows_path):
            for record in build_sparse_records(split, story_id, phrases, vocab, token_to_index=token_to_index):
                line = json.dumps(record, ensure_ascii=False, sort_keys=True)
                records_file.write(line)
                records_file.write("\n")
                if sample_count < sample_limit:
                    samples_file.write(line)
                    samples_file.write("\n")
                    sample_count += 1
                phrase_records_count += 1
    return phrase_records_count, sample_count


def build_outputs(nlp, archive_path, out_dir, splits, limit=None, min_count=1, lowercase=True, sample_limit=100, batch_size=64, tinystories_dir=None, source_format="txt", n_process=1, disable_components=None, progress_every=100, streaming=True, compress=True, resume=False):
    os.makedirs(out_dir, exist_ok=True)
    started_at = time.time()

    if streaming:
        phrase_rows_path = existing_jsonl_path(out_dir, "phrase_rows", compress=compress) if resume else jsonl_output_path(out_dir, "phrase_rows", compress=compress)
        stats, phrase_row_count, phrase_count = collect_phrases_streaming(
            nlp,
            archive_path,
            tinystories_dir,
            source_format,
            splits,
            phrase_rows_path,
            limit=limit,
            lowercase=lowercase,
            batch_size=batch_size,
            n_process=n_process,
            progress_every=progress_every,
            resume=resume,
        )
        vocab = build_vocab_from_stats(stats, min_count=min_count)
        phrase_index_path = jsonl_output_path(out_dir, "phrase_index", compress=compress)
        samples_path = jsonl_output_path(out_dir, "samples", compress=compress)
        phrase_record_count, sample_count = build_records_streaming(
            phrase_rows_path,
            vocab,
            phrase_index_path,
            samples_path,
            sample_limit=sample_limit,
        )
        metadata = {
            "archive": archive_path,
            "tinystories_dir": tinystories_dir,
            "source_format": source_format,
            "splits": splits,
            "limit_per_split": limit,
            "min_count": min_count,
            "lowercase": lowercase,
            "vocab_size": len(vocab),
            "phrase_rows": phrase_row_count,
            "phrases": phrase_count,
            "phrase_records": phrase_record_count,
            "sample_records": sample_count,
            "values": "implicit_ones",
            "streaming": True,
            "compressed": compress,
            "resumed": resume,
            "phrase_rows_cache": phrase_rows_path,
            "phrase_index": phrase_index_path,
            "samples": samples_path,
            "batch_size": batch_size,
            "n_process": n_process,
            "disable_components": disable_components or [],
            "progress_every": progress_every,
            "elapsed_seconds": time.time() - started_at,
        }
        write_json(os.path.join(out_dir, "metadata.json"), metadata)
        write_json(os.path.join(out_dir, "vocab.json"), vocab)
        return metadata

    phrase_rows, all_phrases = collect_phrases(
        nlp,
        archive_path,
        tinystories_dir,
        source_format,
        splits,
        limit=limit,
        lowercase=lowercase,
        batch_size=batch_size,
        n_process=n_process,
        progress_every=progress_every,
    )
    vocab = build_vocab(all_phrases, min_count=min_count)
    phrase_records = []
    for split, story_id, phrases in phrase_rows:
        phrase_records.extend(build_sparse_records(split, story_id, phrases, vocab))

    metadata = {
        "archive": archive_path,
        "tinystories_dir": tinystories_dir,
        "source_format": source_format,
        "splits": splits,
        "limit_per_split": limit,
        "min_count": min_count,
        "lowercase": lowercase,
        "vocab_size": len(vocab),
        "phrase_records": len(phrase_records),
        "values": "implicit_ones",
        "streaming": False,
        "compressed": compress,
        "resumed": False,
        "batch_size": batch_size,
        "n_process": n_process,
        "disable_components": disable_components or [],
        "progress_every": progress_every,
        "elapsed_seconds": time.time() - started_at,
    }

    phrase_index_path = jsonl_output_path(out_dir, "phrase_index", compress=compress)
    samples_path = jsonl_output_path(out_dir, "samples", compress=compress)
    write_jsonl(phrase_index_path, phrase_records)
    write_jsonl(samples_path, phrase_records[:sample_limit])
    metadata["phrase_index"] = phrase_index_path
    metadata["samples"] = samples_path
    write_json(os.path.join(out_dir, "metadata.json"), metadata)
    write_json(os.path.join(out_dir, "vocab.json"), vocab)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Build phrase-position vector schema for TinyStories.")
    parser.add_argument("--archive", default="../archive.zip", help="Path to archive.zip containing train.csv and validation.csv.")
    parser.add_argument("--tinystories-dir", default=None, help="Path to a TinyStories dataset directory containing TinyStories-*.txt files.")
    parser.add_argument("--source-format", default="txt", choices=["txt", "v2-txt"], help="TinyStories directory source format.")
    parser.add_argument("--out-dir", default="phrase_vectors_out", help="Output directory for vocab and phrase records.")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy English model to use.")
    parser.add_argument("--disable-components", nargs="*", default=["ner"], help="spaCy pipeline components to disable.")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], choices=["train", "validation"])
    parser.add_argument("--limit", type=int, default=None, help="Maximum stories per split. Omit for all stories.")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum phrase-token count to include in vocab.")
    parser.add_argument("--batch-size", type=int, default=64, help="spaCy pipe batch size.")
    parser.add_argument("--n-process", type=int, default=1, help="spaCy pipe worker processes. Use >1 for CPU parallelism.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N stories per split. Use 0 to disable.")
    parser.add_argument("--sample-limit", type=int, default=100, help="Number of phrase records to copy to samples.jsonl.")
    parser.add_argument("--streaming", dest="streaming", action="store_true", default=True, help="Stream phrase rows through a disk cache to reduce memory use.")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Use older in-memory phrase row collection.")
    parser.add_argument("--compress", dest="compress", action="store_true", default=True, help="Write JSONL outputs as gzip-compressed .jsonl.gz files.")
    parser.add_argument("--no-compress", dest="compress", action="store_false", help="Write plain .jsonl outputs.")
    parser.add_argument("--resume", action="store_true", help="Reuse and append an existing phrase_rows JSONL cache, then rebuild derived outputs.")
    parser.add_argument("--no-lowercase", action="store_true", help="Keep original token casing.")
    return parser.parse_args()


def main():
    args = parse_args()
    nlp = load_spacy_model(args.spacy_model, disable_components=args.disable_components)
    metadata = build_outputs(
        nlp=nlp,
        archive_path=args.archive,
        out_dir=args.out_dir,
        splits=args.splits,
        limit=args.limit,
        min_count=args.min_count,
        lowercase=not args.no_lowercase,
        sample_limit=args.sample_limit,
        batch_size=args.batch_size,
        tinystories_dir=args.tinystories_dir,
        source_format=args.source_format,
        n_process=args.n_process,
        disable_components=args.disable_components,
        progress_every=args.progress_every,
        streaming=args.streaming,
        compress=args.compress,
        resume=args.resume,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
