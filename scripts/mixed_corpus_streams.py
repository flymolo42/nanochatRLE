"""
Union streams for the mixed-corpus (PG-19 prose + JS/TS code) ordering
experiment. Each corpus keeps its own tokenizer; token strings are pooled into
one vocabulary so shared symbols (punctuation, digits) merge while words and
identifiers stay disjoint.
"""

import json
from collections import defaultdict
from pathlib import Path

from scripts.code_stream import tokenize_code
from scripts.pg19_stream import tokenize_clauses
from scripts.phrase_vectors import build_vocab_from_stats


def prose_file_streams(paths):
    for path in paths:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        yield path.stem, tokenize_clauses(text)


def code_file_streams(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for row_index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                content = json.loads(line).get("content")
                if not content:
                    continue
                yield str(row_index), tokenize_code(content, split_identifiers=False)


def union_census(tagged_stream_iters):
    stats = defaultdict(lambda: {"count": 0, "position_sum": 0.0})
    for _domain, stream in tagged_stream_iters:
        clause_tokens = []
        current = None
        for clause, token in stream:
            if current is not None and clause != current:
                _accumulate(stats, clause_tokens)
                clause_tokens = []
            current = clause
            clause_tokens.append(token)
        if clause_tokens:
            _accumulate(stats, clause_tokens)
    return build_vocab_from_stats(stats, min_count=1)


def _accumulate(stats, clause_tokens):
    denominator = max(len(clause_tokens) - 1, 1)
    for position, token in enumerate(clause_tokens):
        entry = stats[token]
        entry["count"] += 1
        entry["position_sum"] += position / denominator
