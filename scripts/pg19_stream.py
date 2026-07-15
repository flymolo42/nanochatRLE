"""
Lightweight PG-19 tokenizer producing (clause_id, token) streams — the same
canonical-stream shape the TinyStories pipeline consumes, without spaCy.

Conventions match the phrase pipeline where it matters: lowercase; words keep
internal apostrophes; punctuation marks are their own tokens; a clause ends
AFTER each clause-punctuation mark; double quotes alternate open ('"') /
close ('"_close') per book, mirroring scripts/split_quote_token.py.
"""

import re
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*|[^\sa-z0-9]")
CLAUSE_PUNCT = {".", "!", "?", ",", ";", ":", '"', "'_close", '"_close'}


def tokenize_clauses(text):
    stream = []
    clause = 0
    quote_open = False
    pending_break = False
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        if token == '"':
            token = '"' if not quote_open else '"_close'
            quote_open = not quote_open
        if pending_break:
            clause += 1
            pending_break = False
        stream.append((clause, token))
        if token in CLAUSE_PUNCT:
            pending_break = True
    return stream


def book_streams(paths):
    for path in paths:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        yield path.stem, tokenize_clauses(text)
