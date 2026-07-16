"""
Lightweight PG-19 tokenizer producing (clause_id, token) streams — the same
canonical-stream shape the TinyStories pipeline consumes, without spaCy.

Conventions match the phrase pipeline where it matters: lowercase; words keep
internal apostrophes; punctuation marks are their own tokens; a clause ends
AFTER each clause-punctuation mark; double quotes alternate open ('"') /
close ('"_close') per book, mirroring scripts/split_quote_token.py.
"""

import re
import unicodedata
from pathlib import Path

# A token is: a Unicode word (letters/digits of any script, keeping internal
# apostrophes) — [^\W_] is a word char that is not underscore, so accented and
# non-Latin letters stay attached; OR a single non-word, non-space char
# (punctuation); OR a lone underscore (PG-19 uses it as an italics marker).
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*|[^\s\w]|_")
CLAUSE_PUNCT = {".", "!", "?", ",", ";", ":", '"', "'_close", '"_close'}

# Fold typographic Unicode punctuation to the ASCII forms the pipeline reasons
# about: curly single quotes -> straight apostrophe (so contractions stay whole),
# curly/low double quotes -> straight double quote (so the open/close alternation
# below handles them uniformly regardless of typography).
_PUNCT_NORMALIZE = str.maketrans({
    "‘": "'", "’": "'",              # ‘ ’ single quotation marks
    "“": '"', "”": '"',              # “ ” double quotation marks
    "„": '"', "‟": '"',              # „ ‟ low / high-reversed double
})


def tokenize_clauses(text):
    # NFC so decomposed accents (e + combining mark) become one letter, then
    # fold typographic punctuation, then lowercase (Unicode-aware).
    text = unicodedata.normalize("NFC", text).translate(_PUNCT_NORMALIZE).lower()
    stream = []
    clause = 0
    quote_open = False
    pending_break = False
    for match in _TOKEN_RE.finditer(text):
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
