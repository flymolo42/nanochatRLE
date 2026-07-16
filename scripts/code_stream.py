"""
JS/TS code tokenizer producing (clause_id, token) streams — the code analog of
scripts/pg19_stream.py, for the vocab-ordering transfer experiment.

Conventions: lowercase; string/template/regex literals collapse to <str>,
numbers to their literal text, comments dropped; multi-char operators kept
whole; brackets/operators are their own tokens; a "clause" is a statement,
closing after ';', '{', '}', and newlines that follow a value-like token
(a coarse ASI proxy). With split_identifiers=True, camelCase / snake_case /
SCREAMING names split into lowercase word pieces (bounded vocab); with False,
identifiers stay whole (raw code, unbounded vocab).
"""

import json
import re
from pathlib import Path

# order matters: comments and strings first so their contents never tokenize
_MASTER_RE = re.compile(
    r"""
    (?P<line_comment>//[^\n]*)
  | (?P<block_comment>/\*.*?\*/)
  | (?P<string>"(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*' | `(?:\\.|[^`\\])*`)
  | (?P<number>\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)
  | (?P<ident>[A-Za-z_$][A-Za-z0-9_$]*)
  | (?P<newline>\n)
  | (?P<op>===|!==|==|!=|<=|>=|=>|&&|\|\||\?\?|\+\+|--|\.\.\.|[+\-*/%=<>!&|^~?:.,;(){}\[\]@])
  | (?P<ws>[^\S\n]+)
  | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)

_CLAUSE_CLOSERS = {";", "{", "}"}
# a newline closes a statement only after a value-like end (ASI proxy): a name,
# number, string, or closing bracket — never after an operator or open bracket
_NO_BREAK_AFTER = set("+-*/%=<>!&|^~?:.,") | {
    "===", "!==", "==", "!=", "<=", ">=", "=>", "&&", "||", "??", "++", "--", "...",
    "(", "[", "{",
}


def split_identifier(identifier):
    lowered_parts = re.split(r"[_$]+", identifier)
    words = []
    for part in lowered_parts:
        if not part:
            continue
        # split camelCase and ACRONYMWord boundaries
        for match in re.finditer(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+|\d+", part):
            words.append(match.group(0).lower())
    return words or [identifier.lower()]


def tokenize_code(text, split_identifiers=False):
    stream = []
    clause = 0
    pending_break = False
    prev = None

    def emit(token):
        nonlocal clause, pending_break
        if pending_break:
            clause += 1
            pending_break = False
        stream.append((clause, token))

    for match in _MASTER_RE.finditer(text):
        kind = match.lastgroup
        value = match.group()
        if kind in ("line_comment", "block_comment", "ws"):
            continue
        if kind == "newline":
            # ASI proxy: a newline ends a statement unless the last token wants
            # a continuation (operator or open bracket)
            if prev is not None and prev not in _NO_BREAK_AFTER:
                pending_break = True
            continue
        if kind == "string":
            emit("<str>")
            prev = "<str>"
            continue
        if kind == "ident" and split_identifiers:
            pieces = split_identifier(value)
            for piece in pieces:
                emit(piece)
            prev = pieces[-1]  # a value-like token: a following newline may ASI-break
            continue
        token = value.lower() if kind == "ident" else value
        emit(token)
        prev = token
        if token in _CLAUSE_CLOSERS:
            pending_break = True

    return stream


def file_streams(paths, split_identifiers=False):
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                content = record.get("content")
                if not content:
                    continue
                yield str(line_number), tokenize_code(content, split_identifiers=split_identifiers)
