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
  | (?P<ident>(?:[^\W\d]|[$])[\w$]*)
  | (?P<newline>\n)
  | (?P<op>===|!==|==|!=|<=|>=|=>|&&|\|\||\?\?|\+\+|--|\.\.\.|[+\-*/%=<>!&|^~?:.,;(){}\[\]@])
  | (?P<ws>[^\S\n]+)
  | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)

# A regex literal /body/flags on a single line. The body allows escapes (\\.),
# character classes [...] (which may contain an unescaped '/'), and any other
# char that is not a slash, backslash, or newline. Matched only in value-
# expecting position (see below) so it never eats a division operator.
_REGEX_RE = re.compile(r"/(?![*/])(?:\\.|\[(?:\\.|[^\]\\\n])*\]|[^/\\\n])+/[A-Za-z]*")

# A leading '/' begins a regex literal (not division) only in "value-expecting"
# position: at the start of input, right after an operator/opener/comma/etc., or
# right after one of these keywords. If the previous token is a value — a
# non-keyword identifier, a number, a string/regex placeholder, or a closing
# ')' / ']' — then '/' is the division operator.
_REGEX_PRECEDING_KEYWORDS = {
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void",
    "throw", "do", "else", "yield", "await", "case",
}
_VALUE_END_TOKENS = {")", "]"}

_CLAUSE_CLOSERS = {";", "{", "}"}
# a newline closes a statement only after a value-like end (ASI proxy): a name,
# number, string, or closing bracket — never after an operator or open bracket
_NO_BREAK_AFTER = set("+-*/%=<>!&|^~?:.,") | {
    "===", "!==", "==", "!=", "<=", ">=", "=>", "&&", "||", "??", "++", "--", "...",
    "(", "[", "{",
}


def split_identifier(identifier):
    # camelCase/snake splitting is an ASCII convention; keep non-ASCII identifiers
    # whole (lowercased) rather than fragmenting them on ASCII-only case rules
    if not identifier.isascii():
        return [identifier.lower()]
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
    prev_is_value = False  # was the previous token a value? (controls regex vs division)

    def emit(token):
        nonlocal clause, pending_break
        if pending_break:
            clause += 1
            pending_break = False
        stream.append((clause, token))

    pos, n = 0, len(text)
    while pos < n:
        match = _MASTER_RE.match(text, pos)  # <other>. makes this always match
        kind = match.lastgroup
        value = match.group()
        pos = match.end()

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
            prev, prev_is_value = "<str>", True
            continue
        if kind == "op" and value == "/" and not prev_is_value:
            # value-expecting position: a '/' here starts a regex literal
            regex = _REGEX_RE.match(text, match.start())
            if regex is not None:
                emit("<str>")
                prev, prev_is_value = "<str>", True
                pos = regex.end()
                continue
            # otherwise fall through and treat '/' as the division operator
        if kind == "ident":
            if split_identifiers:
                pieces = split_identifier(value)
                for piece in pieces:
                    emit(piece)
                prev = pieces[-1]  # a value-like token: a following newline may ASI-break
            else:
                prev = value.lower()
                emit(prev)
            prev_is_value = prev not in _REGEX_PRECEDING_KEYWORDS
            continue
        # op / other / number
        emit(value)
        prev = value
        if value in _CLAUSE_CLOSERS:
            pending_break = True
        prev_is_value = kind == "number" or value in _VALUE_END_TOKENS

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
