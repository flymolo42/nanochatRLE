import json
import tempfile
import unittest
from pathlib import Path

from scripts.code_stream import file_streams, split_identifier, tokenize_code


class SplitIdentifierTests(unittest.TestCase):
    def test_camel_case(self):
        self.assertEqual(split_identifier("getUserName"), ["get", "user", "name"])

    def test_snake_and_screaming(self):
        self.assertEqual(split_identifier("MAX_LINE_LEN"), ["max", "line", "len"])
        self.assertEqual(split_identifier("my_var2"), ["my", "var2"])

    def test_leading_caps_and_acronym(self):
        self.assertEqual(split_identifier("HTMLParser"), ["html", "parser"])

    def test_plain_word_unchanged(self):
        self.assertEqual(split_identifier("value"), ["value"])


class TokenizeCodeTests(unittest.TestCase):
    def test_whole_identifiers_and_statement_clauses(self):
        stream = tokenize_code("const x = 1;\nreturn x;", split_identifiers=False)
        # two statements -> two clauses; ';' and newline both close
        self.assertEqual(stream, [
            (0, "const"), (0, "x"), (0, "="), (0, "1"), (0, ";"),
            (1, "return"), (1, "x"), (1, ";"),
        ])

    def test_split_identifiers_expands_names(self):
        stream = tokenize_code("let userName = 2;", split_identifiers=True)
        tokens = [t for _, t in stream]
        self.assertEqual(tokens, ["let", "user", "name", "=", "2", ";"])

    def test_braces_and_operators_are_tokens(self):
        stream = tokenize_code("if (a) { b }", split_identifiers=False)
        tokens = [t for _, t in stream]
        self.assertEqual(tokens, ["if", "(", "a", ")", "{", "b", "}"])

    def test_multichar_operators_stay_together(self):
        tokens = [t for _, t in tokenize_code("a === b => c", split_identifiers=False)]
        self.assertEqual(tokens, ["a", "===", "b", "=>", "c"])

    def test_string_literal_collapses_to_placeholder(self):
        tokens = [t for _, t in tokenize_code('x = "hello world";', split_identifiers=False)]
        self.assertEqual(tokens, ["x", "=", "<str>", ";"])

    def test_line_comment_dropped(self):
        tokens = [t for _, t in tokenize_code("a // trailing\nb", split_identifiers=False)]
        self.assertEqual(tokens, ["a", "b"])


class FileStreamsTests(unittest.TestCase):
    def test_streams_content_field_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            path.write_text(
                json.dumps({"content": "const a = 1;", "path": "f.js"}) + "\n" +
                json.dumps({"content": "let b = 2;", "path": "g.js"}) + "\n",
                encoding="utf-8",
            )
            streams = list(file_streams([path], split_identifiers=False))
        self.assertEqual(len(streams), 2)
        self.assertEqual(streams[0][0], "0")
        self.assertEqual([t for _, t in streams[0][1]], ["const", "a", "=", "1", ";"])


if __name__ == "__main__":
    unittest.main()
