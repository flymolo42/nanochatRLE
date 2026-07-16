import json
import tempfile
import unittest
from pathlib import Path

from scripts.code_stream import file_streams, is_minified, split_identifier, tokenize_code


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


class RegexLiteralTests(unittest.TestCase):
    def test_regex_after_assignment_collapses_to_str(self):
        tokens = [t for _, t in tokenize_code("const re = /ab+c/;")]
        self.assertEqual(tokens, ["const", "re", "=", "<str>", ";"])

    def test_regex_as_method_argument(self):
        tokens = [t for _, t in tokenize_code("s.replace(/\\s+/g, '')")]
        self.assertEqual(tokens, ["s", ".", "replace", "(", "<str>", ",", "<str>", ")"])

    def test_regex_after_return_keyword(self):
        tokens = [t for _, t in tokenize_code("return /x/;")]
        self.assertEqual(tokens, ["return", "<str>", ";"])

    def test_regex_at_statement_start(self):
        tokens = [t for _, t in tokenize_code("/^x/.test(s)")]
        self.assertEqual(tokens, ["<str>", ".", "test", "(", "s", ")"])

    def test_division_not_treated_as_regex(self):
        tokens = [t for _, t in tokenize_code("a / b / c")]
        self.assertEqual(tokens, ["a", "/", "b", "/", "c"])

    def test_division_after_closing_paren(self):
        tokens = [t for _, t in tokenize_code("f() / 2")]
        self.assertEqual(tokens, ["f", "(", ")", "/", "2"])

    def test_regex_char_class_contains_slash(self):
        tokens = [t for _, t in tokenize_code("x = /[a/b]/;")]
        self.assertEqual(tokens, ["x", "=", "<str>", ";"])

    def test_regex_escaped_slash_in_body(self):
        tokens = [t for _, t in tokenize_code("x = /a\\/b/;")]
        self.assertEqual(tokens, ["x", "=", "<str>", ";"])

    def test_regex_with_flags(self):
        tokens = [t for _, t in tokenize_code("x = /abc/gi;")]
        self.assertEqual(tokens, ["x", "=", "<str>", ";"])

    def test_line_comment_not_mistaken_for_regex(self):
        tokens = [t for _, t in tokenize_code("x = 1 // /not/ regex\ny")]
        self.assertEqual(tokens, ["x", "=", "1", "y"])


class MinificationFilterTests(unittest.TestCase):
    def test_normal_multiline_code_not_minified(self):
        code = "function add(a, b) {\n    return a + b;\n}\n"
        self.assertFalse(is_minified(code))

    def test_single_giant_line_is_minified(self):
        code = "var x = [" + ",".join(str(i) for i in range(2000)) + "];"
        self.assertTrue(is_minified(code))

    def test_one_very_long_line_among_short_is_minified(self):
        code = "const a = 1;\n" + ("x" * 1500) + "\nconst b = 2;\n"
        self.assertTrue(is_minified(code))

    def test_high_average_line_length_is_minified(self):
        # every line ~200 chars -> mean line length > 100 -> minified
        code = "\n".join(["a" * 200] * 20)
        self.assertTrue(is_minified(code))

    def test_empty_or_whitespace_not_minified(self):
        self.assertFalse(is_minified(""))
        self.assertFalse(is_minified("   \n\n  "))


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

    def test_streams_skips_minified_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.json"
            good = "const a = 1;"
            mini = "z=" + "a+" * 600 + "a;"  # one ~1200-char line -> minified
            path.write_text(
                json.dumps({"content": good}) + "\n" +
                json.dumps({"content": mini}) + "\n",
                encoding="utf-8",
            )
            streams = list(file_streams([path]))
        self.assertEqual(len(streams), 1)
        self.assertEqual([t for _, t in streams[0][1]][:3], ["const", "a", "="])


class UnicodeIdentifierTests(unittest.TestCase):
    def test_accented_identifier_stays_whole(self):
        # café must be ONE token, not caf + é (the fragmentation bug)
        tokens = [t for _, t in tokenize_code("const café = 1;", split_identifiers=False)]
        self.assertEqual(tokens, ["const", "café", "=", "1", ";"])

    def test_mixed_ascii_unicode_identifier_whole(self):
        tokens = [t for _, t in tokenize_code("let naïveCount = 2;", split_identifiers=False)]
        self.assertEqual(tokens, ["const" if False else "let", "naïvecount", "=", "2", ";"])

    def test_non_latin_identifier_whole(self):
        tokens = [t for _, t in tokenize_code("var δfoo = 3;", split_identifiers=False)]
        self.assertEqual(tokens, ["var", "δfoo", "=", "3", ";"])

    def test_split_mode_keeps_non_ascii_identifier_whole(self):
        # camelCase splitting is an ASCII convention; non-ASCII identifiers stay whole (lowercased)
        self.assertEqual(split_identifier("café"), ["café"])
        self.assertEqual(split_identifier("größeWert"), ["größewert"])
        # pure-ASCII camelCase still splits
        self.assertEqual(split_identifier("getUserName"), ["get", "user", "name"])

    def test_ascii_digits_still_not_identifier_start(self):
        tokens = [t for _, t in tokenize_code("2fast", split_identifiers=False)]
        self.assertEqual(tokens, ["2", "fast"])

if __name__ == "__main__":
    unittest.main()
