import tempfile
import unittest
from pathlib import Path

from scripts.pg19_stream import book_streams, tokenize_clauses


class TokenizeClausesTests(unittest.TestCase):
    def test_lowercases_splits_words_and_punct_breaks_clauses(self):
        stream = tokenize_clauses('The cat sat. Then it ran!')
        # clause 0: the cat sat .   clause 1: then it ran !
        self.assertEqual(stream, [
            (0, "the"), (0, "cat"), (0, "sat"), (0, "."),
            (1, "then"), (1, "it"), (1, "ran"), (1, "!"),
        ])

    def test_commas_break_clauses_and_apostrophes_stay_in_words(self):
        stream = tokenize_clauses("don't stop, ever")
        self.assertEqual(stream, [(0, "don't"), (0, "stop"), (0, ","), (1, "ever")])

    def test_quote_alternation_open_close(self):
        stream = tokenize_clauses('"hi." she said. "go now."')
        tokens = [token for _, token in stream]
        self.assertEqual(tokens.count('"'), 2)
        self.assertEqual(tokens.count('"_close'), 2)
        self.assertEqual(tokens[0], '"')                     # first quote opens
        close_pos = tokens.index('"_close')
        self.assertGreater(close_pos, tokens.index("hi"))    # closes after hi.

    def test_empty_and_whitespace_only(self):
        self.assertEqual(tokenize_clauses(""), [])
        self.assertEqual(tokenize_clauses("  \n\n  "), [])


class UnicodeProseTests(unittest.TestCase):
    def test_accented_word_stays_whole(self):
        # café must be ONE token, not caf + é (the fragmentation bug)
        self.assertEqual(tokenize_clauses("café au lait"),
                         [(0, "café"), (0, "au"), (0, "lait")])

    def test_accented_word_lowercased_and_whole(self):
        self.assertEqual(tokenize_clauses("Naïve Zürich"),
                         [(0, "naïve"), (0, "zürich")])

    def test_non_latin_word_stays_whole(self):
        self.assertEqual(tokenize_clauses("δοκιμή"), [(0, "δοκιμή")])

    def test_decomposed_accent_normalized_to_nfc(self):
        # "cafe" + combining acute (NFD) collapses to one precomposed token
        self.assertEqual(tokenize_clauses("cafe\u0301"), [(0, "caf\u00e9")])

    def test_curly_apostrophe_normalized_and_kept_in_word(self):
        self.assertEqual(tokenize_clauses("don’t stop"),
                         [(0, "don't"), (0, "stop")])

    def test_curly_double_quotes_drive_open_close(self):
        tokens = [t for _, t in tokenize_clauses("“Hi.”")]
        self.assertEqual(tokens[0], '"')       # curly-open normalized to open
        self.assertIn('"_close', tokens)       # curly-close normalized to close


class BookStreamsTests(unittest.TestCase):
    def test_streams_books_from_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "101.txt").write_text("A dog. A log.", encoding="utf-8")
            (Path(tmpdir) / "202.txt").write_text("Hello there!", encoding="utf-8")
            streams = dict(book_streams(sorted(Path(tmpdir).glob("*.txt"))))
        self.assertEqual(set(streams), {"101", "202"})
        self.assertEqual(streams["202"], [(0, "hello"), (0, "there"), (0, "!")])


if __name__ == "__main__":
    unittest.main()
