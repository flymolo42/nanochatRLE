import json
import tempfile
import unittest
from pathlib import Path

from scripts.mixed_corpus_streams import code_file_streams, prose_file_streams, union_census


class FileStreamTests(unittest.TestCase):
    def test_prose_streams_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "b.txt").write_text("A dog runs.", encoding="utf-8")
            streams = list(prose_file_streams([Path(tmp) / "b.txt"]))
        self.assertEqual(streams[0][0], "b")
        self.assertEqual([t for _, t in streams[0][1]], ["a", "dog", "runs", "."])

    def test_code_streams_jsonl_whole_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.json"
            p.write_text(json.dumps({"content": "let userName = 1;"}) + "\n", encoding="utf-8")
            streams = list(code_file_streams([p]))
        self.assertEqual([t for _, t in streams[0][1]], ["let", "username", "=", "1", ";"])

    def test_code_streams_skips_minified(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.json"
            good = "let a = 1;"
            mini = "z=" + "a+" * 600 + "a;"  # one ~1200-char line -> minified
            p.write_text(
                json.dumps({"content": good}) + "\n" +
                json.dumps({"content": mini}) + "\n",
                encoding="utf-8",
            )
            streams = list(code_file_streams([p]))
        self.assertEqual(len(streams), 1)
        self.assertEqual([t for _, t in streams[0][1]], ["let", "a", "=", "1", ";"])


class UnionCensusTests(unittest.TestCase):
    def test_shared_symbol_merges_domain_specific_disjoint(self):
        tagged = [
            ("prose", [(0, "the"), (0, "dog"), (0, ".")]),
            ("code", [(0, "let"), (0, "x"), (0, ".")]),
        ]
        vocab = union_census(tagged)
        tokens = {row["token"]: row["count"] for row in vocab}
        self.assertEqual(tokens["."], 2)   # shared symbol: one entry, count 2
        self.assertEqual(tokens["the"], 1)
        self.assertEqual(tokens["let"], 1)
        self.assertNotIn("<unk>", tokens)  # census is raw; unk added by _build_lookup


if __name__ == "__main__":
    unittest.main()
