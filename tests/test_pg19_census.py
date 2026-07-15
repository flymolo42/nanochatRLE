import json
import tempfile
import unittest
from pathlib import Path

from scripts.pg19_census import run_census


class CensusTests(unittest.TestCase):
    def test_census_counts_coverage_and_vocab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = Path(tmpdir) / "train"
            data.mkdir()
            (data / "1.txt").write_text("a cat. a cat runs.", encoding="utf-8")
            (data / "2.txt").write_text("a dog!", encoding="utf-8")
            out = Path(tmpdir) / "out"
            report = run_census(data, out, min_counts=(1, 2), caps=(3,))
            vocab = json.loads((out / "vocab.json").read_text(encoding="utf-8"))

        self.assertEqual(report["books"], 2)
        self.assertEqual(report["tokens"], 10)  # a cat . a cat runs . a dog !
        self.assertEqual(report["vocab_size_by_min_count"]["1"], 6)  # a cat . runs dog !
        self.assertEqual(report["vocab_size_by_min_count"]["2"], 3)  # a(3) cat(2) .(2)
        # top-3 by count: a(3), cat(2), .(2) -> coverage 7/10
        self.assertAlmostEqual(report["coverage_by_cap"]["3"], 0.7)
        self.assertEqual(len(vocab), 6)
        self.assertEqual([row["index"] for row in vocab], list(range(6)))
        self.assertTrue(all({"token", "index", "count", "avg_position"} <= set(row) for row in vocab))


if __name__ == "__main__":
    unittest.main()
