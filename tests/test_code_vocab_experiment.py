import json
import tempfile
import unittest
from pathlib import Path

from scripts.code_vocab_experiment import census_pass, run_variant


def _write_jsonl(path, contents):
    with open(path, "w", encoding="utf-8") as handle:
        for content in contents:
            handle.write(json.dumps({"content": content}) + "\n")


class CensusPassTests(unittest.TestCase):
    def test_split_yields_smaller_vocab_than_whole(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "d.json"
            # heavy identifier reuse: 5 compounds built from 4 shared word-pieces
            _write_jsonl(path, ["let x = fooBar + fooBaz + fooQux + barBaz + barQux;"])
            whole = census_pass([path], split_identifiers=False)
            split = census_pass([path], split_identifiers=True)
        # whole keeps 5 distinct compound names; split shares foo/bar/baz/qux
        self.assertGreater(whole["vocab_size"], split["vocab_size"])
        self.assertGreater(whole["tokens"], 0)
        self.assertIn("foo", split["counts"])  # shared piece present in split census


class RunVariantTests(unittest.TestCase):
    def test_variant_produces_scc_and_order_and_eval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train = Path(tmpdir) / "train.json"
            val = Path(tmpdir) / "val.json"
            body = ["const a = b + c;\nreturn a;"] * 40
            _write_jsonl(train, body)
            _write_jsonl(val, body[:5])
            out = Path(tmpdir) / "out"
            report = run_variant("whole", [train], [val], out, split_identifiers=False,
                                 min_count=1, sample_fraction=1.0, max_passes=10,
                                 ils_restarts=2, ils_generations=1, jobs=1, max_chain_len=9)
            self.assertIn("largest_scc", report["scc"])
            self.assertIn("ascending_fraction", report["order"])
            self.assertGreater(report["validation_chains"]["chains"], 0)
            self.assertTrue((out / "old_to_new_whole.json").exists())


if __name__ == "__main__":
    unittest.main()
