import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.split_quote_token import split_quote_records, run_split
from scripts.train_phrase_vectors import iter_records

QUOTE = 10
CLOSE = 99


def _single(story_id, phrase_id, index, token, token_pos, start=0, split="train", label="punctuation"):
    return {
        "split": split,
        "story_id": story_id,
        "phrase_id": phrase_id,
        "label": label,
        "start": start,
        "end": start + 5,
        "record_type": "single",
        "indices": [index],
        "tokens": [token],
        "token_pos": token_pos,
    }


def _packed(story_id, phrase_id, indices, tokens, start=0, split="train", label="punctuation"):
    return {
        "split": split,
        "story_id": story_id,
        "phrase_id": phrase_id,
        "label": label,
        "start": start,
        "end": start + len(indices),
        "record_type": "packed",
        "indices": list(indices),
        "tokens": list(tokens),
    }


class SplitQuoteRecordsTests(unittest.TestCase):
    def test_alternates_open_close_by_absolute_position(self):
        records = [
            _single(0, 0, QUOTE, '"', token_pos=0, start=0),
            _single(0, 0, 5, "hi", token_pos=1, start=0),
            _single(0, 0, QUOTE, '"', token_pos=2, start=0),
        ]
        out = list(split_quote_records(iter(records), quote_index=QUOTE, close_index=CLOSE, close_token='"_close'))
        self.assertEqual(out[0]["indices"], [QUOTE])
        self.assertEqual(out[0]["tokens"], ['"'])
        self.assertEqual(out[1]["indices"], [5])
        self.assertEqual(out[2]["indices"], [CLOSE])
        self.assertEqual(out[2]["tokens"], ['"_close'])

    def test_same_position_rewritten_consistently_across_records(self):
        # quote at pos 0 appears as a single AND inside a packed record; quote at pos 2 likewise
        records = [
            _single(0, 0, QUOTE, '"', token_pos=0, start=0),
            _packed(0, 1, [QUOTE, 5, QUOTE], ['"', "hi", '"'], start=0),
            _single(0, 2, QUOTE, '"', token_pos=0, start=2),
        ]
        out = list(split_quote_records(iter(records), quote_index=QUOTE, close_index=CLOSE, close_token='"_close'))
        self.assertEqual(out[0]["indices"], [QUOTE])
        self.assertEqual(out[1]["indices"], [QUOTE, 5, CLOSE])
        self.assertEqual(out[1]["tokens"], ['"', "hi", '"_close'])
        self.assertEqual(out[2]["indices"], [CLOSE])

    def test_parity_resets_per_story(self):
        records = [
            _single(0, 0, QUOTE, '"', token_pos=0, start=0),
            _single(1, 0, QUOTE, '"', token_pos=0, start=0),
        ]
        out = list(split_quote_records(iter(records), quote_index=QUOTE, close_index=CLOSE, close_token='"_close'))
        self.assertEqual(out[0]["indices"], [QUOTE])
        self.assertEqual(out[1]["indices"], [QUOTE])

    def test_odd_quotes_keep_alternating(self):
        records = [
            _single(0, 0, QUOTE, '"', token_pos=0, start=0),
            _single(0, 1, QUOTE, '"', token_pos=0, start=1),
            _single(0, 2, QUOTE, '"', token_pos=0, start=2),
        ]
        out = list(split_quote_records(iter(records), quote_index=QUOTE, close_index=CLOSE, close_token='"_close'))
        self.assertEqual([record["indices"][0] for record in out], [QUOTE, CLOSE, QUOTE])

    def test_non_quote_records_pass_through_unchanged(self):
        record = _single(0, 0, 5, "hi", token_pos=0)
        out = list(split_quote_records(iter([record]), quote_index=QUOTE, close_index=CLOSE, close_token='"_close'))
        self.assertEqual(out[0], record)


class RunSplitTests(unittest.TestCase):
    def test_end_to_end_writes_records_and_extended_vocab(self):
        vocab = [
            {"token": '"', "index": 0, "count": 4, "avg_position": 0.5},
            {"token": "hi", "index": 1, "count": 2, "avg_position": 0.5},
        ]
        records = [
            _single(0, 0, 0, '"', token_pos=0, start=0),
            _single(0, 0, 1, "hi", token_pos=1, start=0),
            _single(0, 0, 0, '"', token_pos=2, start=0),
            _single(1, 0, 0, '"', token_pos=0, start=0),
            _single(1, 0, 0, '"', token_pos=1, start=0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            records_path = Path(tmpdir) / "records.jsonl.gz"
            out_dir = Path(tmpdir) / "out"
            vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
            with gzip.open(records_path, "wt", encoding="utf-8") as file:
                for record in records:
                    file.write(json.dumps(record) + "\n")

            report = run_split(vocab_path=vocab_path, records_path=records_path, out_dir=out_dir)

            new_vocab = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))
            new_records = list(iter_records(out_dir / "phrase_index.jsonl.gz"))

        self.assertEqual(len(new_vocab), 3)
        close_row = new_vocab[2]
        self.assertEqual(close_row["token"], '"_close')
        self.assertEqual(close_row["index"], 2)
        self.assertEqual(close_row["count"], 2)
        open_row = new_vocab[0]
        self.assertEqual(open_row["count"], 2)
        self.assertEqual([record["indices"][0] for record in new_records], [0, 1, 2, 0, 2])
        self.assertEqual(report["quote_occurrences"], 4)
        self.assertEqual(report["close_occurrences"], 2)
        self.assertEqual(report["stories"], 2)
        self.assertEqual(report["vocab_size"], 3)


if __name__ == "__main__":
    unittest.main()
