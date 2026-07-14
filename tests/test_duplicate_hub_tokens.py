# tests/test_duplicate_hub_tokens.py
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.duplicate_hub_tokens import duplicate_records, run_transform
from scripts.plan_token_duplicates import renumber_array
from scripts.train_phrase_vectors import iter_records


def _single(story_id, phrase_id, index, token_pos, start=0, split="train", label="punctuation"):
    return {
        "split": split, "story_id": story_id, "phrase_id": phrase_id, "label": label,
        "start": start, "end": start + 5, "record_type": "single",
        "indices": [index], "tokens": [f"tok{index}"], "token_pos": token_pos,
    }


def _plan(vocab_size, parents):
    renumber = renumber_array(vocab_size, parents)
    return {
        "format": "duplicates_plan_v1",
        "vocab_size_old": vocab_size,
        "vocab_size_new": vocab_size + len(parents),
        "num_duplicates": len(parents),
        "parents": [
            {"old_index": p, "token": f"tok{p}", "early_new_index": int(renumber[p]),
             "late_new_index": int(renumber[p]) + 1, "conflict_mass": 1.0, "conflict_share": 0.5}
            for p in sorted(parents)
        ],
    }


_IDENTITY_ILS = list(range(6))  # ils_positions[old] == old: raw-id order matches current order


class DuplicateRecordsTests(unittest.TestCase):
    def test_assignment_by_predecessor_ils_position_identity(self):
        # parent token 2; stream old-indices: 5, 2, 1, 2.
        # With identity ils_positions, current-order position == old index:
        # first 2 has pred 5 (pos 5 > pos 2) => late; second 2 has pred 1 (pos 1 < pos 2) => early
        records = [
            _single(0, 0, 5, token_pos=0),
            _single(0, 0, 2, token_pos=1),
            _single(0, 0, 1, token_pos=2),
            _single(0, 0, 2, token_pos=3),
        ]
        plan = _plan(6, [2])
        out = list(duplicate_records(iter(records), plan, _IDENTITY_ILS))
        # renumber with parent 2: old->new = [0,1,2,4,5,6]; late copy id = 3
        self.assertEqual([r["indices"][0] for r in out], [6, 3, 1, 2])

    def test_assignment_follows_ils_position_not_raw_old_index(self):
        # Same stream (old-indices 5, 2, 1, 2) as above, but ils_positions inverts
        # the current-order relationship between 5 and 2: ils_positions[5]=1 <
        # ils_positions[2]=4, so the predecessor 5 is now EARLIER than 2 in current
        # order -> the first "2" occurrence must become early, not late. This proves
        # the rule compares ils_positions[previous] vs ils_positions[old], not the
        # raw original ids themselves (raw comparison 5>2 would still say late).
        records = [
            _single(0, 0, 5, token_pos=0),
            _single(0, 0, 2, token_pos=1),
            _single(0, 0, 1, token_pos=2),
            _single(0, 0, 2, token_pos=3),
        ]
        plan = _plan(6, [2])
        # original id ->  0  1  2  3  4  5
        ils_positions = [2, 0, 4, 3, 5, 1]
        out = list(duplicate_records(iter(records), plan, ils_positions))
        # both "2" occurrences now resolve to early (parent slot, renumbered id 2):
        # pred 5 -> ils pos 1 < ils pos of 2 (4) => early; pred 1 -> ils pos 0 < 4 => early
        self.assertEqual([r["indices"][0] for r in out], [6, 2, 1, 2])

    def test_first_in_story_is_early(self):
        records = [_single(0, 0, 2, token_pos=0), _single(0, 0, 4, token_pos=1)]
        plan = _plan(6, [2])
        out = list(duplicate_records(iter(records), plan, _IDENTITY_ILS))
        self.assertEqual(out[0]["indices"], [2])  # early copy keeps parent slot

    def test_all_representations_of_a_position_rewritten_consistently(self):
        records = [
            _single(0, 0, 5, token_pos=0),
            _single(0, 0, 2, token_pos=1),
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "root_clause", "start": 0,
             "end": 2, "record_type": "single", "indices": [2], "tokens": ["tok2"], "token_pos": 1},
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0,
             "end": 2, "record_type": "packed", "indices": [5, 2], "tokens": ["tok5", "tok2"]},
        ]
        plan = _plan(6, [2])
        out = list(duplicate_records(iter(records), plan, _IDENTITY_ILS))
        # canonical: pos0=5, pos1=2 (ils pos 5>2 => late id 3); every representation of pos1 gets 3
        self.assertEqual(out[1]["indices"], [3])
        self.assertEqual(out[2]["indices"], [3])
        self.assertEqual(out[3]["indices"], [6, 3])  # packed keeps positional alignment, no sort
        self.assertEqual(out[3]["tokens"], ["tok5", "tok2"])  # surface tokens unchanged

    def test_parity_resets_per_story_and_stats(self):
        records = [
            _single(0, 0, 5, token_pos=0), _single(0, 0, 2, token_pos=1),
            _single(1, 0, 2, token_pos=0),
        ]
        plan = _plan(6, [2])
        stats = {"stories": 0, "early": 0, "late": 0}
        out = list(duplicate_records(iter(records), plan, _IDENTITY_ILS, stats=stats))
        self.assertEqual(out[2]["indices"], [2])  # first-in-story -> early
        self.assertEqual(stats, {"stories": 2, "early": 1, "late": 1})


class RunTransformTests(unittest.TestCase):
    def test_end_to_end_writes_records_vocab_copy_map(self):
        vocab = [{"token": f"tok{i}", "index": i, "count": 10 + i, "avg_position": 0.5} for i in range(6)]
        records = [
            _single(0, 0, 5, token_pos=0), _single(0, 0, 2, token_pos=1),
            _single(1, 0, 2, token_pos=0),
        ]
        plan = _plan(6, [2])
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
            (tmp / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (tmp / "ils_map.json").write_text(json.dumps(_IDENTITY_ILS), encoding="utf-8")
            with gzip.open(tmp / "records.jsonl.gz", "wt", encoding="utf-8") as file:
                for record in records:
                    file.write(json.dumps(record) + "\n")
            report = run_transform(
                tmp / "records.jsonl.gz", tmp / "vocab.json", tmp / "plan.json",
                tmp / "ils_map.json", tmp / "out",
            )
            new_vocab = json.loads((tmp / "out" / "vocab.json").read_text(encoding="utf-8"))
            copy_map = json.loads((tmp / "out" / "copy_map.json").read_text(encoding="utf-8"))
            new_records = list(iter_records(tmp / "out" / "phrase_index.jsonl.gz"))

        self.assertEqual(len(new_vocab), 7)
        by_index = {row["index"]: row for row in new_vocab}
        self.assertEqual(by_index[2]["token"], "tok2")
        self.assertEqual(by_index[2]["count"], 1)   # one early occurrence
        self.assertEqual(by_index[3]["token"], "tok2~dup")
        self.assertEqual(by_index[3]["count"], 1)   # one late occurrence
        self.assertEqual(by_index[4]["token"], "tok3")  # renumbered non-parent, count unchanged
        self.assertEqual(by_index[4]["count"], 13)
        self.assertEqual(copy_map, {"3": 2})
        self.assertEqual([r["indices"][0] for r in new_records], [6, 3, 2])
        self.assertEqual(report["stories"], 2)
        self.assertEqual(report["late_occurrences"], 1)


if __name__ == "__main__":
    unittest.main()
