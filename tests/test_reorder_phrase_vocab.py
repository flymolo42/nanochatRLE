import gzip
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.reorder_phrase_vocab import (
    PairCounter,
    ascending_pair_count,
    optimize_order,
    perturb_order,
    reorder_vocab_rows,
    run_reorder,
    story_index_streams,
)


def _pair_arrays(vocab_size, pair_counts):
    codes = np.array(sorted(left * vocab_size + right for left, right in pair_counts), dtype=np.int64)
    counts = np.array([pair_counts[(int(code) // vocab_size, int(code) % vocab_size)] for code in codes], dtype=np.int64)
    return codes, counts


def _positions(order):
    positions = np.empty(len(order), dtype=np.int64)
    positions[np.asarray(order)] = np.arange(len(order))
    return positions


def _single(split, story_id, phrase_id, index, token_pos, start=0, label="punctuation"):
    return {
        "split": split,
        "story_id": story_id,
        "phrase_id": phrase_id,
        "label": label,
        "start": start,
        "end": start + 10,
        "record_type": "single",
        "indices": [index],
        "token_pos": token_pos,
    }


class StoryIndexStreamsTests(unittest.TestCase):
    def test_extracts_punctuation_singles_in_position_order(self):
        records = [
            _single("train", 0, 0, index=5, token_pos=1, start=0),
            _single("train", 0, 0, index=3, token_pos=0, start=0),
            _single("train", 0, 0, index=9, token_pos=0, start=0, label="subject"),
            _single("train", 0, 1, index=7, token_pos=0, start=2),
        ]
        streams = list(story_index_streams(iter(records)))
        self.assertEqual(streams, [("train", 0, [(0, 3), (0, 5), (1, 7)])])

    def test_dedups_overlapping_absolute_positions_first_wins(self):
        records = [
            _single("train", 0, 0, index=3, token_pos=0, start=0),
            _single("train", 0, 0, index=4, token_pos=1, start=0),
            _single("train", 0, 1, index=8, token_pos=0, start=1),
        ]
        streams = list(story_index_streams(iter(records)))
        self.assertEqual(streams, [("train", 0, [(0, 3), (0, 4)])])

    def test_yields_one_stream_per_story_in_file_order(self):
        records = [
            _single("train", 0, 0, index=1, token_pos=0),
            _single("train", 1, 0, index=2, token_pos=0),
            _single("validation", 0, 0, index=3, token_pos=0),
        ]
        streams = list(story_index_streams(iter(records)))
        self.assertEqual(
            streams,
            [
                ("train", 0, [(0, 1)]),
                ("train", 1, [(0, 2)]),
                ("validation", 0, [(0, 3)]),
            ],
        )

    def test_skips_records_without_indices_and_legacy_records_pass_through(self):
        records = [
            _single("train", 0, 0, index=1, token_pos=0),
            {"split": "train", "story_id": 0, "phrase_id": 0, "label": "punctuation", "start": 0, "end": 3, "record_type": "single", "indices": [], "token_pos": 1},
            # legacy record without record_type expands to singles
            {"split": "train", "story_id": 0, "phrase_id": 1, "label": "punctuation", "start": 2, "end": 4, "indices": [6, 7], "tokens": ["a", "b"]},
        ]
        streams = list(story_index_streams(iter(records)))
        self.assertEqual(streams, [("train", 0, [(0, 1), (1, 6), (1, 7)])])


class PairCounterTests(unittest.TestCase):
    def _counts_dict(self, counter):
        codes, counts = counter.finalize()
        vocab_size = counter.vocab_size
        return {(int(code) // vocab_size, int(code) % vocab_size): int(count) for code, count in zip(codes, counts)}

    def test_counts_adjacent_pairs_within_clause(self):
        counter = PairCounter(vocab_size=16)
        counter.add_stream([(0, 5), (0, 7), (0, 5), (1, 7), (1, 7), (1, 9)])
        self.assertEqual(self._counts_dict(counter), {(5, 7): 1, (7, 5): 1, (7, 9): 1})

    def test_clause_crossings_counted_when_enabled(self):
        counter = PairCounter(vocab_size=16, include_clause_crossings=True)
        counter.add_stream([(0, 5), (1, 7), (1, 9)])
        self.assertEqual(self._counts_dict(counter), {(5, 7): 1, (7, 9): 1})

    def test_self_pairs_are_skipped(self):
        counter = PairCounter(vocab_size=16)
        counter.add_stream([(0, 5), (0, 5), (0, 5)])
        self.assertEqual(self._counts_dict(counter), {})

    def test_chunked_merging_matches_naive_counting(self):
        rng = random.Random(7)
        vocab_size = 12
        counter = PairCounter(vocab_size=vocab_size, chunk_size=4)
        naive = {}
        for _ in range(50):
            stream = [(pos // 3, rng.randrange(vocab_size)) for pos in range(rng.randrange(1, 12))]
            counter.add_stream(stream)
            for (clause_a, left), (clause_b, right) in zip(stream, stream[1:]):
                if clause_a == clause_b and left != right:
                    naive[(left, right)] = naive.get((left, right), 0) + 1
        self.assertEqual(self._counts_dict(counter), naive)

    def test_finalize_is_idempotent_and_sorted(self):
        counter = PairCounter(vocab_size=8, chunk_size=2)
        counter.add_stream([(0, 1), (0, 2), (0, 3), (0, 1), (0, 2)])
        codes_a, counts_a = counter.finalize()
        codes_b, counts_b = counter.finalize()
        self.assertEqual(codes_a.tolist(), sorted(codes_a.tolist()))
        self.assertEqual(codes_a.tolist(), codes_b.tolist())
        self.assertEqual(counts_a.tolist(), counts_b.tolist())


class AscendingPairCountTests(unittest.TestCase):
    def test_counts_weighted_pairs_where_left_precedes_right(self):
        vocab_size = 4
        codes, counts = _pair_arrays(vocab_size, {(0, 1): 3, (2, 1): 2, (1, 3): 1})
        positions = _positions([0, 1, 2, 3])
        self.assertEqual(ascending_pair_count(codes, counts, positions, vocab_size), 4)
        self.assertEqual(ascending_pair_count(codes, counts, _positions([3, 2, 1, 0]), vocab_size), 2)


class OptimizeOrderTests(unittest.TestCase):
    def _ascending(self, codes, counts, order, vocab_size):
        return ascending_pair_count(codes, counts, _positions(order), vocab_size)

    def test_acyclic_conflicts_reach_optimal(self):
        vocab_size = 4
        codes, counts = _pair_arrays(vocab_size, {(3, 2): 5, (2, 1): 5, (1, 0): 5})
        warm_order = np.arange(vocab_size)
        order, stats = optimize_order(codes, counts, vocab_size, warm_order)
        self.assertEqual(self._ascending(codes, counts, order, vocab_size), 15)
        self.assertEqual(stats["ascending_after"], 15)
        self.assertEqual(stats["total_pairs"], 15)

    def test_long_reversed_chain_reaches_optimal(self):
        vocab_size = 50
        codes, counts = _pair_arrays(vocab_size, {(i + 1, i): 1 for i in range(vocab_size - 1)})
        order, stats = optimize_order(codes, counts, vocab_size, np.arange(vocab_size))
        self.assertEqual(self._ascending(codes, counts, order, vocab_size), vocab_size - 1)

    def test_two_token_cycle_resolved_to_heavier_direction(self):
        vocab_size = 3
        codes, counts = _pair_arrays(vocab_size, {(0, 1): 5, (1, 0): 2, (1, 2): 4})
        order, stats = optimize_order(codes, counts, vocab_size, np.array([2, 1, 0]))
        self.assertEqual(self._ascending(codes, counts, order, vocab_size), 9)

    def test_three_cycle_single_component_needs_local_search(self):
        vocab_size = 3
        codes, counts = _pair_arrays(vocab_size, {(0, 1): 10, (1, 2): 10, (2, 0): 1})
        order, stats = optimize_order(codes, counts, vocab_size, np.array([2, 1, 0]))
        self.assertEqual(self._ascending(codes, counts, order, vocab_size), 20)

    def test_never_worse_than_warm_start_on_random_instances(self):
        rng = random.Random(11)
        for trial in range(5):
            vocab_size = 20
            pair_counts = {}
            for _ in range(60):
                left = rng.randrange(vocab_size)
                right = rng.randrange(vocab_size)
                if left != right:
                    pair_counts[(left, right)] = pair_counts.get((left, right), 0) + rng.randrange(1, 5)
            codes, counts = _pair_arrays(vocab_size, pair_counts)
            warm_order = np.array(rng.sample(range(vocab_size), vocab_size))
            order, stats = optimize_order(codes, counts, vocab_size, warm_order)
            self.assertEqual(sorted(order.tolist()), list(range(vocab_size)))
            self.assertGreaterEqual(
                self._ascending(codes, counts, order, vocab_size),
                self._ascending(codes, counts, warm_order, vocab_size),
            )
            self.assertEqual(stats["ascending_after"], self._ascending(codes, counts, order, vocab_size))

    def test_tokens_without_pairs_keep_relative_warm_order(self):
        vocab_size = 6
        codes, counts = _pair_arrays(vocab_size, {(3, 1): 2})
        warm_order = np.array([5, 0, 3, 4, 1, 2])
        order, _ = optimize_order(codes, counts, vocab_size, warm_order)
        untouched = [token for token in order.tolist() if token in {5, 0, 4, 2}]
        self.assertEqual(untouched, [5, 0, 4, 2])


class ReorderVocabRowsTests(unittest.TestCase):
    def test_reindexes_rows_and_returns_old_to_new_map(self):
        rows = [
            {"token": "a", "index": 0, "count": 3, "avg_position": 0.1},
            {"token": "b", "index": 1, "count": 5, "avg_position": 0.2},
            {"token": "c", "index": 2, "count": 2, "avg_position": 0.9},
        ]
        new_rows, old_to_new = reorder_vocab_rows(rows, np.array([1, 0, 2]))
        self.assertEqual([row["token"] for row in new_rows], ["b", "a", "c"])
        self.assertEqual([row["index"] for row in new_rows], [0, 1, 2])
        self.assertEqual(old_to_new, [1, 0, 2])
        self.assertEqual(new_rows[0]["count"], 5)
        self.assertEqual(new_rows[0]["avg_position"], 0.2)
        # input rows untouched
        self.assertEqual(rows[1]["index"], 1)


class RunReorderTests(unittest.TestCase):
    def test_end_to_end_improves_order_and_writes_outputs(self):
        vocab = [
            {"token": "a", "index": 0, "count": 10, "avg_position": 0.0},
            {"token": "b", "index": 1, "count": 10, "avg_position": 0.5},
            {"token": "c", "index": 2, "count": 10, "avg_position": 1.0},
        ]
        records = []
        for story_id in range(4):
            # canonical stream b -> a -> c per story: warm order gets only (a, c) ascending
            records.append(_single("train", story_id, 0, index=1, token_pos=0))
            records.append(_single("train", story_id, 0, index=0, token_pos=1))
            records.append(_single("train", story_id, 0, index=2, token_pos=2))

        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.json"
            records_path = Path(tmpdir) / "records.jsonl.gz"
            out_dir = Path(tmpdir) / "out"
            vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
            with gzip.open(records_path, "wt", encoding="utf-8") as file:
                for record in records:
                    file.write(json.dumps(record) + "\n")

            report = run_reorder(vocab_path=vocab_path, records_path=records_path, out_dir=out_dir)

            new_vocab = json.loads((out_dir / "vocab.json").read_text(encoding="utf-8"))
            old_to_new = json.loads((out_dir / "old_to_new.json").read_text(encoding="utf-8"))
            saved_report = json.loads((out_dir / "reorder_report.json").read_text(encoding="utf-8"))

        self.assertEqual(saved_report, report)
        self.assertEqual(report["total_pairs"], 8)
        self.assertEqual(report["ascending_before"], 4)
        self.assertEqual(report["ascending_after"], 8)
        self.assertEqual(report["vocab_size"], 3)
        self.assertEqual(report["stories"], 4)

        self.assertEqual(sorted(row["index"] for row in new_vocab), [0, 1, 2])
        token_to_new = {row["token"]: row["index"] for row in new_vocab}
        self.assertLess(token_to_new["b"], token_to_new["a"])
        self.assertLess(token_to_new["a"], token_to_new["c"])
        self.assertEqual(old_to_new, [token_to_new["a"], token_to_new["b"], token_to_new["c"]])
        # counts survive the reindex
        self.assertEqual({row["count"] for row in new_vocab}, {10})


class PerturbOrderTests(unittest.TestCase):
    def test_returns_new_permutation_deterministically(self):
        order = np.arange(200, dtype=np.int64)
        perturbed_a = perturb_order(order, np.random.default_rng(3))
        perturbed_b = perturb_order(order, np.random.default_rng(3))
        self.assertEqual(sorted(perturbed_a.tolist()), list(range(200)))
        self.assertNotEqual(perturbed_a.tolist(), order.tolist())
        self.assertEqual(perturbed_a.tolist(), perturbed_b.tolist())
        # input untouched
        self.assertEqual(order.tolist(), list(range(200)))

    def test_different_seeds_differ(self):
        order = np.arange(200, dtype=np.int64)
        self.assertNotEqual(
            perturb_order(order, np.random.default_rng(1)).tolist(),
            perturb_order(order, np.random.default_rng(2)).tolist(),
        )


def _random_instance(seed, vocab_size=20, num_draws=80):
    rng = random.Random(seed)
    pair_counts = {}
    for _ in range(num_draws):
        left = rng.randrange(vocab_size)
        right = rng.randrange(vocab_size)
        if left != right:
            pair_counts[(left, right)] = pair_counts.get((left, right), 0) + rng.randrange(1, 5)
    return _pair_arrays(vocab_size, pair_counts)


class ILSTests(unittest.TestCase):
    def test_ils_reports_improvement_metrics_and_is_never_worse(self):
        vocab_size = 20
        codes, counts = _random_instance(23, vocab_size)
        warm_order = np.arange(vocab_size)
        base_order, base_stats = optimize_order(codes, counts, vocab_size, warm_order, max_passes=20)
        order, stats = optimize_order(
            codes, counts, vocab_size, warm_order, max_passes=20,
            ils_restarts=3, ils_generations=2, ils_seed=7,
        )
        self.assertIn("ils", stats)
        ils = stats["ils"]
        self.assertEqual(ils["restarts"], 3)
        self.assertEqual(len(ils["generations"]), 2)
        self.assertEqual(ils["base_ascending"], base_stats["ascending_after"])
        self.assertEqual(ils["improvement_pairs"], stats["ascending_after"] - ils["base_ascending"])
        self.assertGreaterEqual(ils["improvement_pairs"], 0)
        positions = np.empty(vocab_size, dtype=np.int64)
        positions[order] = np.arange(vocab_size)
        self.assertEqual(stats["ascending_after"], ascending_pair_count(codes, counts, positions, vocab_size))

    def test_parallel_jobs_match_sequential(self):
        vocab_size = 20
        codes, counts = _random_instance(31, vocab_size)
        warm_order = np.arange(vocab_size)
        kwargs = dict(max_passes=20, ils_restarts=4, ils_generations=2, ils_seed=11)
        order_seq, stats_seq = optimize_order(codes, counts, vocab_size, warm_order, jobs=1, **kwargs)
        order_par, stats_par = optimize_order(codes, counts, vocab_size, warm_order, jobs=2, **kwargs)
        self.assertEqual(order_seq.tolist(), order_par.tolist())
        self.assertEqual(stats_seq["ascending_after"], stats_par["ascending_after"])


class PairCountsPersistenceTests(unittest.TestCase):
    def _write_inputs(self, tmpdir):
        vocab = [
            {"token": "a", "index": 0, "count": 10, "avg_position": 0.0},
            {"token": "b", "index": 1, "count": 10, "avg_position": 0.5},
            {"token": "c", "index": 2, "count": 10, "avg_position": 1.0},
        ]
        records = []
        for story_id in range(4):
            records.append(_single("train", story_id, 0, index=1, token_pos=0))
            records.append(_single("train", story_id, 0, index=0, token_pos=1))
            records.append(_single("train", story_id, 0, index=2, token_pos=2))
        vocab_path = Path(tmpdir) / "vocab.json"
        records_path = Path(tmpdir) / "records.jsonl.gz"
        vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
        with gzip.open(records_path, "wt", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record) + "\n")
        return vocab_path, records_path

    def test_run_reorder_saves_pair_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path, records_path = self._write_inputs(tmpdir)
            out_dir = Path(tmpdir) / "out"
            run_reorder(vocab_path=vocab_path, records_path=records_path, out_dir=out_dir)
            saved = np.load(out_dir / "pair_counts.npz")
        self.assertEqual(int(saved["vocab_size"]), 3)
        self.assertEqual(int(saved["stories"]), 4)
        self.assertEqual(saved["codes"].tolist(), sorted(saved["codes"].tolist()))
        self.assertEqual(int(saved["counts"].sum()), 8)

    def test_run_reorder_from_saved_counts_skips_streaming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path, records_path = self._write_inputs(tmpdir)
            first_out = Path(tmpdir) / "first"
            report_a = run_reorder(vocab_path=vocab_path, records_path=records_path, out_dir=first_out)
            second_out = Path(tmpdir) / "second"
            report_b = run_reorder(
                vocab_path=vocab_path,
                records_path=None,
                out_dir=second_out,
                pair_counts_path=first_out / "pair_counts.npz",
            )
            vocab_a = json.loads((first_out / "vocab.json").read_text(encoding="utf-8"))
            vocab_b = json.loads((second_out / "vocab.json").read_text(encoding="utf-8"))
        self.assertEqual(report_b["ascending_after"], report_a["ascending_after"])
        self.assertEqual(report_b["stories"], 4)
        self.assertEqual(vocab_a, vocab_b)


if __name__ == "__main__":
    unittest.main()
