import unittest

from scripts.pg19_order_experiment import build_token_lookup, sample_books, stream_to_ids


class LookupTests(unittest.TestCase):
    def test_build_token_lookup_keeps_min_count_and_appends_unk(self):
        vocab = [
            {"token": "a", "index": 0, "count": 30, "avg_position": 0.1},
            {"token": "rare", "index": 1, "count": 2, "avg_position": 0.5},
            {"token": "b", "index": 2, "count": 25, "avg_position": 0.9},
        ]
        lookup, kept = build_token_lookup(vocab, min_count=25)
        # kept preserves census (avg-position) order; unk appended last
        self.assertEqual(kept, ["a", "b", "<unk>"])
        self.assertEqual(lookup["a"], 0)
        self.assertEqual(lookup["b"], 1)
        self.assertNotIn("rare", lookup)

    def test_stream_to_ids_maps_unknown_to_unk(self):
        vocab = [{"token": "a", "index": 0, "count": 30, "avg_position": 0.1}]
        lookup, kept = build_token_lookup(vocab, min_count=25)
        unk = len(kept) - 1
        stream = [(0, "a"), (0, "mystery"), (1, "a")]
        self.assertEqual(stream_to_ids(stream, lookup, unk), [(0, 0), (0, unk), (1, 0)])


class SamplingTests(unittest.TestCase):
    def test_sample_books_is_deterministic_fraction(self):
        paths = [f"book_{i}.txt" for i in range(100)]
        sample_a = sample_books(paths, fraction=0.1, seed=7)
        sample_b = sample_books(paths, fraction=0.1, seed=7)
        self.assertEqual(sample_a, sample_b)
        self.assertEqual(len(sample_a), 10)
        self.assertTrue(set(sample_a) <= set(paths))
        different = sample_books(paths, fraction=0.1, seed=8)
        self.assertNotEqual(sample_a, different)


if __name__ == "__main__":
    unittest.main()
