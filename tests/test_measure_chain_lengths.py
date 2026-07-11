import unittest

from scripts.measure_chain_lengths import chain_length_histogram


class ChainLengthHistogramTests(unittest.TestCase):
    def test_reset_on_clause_breaks_at_boundaries(self):
        stream = [(0, 1), (0, 3), (1, 5), (1, 2), (1, 4)]
        hist = chain_length_histogram(stream, mapping=None, reset_on_clause=True)
        # [1,3] | [5] | [2,4]
        self.assertEqual(hist, {2: 2, 1: 1})

    def test_cross_clause_chains_continue_when_ascending(self):
        stream = [(0, 1), (0, 3), (1, 5), (1, 2), (1, 4)]
        hist = chain_length_histogram(stream, mapping=None, reset_on_clause=False)
        # [1,3,5] | [2,4]
        self.assertEqual(hist, {3: 1, 2: 1})

    def test_mapping_applied_and_equal_breaks(self):
        stream = [(0, 0), (0, 1), (0, 2)]
        mapping = [7, 7, 9]  # 0->7, 1->7 (equal breaks), 2->9
        hist = chain_length_histogram(stream, mapping=mapping, reset_on_clause=True)
        # [7] | [7,9]
        self.assertEqual(hist, {1: 1, 2: 1})

    def test_empty_stream(self):
        self.assertEqual(chain_length_histogram([], mapping=None, reset_on_clause=True), {})


if __name__ == "__main__":
    unittest.main()
