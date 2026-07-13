import tempfile
import unittest
from pathlib import Path

import torch

from scripts.extract_sae_bags import (
    chain_bags_from_stream,
    load_bags,
    remap_bags,
    save_bags,
    window_bags_from_stream,
)


class BagExtractionTests(unittest.TestCase):
    def test_chain_bags_cross_clause_ascending(self):
        stream = [(0, 1), (0, 3), (1, 5), (1, 2), (1, 4)]
        # identity map; cross-clause: chains [1,3,5] | [2,4]
        self.assertEqual(chain_bags_from_stream(stream, index_map=None), [[1, 3, 5], [2, 4]])

    def test_chain_bags_apply_index_map(self):
        stream = [(0, 0), (0, 1)]
        # map 0->5, 1->2: mapped stream [5, 2] descends -> two chains
        self.assertEqual(chain_bags_from_stream(stream, index_map=[5, 2]), [[5], [2]])

    def test_window_bags_fixed_length(self):
        stream = [(0, 1), (0, 3), (1, 5), (1, 2), (1, 4)]
        self.assertEqual(window_bags_from_stream(stream, index_map=None, window=4), [[1, 3, 5, 2], [4]])

    def test_remap_bags_dedups_and_sorts(self):
        lookup = torch.tensor([7, 7, 3])
        self.assertEqual(remap_bags([[0, 1, 2]], lookup), [[3, 7]])

    def test_save_load_roundtrip(self):
        bags = [[1, 3, 5], [2], [0, 4]]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bags.npz"
            save_bags(bags, path, input_dim=8)
            loaded, input_dim = load_bags(path)
        self.assertEqual(loaded, bags)
        self.assertEqual(input_dim, 8)


if __name__ == "__main__":
    unittest.main()
