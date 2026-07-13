import json
import tempfile
import unittest
from pathlib import Path

from tests.test_sae import _planted_bags
from scripts.train_sae_grid import run_grid


class GridTests(unittest.TestCase):
    def test_grid_reports_and_gates(self):
        train = _planted_bags(1500)
        val = _planted_bags(300, seed=11)
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_grid(
                train, val, input_dim=64,
                grid=[(32, 2), (32, 8)],
                epochs=15, batch_size=256, device="cpu", out_dir=tmpdir,
                gate_f1=0.9, gate_max_len=6, seed=0,
            )
            saved = json.loads((Path(tmpdir) / "grid_report.json").read_text())
            self.assertTrue((Path(tmpdir) / "sae_best.pt").exists())
        self.assertEqual(saved["best"], report["best"])
        self.assertEqual(len(report["configs"]), 2)
        for config in report["configs"]:
            self.assertIn("f1", config)
            self.assertIn("gate_f1", config)
        # planted concepts (len 4 <= gate_max_len) are learnable -> gate passes
        self.assertTrue(report["gate_passed"])


if __name__ == "__main__":
    unittest.main()
