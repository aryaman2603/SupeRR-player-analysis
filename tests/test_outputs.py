"""Smoke tests for the generated pressure-performance outputs.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bat = pd.read_csv(OUT / "top10_batters.csv")
        cls.bowl = pd.read_csv(OUT / "top10_bowlers.csv")
        cls.balls = pd.read_parquet(
            OUT / "pressure_balls.parquet",
            columns=["match_id", "wp_usable", "pressure_index", "is_super_over"],
        )
        cls.val = json.loads((OUT / "validation.json").read_text())
        cls.wp = json.loads((OUT / "wp_metrics.json").read_text())

    def test_top10_shapes(self):
        self.assertEqual(len(self.bat), 10)
        self.assertEqual(len(self.bowl), 10)
        self.assertEqual(self.bat["rank"].tolist(), list(range(1, 11)))
        self.assertEqual(self.bowl["rank"].tolist(), list(range(1, 11)))
        self.assertEqual(self.bat["player_name"].nunique(), 10)
        self.assertEqual(self.bowl["player_name"].nunique(), 10)

    def test_index_bounds(self):
        for df in (self.bat, self.bowl):
            self.assertTrue(df["score_index"].between(0, 100).all())
            self.assertTrue((df["rank_lo"] <= df["rank"]).all())
            self.assertTrue((df["rank_hi"] >= df["rank"]).all())

    def test_pressure_index_bounds(self):
        pi = self.balls.loc[self.balls["wp_usable"], "pressure_index"].dropna()
        self.assertTrue(pi.between(0, 100).all())
        self.assertGreater((pi >= 40).sum(), 30_000)
        usable = self.balls[self.balls["wp_usable"].astype(bool)]
        self.assertFalse(usable["is_super_over"].any())

    def test_wp_beats_baseline(self):
        for key in ("innings_1", "innings_2"):
            self.assertLess(self.wp[key]["brier"], self.wp[key]["baseline_brier"])

    def test_validation_fields(self):
        self.assertIn("split_half", self.val)
        self.assertGreater(self.val["split_half"]["batter"]["spearman"], 0)
        self.assertGreater(self.val["split_half"]["bowler"]["spearman"], 0)


if __name__ == "__main__":
    unittest.main()
