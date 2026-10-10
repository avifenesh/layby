"""Regime classification from r and w (regime.py, BENCH_PLAN section 1).

Run: python engine/phase_r/tests/test_regime.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import regime  # noqa: E402

MiB = 1 << 20
PILOT = {"kv_bytes_per_token": 147456,
         "loads": {"heavy": {"prefill_tok_s": 20000.0, "wt_offered_write_Bps": 6.0e8},
                   "light": {"prefill_tok_s": 24000.0, "wt_offered_write_Bps": 2.2e8}}}


class Classify(unittest.TestCase):
    def test_boundaries(self):
        c = regime.classify
        self.assertEqual(c(0.49, 0.1), "slow")
        self.assertEqual(c(0.5, 0.1), "medium")       # r = 0.5 is not slow
        self.assertEqual(c(1.0, 1.0), "medium")       # w = 1 is not slow
        self.assertEqual(c(1.0, 1.01), "slow")        # disk cannot absorb write-through
        self.assertEqual(c(3.0, 1.5), "slow")         # w alone makes it slow even with a fast read
        self.assertEqual(c(2.0, 0.49), "fast")
        self.assertEqual(c(2.0, 0.5), "medium")       # fast needs w < 0.5
        self.assertEqual(c(1.99, 0.1), "medium")      # fast needs r >= 2
        self.assertEqual(c(0.0, float("inf")), "slow")

    def test_ratios(self):
        # 450 MiB/s read of 144 KiB tokens = 3200 tok/s; over 20000 tok/s prefill r = 0.16
        r, w = regime.ratios(450 * MiB, 450 * MiB, 147456, 20000.0, 6.0e8)
        self.assertAlmostEqual(r, 450 * MiB / 147456 / 20000.0)
        self.assertAlmostEqual(r, 0.16, places=2)
        self.assertAlmostEqual(w, 6.0e8 / (450 * MiB))
        self.assertEqual(regime.ratios(1, 0, 1, 1, 1)[1], float("inf"))

    def test_plan_cells(self):
        # slow: NVMe throttled to 450 MiB/s; fast: a RAM tier; medium: an NVMe at about 4 GB/s
        self.assertEqual(regime.cell_regime(PILOT, "heavy", 450 * MiB, 450 * MiB)["regime"], "slow")
        self.assertEqual(regime.cell_regime(PILOT, "heavy", 2.0e10, 8.0e9)["regime"], "fast")
        self.assertEqual(regime.cell_regime(PILOT, "heavy", 4.0e9, 2.5e9)["regime"], "medium")

    def test_regime_depends_on_load(self):
        # the same disk can be slow at heavy load (by w) and medium at light load
        rb, wb = 3.0e9, 5.0e8
        self.assertEqual(regime.cell_regime(PILOT, "heavy", rb, wb)["regime"], "slow")
        self.assertEqual(regime.cell_regime(PILOT, "light", rb, wb)["regime"], "medium")

    def test_bounds_agree_with_classify(self):
        b = regime.bandwidth_bounds(PILOT, "heavy")
        eps = 1.0
        self.assertEqual(regime.cell_regime(PILOT, "heavy", b["slow_read_below"] - eps, 1e12)["regime"], "slow")
        self.assertNotEqual(regime.cell_regime(PILOT, "heavy", b["slow_read_below"] + eps, 1e12)["regime"], "slow")
        self.assertEqual(regime.cell_regime(PILOT, "heavy", 1e12, b["slow_write_below"] - eps)["regime"], "slow")
        self.assertEqual(regime.cell_regime(PILOT, "heavy", b["fast_read_at_least"], b["fast_write_above"] + eps)
                         ["regime"], "fast")
        self.assertEqual(regime.cell_regime(PILOT, "heavy", b["fast_read_at_least"], b["fast_write_above"])
                         ["regime"], "medium")

    def test_check_and_missing_load(self):
        ok, d = regime.check("slow", PILOT, "heavy", 450 * MiB, 450 * MiB)
        self.assertTrue(ok)
        ok, d = regime.check("medium", PILOT, "heavy", 450 * MiB, 450 * MiB)
        self.assertFalse(ok)
        self.assertEqual(d["regime"], "slow")
        with self.assertRaises(ValueError):
            regime.cell_regime(PILOT, "sweep3", 1e9, 1e9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
