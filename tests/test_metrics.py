"""Unit tests for metrics and questionnaire scoring.

Runnable via both:
    python -m unittest discover tests
and:
    pytest tests/
"""

import math
import os
import sys
import unittest
from datetime import date

# Ensure parent directory (sleep_mood_app) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from metrics import (
    build_sleep_dataframe,
    compute_sleep_metrics,
    compute_social_jetlag,
    compute_sleep_debt,
    compute_affective_instability_mssd,
    daily_flags,
    fisher_z_interval,
    merge_sleep_mood,
    sleep_regularity_simplified,
    spearman_pair,
)
from instruments import (
    score_phq9,
    score_gad7,
    score_isi,
    score_mdq,
)
import analytics


class TestMetricsAndInstruments(unittest.TestCase):
    # ---------- compute_sleep_metrics ----------

    def test_overnight_sleep(self):
        m = compute_sleep_metrics("2025-01-01", "23:00", "07:00", 20, 10)
        self.assertTrue(m["valid"])
        self.assertAlmostEqual(m["tib_min"], 480.0)
        self.assertAlmostEqual(m["tst_min"], 450.0)
        self.assertAlmostEqual(m["sleep_efficiency_pct"], 93.75)

    def test_after_midnight_bedtime_same_day(self):
        m = compute_sleep_metrics("2025-01-01", "01:00", "08:00", 15, 30)
        self.assertAlmostEqual(m["tib_min"], 420.0)
        self.assertAlmostEqual(m["tst_min"], 375.0)
        self.assertAlmostEqual(m["sleep_efficiency_pct"], 89.2857, places=3)

    def test_missing_fields(self):
        m = compute_sleep_metrics("2025-01-01", None, "07:00", 20, 10)
        self.assertFalse(m["valid"])

    def test_negative_tst_clamped_to_zero(self):
        m = compute_sleep_metrics("2025-01-01", "23:00", "23:30", 60, 30)
        self.assertTrue(m["valid"])
        self.assertEqual(m["tst_min"], 0.0)
        self.assertEqual(m["sleep_efficiency_pct"], 0.0)

    def test_invalid_time_string(self):
        m = compute_sleep_metrics("2025-01-01", "not-a-time", "07:00", 20, 10)
        self.assertFalse(m["valid"])

    # ---------- sleep_regularity_simplified ----------

    def test_regularity_perfect(self):
        entries = [
            {"entry_date": f"2025-01-0{i}", "bedtime": "23:00", "wake_time": "07:00",
             "sleep_onset_latency_min": 10, "waso_min": 10}
            for i in range(1, 6)
        ]
        df = build_sleep_dataframe(entries)
        self.assertAlmostEqual(sleep_regularity_simplified(df), 100.0)

    def test_regularity_insufficient_data(self):
        entries = [{"entry_date": "2025-01-01", "bedtime": "23:00", "wake_time": "07:00",
                    "sleep_onset_latency_min": 10, "waso_min": 10}]
        df = build_sleep_dataframe(entries)
        self.assertIsNone(sleep_regularity_simplified(df))

    # ---------- social jetlag & sleep debt ----------

    def test_social_jetlag_calculation(self):
        # 5 workdays (Sun-Thu nights): Bed 23:00, Wake 07:00 -> mid-sleep = 03:00 (180 min)
        # 2 weekend nights (Fri-Sat nights): Bed 02:00, Wake 10:00 -> mid-sleep = 06:00 (360 min)
        # Expected social jetlag = |6.0 - 3.0| = 3.0 hours
        entries = []
        # Jan 5 2025 was Sunday (weekday sleep night)
        dates_and_times = [
            ("2025-01-05", "23:00", "07:00"),  # Sun
            ("2025-01-06", "23:00", "07:00"),  # Mon
            ("2025-01-07", "23:00", "07:00"),  # Tue
            ("2025-01-08", "23:00", "07:00"),  # Wed
            ("2025-01-09", "23:00", "07:00"),  # Thu
            ("2025-01-10", "02:00", "10:00"),  # Fri (weekend night)
            ("2025-01-11", "02:00", "10:00"),  # Sat (weekend night)
        ]
        for d, b, w in dates_and_times:
            entries.append({
                "entry_date": d, "bedtime": b, "wake_time": w,
                "sleep_onset_latency_min": 15, "waso_min": 15,
            })
        df = build_sleep_dataframe(entries)
        sj = compute_social_jetlag(df)
        self.assertTrue(sj["sufficient_data"])
        self.assertAlmostEqual(sj["social_jetlag_hours"], 3.0, places=1)
        self.assertTrue(sj["flagged"])
        self.assertEqual(sj["workday_midsleep_hhmm"], "03:00 AM")
        self.assertEqual(sj["freeday_midsleep_hhmm"], "06:00 AM")

    def test_format_time_ampm(self):
        from metrics import format_time_ampm
        self.assertEqual(format_time_ampm("23:00"), "11:00 PM")
        self.assertEqual(format_time_ampm("07:00"), "07:00 AM")
        self.assertEqual(format_time_ampm("00:15"), "12:15 AM")
        self.assertEqual(format_time_ampm("12:30"), "12:30 PM")
        self.assertEqual(format_time_ampm(None), "—")

    def test_sleep_debt_accumulation(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "tst_hours": [5.5, 6.0, 7.5],  # Baseline 7.5h -> deficits: 2.0h, 1.5h, 0.0h
        })
        debt_df = compute_sleep_debt(df, baseline_hours=7.5)
        self.assertAlmostEqual(debt_df["nightly_deficit_hours"].iloc[0], 2.0)
        self.assertAlmostEqual(debt_df["nightly_deficit_hours"].iloc[1], 1.5)
        self.assertAlmostEqual(debt_df["nightly_deficit_hours"].iloc[2], 0.0)
        self.assertAlmostEqual(debt_df["cumulative_debt_hours"].iloc[-1], 3.5)

    def test_affective_instability_mssd(self):
        # Constant mood has 0 MSSD
        s_const = pd.Series([5.0, 5.0, 5.0, 5.0])
        self.assertEqual(compute_affective_instability_mssd(s_const), 0.0)

        # Alternating mood 4, 6, 4, 6 -> diffs = 2, -2, 2 -> squared diffs = 4, 4, 4 -> mean = 4.0
        s_oscillating = pd.Series([4.0, 6.0, 4.0, 6.0])
        self.assertAlmostEqual(compute_affective_instability_mssd(s_oscillating), 4.0)

        # Insufficient data
        self.assertIsNone(compute_affective_instability_mssd(pd.Series([5.0, 6.0])))

    # ---------- spearman_pair / fisher_z_interval ----------

    def test_spearman_insufficient_n_has_flag(self):
        n = 5
        df = pd.DataFrame({"x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float)})
        res = spearman_pair(df, "x", "y", min_pairs=7)
        self.assertFalse(res["sufficient"])
        self.assertEqual(res["n"], n)

    def test_spearman_perfect_positive(self):
        n = 10
        df = pd.DataFrame({"x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float)})
        res = spearman_pair(df, "x", "y", min_pairs=7)
        self.assertTrue(res["sufficient"])
        self.assertAlmostEqual(res["rho"], 1.0)
        self.assertIsNone(res["ci"])  # rho = 1 => Fisher-z interval degenerate/omitted

    def test_spearman_missing_column_returns_None(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        self.assertIsNone(spearman_pair(df, "x", "missing"))

    def test_fisher_z_symmetric_and_bracketing(self):
        lo, hi = fisher_z_interval(0.5, 20)
        self.assertTrue(lo < 0.5 < hi)
        lo2, hi2 = fisher_z_interval(-0.5, 20)
        self.assertTrue(lo2 < -0.5 < hi2)
        self.assertIsNone(fisher_z_interval(0.0, 3))

    def test_spearman_on_shared_seasonal_trend_is_perfect(self):
        """Two sinusoids in phase rank-correlate exactly despite neither causing the other."""
        t = np.arange(60)
        df = pd.DataFrame({"x": np.sin(t / 6.0), "y": np.sin(t / 6.0)})
        res = spearman_pair(df, "x", "y")
        self.assertAlmostEqual(res["rho"], 1.0, places=5)

    # ---------- merge / flags ----------

    def test_merge_sleep_mood_outer_keeps_all_dates(self):
        sleep_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]), "tst_hours": [7.0], "valid_sleep": [True]})
        mood_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), "mood_mean": [6.0]})
        merged = merge_sleep_mood(sleep_df, mood_df)
        self.assertEqual(len(merged), 2)
        self.assertTrue(pd.isna(merged.loc[0, "mood_mean"]))
        self.assertTrue(pd.isna(merged.loc[1, "tst_hours"]))
        # valid_sleep must be strictly boolean without NA/NaN values
        self.assertFalse(merged["valid_sleep"].isna().any())
        self.assertFalse(bool(merged.loc[1, "valid_sleep"]))
        self.assertTrue(bool(merged.loc[0, "valid_sleep"]))

    def test_analytics_with_disjoint_sleep_and_mood(self):
        # Patient records mood on day 2 and sleep on day 1
        sleep_df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"]),
            "tst_hours": [7.0],
            "sleep_efficiency_pct": [88.0],
            "sol_min": [15.0],
            "valid_sleep": [True]
        })
        mood_df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-02"]),
            "mood_mean": [6.5],
            "anxiety_mean": [3.0]
        })
        merged = merge_sleep_mood(sleep_df, mood_df)
        # Should not raise ValueError: Cannot mask with non-boolean array containing NA / NaN values
        obs = analytics.all_observations(merged, date(2025, 1, 1), date(2025, 1, 2))
        self.assertIsInstance(obs, list)
        self.assertTrue(len(obs) > 0)
        s_summary = analytics.sleep_summary(merged)
        self.assertTrue(any("Across 1 recorded night" in s for s in s_summary))
        m_summary = analytics.mood_summary(merged)
        self.assertTrue(any("Across 1 day(s) with mood data" in m for m in m_summary))

    def test_daily_flags_trips_low_tst(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"]),
            "tst_hours": [5.0],
            "sleep_efficiency_pct": [90.0],
            "sol_min": [10.0], "waso_min": [10.0],
            "mood_mean": [7.0], "anxiety_mean": [3.0], "energy_mean": [6.0],
        })
        fl = daily_flags(df)
        self.assertTrue(bool(fl.loc[0, "flagged"]))
        self.assertIn("5.0 h", fl.loc[0, "flags"])

    def test_daily_flags_empty_on_none(self):
        self.assertTrue(daily_flags(None).empty)

    # ---------- questionnaire scorers ----------

    def test_score_phq9_bands(self):
        self.assertEqual(score_phq9({i: 0 for i in range(9)})[0], 0)
        band = score_phq9({i: 1 for i in range(9)})[1]  # total 9
        self.assertEqual(band, "Mild")
        band = score_phq9({i: 2 for i in range(9)})[1]  # total 18
        self.assertEqual(band, "Moderately severe")
        band = score_phq9({i: 3 for i in range(9)})[1]  # total 27
        self.assertEqual(band, "Severe")

    def test_score_gad7_bands(self):
        self.assertEqual(score_gad7({i: 2 for i in range(7)})[1], "Moderate")  # 14
        self.assertEqual(score_gad7({i: 3 for i in range(7)})[1], "Severe")    # 21

    def test_score_isi_bands(self):
        self.assertEqual(score_isi({i: 2 for i in range(7)})[1], "Subthreshold insomnia")
        self.assertEqual(score_isi({i: 4 for i in range(7)})[1], "Severe clinical insomnia")

    def test_score_mdq_screen_rules(self):
        yes = {i: 1 for i in range(7)}  # 7 symptoms
        self.assertEqual(score_mdq({**yes, "cooc": 1, "problem": "Serious problem"})[1], "Screen positive")
        self.assertEqual(score_mdq({**yes, "cooc": 1, "problem": "Minor problem"})[1], "Screen negative")
        self.assertEqual(score_mdq({**yes, "cooc": 0, "problem": "Serious problem"})[1], "Screen negative")
        few = {i: 1 for i in range(5)}
        self.assertEqual(score_mdq({**few, "cooc": 1, "problem": "Serious problem"})[1], "Screen negative")


if __name__ == "__main__":
    unittest.main()
