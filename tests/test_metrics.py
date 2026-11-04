"""Unit tests for metrics and questionnaire scoring.

Run with:
    python -m pytest tests/ -q
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from metrics import (
    build_sleep_dataframe,
    compute_sleep_metrics,
    daily_flags,
    fisher_z_interval,
    merge_sleep_mood,
    sleep_regularity_simplified,
    spearman_pair,
)


# ---------- compute_sleep_metrics ----------

def test_overnight_sleep():
    m = compute_sleep_metrics("2025-01-01", "23:00", "07:00", 20, 10)
    assert m["valid"] is True
    assert m["tib_min"] == pytest.approx(480)
    assert m["tst_min"] == pytest.approx(450)
    assert m["sleep_efficiency_pct"] == pytest.approx(93.75)


def test_after_midnight_bedtime_same_day():
    m = compute_sleep_metrics("2025-01-01", "01:00", "08:00", 15, 30)
    assert m["tib_min"] == pytest.approx(420)
    assert m["tst_min"] == pytest.approx(375)
    assert m["sleep_efficiency_pct"] == pytest.approx(89.2857, rel=1e-3)


def test_missing_fields():
    m = compute_sleep_metrics("2025-01-01", None, "07:00", 20, 10)
    assert m["valid"] is False


def test_negative_tst_clamped_to_zero():
    m = compute_sleep_metrics("2025-01-01", "23:00", "23:30", 60, 30)
    assert m["valid"] is True
    assert m["tst_min"] == 0.0
    assert m["sleep_efficiency_pct"] == 0.0


def test_invalid_time_string():
    m = compute_sleep_metrics("2025-01-01", "not-a-time", "07:00", 20, 10)
    assert m["valid"] is False


# ---------- sleep_regularity_simplified ----------

def test_regularity_perfect():
    entries = [
        {"entry_date": f"2025-01-0{i}", "bedtime": "23:00", "wake_time": "07:00",
         "sleep_onset_latency_min": 10, "waso_min": 10}
        for i in range(1, 6)
    ]
    df = build_sleep_dataframe(entries)
    assert sleep_regularity_simplified(df) == pytest.approx(100.0)


def test_regularity_insufficient_data():
    entries = [{"entry_date": "2025-01-01", "bedtime": "23:00", "wake_time": "07:00",
                "sleep_onset_latency_min": 10, "waso_min": 10}]
    df = build_sleep_dataframe(entries)
    assert sleep_regularity_simplified(df) is None


# ---------- spearman_pair / fisher_z_interval ----------

def test_spearman_insufficient_n_has_flag():
    n = 5
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float)})
    res = spearman_pair(df, "x", "y", min_pairs=7)
    assert res["sufficient"] is False
    assert res["n"] == n


def test_spearman_perfect_positive():
    n = 10
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float)})
    res = spearman_pair(df, "x", "y", min_pairs=7)
    assert res["sufficient"] is True
    assert res["rho"] == pytest.approx(1.0)
    assert res["ci"] is None  # rho = 1 ⇒ Fisher-z interval degenerate/omitted


def test_spearman_missing_column_returns_None():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert spearman_pair(df, "x", "missing") is None


def test_fisher_z_symmetric_and_bracketing():
    lo, hi = fisher_z_interval(0.5, 20)
    assert lo < 0.5 < hi
    lo2, hi2 = fisher_z_interval(-0.5, 20)
    assert lo2 < -0.5 < hi2
    assert fisher_z_interval(0.0, 3) is None


# ---------- seasonal trend sanity (shared-trend correlation pitfall) ----------

def test_spearman_on_shared_seasonal_trend_is_perfect():
    """Two sinusoids in phase rank-correlate exactly despite neither causing the other."""
    t = np.arange(60)
    df = pd.DataFrame({"x": np.sin(t / 6), "y": np.sin(t / 6)})
    res = spearman_pair(df, "x", "y")
    assert res["rho"] == pytest.approx(1.0, abs=1e-6)


# ---------- merge / flags ----------

def test_merge_sleep_mood_outer_keeps_all_dates():
    sleep_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]), "tst_hours": [7.0], "valid_sleep": [True]})
    mood_df = pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), "mood_mean": [6.0]})
    merged = merge_sleep_mood(sleep_df, mood_df)
    assert len(merged) == 2
    assert pd.isna(merged.loc[0, "mood_mean"])
    assert pd.isna(merged.loc[1, "tst_hours"])


def test_daily_flags_trips_low_tst():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01"]),
        "tst_hours": [5.0],
        "sleep_efficiency_pct": [90.0],
        "sol_min": [10.0], "waso_min": [10.0],
        "mood_mean": [7.0], "anxiety_mean": [3.0], "energy_mean": [6.0],
    })
    fl = daily_flags(df)
    assert bool(fl.loc[0, "flagged"]) is True
    assert "5.0 h" in fl.loc[0, "flags"]


def test_daily_flags_empty_on_none():
    assert daily_flags(None).empty


# ---------- questionnaire scorers ----------

def test_score_phq9_bands():
    from instruments import score_phq9
    assert score_phq9({i: 0 for i in range(9)})[0] == 0
    band = score_phq9({i: 1 for i in range(9)})[1]  # total 9
    assert band == "Mild"
    band = score_phq9({i: 2 for i in range(9)})[1]  # total 18
    assert band == "Moderately severe"
    band = score_phq9({i: 3 for i in range(9)})[1]  # total 27
    assert band == "Severe"


def test_score_gad7_bands():
    from instruments import score_gad7
    assert score_gad7({i: 2 for i in range(7)})[1] == "Moderate"  # 14
    assert score_gad7({i: 3 for i in range(7)})[1] == "Severe"    # 21


def test_score_isi_bands():
    from instruments import score_isi
    # 7 items x 2 = 14 ⇒ subthreshold
    assert score_isi({i: 2 for i in range(7)})[1] == "Subthreshold insomnia"
    # 7 items x 4 = 28 ⇒ severe
    assert score_isi({i: 4 for i in range(7)})[1] == "Severe clinical insomnia"


def test_score_mdq_screen_rules():
    from instruments import score_mdq
    yes = {i: 1 for i in range(7)}          # 7 symptoms
    assert score_mdq({**yes, "cooc": 1, "problem": "Serious problem"})[1] == "Screen positive"
    assert score_mdq({**yes, "cooc": 1, "problem": "Minor problem"})[1] == "Screen negative"
    assert score_mdq({**yes, "cooc": 0, "problem": "Serious problem"})[1] == "Screen negative"
    few = {i: 1 for i in range(5)}
    assert score_mdq({**few, "cooc": 1, "problem": "Serious problem"})[1] == "Screen negative"
