"""Generate evidence-based observations and trend summaries.

This module intentionally avoids diagnostic language. It reports what the data
show, with sample sizes and uncertainty, and explicitly separates patient-
reported data from calculated metrics and generated observations.

Association reporting discipline: every Spearman rho is reported with n, p,
and a 95% Fisher-z confidence interval, and always labelled exploratory. When
n is too small, we say so instead of showing an unstable number.
"""

from datetime import date

import numpy as np
import pandas as pd

import config
from metrics import completeness, spearman_pair, sleep_regularity_simplified


def _fmt(value, digits=1):
    if value is None or pd.isna(value):
        return "not available"
    return f"{value:.{digits}f}"


def _ci_text(ci) -> str:
    if not ci:
        return ""
    return f", 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"


def sleep_summary(df: pd.DataFrame) -> list[str]:
    if df.empty or "valid_sleep" not in df or df["valid_sleep"].sum() == 0:
        return ["No valid sleep-diary nights are available for the selected period."]
    sub = df[df["valid_sleep"]]
    n = len(sub)
    mean_tst = sub["tst_hours"].mean()
    mean_se = sub["sleep_efficiency_pct"].mean()
    mean_sol = sub["sol_min"].mean()
    pct_short = (sub["tst_hours"] < config.SLEEP_DURATION_REFERENCE_HOURS).mean() * 100
    pct_low_se = (sub["sleep_efficiency_pct"] < config.SLEEP_EFFICIENCY_REFERENCE_PCT).mean() * 100
    pct_long_sol = (sub["sol_min"] > config.SOL_REFERENCE_MIN).mean() * 100
    out = [
        f"Across {n} recorded night(s), mean diary-derived total sleep time was {_fmt(mean_tst)} h. "
        f"The {config.SLEEP_DURATION_REFERENCE_HOURS:.0f}-h population reference "
        f"(AASM/SRS, 2015) was not met on {pct_short:.0f}% of nights.",
        f"Mean diary-derived sleep efficiency (TST/TIB) was {_fmt(mean_se)}%. "
        f"The {config.SLEEP_EFFICIENCY_REFERENCE_PCT:.0f}% reference was not met on "
        f"{pct_low_se:.0f}% of nights.",
        f"Mean diary sleep-onset latency was {_fmt(mean_sol)} min; latency exceeded "
        f"{config.SOL_REFERENCE_MIN:.0f} min on {pct_long_sol:.0f}% of nights.",
    ]
    if sub["sleep_quality"].notna().any():
        out.append(
            f"Mean patient-rated sleep quality was {_fmt(sub['sleep_quality'].mean())} / 5 "
            f"(patient-reported; n={int(sub['sleep_quality'].notna().sum())})."
        )
    return out


def regularity_summary(df: pd.DataFrame) -> list[str]:
    """Report a simplified day-to-day sleep-timing regularity index (0–100)."""
    sri = sleep_regularity_simplified(df)
    if sri is None:
        return [
            "Sleep-timing regularity needs at least 2 adjacent recorded nights with "
            "bed and wake times; it was not calculated."
        ]
    if sri >= 85:
        label = "relatively regular"
    elif sri >= 70:
        label = "moderately regular"
    else:
        label = "relatively irregular"
    return [
        f"Simplified sleep-timing regularity (how closely bed and wake times repeat "
        f"day to day, 0–100): {sri:.0f} ({label}). Diary-based regularity is a "
        f"simplified proxy for epoch-based indices (e.g., the Sleep Regularity "
        f"Index of Phillips et al., 2017) and does not capture naps or mid-night wake."
    ]


def weekly_summary(df: pd.DataFrame) -> list[str]:
    """Short week-over-week aggregates so the clinician can see drift."""
    if df.empty or "tst_hours" not in df:
        return ["Weekly aggregation needs sleep data."]
    d = df.copy()
    d["week"] = d["date"].dt.strftime("%G-W%V")  # ISO week label
    lines = []
    g = d.groupby("week")
    for week, sub in g:
        n_sleep = int(sub["valid_sleep"].sum()) if "valid_sleep" in sub else 0
        n_mood = int(sub["mood_mean"].notna().sum()) if "mood_mean" in sub else 0
        mean_tst = sub.loc[sub["valid_sleep"], "tst_hours"].mean() if n_sleep else np.nan
        mean_mood = sub.loc[sub["mood_mean"].notna(), "mood_mean"].mean() if n_mood else np.nan
        lines.append(
            f"Week {week}: {n_sleep} sleep night(s) (mean TST {_fmt(mean_tst)} h), "
            f"{n_mood} mood day(s) (mean mood {_fmt(mean_mood)} / 10)."
        )
    return lines


def mood_summary(df: pd.DataFrame) -> list[str]:
    if df.empty or "mood_mean" not in df or df["mood_mean"].notna().sum() == 0:
        return ["No mood entries are available for the selected period."]
    sub = df[df["mood_mean"].notna()]
    n = len(sub)
    mean_mood = sub["mood_mean"].mean()
    sd_mood = sub["mood_mean"].std(ddof=0)
    min_mood = sub["mood_mean"].min()
    max_mood = sub["mood_mean"].max()
    out = [
        f"Across {n} day(s) with mood data, mean daily mood was {_fmt(mean_mood)} / 10 "
        f"(patient-reported), with a between-day standard deviation of {_fmt(sd_mood)}.",
        f"Daily mood ranged from {_fmt(min_mood)} to {_fmt(max_mood)} / 10. "
        f"This describes observed variability; it is not a clinical interpretation.",
    ]
    if "mood_sd" in sub and sub["mood_sd"].notna().any():
        out.append(
            f"Within-day mood variability (standard deviation of multiple same-day entries) "
            f"was available on {int(sub['mood_sd'].notna().sum())} day(s); mean within-day "
            f"SD was {_fmt(sub['mood_sd'].mean())}."
        )
    if "anxiety_mean" in sub and sub["anxiety_mean"].notna().any():
        out.append(f"Mean daily anxiety rating was {_fmt(sub['anxiety_mean'].mean())} / 10 (patient-reported).")
    if "energy_mean" in sub and sub["energy_mean"].notna().any():
        out.append(f"Mean daily energy rating was {_fmt(sub['energy_mean'].mean())} / 10 (patient-reported).")
    return out


def association_summary(df: pd.DataFrame) -> list[str]:
    """Report exploratory Spearman associations with n, p, and 95% CI."""
    if df.empty:
        return ["No paired sleep–mood data are available for association analysis."]
    pairs = [
        ("tst_hours", "mood_mean", "same-day total sleep time and daily mood"),
        ("sleep_efficiency_pct", "mood_mean", "same-day sleep efficiency and daily mood"),
        ("sleep_quality", "mood_mean", "same-day patient-rated sleep quality and daily mood"),
        ("tst_hours", "mood_next_day", "total sleep time and next-day mood"),
        ("sleep_efficiency_pct", "mood_next_day", "sleep efficiency and next-day mood"),
        ("sleep_quality", "mood_next_day", "patient-rated sleep quality and next-day mood"),
        ("mood_mean", "sleep_quality_next_night", "daily mood and the following night's sleep quality"),
    ]
    out = []
    for x, y, label in pairs:
        res = spearman_pair(df, x, y)
        if res is None:
            continue
        if not res["sufficient"]:
            out.append(
                f"{label.capitalize()}: fewer than {config.MIN_PAIRS_FOR_CORRELATION} paired "
                f"observations (n={res['n']}); no correlation is reported."
            )
            continue
        direction = "positive" if res["rho"] > 0 else "negative" if res["rho"] < 0 else "near-zero"
        out.append(
            f"{label.capitalize()}: Spearman ρ = {res['rho']:.2f} (p = {res['p']:.3f}, "
            f"n = {res['n']}{_ci_text(res['ci'])}). This is a {direction} rank association. "
            f"It is exploratory, affected by autocorrelation and shared trends, and does "
            f"not establish causation or a temporal direction."
        )
    if not out:
        out.append("No association could be computed from the available paired data.")
    return out


def flagged_days_report(df: pd.DataFrame) -> list[str]:
    """List diary days that trip pre-specified review cutpoints."""
    from metrics import daily_flags

    fl = daily_flags(df)
    if fl.empty:
        return ["No diary data to scan."]
    flagged = fl[fl["flagged"]]
    if flagged.empty:
        return [
            f"No diary days tripped the pre-specified review cutpoints in this period "
            f"(TST < {config.FLAG_TST_SHORT_HOURS} h or > {config.FLAG_TST_LONG_HOURS} h, "
            f"sleep efficiency < {config.FLAG_SLEEP_EFFICIENCY_LOW_PCT:.0f}%, SOL > "
            f"{config.FLAG_SOL_LONG_MIN:.0f} min, WASO > {config.FLAG_WASO_HIGH_MIN:.0f} min, "
            f"mood ≤ {config.FLAG_MOOD_VERY_LOW} or anxiety ≥ {config.FLAG_ANXIETY_VERY_HIGH})."
        ]
    lines = [
        "Days below tripped pre-specified review cutpoints (transparent cutpoints for "
        "discussion; not diagnostic findings):"
    ]
    for _, row in flagged.iterrows():
        lines.append(f"**{row['date'].date()}** — {row['flags']}")
    return lines


def data_quality_summary(df: pd.DataFrame, start: date, end: date) -> list[str]:
    c = completeness(df, start, end)
    return [
        f"Expected calendar days in period: {c['expected_days']}.",
        f"Valid sleep-diary nights: {c['sleep_nights']} "
        f"({c['sleep_completeness_pct']}% of expected days).",
        f"Days with at least one mood entry: {c['mood_days']} "
        f"({c['mood_completeness_pct']}% of expected days).",
        "Missing entries are left missing; they are not imputed. "
        "Calculated metrics are based only on valid recorded data.",
    ]


def all_observations(df: pd.DataFrame, start: date, end: date) -> list[str]:
    lines = []
    lines.append("**Patient-reported sleep data and calculated sleep metrics**")
    lines.extend(sleep_summary(df))
    lines.append("")
    lines.append("**Sleep-timing regularity**")
    lines.extend(regularity_summary(df))
    lines.append("")
    lines.append("**Patient-reported mood data**")
    lines.extend(mood_summary(df))
    lines.append("")
    lines.append("**Exploratory sleep–mood associations**")
    lines.extend(association_summary(df))
    lines.append("")
    lines.append("**Days tripping review cutpoints**")
    lines.extend(flagged_days_report(df))
    lines.append("")
    lines.append("**Data completeness and missingness**")
    lines.extend(data_quality_summary(df, start, end))
    return lines
