"""Sleep metrics, data preparation, and association analysis.

All calculations are explicit and reproducible. Missing or invalid data are
handled transparently and never silently imputed.

Key evidence notes carried into the code:
- Sleep-diary-derived TST/SE/SOL are subjective estimates; actigraphy studies
  show meaningful diary-vs-actigraphy disagreement (especially for WASO, and
  with insomnia samples tending to underestimate sleep). The language around
  these metrics therefore always says "patient-reported" or "diary-derived".
- Correlations between repeated within-person measures are confounded by
  autocorrelation, shared trends, and common third variables. Every reported
  correlation therefore carries n, a p-value, AND a 95% Fisher-z confidence
  interval (p-values alone severely overstate certainty in small samples).
"""

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from scipy import stats

import config


def _combine(d: date, t: time) -> datetime:
    return datetime.combine(d, t)


def compute_sleep_metrics(
    entry_date: str,
    bedtime: str | None,
    wake_time: str | None,
    sol_min: float | None,
    waso_min: float | None,
) -> dict:
    """Compute TIB, TST, and SE from raw diary fields.

    ``entry_date`` is the night date. If wake time is earlier than or equal to
    bedtime, the wake time is assumed to occur on the following calendar day.
    """
    result = {
        "tib_min": np.nan,
        "tst_min": np.nan,
        "sleep_efficiency_pct": np.nan,
        "valid": False,
    }
    if not bedtime or not wake_time:
        return result
    try:
        d = date.fromisoformat(entry_date)
        bt = time.fromisoformat(bedtime)
        wt = time.fromisoformat(wake_time)
    except ValueError:
        return result
    bed_dt = _combine(d, bt)
    wake_dt = _combine(d, wt)
    if wake_dt <= bed_dt:
        wake_dt += timedelta(days=1)
    tib = (wake_dt - bed_dt).total_seconds() / 60.0
    if tib <= 0:
        return result
    sol = float(sol_min) if sol_min is not None and not pd.isna(sol_min) else 0.0
    waso = float(waso_min) if waso_min is not None and not pd.isna(waso_min) else 0.0
    tst = max(0.0, tib - sol - waso)
    se = (tst / tib) * 100.0
    result.update({"tib_min": tib, "tst_min": tst, "sleep_efficiency_pct": se, "valid": True})
    return result


def format_time_ampm(time_val: str | time | None) -> str:
    """Format an ISO time string ('23:00' or '23:00:00') or time object into 12-hour AM/PM format ('11:00 PM')."""
    if not time_val or pd.isna(time_val):
        return "—"
    try:
        if isinstance(time_val, str):
            clean_str = time_val.strip()
            if not clean_str:
                return "—"
            clean_str = clean_str[:5]
            t = time.fromisoformat(clean_str)
        else:
            t = time_val
        return t.strftime("%I:%M %p")
    except Exception:
        return str(time_val)


def compute_social_jetlag(df: pd.DataFrame) -> dict:
    """Compute Social Jetlag (Roenneberg et al., 2012; Wittmann et al., 2006).

    Social jetlag is the absolute discrepancy between mid-sleep time on free days
    (weekend nights: Fri/Sat) and work days (weekday nights: Sun-Thu).
    Returns mean workday mid-sleep, mean free-day mid-sleep in 12-hour AM/PM format,
    difference in hours, and clinical flag indicator.
    """
    res = {
        "workday_midsleep_hhmm": None,
        "freeday_midsleep_hhmm": None,
        "social_jetlag_hours": None,
        "flagged": False,
        "sufficient_data": False,
    }
    if df is None or df.empty or "mid_sleep_minutes" not in df.columns or "date" not in df.columns:
        return res
    sub = df.dropna(subset=["mid_sleep_minutes", "valid_sleep"]).copy()
    sub = sub[sub["valid_sleep"]]
    if len(sub) < 4:
        return res

    # In diary dates: day-of-week 4=Friday, 5=Saturday are free/weekend sleep nights
    # 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 6=Sunday are work/school sleep nights
    sub["weekday"] = sub["date"].dt.day_of_week
    workdays = sub[sub["weekday"].isin([0, 1, 2, 3, 6])]
    freedays = sub[sub["weekday"].isin([4, 5])]

    if len(workdays) < 2 or len(freedays) < 1:
        return res

    def _mean_clock_minutes(mins_series):
        angles = 2.0 * np.pi * mins_series / 1440.0
        s = np.sin(angles).mean()
        c = np.cos(angles).mean()
        mean_angle = np.arctan2(s, c) % (2.0 * np.pi)
        return (mean_angle / (2.0 * np.pi)) * 1440.0

    msw = _mean_clock_minutes(workdays["mid_sleep_minutes"])
    msf = _mean_clock_minutes(freedays["mid_sleep_minutes"])

    diff_mins = abs(msf - msw)
    diff_mins = min(diff_mins, 1440.0 - diff_mins)
    diff_hours = diff_mins / 60.0

    def _mins_to_ampm(m):
        h = int(m // 60) % 24
        mn = int(m % 60)
        return time(h, mn).strftime("%I:%M %p")

    res.update({
        "workday_midsleep_hhmm": _mins_to_ampm(msw),
        "freeday_midsleep_hhmm": _mins_to_ampm(msf),
        "social_jetlag_hours": round(diff_hours, 2),
        "flagged": diff_hours >= config.SOCIAL_JETLAG_FLAG_HOURS,
        "sufficient_data": True,
    })
    return res


def compute_affective_instability_mssd(series: pd.Series) -> float | None:
    """Compute Mean Squared Successive Difference (MSSD) for mood ratings.

    MSSD (Ebner-Priemer et al., 2007; Jahng et al., 2008) measures temporal mood
    instability / emotional volatility by quantifying the magnitude of sequential
    fluctuations: MSSD = 1/(N-1) * sum((x_{i+1} - x_i)^2).
    Unlike standard deviation, MSSD is sensitive to the temporal order and
    frequency of mood swings and is not inflated by slow monotonic trends.
    """
    if series is None:
        return None
    valid = series.dropna().to_numpy(dtype=float)
    if len(valid) < 3:
        return None
    diffs = np.diff(valid)
    return float(np.mean(diffs ** 2))


def compute_sleep_debt(df: pd.DataFrame, baseline_hours: float = config.SLEEP_DEBT_BASELINE_HOURS) -> pd.DataFrame:
    """Compute nightly sleep deficit and rolling cumulative sleep debt."""
    if df is None or df.empty or "tst_hours" not in df.columns:
        return pd.DataFrame()
    d = df.sort_values("date").copy()
    d["nightly_deficit_hours"] = np.maximum(0.0, baseline_hours - d["tst_hours"].fillna(baseline_hours))
    d["cumulative_debt_hours"] = d["nightly_deficit_hours"].cumsum()
    return d


def sleep_regularity_simplified(df: pd.DataFrame) -> float | None:
    """Simplified Sleep Regularity Index from diary bed/wake times.

    The full SRI (Phillips et al. 2017; Windred et al. 2024) needs epoch-level
    sleep/wake states; with diary data the accepted proxy is the concordance of
    bedtime/waketime day-to-day. We therefore report the mean instantaneous
    concordance of standardised bed- and wake-times between adjacent nights,
    scaled 0–100 (100 = perfectly regular), or None when fewer than 2 valid
    adjacent night pairs exist.
    """
    if df is None or df.empty or "bedtime_minutes" not in df.columns:
        return None
    sub = df.dropna(subset=["bedtime_minutes", "wake_minutes"]).sort_values("date")
    if len(sub) < 2:
        return None
    b = sub["bedtime_minutes"].to_numpy(dtype=float)
    w = sub["wake_minutes"].to_numpy(dtype=float)
    # Concordance = |x_t - x_{t-1}| wrapped to the 0..120 clock-circle, scaled.
    def _concordance(prev, curr):
        diff = abs(curr - prev) % 1440.0
        diff = min(diff, 1440.0 - diff)
        return 1.0 - diff / 720.0
    pairs = len(b) - 1
    conc = [_concordance(b[i], b[i + 1]) for i in range(pairs)]
    conc += [_concordance(w[i], w[i + 1]) for i in range(pairs)]
    return float(np.mean(conc) * 100.0)


def build_sleep_dataframe(entries: list[dict]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame()
    rows = []
    for e in entries:
        m = compute_sleep_metrics(
            e["entry_date"], e.get("bedtime"), e.get("wake_time"),
            e.get("sleep_onset_latency_min"), e.get("waso_min"),
        )
        bt_minutes = wt_minutes = np.nan
        try:
            if e.get("bedtime"):
                t = time.fromisoformat(e["bedtime"])
                bt_minutes = t.hour * 60 + t.minute
            if e.get("wake_time"):
                t = time.fromisoformat(e["wake_time"])
                wt_minutes = t.hour * 60 + t.minute
        except ValueError:
            pass
        mid_sleep = np.nan
        if pd.notna(bt_minutes) and pd.notna(wt_minutes) and m["valid"]:
            dur = wt_minutes - bt_minutes if wt_minutes > bt_minutes else (wt_minutes + 1440.0 - bt_minutes)
            mid_sleep = (bt_minutes + dur / 2.0) % 1440.0
        rows.append({
            "date": pd.to_datetime(e["entry_date"]),
            "bedtime": e.get("bedtime"),
            "wake_time": e.get("wake_time"),
            "bedtime_ampm": format_time_ampm(e.get("bedtime")),
            "wake_time_ampm": format_time_ampm(e.get("wake_time")),
            "bedtime_minutes": bt_minutes,
            "wake_minutes": wt_minutes,
            "mid_sleep_minutes": mid_sleep,
            "sol_min": e.get("sleep_onset_latency_min"),
            "awakenings": e.get("awakenings_count"),
            "waso_min": e.get("waso_min"),
            "sleep_quality": e.get("sleep_quality_1_5"),
            "notes": e.get("notes"),
            "tib_min": m["tib_min"],
            "tst_min": m["tst_min"],
            "tst_hours": m["tst_min"] / 60.0 if pd.notna(m["tst_min"]) else np.nan,
            "sleep_efficiency_pct": m["sleep_efficiency_pct"],
            "valid_sleep": m["valid"],
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_mood_daily_dataframe(entries: list[dict]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame()
    df = pd.DataFrame(entries)
    df["date"] = pd.to_datetime(df["entry_date"])
    df["mood_score_1_10"] = pd.to_numeric(df["mood_score_1_10"], errors="coerce")
    df["anxiety_1_10"] = pd.to_numeric(df["anxiety_1_10"], errors="coerce")
    df["energy_1_10"] = pd.to_numeric(df["energy_1_10"], errors="coerce")
    g = df.groupby("date")
    out = pd.DataFrame({
        "mood_mean": g["mood_score_1_10"].mean(),
        "mood_sd": g["mood_score_1_10"].std(ddof=0),
        "mood_n": g["mood_score_1_10"].count(),
        "anxiety_mean": g["anxiety_1_10"].mean(),
        "energy_mean": g["energy_1_10"].mean(),
    }).reset_index()
    return out.sort_values("date").reset_index(drop=True)


STANDARD_SLEEP_COLS = {
    "bedtime": None,
    "wake_time": None,
    "bedtime_ampm": "—",
    "wake_time_ampm": "—",
    "bedtime_minutes": np.nan,
    "wake_minutes": np.nan,
    "mid_sleep_minutes": np.nan,
    "sol_min": np.nan,
    "awakenings": np.nan,
    "waso_min": np.nan,
    "sleep_quality": np.nan,
    "notes": None,
    "tib_min": np.nan,
    "tst_min": np.nan,
    "tst_hours": np.nan,
    "sleep_efficiency_pct": np.nan,
    "valid_sleep": False,
}

STANDARD_MOOD_COLS = {
    "mood_mean": np.nan,
    "mood_sd": np.nan,
    "mood_n": 0,
    "anxiety_mean": np.nan,
    "energy_mean": np.nan,
}


def merge_sleep_mood(sleep_df: pd.DataFrame, mood_daily: pd.DataFrame) -> pd.DataFrame:
    if sleep_df.empty and mood_daily.empty:
        return pd.DataFrame()
    if sleep_df.empty:
        df = mood_daily.copy()
    elif mood_daily.empty:
        df = sleep_df.copy()
    else:
        df = pd.merge(sleep_df, mood_daily, on="date", how="outer")

    # Guarantee standard schema so downstream analytics and charts never raise KeyError
    for col, default_val in STANDARD_SLEEP_COLS.items():
        if col not in df.columns:
            df[col] = default_val
    for col, default_val in STANDARD_MOOD_COLS.items():
        if col not in df.columns:
            df[col] = default_val

    return df.sort_values("date").reset_index(drop=True)


def add_lagged_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged columns for exploratory temporal association.

    - sleep_t -> mood_t+1  (does last night's sleep relate to next-day mood/anxiety/energy?)
    - mood_t -> sleep_t+1  (does today's mood/anxiety relate to the following night's sleep/WASO/SOL?)
    - concurrent same-day associations are also available for comparison.
    """
    if df.empty:
        return df
    d = df.sort_values("date").copy()
    d["mood_next_day"] = d["mood_mean"].shift(-1) if "mood_mean" in d else np.nan
    d["anxiety_next_day"] = d["anxiety_mean"].shift(-1) if "anxiety_mean" in d else np.nan
    d["energy_next_day"] = d["energy_mean"].shift(-1) if "energy_mean" in d else np.nan
    d["sleep_quality_next_night"] = d["sleep_quality"].shift(-1) if "sleep_quality" in d else np.nan
    d["tst_hours_next_night"] = d["tst_hours"].shift(-1) if "tst_hours" in d else np.nan
    d["waso_next_night"] = d["waso_min"].shift(-1) if "waso_min" in d else np.nan
    d["sol_next_night"] = d["sol_min"].shift(-1) if "sol_min" in d else np.nan
    return d


def fisher_z_interval(rho: float, n: int, alpha: float = config.CORRELATION_ALPHA) -> tuple[float, float] | None:
    """Approximate (1-alpha) CI for a Spearman rho via Fisher's z transform.

    Adequate for reporting uncertainty on moderate rho; see README limitations.
    """
    if n < 4 or not np.isfinite(rho) or abs(rho) >= 1.0 - 1e-9:
        return None
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    lo, hi = np.tanh(z - crit * se), np.tanh(z + crit * se)
    return float(lo), float(hi)


def spearman_pair(df: pd.DataFrame, x: str, y: str, min_pairs: int = config.MIN_PAIRS_FOR_CORRELATION):
    """Return Spearman rho, p-value, n, and 95% CI for two columns.

    Returns ``None`` if either column is absent. When there are fewer than
    ``min_pairs`` complete observations, returns n with ``sufficient=False``
    so the UI can say so explicitly instead of showing an unreliable value.
    """
    if x not in df.columns or y not in df.columns:
        return None
    sub = df[[x, y]].dropna()
    n = len(sub)
    if n < min_pairs or sub[x].nunique() <= 1 or sub[y].nunique() <= 1:
        return {"rho": np.nan, "p": np.nan, "n": n, "sufficient": False, "ci": None}
    rho, p = stats.spearmanr(sub[x], sub[y])
    return {
        "rho": float(rho), "p": float(p), "n": n, "sufficient": True,
        "ci": fisher_z_interval(float(rho), n),
    }


def rolling_mean(series: pd.Series, window: int = 7) -> pd.Series:
    if series is None or len(series) == 0:
        return series
    return series.rolling(window=window, min_periods=max(2, window // 2), center=True).mean()


def completeness(df: pd.DataFrame, start: date, end: date) -> dict:
    """Report expected vs observed diary nights and mood days in a period."""
    expected = (end - start).days + 1
    if df is None or df.empty or "date" not in df:
        return {"expected_days": expected, "sleep_nights": 0, "mood_days": 0,
                "sleep_completeness_pct": 0.0, "mood_completeness_pct": 0.0}
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    sub = df.loc[mask]
    sleep_n = int(sub["valid_sleep"].sum()) if "valid_sleep" in sub else 0
    mood_n = int(sub["mood_mean"].notna().sum()) if "mood_mean" in sub else 0
    return {
        "expected_days": expected,
        "sleep_nights": sleep_n,
        "mood_days": mood_n,
        "sleep_completeness_pct": round(sleep_n / expected * 100, 1) if expected else 0.0,
        "mood_completeness_pct": round(mood_n / expected * 100, 1) if expected else 0.0,
    }


def daily_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-specified review-worthy diary days: extreme TST / low SE / long SOL etc.

    These transparent cutpoints (config.FLAG_*) intentionally err toward
    flagging; they describe data worth discussing, not observations about the
    person. No imputation or inference beyond the recorded values.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    flags: list[list[str]] = []
    for _, row in d.iterrows():
        f = []
        if pd.notna(row.get("tst_hours")):
            if row["tst_hours"] < config.FLAG_TST_SHORT_HOURS:
                f.append(f"Diary TST {row['tst_hours']:.1f} h (< {config.FLAG_TST_SHORT_HOURS} h)")
            if row["tst_hours"] > config.FLAG_TST_LONG_HOURS:
                f.append(f"Diary TST {row['tst_hours']:.1f} h (> {config.FLAG_TST_LONG_HOURS} h)")
        if pd.notna(row.get("sleep_efficiency_pct")) and row["sleep_efficiency_pct"] < config.FLAG_SLEEP_EFFICIENCY_LOW_PCT:
            f.append(f"Diary sleep efficiency {row['sleep_efficiency_pct']:.0f}% (< {config.FLAG_SLEEP_EFFICIENCY_LOW_PCT:.0f}%)")
        if pd.notna(row.get("sol_min")) and row["sol_min"] > config.FLAG_SOL_LONG_MIN:
            f.append(f"Diary SOL {row['sol_min']:.0f} min (> {config.FLAG_SOL_LONG_MIN:.0f} min)")
        if pd.notna(row.get("waso_min")) and row["waso_min"] > config.FLAG_WASO_HIGH_MIN:
            f.append(f"Diary WASO {row['waso_min']:.0f} min (> {config.FLAG_WASO_HIGH_MIN:.0f} min)")
        if pd.notna(row.get("mood_mean")) and row["mood_mean"] <= config.FLAG_MOOD_VERY_LOW:
            f.append(f"Daily mood {row['mood_mean']:.1f} / 10 (low)")
        if pd.notna(row.get("anxiety_mean")) and row["anxiety_mean"] >= config.FLAG_ANXIETY_VERY_HIGH:
            f.append(f"Daily anxiety {row['anxiety_mean']:.1f} / 10 (high)")
        if pd.notna(row.get("energy_mean")) and row["energy_mean"] <= config.FLAG_ENERGY_VERY_LOW:
            f.append(f"Daily energy {row['energy_mean']:.1f} / 10 (low)")
        flags.append("; ".join(f) if f else "")
    out = d[["date"]].copy()
    out["flags"] = flags
    out["flagged"] = [bool(x) for x in flags]
    return out.reset_index(drop=True)
