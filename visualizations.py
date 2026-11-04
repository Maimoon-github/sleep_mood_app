"""Altair visualisations for sleep and mood data.

Design principles: line charts for trends, threshold annotations, missing-data
markers, consistent colour encoding, and accessible axis labels. Reference
lines correspond to the documented population values in config.py and are
always labelled as population guides, not individual targets.
"""

import altair as alt
import pandas as pd

import config


def _base() -> alt.Chart:
    return alt.Chart()


def sleep_mood_timeline(df: pd.DataFrame) -> alt.LayerChart | None:
    if df.empty:
        return None
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    sleep_line = (
        alt.Chart(d)
        .mark_line(point=True, color="#1f77b4", strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("tst_hours:Q", title="Total sleep time (hours)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("tst_hours:Q", title="TST (h)", format=".1f"),
                alt.Tooltip("sleep_efficiency_pct:Q", title="SE (%)", format=".0f"),
                alt.Tooltip("mood_mean:Q", title="Mood (/10)", format=".1f"),
            ],
        )
    )
    mood_line = (
        alt.Chart(d)
        .mark_line(point=True, color="#d62728", strokeWidth=2)
        .encode(
            x="date:T",
            y=alt.Y("mood_mean:Q", title="Daily mood (/10)", scale=alt.Scale(domain=[0, 10])),
        )
    )
    ref = (
        alt.Chart(pd.DataFrame({"y": [config.SLEEP_DURATION_REFERENCE_HOURS]}))
        .mark_rule(strokeDash=[6, 4], color="#1f77b4", opacity=0.6)
        .encode(y="y:Q")
    )
    return alt.layer(sleep_line, mood_line + ref).resolve_scale(y="independent").properties(
        height=340, title="Sleep duration and daily mood over time"
    ).interactive()


def sleep_timing_chart(df: pd.DataFrame) -> alt.Chart | None:
    """Bed and wake times as floating bars; visually exposes late/early drift."""
    if df.empty or "bedtime" not in df:
        return None
    d = df.dropna(subset=["bedtime", "wake_time"]).copy()
    if d.empty:
        return None
    d["date"] = pd.to_datetime(d["date"])
    d["bed_dt"] = pd.to_datetime(d["bedtime"], format="%H:%M")
    d["wake_dt"] = pd.to_datetime(d["wake_time"], format="%H:%M")
    # Wake earlier than bedtime ⇒ crossed midnight; shift forward for plotting.
    d.loc[d["wake_dt"] <= d["bed_dt"], "wake_dt"] += pd.Timedelta(days=1)
    d["mid_dt"] = d["bed_dt"] + (d["wake_dt"] - d["bed_dt"]) / 2
    return (
        alt.Chart(d)
        .mark_bar(color="#5ba3d0", size=18)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("bed_dt:Q", title="" , axis=None, scale=alt.Scale(zero=False)),
            y2="wake_dt:Q",
            detail="mid_dt:T",
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("bedtime:N", title="Bedtime"),
                alt.Tooltip("wake_time:N", title="Wake time"),
            ],
        )
        .properties(height=240, title="Sleep timing (diary bed/wake times)")
    )


def sleep_duration_chart(df: pd.DataFrame) -> alt.Chart | None:
    if df.empty:
        return None
    d = df.dropna(subset=["tst_hours"]).copy()
    if d.empty:
        return None
    bars = (
        alt.Chart(d)
        .mark_bar(color="#4c78a8")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("tst_hours:Q", title="Total sleep time (hours)"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("tst_hours:Q", title="TST (h)", format=".1f"),
                alt.Tooltip("sleep_efficiency_pct:Q", title="SE (%)", format=".0f"),
                alt.Tooltip("sleep_quality:Q", title="Quality (/5)", format=".0f"),
            ],
        )
    )
    ref = (
        alt.Chart(pd.DataFrame({"y": [config.SLEEP_DURATION_REFERENCE_HOURS]}))
        .mark_rule(strokeDash=[6, 4], color="#d62728", opacity=0.7)
        .encode(y="y:Q")
    )
    return (bars + ref).properties(height=260, title="Nightly sleep duration with 7-hour population reference")


def sleep_efficiency_chart(df: pd.DataFrame) -> alt.Chart | None:
    if df.empty:
        return None
    d = df.dropna(subset=["sleep_efficiency_pct"]).copy()
    if d.empty:
        return None
    line = (
        alt.Chart(d)
        .mark_line(point=True, color="#2ca02c")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("sleep_efficiency_pct:Q", title="Sleep efficiency (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("sleep_efficiency_pct:Q", title="SE (%)", format=".0f"),
                alt.Tooltip("tst_hours:Q", title="TST (h)", format=".1f"),
                alt.Tooltip("tib_min:Q", title="TIB (min)", format=".0f"),
            ],
        )
    )
    ref = (
        alt.Chart(pd.DataFrame({"y": [config.SLEEP_EFFICIENCY_REFERENCE_PCT]}))
        .mark_rule(strokeDash=[6, 4], color="#d62728", opacity=0.7)
        .encode(y="y:Q")
    )
    return (line + ref).properties(height=260, title="Sleep efficiency with 85% population reference")


def mood_trajectory(df: pd.DataFrame, rolling_window: int = 0) -> alt.Chart | None:
    if df.empty or "mood_mean" not in df:
        return None
    d = df.dropna(subset=["mood_mean"]).copy()
    if d.empty:
        return None
    layer_specs = []
    if rolling_window and rolling_window >= 2:
        smooth = d.copy()
        smooth["mood_smooth"] = d["mood_mean"].rolling(
            window=rolling_window, min_periods=max(2, rolling_window // 2), center=True
        ).mean()
        layer_specs.append(
            alt.Chart(smooth)
            .mark_line(color="#f2b544", strokeWidth=3)
            .encode(x="date:T", y=alt.Y("mood_smooth:Q", title="Daily mood (/10)", scale=alt.Scale(domain=[0, 10])))
        )
    line = (
        alt.Chart(d)
        .mark_line(point=True, color="#9467bd", opacity=0.55)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("mood_mean:Q", title="Daily mood (/10)", scale=alt.Scale(domain=[0, 10])),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("mood_mean:Q", title="Mood (/10)", format=".1f"),
                alt.Tooltip("anxiety_mean:Q", title="Anxiety (/10)", format=".1f"),
                alt.Tooltip("mood_sd:Q", title="Within-day SD", format=".2f"),
            ],
        )
    )
    layer_specs.append(line)
    combined = layer_specs[0]
    for spec in layer_specs[1:]:
        combined = combined + spec
    return combined.properties(height=260, title="Daily mood trajectory (patient-reported)")


def mood_distribution(df: pd.DataFrame) -> alt.Chart | None:
    if df.empty or "mood_mean" not in df:
        return None
    d = df.dropna(subset=["mood_mean"])
    if d.empty:
        return None
    hist = (
        alt.Chart(d)
        .mark_bar(color="#9467bd", opacity=0.8)
        .encode(
            x=alt.X("mood_mean:Q", bin=alt.Bin(maxbins=10), title="Daily mood (/10)"),
            y=alt.Y("count():Q", title="Number of days"),
            tooltip=[alt.Tooltip("count():Q", title="Days")],
        )
    )
    return hist.properties(height=240, title="Distribution of daily mood scores")


def correlation_heatmap(df: pd.DataFrame) -> alt.Chart | None:
    cols = [
        "tst_hours", "sleep_efficiency_pct", "sol_min", "waso_min",
        "sleep_quality", "mood_mean", "anxiety_mean", "energy_mean",
    ]
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        return None
    sub = df[available].dropna(how="all")
    if len(sub) < config.MIN_PAIRS_FOR_CORRELATION:
        return None
    corr = sub.corr(method="spearman")
    corr_long = corr.reset_index().melt(id_vars="index", var_name="variable", value_name="rho")
    corr_long = corr_long.rename(columns={"index": "row"})
    heat = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("variable:N", title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("row:N", title=None),
            color=alt.Color("rho:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="Spearman ρ"),
            tooltip=[
                alt.Tooltip("row:N", title="Variable A"),
                alt.Tooltip("variable:N", title="Variable B"),
                alt.Tooltip("rho:Q", title="ρ", format=".2f"),
            ],
        )
    )
    text = heat.mark_text(size=9).encode(
        text=alt.Text("rho:Q", format=".2f"),
        color=alt.condition(alt.datum.rho > 0.5, alt.value("white"), alt.value("black")),
    )
    return (heat + text).properties(height=320, title="Spearman correlation matrix (exploratory)")


def calendar_heatmap(df: pd.DataFrame, value_col: str = "mood_mean", title: str = "Calendar heatmap") -> alt.Chart | None:
    if df.empty or value_col not in df:
        return None
    d = df.dropna(subset=[value_col]).copy()
    if d.empty:
        return None
    d["date"] = pd.to_datetime(d["date"])
    d["week"] = d["date"].dt.isocalendar().week.astype(int)
    d["weekday"] = d["date"].dt.day_name()
    heat = (
        alt.Chart(d)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("week:O", title="ISO week"),
            y=alt.Y("weekday:N", title=None, sort=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
            color=alt.Color(f"{value_col}:Q", scale=alt.Scale(scheme="viridis"), title=value_col),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip(f"{value_col}:Q", title=value_col, format=".1f"),
            ],
        )
    )
    return heat.properties(height=220, title=title)


def questionnaire_trajectory(qdf: pd.DataFrame, instrument: str) -> alt.Chart | None:
    if qdf.empty:
        return None
    d = qdf.copy()
    d["date"] = pd.to_datetime(d["administered_at"])
    line = (
        alt.Chart(d)
        .mark_line(point=True, color="#e377c2")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("total_score:Q", title=f"{instrument} total score"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("total_score:Q", title="Score", format=".0f"),
                alt.Tooltip("severity_band:N", title="Band"),
            ],
        )
    )
    max_score = {"PHQ-9": 27, "GAD-7": 21, "ISI": 28, "MDQ": 13}.get(instrument)
    if max_score:
        for lo, hi, label in (
            {"PHQ-9": config.PHQ9_BANDS, "GAD-7": config.GAD7_BANDS,
             "ISI": config.ISI_BANDS}.get(instrument, [])
        ):
            band_rule = (
                alt.Chart(pd.DataFrame({"y": [hi]}))
                .mark_rule(strokeDash=[4, 4], opacity=0.35)
                .encode(y="y:Q")
            )
            line = line + band_rule
    return line.properties(height=220, title=f"{instrument} scores over time (patient-reported)")
