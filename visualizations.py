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


# ---------------------------------------------------------------------------
# Specialized Multi-Scale Visualizations (Day, Week, Month, Year)
# ---------------------------------------------------------------------------

def intraday_mood_chart(entries: list[dict]) -> alt.Chart | None:
    """Intraday timeline of momentary mood, anxiety, and energy ratings."""
    if not entries:
        return None
    d = pd.DataFrame(entries).copy()
    if "entry_timestamp" not in d.columns:
        return None
    d["time_dt"] = pd.to_datetime(d["entry_timestamp"])
    d["time_str"] = d["time_dt"].dt.strftime("%H:%M")
    cols = [c for c in ["mood_score_1_10", "anxiety_1_10", "energy_1_10"] if c in d.columns]
    if not cols:
        return None

    melted = d.melt(
        id_vars=["time_dt", "time_str"],
        value_vars=cols,
        var_name="metric",
        value_name="rating",
    )
    label_map = {
        "mood_score_1_10": "Mood (Valence)",
        "anxiety_1_10": "Anxiety (Tension)",
        "energy_1_10": "Energy (Arousal)",
    }
    melted["metric_label"] = melted["metric"].map(label_map)

    color_scale = alt.Scale(
        domain=["Mood (Valence)", "Anxiety (Tension)", "Energy (Arousal)"],
        range=["#2b83ba", "#d7191c", "#fdae61"],
    )

    line = (
        alt.Chart(melted)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("time_dt:T", title="Time of Day", axis=alt.Axis(format="%H:%M")),
            y=alt.Y("rating:Q", title="Rating (1–10)", scale=alt.Scale(domain=[0, 10])),
            color=alt.Color("metric_label:N", scale=color_scale, title="Dimension"),
            tooltip=[
                alt.Tooltip("time_str:N", title="Timestamp"),
                alt.Tooltip("metric_label:N", title="Metric"),
                alt.Tooltip("rating:Q", title="Score (/10)"),
            ],
        )
    )
    return line.properties(height=240, title="Intraday Momentary Affect Fluctuations (EMA)")


def circumplex_affect_chart(entries: list[dict]) -> alt.Chart | None:
    """Russell Circumplex Model of Affect (Valence x Arousal).

    Pleasure/Valence on X axis (1-10); Energy/Arousal on Y axis (1-10).
    Quad 1 (top-right): High valence, High arousal (Alert, Excited, Joyful)
    Quad 2 (bottom-right): High valence, Low arousal (Calm, Content, Serene)
    Quad 3 (top-left): Low valence, High arousal (Tense, Anxious, Agitated)
    Quad 4 (bottom-left): Low valence, Low arousal (Sad, Depressed, Fatigued)
    """
    if not entries:
        return None
    d = pd.DataFrame(entries).copy()
    if "mood_score_1_10" not in d.columns or "energy_1_10" not in d.columns:
        return None
    d = d.dropna(subset=["mood_score_1_10", "energy_1_10"]).copy()
    if d.empty:
        return None

    if "dominant_emotion" not in d.columns or d["dominant_emotion"].isna().all():
        d["dominant_emotion"] = "Check-in"

    pts = (
        alt.Chart(d)
        .mark_circle(size=140, opacity=0.85)
        .encode(
            x=alt.X("mood_score_1_10:Q", title="Valence: Displeasure (1) → Pleasure (10)", scale=alt.Scale(domain=[0.5, 10.5])),
            y=alt.Y("energy_1_10:Q", title="Arousal: Deactivation/Fatigue (1) → Activation/Energy (10)", scale=alt.Scale(domain=[0.5, 10.5])),
            color=alt.Color("dominant_emotion:N", title="Emotion / State", scale=alt.Scale(scheme="category10")),
            tooltip=[
                alt.Tooltip("entry_timestamp:N", title="Time"),
                alt.Tooltip("mood_score_1_10:Q", title="Valence (/10)"),
                alt.Tooltip("energy_1_10:Q", title="Arousal (/10)"),
                alt.Tooltip("anxiety_1_10:Q", title="Anxiety (/10)"),
                alt.Tooltip("dominant_emotion:N", title="Emotion"),
            ],
        )
    )

    # Reference crosshairs at neutral midpoint (5.5, 5.5)
    v_line = alt.Chart(pd.DataFrame({"x": [5.5]})).mark_rule(strokeDash=[4, 4], color="#888888", opacity=0.6).encode(x="x:Q")
    h_line = alt.Chart(pd.DataFrame({"y": [5.5]})).mark_rule(strokeDash=[4, 4], color="#888888", opacity=0.6).encode(y="y:Q")

    # Quadrant descriptive annotations
    quad_data = pd.DataFrame([
        {"x": 2.5, "y": 9.5, "text": "Tense / Anxious"},
        {"x": 8.5, "y": 9.5, "text": "Alert / Joyful"},
        {"x": 2.5, "y": 1.5, "text": "Fatigued / Depressed"},
        {"x": 8.5, "y": 1.5, "text": "Calm / Content"},
    ])
    labels = alt.Chart(quad_data).mark_text(size=11, opacity=0.5, fontWeight="bold").encode(
        x="x:Q", y="y:Q", text="text:N"
    )

    return (v_line + h_line + labels + pts).properties(
        height=320, title="Russell Circumplex Model of Affect (Valence vs. Arousal)"
    )


def day_of_week_chart(df: pd.DataFrame) -> alt.Chart | None:
    """Sleep duration and mood profiles broken down by day of the week."""
    if df is None or df.empty or "date" not in df.columns:
        return None
    d = df.dropna(subset=["date"]).copy()
    d["weekday_name"] = d["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    g = d.groupby("weekday_name").agg(
        mean_tst=("tst_hours", "mean"),
        mean_mood=("mood_mean", "mean"),
        count=("valid_sleep", "sum"),
    ).reset_index()

    if g.empty or g["count"].sum() == 0:
        return None

    bars = (
        alt.Chart(g)
        .mark_bar(color="#4c78a8", opacity=0.85, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("weekday_name:N", sort=order, title="Day of Week"),
            y=alt.Y("mean_tst:Q", title="Mean Total Sleep Time (hours)"),
            tooltip=[
                alt.Tooltip("weekday_name:N", title="Day"),
                alt.Tooltip("mean_tst:Q", title="Mean TST (h)", format=".1f"),
                alt.Tooltip("mean_mood:Q", title="Mean Mood (/10)", format=".1f"),
                alt.Tooltip("count:Q", title="Recorded Nights"),
            ],
        )
    )

    ref = (
        alt.Chart(pd.DataFrame({"y": [config.SLEEP_DURATION_REFERENCE_HOURS]}))
        .mark_rule(strokeDash=[6, 4], color="#d62728", opacity=0.7)
        .encode(y="y:Q")
    )
    return (bars + ref).properties(height=260, title="Day-of-Week Sleep Duration Profile (vs. 7 h Reference)")


def weekday_vs_weekend_comparison_chart(df: pd.DataFrame) -> alt.Chart | None:
    """Grouped bar comparison contrasting workdays vs free days / weekends."""
    if df is None or df.empty or "date" not in df.columns:
        return None
    d = df.dropna(subset=["date"]).copy()
    d["day_type"] = d["date"].dt.day_of_week.apply(lambda w: "Weekend (Fri–Sat)" if w in [4, 5] else "Weekday (Sun–Thu)")

    rows = []
    for dtype, group in d.groupby("day_type"):
        if "tst_hours" in group and group["tst_hours"].notna().any():
            rows.append({"Day Type": dtype, "Metric": "Sleep Duration (h)", "Value": group["tst_hours"].mean()})
        if "sleep_efficiency_pct" in group and group["sleep_efficiency_pct"].notna().any():
            rows.append({"Day Type": dtype, "Metric": "Sleep Efficiency (%)", "Value": group["sleep_efficiency_pct"].mean() / 10.0}) # scaled to 0-10
        if "mood_mean" in group and group["mood_mean"].notna().any():
            rows.append({"Day Type": dtype, "Metric": "Mood Score (/10)", "Value": group["mood_mean"].mean()})

    if not rows:
        return None
    comp_df = pd.DataFrame(rows)

    chart = (
        alt.Chart(comp_df)
        .mark_bar(opacity=0.85)
        .encode(
            x=alt.X("Metric:N", title=None),
            y=alt.Y("Value:Q", title="Score / Scaled Value"),
            color=alt.Color("Day Type:N", scale=alt.Scale(domain=["Weekday (Sun–Thu)", "Weekend (Fri–Sat)"], range=["#4c78a8", "#f58518"])),
            xOffset="Day Type:N",
            tooltip=[
                alt.Tooltip("Day Type:N"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Value:Q", format=".1f"),
            ],
        )
    )
    return chart.properties(height=260, title="Weekday vs. Weekend Sleep and Mood Contrast (Social Jetlag Exploration)")


def sleep_debt_chart(df: pd.DataFrame) -> alt.Chart | None:
    """Cumulative sleep deficit curve relative to standard 7.5h baseline."""
    from metrics import compute_sleep_debt
    debt_df = compute_sleep_debt(df)
    if debt_df.empty or "cumulative_debt_hours" not in debt_df.columns:
        return None
    d = debt_df.dropna(subset=["date", "cumulative_debt_hours"]).copy()
    if d.empty:
        return None

    area = (
        alt.Chart(d)
        .mark_area(color="#e45756", opacity=0.35)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("cumulative_debt_hours:Q", title="Cumulative Sleep Debt (hours)"),
        )
    )
    line = (
        alt.Chart(d)
        .mark_line(color="#e45756", strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("cumulative_debt_hours:Q", title="Cumulative Sleep Debt (hours)"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("cumulative_debt_hours:Q", title="Cumulative Debt (h)", format=".1f"),
                alt.Tooltip("nightly_deficit_hours:Q", title="Nightly Deficit (h)", format=".1f"),
            ],
        )
    )
    return (area + line).properties(height=240, title="Cumulative Sleep Debt Accumulation Curve")


def cross_lagged_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> alt.Chart | None:
    """Scatter plot with trendline demonstrating temporal associations."""
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    d = df.dropna(subset=[x_col, y_col]).copy()
    if len(d) < config.MIN_PAIRS_FOR_CORRELATION:
        return None

    points = (
        alt.Chart(d)
        .mark_circle(size=80, color="#1f77b4", opacity=0.7)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_label),
            y=alt.Y(f"{y_col}:Q", title=y_label),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip(f"{x_col}:Q", title=x_label, format=".1f"),
                alt.Tooltip(f"{y_col}:Q", title=y_label, format=".1f"),
            ],
        )
    )
    trend = points.transform_regression(x_col, y_col).mark_line(color="#d62728", strokeWidth=2)
    return (points + trend).properties(height=280, title=title)


def annual_monthly_trajectory(df: pd.DataFrame) -> alt.Chart | None:
    """12-month longitudinal monthly averages of sleep duration and mood."""
    if df is None or df.empty or "date" not in df.columns:
        return None
    d = df.dropna(subset=["date"]).copy()
    d["year_month"] = d["date"].dt.strftime("%Y-%m")
    g = d.groupby("year_month").agg(
        mean_tst=("tst_hours", "mean"),
        mean_mood=("mood_mean", "mean"),
        nights=("valid_sleep", "sum"),
    ).reset_index()

    if len(g) < 2:
        return None

    tst_bar = (
        alt.Chart(g)
        .mark_bar(color="#5b9bd5", opacity=0.75, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("year_month:N", title="Month"),
            y=alt.Y("mean_tst:Q", title="Mean TST (hours)"),
            tooltip=[
                alt.Tooltip("year_month:N", title="Month"),
                alt.Tooltip("mean_tst:Q", title="Mean TST (h)", format=".1f"),
                alt.Tooltip("nights:Q", title="Sleep Nights"),
            ],
        )
    )
    mood_line = (
        alt.Chart(g)
        .mark_line(point=True, color="#ed7d31", strokeWidth=2.5)
        .encode(
            x="year_month:N",
            y=alt.Y("mean_mood:Q", title="Mean Mood (/10)", scale=alt.Scale(domain=[0, 10])),
            tooltip=[
                alt.Tooltip("year_month:N", title="Month"),
                alt.Tooltip("mean_mood:Q", title="Mean Mood (/10)", format=".1f"),
            ],
        )
    )
    ref = (
        alt.Chart(pd.DataFrame({"y": [config.SLEEP_DURATION_REFERENCE_HOURS]}))
        .mark_rule(strokeDash=[4, 4], color="#5b9bd5", opacity=0.6)
        .encode(y="y:Q")
    )
    return (tst_bar + ref + mood_line).resolve_scale(y="independent").properties(
        height=280, title="12-Month Macro Assessment: Monthly Sleep Duration & Mood Trajectories"
    )
