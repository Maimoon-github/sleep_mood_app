"""Patient-facing views: dashboard, sleep entry, mood entry, notes, questionnaires, data control."""

from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

import analytics
import auth
import config
import database
import instruments
import metrics
import visualizations as viz


def _patient_id() -> int:
    return st.session_state.user["id"]


def _check_consent() -> bool:
    profile = database.get_patient_profile(_patient_id())
    if profile and profile.get("consent_given"):
        return True
    st.warning(
        "Before recording health data, please confirm that you understand how your "
        "sleep and mood information will be stored and used."
    )
    with st.form("consent_form"):
        st.markdown(
            "**Consent for recording health data**\n\n"
            "- This application stores sleep, mood, and note data you enter.\n"
            "- Data is stored locally in a SQLite database for this research prototype.\n"
            "- Your clinician can view your data only if you are linked to them; questionnaire "
            "entries can be kept private from your clinician.\n"
            "- From **My Data** you can export all of your records or delete your account and data.\n"
            "- This application does not provide diagnoses or emergency care."
        )
        agreed = st.checkbox("I understand and consent to the recording of my health data.")
        submitted = st.form_submit_button("Confirm consent")
        if submitted:
            if agreed:
                database.set_consent(_patient_id())
                st.success("Consent recorded.")
                st.rerun()
            else:
                st.error("Consent is required before data entry.")
    return False


def dashboard():
    st.subheader("Interactive Longitudinal Sleep & Mood Assessment")
    if not _check_consent():
        return

    # Assessment period selector
    col1, col2 = st.columns([1, 1])
    with col1:
        period = st.selectbox("Assessment period", ["Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days", "Last 365 days", "Custom"], index=2)
    with col2:
        if period == "Custom":
            start = st.date_input("Start date", value=date.today() - timedelta(days=30))
            end = st.date_input("End date", value=date.today())
        else:
            days = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30, "Last 90 days": 90, "Last 365 days": 365}[period]
            end = date.today()
            start = end - timedelta(days=days - 1)

    sleep_entries = database.get_sleep_entries(_patient_id(), start.isoformat(), end.isoformat())
    mood_entries = database.get_mood_entries(_patient_id(), start.isoformat(), end.isoformat())
    sleep_df = metrics.build_sleep_dataframe(sleep_entries)
    mood_daily = metrics.build_mood_daily_dataframe(mood_entries)
    df = metrics.merge_sleep_mood(sleep_df, mood_daily)
    df = metrics.add_lagged_columns(df)

    c = metrics.completeness(df, start, end)
    sri = metrics.sleep_regularity_simplified(sleep_df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Valid sleep nights", f"{c['sleep_nights']} / {c['expected_days']}")
    m2.metric("Days with mood data", f"{c['mood_days']} / {c['expected_days']}")
    if not sleep_df.empty and sleep_df["valid_sleep"].sum() > 0:
        m3.metric("Mean TST", f"{sleep_df.loc[sleep_df['valid_sleep'], 'tst_hours'].mean():.1f} h")
        m4.metric("Mean SE", f"{sleep_df.loc[sleep_df['valid_sleep'], 'sleep_efficiency_pct'].mean():.0f}%")
    else:
        m3.metric("Mean TST", "—")
        m4.metric("Mean SE", "—")
    m5.metric("Sleep regularity (SRI)", f"{sri:.0f} / 100" if sri is not None else "—")

    st.caption(
        "Blue/purple = yours (patient-reported). TST (Total Sleep Time), SE (Sleep Efficiency = TST/TIB), and TIB (Time in Bed) "
        "are derived from your daily diary. Reference lines are population-level guides (AASM/SRS ≥ 7 h; SE ≥ 85%), not individual targets."
    )

    tab_day, tab_week, tab_month, tab_year, tab_assoc, tab_obs = st.tabs([
        "📅 Daily Assessment",
        "📊 Weekly Assessment",
        "🗓️ Monthly Assessment",
        "📈 Yearly Assessment",
        "🔗 Associations",
        "📝 Evidence Observations",
    ])

    with tab_day:
        st.markdown("### 24-Hour Day Level Assessment")
        st.caption("Deep-dive into a specific 24-hour cycle: overnight sleep architecture, within-day mood fluctuations, and contextual observations.")
        avail_dates = sorted(list(set(
            [e["entry_date"] for e in sleep_entries] + [e["entry_date"] for e in mood_entries]
        )), reverse=True)
        if not avail_dates:
            st.info("No sleep or mood entries recorded in this period.")
        else:
            selected_date_str = st.selectbox("Select day to inspect", avail_dates, index=0)
            sel_d = date.fromisoformat(selected_date_str)

            # Sleep details for that date
            s_match = [e for e in sleep_entries if e["entry_date"] == selected_date_str]
            m_match = [e for e in mood_entries if e["entry_date"] == selected_date_str]

            st.markdown(f"#### Sleep Architecture: Night of {sel_d.strftime('%A, %B %d, %Y')}")
            if s_match:
                s = s_match[0]
                sm = metrics.compute_sleep_metrics(selected_date_str, s.get("bedtime"), s.get("wake_time"), s.get("sleep_onset_latency_min"), s.get("waso_min"))
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                bed_str = metrics.format_time_ampm(s.get('bedtime'))
                wake_str = metrics.format_time_ampm(s.get('wake_time'))
                sc1.metric("Bedtime → Wake", f"{bed_str} → {wake_str}")
                sc2.metric("Time in Bed (TIB)", f"{sm['tib_min'] / 60.0:.1f} h" if sm['valid'] else "—")
                sc3.metric("Total Sleep (TST)", f"{sm['tst_min'] / 60.0:.1f} h" if sm['valid'] else "—")
                sc4.metric("Sleep Efficiency", f"{sm['sleep_efficiency_pct']:.0f}%" if sm['valid'] else "—")
                sc5.metric("Sleep Quality", f"{s.get('sleep_quality_1_5', '—')} / 5")

                sc_sub1, sc_sub2, sc_sub3 = st.columns(3)
                sc_sub1.info(f"**Sleep Onset Latency (SOL):** {s.get('sleep_onset_latency_min', 0):.0f} min")
                sc_sub2.info(f"**Awakenings Count:** {s.get('awakenings_count', 0)}")
                sc_sub3.info(f"**Wake After Sleep Onset (WASO):** {s.get('waso_min', 0):.0f} min")
                if s.get("notes"):
                    st.markdown(f"**Sleep Notes:** *{s['notes']}*")
            else:
                st.info(f"No sleep diary entry recorded for {selected_date_str}.")

            st.markdown("---")
            st.markdown(f"#### Daytime Momentary Affect & Circumplex: {sel_d.strftime('%A, %B %d')}")
            if m_match:
                mc1, mc2 = st.columns([1, 1])
                with mc1:
                    chart_intra = viz.intraday_mood_chart(m_match)
                    if chart_intra:
                        st.altair_chart(chart_intra, use_container_width=True)
                with mc2:
                    chart_circ = viz.circumplex_affect_chart(m_match)
                    if chart_circ:
                        st.altair_chart(chart_circ, use_container_width=True)

                st.markdown("**Logged Momentary Check-ins**")
                m_table = pd.DataFrame(m_match).copy()
                m_table["Time (AM/PM)"] = pd.to_datetime(m_table["entry_timestamp"]).dt.strftime("%I:%M %p")
                cols_to_disp = [c for c in ["Time (AM/PM)", "mood_score_1_10", "anxiety_1_10", "energy_1_10", "dominant_emotion", "notes"] if c in m_table.columns]
                st.dataframe(
                    m_table[cols_to_disp].rename(
                        columns={
                            "mood_score_1_10": "Mood (Valence)",
                            "anxiety_1_10": "Anxiety",
                            "energy_1_10": "Energy",
                            "dominant_emotion": "Emotion",
                            "notes": "Notes",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info(f"No mood check-ins logged on {selected_date_str}.")

    with tab_week:
        st.markdown("### Weekly Cyclical Assessment")
        st.caption("Day-of-week patterns, weekday vs. weekend sleep shifts (social jetlag), and weekly sleep debt accumulation.")

        wc1, wc2 = st.columns(2)
        with wc1:
            dow_chart = viz.day_of_week_chart(df)
            if dow_chart:
                st.altair_chart(dow_chart, use_container_width=True)
            else:
                st.info("Insufficient data for day-of-week breakdown.")
        with wc2:
            comp_chart = viz.weekday_vs_weekend_comparison_chart(df)
            if comp_chart:
                st.altair_chart(comp_chart, use_container_width=True)
            else:
                st.info("Need both weekday and weekend entries for comparison.")

        st.markdown("#### Circadian Alignment & Sleep Debt")
        sj = metrics.compute_social_jetlag(df)
        sjc1, sjc2, sjc3 = st.columns(3)
        if sj["sufficient_data"]:
            sjc1.metric("Workday Mid-Sleep", sj["workday_midsleep_hhmm"])
            sjc2.metric("Weekend Mid-Sleep", sj["freeday_midsleep_hhmm"])
            sjc3.metric("Social Jetlag (|MSF - MSW|)", f"{sj['social_jetlag_hours']:.1f} h", delta="Notable shift" if sj["flagged"] else "Normal", delta_color="inverse")
        else:
            sjc1.metric("Workday Mid-Sleep", "—")
            sjc2.metric("Weekend Mid-Sleep", "—")
            sjc3.metric("Social Jetlag", "Need ≥ 4 nights")

        debt_chart = viz.sleep_debt_chart(df)
        if debt_chart:
            st.altair_chart(debt_chart, use_container_width=True)

    with tab_month:
        st.markdown("### Monthly Macro Assessment (30-Day Dynamics)")
        st.caption("Synchronized timelines, calendar heatmaps, mood distribution, and bedtime/waketime drift.")

        timeline = viz.sleep_mood_timeline(df)
        if timeline:
            st.altair_chart(timeline, use_container_width=True)
        else:
            st.info("No sleep or mood data available.")

        mc1, mc2 = st.columns(2)
        with mc1:
            cal_mood = viz.calendar_heatmap(df, "mood_mean", "Daily Mood Calendar Heatmap")
            if cal_mood:
                st.altair_chart(cal_mood, use_container_width=True)
        with mc2:
            cal_sleep = viz.calendar_heatmap(df, "tst_hours", "Daily Sleep Duration (TST) Calendar Heatmap")
            if cal_sleep:
                st.altair_chart(cal_sleep, use_container_width=True)

        mc3, mc4 = st.columns(2)
        with mc3:
            dist_chart = viz.mood_distribution(df)
            if dist_chart:
                st.altair_chart(dist_chart, use_container_width=True)
        with mc4:
            timing_chart = viz.sleep_timing_chart(df)
            if timing_chart:
                st.altair_chart(timing_chart, use_container_width=True)

    with tab_year:
        st.markdown("### Yearly & Longitudinal Assessment")
        st.caption("12-month overview of seasonal patterns, monthly aggregates, long-term sleep regularity, and questionnaire trends.")

        ann_chart = viz.annual_monthly_trajectory(df)
        if ann_chart:
            st.altair_chart(ann_chart, use_container_width=True)
        else:
            st.info("Annual macro trajectory requires entries across at least two distinct calendar months.")

        st.markdown("#### Longitudinal Questionnaire Progressions")
        q_cols = st.columns(3)
        for i, q_name in enumerate(["PHQ-9", "GAD-7", "ISI"]):
            with q_cols[i]:
                qdf = pd.DataFrame(database.get_questionnaires(_patient_id(), q_name, viewer_role="patient"))
                if not qdf.empty:
                    st.markdown(f"**{q_name} Trajectory**")
                    q_chart = viz.questionnaire_trajectory(qdf, q_name)
                    if q_chart:
                        st.altair_chart(q_chart, use_container_width=True)
                else:
                    st.info(f"No {q_name} records.")

    with tab_assoc:
        st.markdown("### Exploratory Sleep–Mood Associations")
        st.caption(
            "Correlations describe observed associations across paired observations; they do not establish causation "
            "or direct treatment targets. All values require at least 7 complete pairs."
        )
        heat = viz.correlation_heatmap(df)
        if heat:
            st.altair_chart(heat, use_container_width=True)
        else:
            st.info(f"At least {config.MIN_PAIRS_FOR_CORRELATION} paired observations are required for a correlation matrix.")

        st.markdown("#### Cross-Lagged Directional Scatter Plots")
        ac1, ac2 = st.columns(2)
        with ac1:
            sc1 = viz.cross_lagged_scatter(
                df, "tst_hours", "mood_next_day",
                "Sleep Duration Tonight vs. Next-Day Mood",
                "Sleep Tonight (TST hours)", "Next-Day Mood (/10)",
            )
            if sc1:
                st.altair_chart(sc1, use_container_width=True)
            else:
                st.info("Insufficient paired data for sleep → next-day mood.")
        with ac2:
            sc2 = viz.cross_lagged_scatter(
                df, "mood_mean", "sleep_quality_next_night",
                "Today's Mood vs. Tonight's Sleep Quality",
                "Today's Mood (/10)", "Tonight's Sleep Quality (/5)",
            )
            if sc2:
                st.altair_chart(sc2, use_container_width=True)
            else:
                st.info("Insufficient paired data for mood → night sleep quality.")

        st.markdown("#### Statistical Summaries with 95% Confidence Intervals")
        for line in analytics.association_summary(df):
            st.markdown(line)

    with tab_obs:
        st.markdown("### Evidence-Based Observations & Flagged Clinical Days")
        st.caption("Synthesized observations of patient-reported data and calculated metrics. Strictly non-diagnostic.")
        for line in analytics.all_observations(df, start, end):
            if line:
                st.markdown(line)


def sleep_entry():
    st.subheader("Log Sleep (Consensus Sleep Diary)")
    if not _check_consent():
        return
    st.caption(
        "Standardized morning self-monitoring based on the Consensus Sleep Diary (Carney et al., 2012). "
        "Best logged upon awakening. Estimates are acceptable; do not watch the clock during the night."
    )
    with st.form("sleep_form"):
        entry_date = st.date_input("Night date (the date you went to bed)", value=date.today() - timedelta(days=1))
        c1, c2 = st.columns(2)
        with c1:
            bedtime = st.time_input("Bedtime (lights out, AM/PM)", value=datetime.strptime("23:00", "%H:%M").time())
            sol = st.number_input("Minutes to fall asleep (Sleep Onset Latency - SOL)", min_value=0.0, max_value=600.0, value=20.0, step=5.0)
            awakenings = st.number_input("Number of awakenings during night", min_value=0, max_value=20, value=1, step=1)
        with c2:
            wake_time = st.time_input("Final wake-up time (AM/PM)", value=datetime.strptime("07:00", "%H:%M").time())
            waso = st.number_input("Minutes awake after falling asleep (WASO)", min_value=0.0, max_value=600.0, value=15.0, step=5.0)
            quality = st.slider("Sleep quality (1 = very poor, 5 = very good)", 1, 5, 3)
        context_selected = st.multiselect("Relevant night context factors (optional)", config.CONTEXT_TAG_CHOICES)
        notes = st.text_area("Notes (optional)", placeholder="e.g., late caffeine, stressful evening, early meeting, noise, ambient temperature")
        submitted = st.form_submit_button("Save sleep entry")
        if submitted:
            tags_str = ", ".join(context_selected) if context_selected else ""
            comb_notes = f"[Tags: {tags_str}] {notes}".strip() if tags_str else (notes or None)
            database.upsert_sleep_entry(
                _patient_id(), entry_date.isoformat(), bedtime.strftime("%H:%M"), wake_time.strftime("%H:%M"),
                sol, awakenings, waso, quality, comb_notes,
            )
            st.success("Sleep entry saved. You can update it by submitting the same night date again.")
            st.rerun()
    st.markdown("---")
    st.markdown("**Recent entries**")
    entries = database.get_sleep_entries(_patient_id())
    if entries:
        df = metrics.build_sleep_dataframe(entries[-14:])
        disp_cols = [c for c in ["date", "bedtime_ampm", "wake_time_ampm", "tst_hours", "sleep_efficiency_pct", "sleep_quality", "notes"] if c in df.columns]
        st.dataframe(
            df[disp_cols].rename(columns={
                "bedtime_ampm": "Bedtime (AM/PM)",
                "wake_time_ampm": "Wake Time (AM/PM)",
                "tst_hours": "TST (h)",
                "sleep_efficiency_pct": "SE (%)",
                "sleep_quality": "Quality (/5)",
            }),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**Delete an entry**")
        with st.form("sleep_delete_form"):
            dates = [e["entry_date"] for e in entries]
            pick = st.selectbox("Night date to delete", dates)
            confirm = st.checkbox("Yes, permanently delete this sleep entry.")
            if st.form_submit_button("Delete") :
                if confirm:
                    database.delete_sleep_entry(_patient_id(), pick)
                    st.success(f"Deleted sleep entry for {pick}.")
                    st.rerun()
                else:
                    st.error("Please tick confirm to delete.")
    else:
        st.info("No sleep entries yet.")


def mood_entry():
    st.subheader("Log Momentary Affect & Mood Check-In")
    if not _check_consent():
        return
    st.caption("Ecological Momentary Assessment (EMA). Multiple check-ins per day capture real-time affective dynamics without recall bias.")
    with st.form("mood_form"):
        c1, c2 = st.columns(2)
        with c1:
            entry_date = st.date_input("Date", value=date.today())
            entry_time = st.time_input("Time (AM/PM)", value=datetime.now().time())
            dominant_emotion = st.selectbox(
                "Primary Emotion / State",
                ["Calm", "Content", "Happy", "Energetic", "Anxious", "Sad", "Irritable", "Exhausted", "Numb"],
            )
        with c2:
            mood = st.slider("Valence / Pleasure (1 = very unpleasant/low, 10 = very pleasant/happy)", 1, 10, 5)
            anxiety = st.slider("Tension / Anxiety (1 = completely calm, 10 = extreme panic/tension)", 1, 10, 5)
            energy = st.slider("Arousal / Energy (1 = exhausted/deactivated, 10 = energetic/activated)", 1, 10, 5)

        tags_selected = st.multiselect("Contextual triggers / factors (optional)", config.CONTEXT_TAG_CHOICES)
        notes = st.text_area("Notes / context (optional)", placeholder="e.g., meeting presentation, argument, physical activity, meditation")
        submitted = st.form_submit_button("Save mood check-in")
        if submitted:
            ts = datetime.combine(entry_date, entry_time).isoformat()
            tags_str = ", ".join(tags_selected) if tags_selected else None
            database.add_mood_entry(
                _patient_id(), ts, entry_date.isoformat(),
                mood, anxiety, energy, notes or None,
                dominant_emotion=dominant_emotion,
                context_tags=tags_str,
            )
            st.success("Mood check-in recorded.")
            st.rerun()
    st.markdown("---")
    st.markdown("**Recent mood entries**")
    entries = database.get_mood_entries(_patient_id())
    if entries:
        df = pd.DataFrame(entries).tail(20).copy()
        df["Time (AM/PM)"] = pd.to_datetime(df["entry_timestamp"]).dt.strftime("%I:%M %p")
        disp_m_cols = [c for c in ["id", "entry_date", "Time (AM/PM)", "mood_score_1_10", "anxiety_1_10", "energy_1_10", "dominant_emotion", "notes"] if c in df.columns]
        st.dataframe(
            df[disp_m_cols].rename(columns={
                "mood_score_1_10": "Mood (/10)",
                "anxiety_1_10": "Anxiety (/10)",
                "energy_1_10": "Energy (/10)",
                "dominant_emotion": "Emotion",
            }),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**Delete an entry**")
        with st.form("mood_delete_form"):
            ids = {f"{e['entry_date']} at {pd.to_datetime(e['entry_timestamp']).strftime('%I:%M %p')} (Mood: {e['mood_score_1_10']}/10)": e["id"] for e in entries}
            pick = st.selectbox("Entry to delete", list(ids.keys()))
            confirm = st.checkbox("Yes, permanently delete this mood entry.")
            if st.form_submit_button("Delete"):
                if confirm:
                    database.delete_mood_entry(_patient_id(), ids[pick])
                    st.success("Deleted mood entry.")
                    st.rerun()
                else:
                    st.error("Please tick confirm to delete.")


def daily_notes():
    st.subheader("Daily notes and contextual observations")
    if not _check_consent():
        return
    with st.form("obs_form"):
        obs_date = st.date_input("Date", value=date.today())
        tags = st.text_input("Context tags (comma-separated, optional)", placeholder="work stress, exercise, alcohol, travel")
        notes = st.text_area("Observation", placeholder="Describe anything that may be relevant to your sleep or mood.")
        submitted = st.form_submit_button("Save observation")
        if submitted and notes.strip():
            database.add_observation(_patient_id(), obs_date.isoformat(), notes.strip(), tags.strip() or None)
            st.success("Observation saved.")
            st.rerun()
        elif submitted:
            st.error("Please enter an observation before saving.")
    st.markdown("---")
    st.markdown("**Recent observations**")
    obs = database.get_observations(_patient_id())
    if obs:
        for o in reversed(obs[-15:]):
            st.markdown(f"**{o['obs_date']}** — {o['notes']}")
            if o.get("context_tags"):
                st.caption(f"Tags: {o['context_tags']}")
    else:
        st.info("No observations yet.")


def questionnaires():
    st.subheader("Optional validated questionnaires")
    if not _check_consent():
        return
    st.info(
        "These questionnaires are screening and severity measures. They are not diagnoses. "
        "Scores are stored with the date so that your clinician can see change over time — "
        "unless below you choose to keep a particular questionnaire private from your clinician. "
        "You may complete them at your own pace."
    )
    instrument_key = st.selectbox("Choose a questionnaire", list(instruments.INSTRUMENTS.keys()))
    spec = instruments.INSTRUMENTS[instrument_key]
    st.markdown(f"### {spec['name']}")
    st.caption(spec["description"])
    share = st.checkbox("Share this entry with my linked clinician", value=True,
                        help="Leave unticked to keep this questionnaire entry visible only to you.")
    with st.form(f"form_{instrument_key}"):
        responses = {}
        for i, item in enumerate(spec["items"]):
            options = [label for _, label in spec["options"]]
            values = [value for value, _ in spec["options"]]
            choice = st.radio(f"{i + 1}. {item}", options=options, index=0, key=f"{instrument_key}_{i}")
            responses[i] = values[options.index(choice)]
        extra = {}
        if "extra_questions" in spec:
            cooc_options = [(0, "No"), (1, "Yes")]
            opts = [lbl for _, lbl in cooc_options]
            vals = [v for v, _ in cooc_options]
            cooc_choice = st.radio(spec["extra_questions"]["cooc"], options=opts, key=f"{instrument_key}_cooc")
            extra["cooc"] = vals[opts.index(cooc_choice)]
            prob = st.radio(spec["extra_questions"]["problem"], options=spec["extra_options"]["problem"],
                            index=0, key=f"{instrument_key}_prob")
            extra["problem"] = prob
        submitted = st.form_submit_button("Score and save")
        if submitted:
            if "extra_questions" in spec:
                responses.update(extra)
            total, band = spec["scorer"](responses)
            database.save_questionnaire(
                _patient_id(), instrument_key, datetime.now().isoformat(),
                responses, float(total), band, visible_to_clinician=share,
            )
            st.success(f"Saved. {instrument_key} total = {total} ({band}). This is not a diagnosis.")
            # Safety-sensitive item handling: PHQ-9 item 9.
            if instrument_key == "PHQ-9" and int(responses.get(config.PHQ9_ITEM9_INDEX, 0)) > 0:
                st.markdown(config.SAFETY_DISCLAIMER)
                st.markdown(config.CRISIS_RESOURCES)
            st.rerun()

    st.markdown("---")
    for key in instruments.INSTRUMENTS:
        qdf = pd.DataFrame(database.get_questionnaires(_patient_id(), key, viewer_role="patient"))
        if not qdf.empty:
            visible = qdf["visible_to_clinician"].sum()
            st.markdown(f"**{key} history** ({len(qdf)} entries; {len(qdf) - visible} private)")
            st.dataframe(
                qdf[["administered_at", "total_score", "severity_band", "visible_to_clinician"]],
                hide_index=True, use_container_width=True,
            )
            chart = viz.questionnaire_trajectory(qdf, key)
            if chart:
                st.altair_chart(chart, use_container_width=True)


def my_data():
    """Export and deletion controls, as promised in the consent text."""
    st.subheader("My data")
    if not _check_consent():
        return
    st.caption("You can download everything this app stores about you, or permanently delete your account and data.")

    if st.button("Prepare my data for download"):
        data = database.export_all_patient_data(_patient_id())
        st.download_button(
            "Download my data (JSON)",
            data=pd.Series(data).to_json(orient="records", indent=2).encode("utf-8"),
            file_name=f"my_sleep_mood_data_{date.today().isoformat()}.json",
            mime="application/json",
        )
        st.dataframe(pd.DataFrame(data["sleep_entries"]).tail(5), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame(data["mood_entries"]).tail(5), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("**Delete my account and all my data**")
    st.caption("This permanently removes all sleep, mood, note, and questionnaire records, and your login. It cannot be undone.")
    confirm_text = st.text_input("Type DELETE to enable deletion")
    if st.button("Delete my account and data", type="primary",
                 disabled=confirm_text != "DELETE"):
        database.delete_account_and_data(_patient_id())
        st.success("All of your data has been deleted.")
        st.session_state.user = None
        st.rerun()
