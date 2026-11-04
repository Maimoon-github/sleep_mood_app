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
    st.subheader("My sleep and mood dashboard")
    if not _check_consent():
        return

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
    m5.metric("Sleep regularity", f"{sri:.0f}" if sri is not None else "—")

    st.caption(
        "Blue/purple = yours (patient-reported). TST, SE, and TIB are calculated from your diary "
        "fields. Reference lines are population-level guides (AASM/SRS ≥ 7 h; SE ≥ 85%), not "
        "individual targets."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Timeline", "Sleep", "Mood", "Associations", "Observations"])
    with tab1:
        chart = viz.sleep_mood_timeline(df)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No sleep or mood data in this period.")
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            chart = viz.sleep_duration_chart(df)
            if chart:
                st.altair_chart(chart, use_container_width=True)
        with c2:
            chart = viz.sleep_efficiency_chart(df)
            if chart:
                st.altair_chart(chart, use_container_width=True)
        chart = viz.sleep_timing_chart(df)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        if not sleep_df.empty:
            st.markdown("**Recorded sleep entries**")
            st.dataframe(
                sleep_df[["date", "bedtime", "wake_time", "sol_min", "awakenings", "waso_min", "tst_hours", "sleep_efficiency_pct", "sleep_quality"]],
                use_container_width=True, hide_index=True,
            )
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            chart = viz.mood_trajectory(df, rolling_window=7)
            if chart:
                st.altair_chart(chart, use_container_width=True)
        with c2:
            chart = viz.mood_distribution(df)
            if chart:
                st.altair_chart(chart, use_container_width=True)
        chart = viz.calendar_heatmap(df, "mood_mean", "Daily mood calendar heatmap")
        if chart:
            st.altair_chart(chart, use_container_width=True)
    with tab4:
        st.markdown(
            "Correlations are **exploratory** and based on the data available. "
            "A correlation does not establish causation or a treatment target."
        )
        chart = viz.correlation_heatmap(df)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info(f"At least {config.MIN_PAIRS_FOR_CORRELATION} paired observations are needed for a correlation matrix.")
        st.markdown("**Association details**")
        for line in analytics.association_summary(df):
            st.markdown(line)
    with tab5:
        for line in analytics.all_observations(df, start, end):
            if line:
                st.markdown(line)


def sleep_entry():
    st.subheader("Log last night's sleep")
    if not _check_consent():
        return
    st.caption(
        "Record one entry per night, ideally in the morning. Estimates are acceptable; "
        "do not watch the clock during the night."
    )
    with st.form("sleep_form"):
        entry_date = st.date_input("Night date (the date you went to bed)", value=date.today() - timedelta(days=1))
        c1, c2 = st.columns(2)
        with c1:
            bedtime = st.time_input("Bedtime", value=datetime.strptime("23:00", "%H:%M").time())
            sol = st.number_input("Minutes to fall asleep (SOL)", min_value=0.0, max_value=600.0, value=20.0, step=5.0)
            awakenings = st.number_input("Number of awakenings", min_value=0, max_value=20, value=1, step=1)
        with c2:
            wake_time = st.time_input("Final wake time", value=datetime.strptime("07:00", "%H:%M").time())
            waso = st.number_input("Minutes awake after falling asleep (WASO)", min_value=0.0, max_value=600.0, value=15.0, step=5.0)
            quality = st.slider("Sleep quality (1 = very poor, 5 = very good)", 1, 5, 3)
        notes = st.text_area("Notes (optional)", placeholder="e.g., late caffeine, stressful evening, early meeting")
        submitted = st.form_submit_button("Save sleep entry")
        if submitted:
            database.upsert_sleep_entry(
                _patient_id(), entry_date.isoformat(), bedtime.strftime("%H:%M"), wake_time.strftime("%H:%M"),
                sol, awakenings, waso, quality, notes or None,
            )
            st.success("Sleep entry saved. You can update it by submitting the same night date again.")
            st.rerun()
    st.markdown("---")
    st.markdown("**Recent entries**")
    entries = database.get_sleep_entries(_patient_id())
    if entries:
        df = metrics.build_sleep_dataframe(entries[-14:])
        st.dataframe(df[["date", "bedtime", "wake_time", "tst_hours", "sleep_efficiency_pct", "sleep_quality"]], hide_index=True, use_container_width=True)
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
    st.subheader("Log a mood observation")
    if not _check_consent():
        return
    st.caption("You may log more than once per day. Each entry is timestamped and treated as a separate observation.")
    with st.form("mood_form"):
        c1, c2 = st.columns(2)
        with c1:
            entry_date = st.date_input("Date", value=date.today())
            entry_time = st.time_input("Time", value=datetime.now().time())
        with c2:
            mood = st.slider("Mood (1 = very low, 10 = very good)", 1, 10, 5)
            anxiety = st.slider("Anxiety (1 = none, 10 = extreme)", 1, 10, 5)
            energy = st.slider("Energy (1 = exhausted, 10 = energetic)", 1, 10, 5)
        notes = st.text_area("Notes / context (optional)", placeholder="e.g., argument, good news, physical activity")
        submitted = st.form_submit_button("Save mood entry")
        if submitted:
            ts = datetime.combine(entry_date, entry_time).isoformat()
            database.add_mood_entry(_patient_id(), ts, entry_date.isoformat(), mood, anxiety, energy, notes or None)
            st.success("Mood entry saved.")
            st.rerun()
    st.markdown("---")
    st.markdown("**Recent mood entries**")
    entries = database.get_mood_entries(_patient_id())
    if entries:
        df = pd.DataFrame(entries).tail(20)
        st.dataframe(df[["id", "entry_timestamp", "mood_score_1_10", "anxiety_1_10", "energy_1_10", "notes"]], hide_index=True, use_container_width=True)
        st.markdown("**Delete an entry**")
        with st.form("mood_delete_form"):
            ids = {f"{e['entry_timestamp']}  (mood {e['mood_score_1_10']})": e["id"] for e in entries}
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
