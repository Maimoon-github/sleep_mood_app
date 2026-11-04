"""Clinician-facing views: patient list, longitudinal review, data quality, audit.

Every patient-review action is written to the access log. Clinicians only ever
see patients linked to them, and only questionnaire entries the patient chose
to share.
"""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

import analytics
import config
import database
import metrics
import visualizations as viz


def _clinician_id() -> int:
    return st.session_state.user["id"]


def patient_list():
    st.subheader("My linked patients")
    patients = database.get_patients_for_clinician(_clinician_id())
    if not patients:
        st.info("No patients are linked to you yet.")
        st.markdown("**Link a patient by username**")
        with st.form("link_patient"):
            username = st.text_input("Patient username")
            submitted = st.form_submit_button("Link patient")
            if submitted:
                p = database.find_patient_by_username(username)
                if not p:
                    st.error("No patient with that username was found.")
                elif p.get("clinician_id"):
                    st.warning("This patient is already linked to a clinician.")
                else:
                    database.link_patient_to_clinician(p["user_id"], _clinician_id())
                    st.success(f"Linked {p['display_name'] or p['username']}.")
                    st.rerun()
        return
    st.caption("Only patients linked to you are shown. Every review is written to an access log.")
    for p in patients:
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**{p['display_name'] or p['username']}**  \n`{p['username']}`")
        with cols[1]:
            if st.button("Review", key=f"review_{p['user_id']}"):
                st.session_state["review_patient_id"] = p["user_id"]
                st.session_state["review_patient_name"] = p["display_name"] or p["username"]
                st.rerun()
    st.markdown("---")
    st.markdown("**Link another patient**")
    with st.form("link_patient_more"):
        username = st.text_input("Patient username", key="link_more")
        submitted = st.form_submit_button("Link patient")
        if submitted:
            p = database.find_patient_by_username(username)
            if not p:
                st.error("No patient with that username was found.")
            elif p.get("clinician_id"):
                st.warning("This patient is already linked to a clinician.")
            else:
                database.link_patient_to_clinician(p["user_id"], _clinician_id())
                st.success(f"Linked {p['display_name'] or p['username']}.")
                st.rerun()


def patient_review():
    patient_id = st.session_state.get("review_patient_id")
    if not patient_id:
        st.info("Select a patient from the patient list.")
        return
    patient_name = st.session_state.get("review_patient_name", f"Patient {patient_id}")
    database.log_access(_clinician_id(), patient_id, "review_patient_dashboard")
    st.subheader(f"Longitudinal review: {patient_name}")
    st.caption(
        "This view presents patient-reported data, calculated metrics, and exploratory observations. "
        "It does not provide a diagnosis. Use clinical judgement and discuss findings with the patient."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        period = st.selectbox("Review period", ["Last 14 days", "Last 30 days", "Last 90 days", "Last 365 days", "Custom"], index=1)
    with col2:
        if period == "Custom":
            start = st.date_input("Start", value=date.today() - timedelta(days=30))
            end = st.date_input("End", value=date.today())
        else:
            days = {"Last 14 days": 14, "Last 30 days": 30, "Last 90 days": 90, "Last 365 days": 365}[period]
            end = date.today()
            start = end - timedelta(days=days - 1)

    sleep_entries = database.get_sleep_entries(patient_id, start.isoformat(), end.isoformat())
    mood_entries = database.get_mood_entries(patient_id, start.isoformat(), end.isoformat())
    sleep_df = metrics.build_sleep_dataframe(sleep_entries)
    mood_daily = metrics.build_mood_daily_dataframe(mood_entries)
    df = metrics.merge_sleep_mood(sleep_df, mood_daily)
    df = metrics.add_lagged_columns(df)

    c = metrics.completeness(df, start, end)
    sri = metrics.sleep_regularity_simplified(sleep_df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sleep nights", f"{c['sleep_nights']} / {c['expected_days']}")
    m2.metric("Mood days", f"{c['mood_days']} / {c['expected_days']}")
    if not sleep_df.empty and sleep_df["valid_sleep"].sum() > 0:
        m3.metric("Mean TST", f"{sleep_df.loc[sleep_df['valid_sleep'], 'tst_hours'].mean():.1f} h")
        m4.metric("Mean SE", f"{sleep_df.loc[sleep_df['valid_sleep'], 'sleep_efficiency_pct'].mean():.0f}%")
    else:
        m3.metric("Mean TST", "—")
        m4.metric("Mean SE", "—")
    m5.metric("Sleep regularity", f"{sri:.0f}" if sri is not None else "—")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Timeline", "Sleep", "Mood", "Associations", "Questionnaires", "Notes & observations"]
    )
    with tab1:
        chart = viz.sleep_mood_timeline(df)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data in this period.")
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
            st.dataframe(
                sleep_df[["date", "bedtime", "wake_time", "sol_min", "awakenings", "waso_min", "tst_hours", "sleep_efficiency_pct", "sleep_quality", "notes"]],
                hide_index=True, use_container_width=True,
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
        chart = viz.calendar_heatmap(df, "mood_mean", "Daily mood calendar")
        if chart:
            st.altair_chart(chart, use_container_width=True)
    with tab4:
        st.markdown(
            "Correlations are exploratory, based on available paired observations. "
            "They do not establish causation. Sample size, p-values, and 95% CIs are "
            "reported in the observations tab."
        )
        chart = viz.correlation_heatmap(df)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Insufficient paired observations for a correlation matrix.")
        st.markdown("**Association details**")
        for line in analytics.association_summary(df):
            st.markdown(line)
    with tab5:
        for key in ["PHQ-9", "GAD-7", "ISI", "MDQ"]:
            qdf = pd.DataFrame(database.get_questionnaires(patient_id, key, viewer_role="clinician"))
            if not qdf.empty:
                st.markdown(f"**{key}** (shared by patient)")
                st.dataframe(qdf[["administered_at", "total_score", "severity_band"]], hide_index=True, use_container_width=True)
                chart = viz.questionnaire_trajectory(qdf, key)
                if chart:
                    st.altair_chart(chart, use_container_width=True)
                if key == "PHQ-9":
                    positives = []
                    for _, row in qdf.iterrows():
                        if int(row["responses"].get(config.PHQ9_ITEM9_INDEX, 0)) > 0:
                            positives.append(row["administered_at"][:10])
                    if positives:
                        st.warning(
                            "PHQ-9 item 9 (thoughts of death or self-harm) was reported above zero "
                            f"on: {', '.join(positives)}. This is a flag for a clinical conversation, "
                            "not an automated assessment of risk."
                        )
        if not any(not pd.DataFrame(database.get_questionnaires(patient_id, k, viewer_role="clinician")).empty
                   for k in ["PHQ-9", "GAD-7", "ISI", "MDQ"]):
            st.info("No questionnaire data has been shared with you.")
    with tab6:
        st.markdown("**Patient observations and contextual notes**")
        obs = database.get_observations(patient_id, start.isoformat(), end.isoformat())
        if obs:
            for o in obs:
                st.markdown(f"**{o['obs_date']}** — {o['notes']}")
                if o.get("context_tags"):
                    st.caption(f"Tags: {o['context_tags']}")
        else:
            st.info("No free-text observations in this period.")
        st.markdown("---")
        st.markdown("**Generated observations (not diagnoses)**")
        for line in analytics.all_observations(df, start, end):
            if line:
                st.markdown(line)
        st.markdown("---")
        st.markdown("**Access log for this patient**")
        logs = database.get_access_log(patient_id)
        if logs:
            st.dataframe(pd.DataFrame(logs)[["username", "action", "timestamp"]], hide_index=True, use_container_width=True)
        else:
            st.info("No access events recorded.")


def data_quality():
    st.subheader("Data quality and audit overview")
    patients = database.get_patients_for_clinician(_clinician_id())
    if not patients:
        st.info("No linked patients.")
        return
    rows = []
    for p in patients:
        entries = database.get_sleep_entries(p["user_id"])
        mood = database.get_mood_entries(p["user_id"])
        rows.append({
            "patient": p["display_name"] or p["username"],
            "sleep_entries": len(entries),
            "mood_entries": len(mood),
            "first_sleep_date": entries[0]["entry_date"] if entries else None,
            "last_sleep_date": entries[-1]["entry_date"] if entries else None,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Missing data are not imputed. Encourage patients to complete the diary in the morning "
        "and to record bad nights as well as good nights."
    )
