"""Clinician-facing views: patient list, multi-scale longitudinal assessment,
clinical decision support, psychologist notes, report generator, and audit logging.

Strict clinical governance:
- All generated output is explicitly non-diagnostic decision support.
- Clinicians only ever see patients linked to them.
- Questionnaire entries respect patient privacy sharing switches.
- Every review action is audit-logged with user, patient, and timestamp.
"""

from datetime import date, datetime, timedelta

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
    st.subheader("Clinical Patient Cohort Management")
    st.info(config.CLINICAL_DECISION_SUPPORT_NOTICE)

    # Top action bar: Link new patient or Load Demo Cohort
    c1, c2 = st.columns([3, 2])
    with c2:
        if st.button("⚡ Load Clinical Demo Cohort", type="primary",
                     help="Populates 4 realistic clinical phenotype demonstration patients (Alex, Sarah, David, Elena) with 90 days of longitudinal data for immediate exploration."):
            created = database.seed_demo_patients(_clinician_id())
            st.success(f"Demonstration cohort loaded: {', '.join(created) if created else 'Already populated'}.")
            st.rerun()

    patients = database.get_patients_for_clinician(_clinician_id())
    if not patients:
        st.warning("No patients are linked to your clinician account yet.")
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

    st.markdown(f"**Linked Patients ({len(patients)})**")
    st.caption("Review patient records, assess longitudinal sleep architecture and affective dynamics. All reviews are audit-logged.")

    # Render a clinically informative card for each linked patient
    for p in patients:
        p_id = p["user_id"]
        p_name = p["display_name"] or p["username"]

        # Quick 30-day stats
        s_entries = database.get_sleep_entries(p_id)
        m_entries = database.get_mood_entries(p_id)
        s_df = metrics.build_sleep_dataframe(s_entries[-30:]) if s_entries else pd.DataFrame()
        m_df = metrics.build_mood_daily_dataframe(m_entries[-60:]) if m_entries else pd.DataFrame()

        # Check for PHQ-9 item 9 flag
        phq_list = database.get_questionnaires(p_id, "PHQ-9", viewer_role="clinician")
        has_item9_flag = any(int(q["responses"].get(config.PHQ9_ITEM9_INDEX, 0)) > 0 for q in phq_list)

        with st.container(border=True):
            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns([3, 2, 2, 2, 2])
            with r_col1:
                st.markdown(f"### {p_name}")
                st.caption(f"Username: `{p['username']}` | Total records: {len(s_entries)} sleep, {len(m_entries)} mood")
                if has_item9_flag:
                    st.error("⚠️ PHQ-9 Item 9 Flag: Thoughts of self-harm reported")
            with r_col2:
                if not s_df.empty and s_df["valid_sleep"].sum() > 0:
                    mean_tst = s_df.loc[s_df["valid_sleep"], "tst_hours"].mean()
                    mean_se = s_df.loc[s_df["valid_sleep"], "sleep_efficiency_pct"].mean()
                    st.metric("30-Day Mean TST", f"{mean_tst:.1f} h")
                    st.caption(f"Sleep Efficiency: {mean_se:.0f}%")
                else:
                    st.metric("30-Day Mean TST", "—")
            with r_col3:
                if not m_df.empty and "mood_mean" in m_df:
                    m_mean = m_df["mood_mean"].mean()
                    mssd = metrics.compute_affective_instability_mssd(m_df["mood_mean"])
                    st.metric("Recent Mood", f"{m_mean:.1f} / 10")
                    st.caption(f"Affective Instability (MSSD): {f'{mssd:.2f}' if mssd is not None else '—'}")
                else:
                    st.metric("Recent Mood", "—")
            with r_col4:
                # Latest questionnaires summary
                recent_q = database.get_questionnaires(p_id, viewer_role="clinician")
                if recent_q:
                    latest_q = recent_q[-1]
                    st.markdown(f"**Latest {latest_q['instrument']}**")
                    st.markdown(f"Score: `{latest_q['total_score']:.0f}` (*{latest_q['severity_band']}*)")
                else:
                    st.caption("No questionnaires shared")
            with r_col5:
                st.write("")
                if st.button("Review Patient", key=f"rev_{p_id}", type="primary", use_container_width=True):
                    st.session_state["review_patient_id"] = p_id
                    st.session_state["review_patient_name"] = p_name
                    # Switch page to patient review in session state
                    st.rerun()

    st.markdown("---")
    with st.expander("Link Another Patient"):
        with st.form("link_patient_more"):
            username = st.text_input("Patient username to link", key="link_more")
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
        st.info("Please select a patient from the Patient List to review.")
        return
    patient_name = st.session_state.get("review_patient_name", f"Patient {patient_id}")

    # Audit log every patient review event
    database.log_access(_clinician_id(), patient_id, "review_patient_dashboard")

    st.subheader(f"Longitudinal Review: {patient_name}")
    st.warning(config.CLINICAL_DECISION_SUPPORT_NOTICE)

    col1, col2 = st.columns([1, 1])
    with col1:
        period = st.selectbox("Review Time Horizon", ["Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days", "Last 365 days", "Custom"], index=3)
    with col2:
        if period == "Custom":
            start = st.date_input("Start date", value=date.today() - timedelta(days=90))
            end = st.date_input("End date", value=date.today())
        else:
            days = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30, "Last 90 days": 90, "Last 365 days": 365}[period]
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
    mssd = metrics.compute_affective_instability_mssd(df["mood_mean"]) if not df.empty and "mood_mean" in df else None

    # Key Summary Metrics
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Sleep Nights", f"{c['sleep_nights']} / {c['expected_days']}")
    m2.metric("Mood Days", f"{c['mood_days']} / {c['expected_days']}")
    if not sleep_df.empty and sleep_df["valid_sleep"].sum() > 0:
        m3.metric("Mean TST", f"{sleep_df.loc[sleep_df['valid_sleep'], 'tst_hours'].mean():.1f} h")
        m4.metric("Mean SE", f"{sleep_df.loc[sleep_df['valid_sleep'], 'sleep_efficiency_pct'].mean():.0f}%")
    else:
        m3.metric("Mean TST", "—")
        m4.metric("Mean SE", "—")
    m5.metric("Sleep Regularity", f"{sri:.0f} / 100" if sri is not None else "—")
    m6.metric("Affective Instab. (MSSD)", f"{mssd:.2f}" if mssd is not None else "—")

    # Multi-Scale Assessment Tabs
    tab_day, tab_week, tab_month, tab_year, tab_assoc, tab_quest, tab_notes, tab_report = st.tabs([
        "📅 Daily Assessment",
        "📊 Weekly Assessment",
        "🗓️ Monthly Assessment",
        "📈 Yearly Assessment",
        "🔗 Associations",
        "📋 Questionnaires",
        "🩺 Psychologist Notes",
        "📄 Clinical Report",
    ])

    with tab_day:
        st.markdown("### 24-Hour Day Level Clinical Assessment")
        st.caption("Granular inspection of overnight sleep architecture, within-day momentary affect (EMA), and contextual observations.")
        avail_dates = sorted(list(set(
            [e["entry_date"] for e in sleep_entries] + [e["entry_date"] for e in mood_entries]
        )), reverse=True)

        if not avail_dates:
            st.info("No sleep or mood entries in this period.")
        else:
            selected_date_str = st.selectbox("Select day to inspect", avail_dates, key="clin_day_select")
            sel_d = date.fromisoformat(selected_date_str)
            s_match = [e for e in sleep_entries if e["entry_date"] == selected_date_str]
            m_match = [e for e in mood_entries if e["entry_date"] == selected_date_str]

            st.markdown(f"#### Sleep Architecture: Night of {sel_d.strftime('%A, %B %d, %Y')}")
            if s_match:
                s = s_match[0]
                sm = metrics.compute_sleep_metrics(selected_date_str, s.get("bedtime"), s.get("wake_time"), s.get("sleep_onset_latency_min"), s.get("waso_min"))
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Bedtime → Wake", f"{s.get('bedtime', '—')} → {s.get('wake_time', '—')}")
                sc2.metric("Time in Bed (TIB)", f"{sm['tib_min'] / 60.0:.1f} h" if sm['valid'] else "—")
                sc3.metric("Total Sleep Time (TST)", f"{sm['tst_min'] / 60.0:.1f} h" if sm['valid'] else "—")
                sc4.metric("Sleep Efficiency", f"{sm['sleep_efficiency_pct']:.0f}%" if sm['valid'] else "—")
                sc5.metric("Sleep Quality", f"{s.get('sleep_quality_1_5', '—')} / 5")

                c_sol, c_waso, c_awak = st.columns(3)
                c_sol.info(f"**Sleep Onset Latency (SOL):** {s.get('sleep_onset_latency_min', 0):.0f} min (Ref ≤ 30 min)")
                c_waso.info(f"**WASO (Awake after onset):** {s.get('waso_min', 0):.0f} min (Ref ≤ 30 min)")
                c_awak.info(f"**Nighttime Awakenings:** {s.get('awakenings_count', 0)}")
                if s.get("notes"):
                    st.markdown(f"**Patient Diary Notes:** *{s['notes']}*")
            else:
                st.info(f"No sleep diary entry recorded for {selected_date_str}.")

            st.markdown("---")
            st.markdown(f"#### Daytime Momentary Affect Dynamics & Circumplex: {sel_d.strftime('%A, %B %d')}")
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

                st.markdown("**Logged Momentary Check-Ins**")
                m_table = pd.DataFrame(m_match)[["entry_timestamp", "mood_score_1_10", "anxiety_1_10", "energy_1_10", "dominant_emotion", "notes"]]
                st.dataframe(m_table, hide_index=True, use_container_width=True)
            else:
                st.info(f"No mood check-ins logged on {selected_date_str}.")

    with tab_week:
        st.markdown("### Weekly Cyclical Assessment")
        st.caption("Day-of-week profiles, workday vs. free day circadian shifts (social jetlag), and weekly sleep debt.")

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

        st.markdown("#### Circadian Alignment (Social Jetlag) & Sleep Debt Accumulation")
        sj = metrics.compute_social_jetlag(df)
        sjc1, sjc2, sjc3 = st.columns(3)
        if sj["sufficient_data"]:
            sjc1.metric("Workday Mid-Sleep", sj["workday_midsleep_hhmm"])
            sjc2.metric("Weekend Mid-Sleep", sj["freeday_midsleep_hhmm"])
            sjc3.metric("Social Jetlag (|MSF - MSW|)", f"{sj['social_jetlag_hours']:.1f} h",
                        delta="Circadian phase shift" if sj["flagged"] else "Within normal limits", delta_color="inverse")
        else:
            sjc1.metric("Workday Mid-Sleep", "—")
            sjc2.metric("Weekend Mid-Sleep", "—")
            sjc3.metric("Social Jetlag", "Need ≥ 4 nights")

        debt_chart = viz.sleep_debt_chart(df)
        if debt_chart:
            st.altair_chart(debt_chart, use_container_width=True)

    with tab_month:
        st.markdown("### Monthly Macro Assessment (30-Day Dynamics)")
        st.caption("Longitudinal trajectories, calendar heatmaps, mood distribution, and bedtime/waketime shifts.")

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
        st.caption("12-month macro trajectory, seasonal patterns, long-term sleep regularity, and questionnaire trends.")

        ann_chart = viz.annual_monthly_trajectory(df)
        if ann_chart:
            st.altair_chart(ann_chart, use_container_width=True)
        else:
            st.info("Annual macro trajectory requires entries across at least two distinct calendar months.")

        st.markdown("#### Longitudinal Questionnaire Progressions")
        q_cols = st.columns(3)
        for i, q_name in enumerate(["PHQ-9", "GAD-7", "ISI"]):
            with q_cols[i]:
                qdf = pd.DataFrame(database.get_questionnaires(patient_id, q_name, viewer_role="clinician"))
                if not qdf.empty:
                    st.markdown(f"**{q_name} Trajectory**")
                    q_chart = viz.questionnaire_trajectory(qdf, q_name)
                    if q_chart:
                        st.altair_chart(q_chart, use_container_width=True)
                else:
                    st.info(f"No {q_name} records shared.")

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

        st.markdown("#### Statistical Association Details (Exact p, n, and 95% Fisher-z CIs)")
        for line in analytics.association_summary(df):
            st.markdown(line)

    with tab_quest:
        st.markdown("### Standardized Questionnaires (Shared by Patient)")
        st.caption("Screening and severity measures only. Not stand-alone diagnoses.")
        for key in ["PHQ-9", "GAD-7", "ISI", "MDQ"]:
            qdf = pd.DataFrame(database.get_questionnaires(patient_id, key, viewer_role="clinician"))
            if not qdf.empty:
                st.markdown(f"#### {key}")
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
                        st.error(
                            "⚠️ SAFETY ALERT: PHQ-9 item 9 (thoughts of death or self-harm) was reported above zero "
                            f"on: {', '.join(positives)}. This is an immediate prompt for a clinical safety interview."
                        )
                        st.info(config.CRISIS_RESOURCES)

        all_shared = database.get_questionnaires(patient_id, viewer_role="clinician")
        if not all_shared:
            st.info("No questionnaire data has been shared with you by this patient.")

    with tab_notes:
        st.markdown("### Psychologist Clinical Consultation Notes & Action Plan")
        st.caption("Record session formulations, behavioral interventions (CBT-I / CBT-D components), and clinical notes.")

        with st.form("clinical_note_form"):
            n_date = st.date_input("Session / Formulation Date", value=date.today())
            n_type = st.selectbox("Note Type", ["Clinical Formulation", "Session Note", "Behavioral Intervention Plan", "Safety Check", "Follow-up Observation"])
            n_content = st.text_area("Clinical Note Content", placeholder="Enter clinical formulation, sleep hygiene targets, cognitive reframing, or session summary...")
            submitted = st.form_submit_button("Save Clinical Note")
            if submitted and n_content.strip():
                database.add_clinical_note(_clinician_id(), patient_id, n_date.isoformat(), n_type, n_content.strip())
                st.success("Clinical note recorded.")
                st.rerun()

        st.markdown("---")
        st.markdown("**Recorded Clinical Notes**")
        notes = database.get_clinical_notes(patient_id)
        if notes:
            for note in notes:
                with st.expander(f"{note['session_date']} — {note['note_type']} (by {note.get('clinician_username', 'clinician')})", expanded=True):
                    st.markdown(note["content"])
                    if st.button("Delete Note", key=f"del_note_{note['id']}"):
                        database.delete_clinical_note(note["id"], _clinician_id())
                        st.success("Note deleted.")
                        st.rerun()
        else:
            st.info("No clinical notes recorded yet for this patient.")

    with tab_report:
        st.markdown("### Comprehensive Clinical Longitudinal Report Generator")
        st.caption("Generate a formatted clinical summary report for electronic medical records (EMR) or consultation review.")

        q_list = database.get_questionnaires(patient_id, viewer_role="clinician")
        c_notes = database.get_clinical_notes(patient_id)
        report_md = analytics.generate_clinical_report_markdown(patient_name, df, q_list, c_notes, start, end)

        st.download_button(
            label="📥 Download Clinical Report (Markdown)",
            data=report_md.encode("utf-8"),
            file_name=f"clinical_report_{patient_name.replace(' ', '_')}_{date.today().isoformat()}.md",
            mime="text/markdown",
            type="primary",
        )

        st.markdown("---")
        st.markdown(report_md)


def data_quality():
    st.subheader("Data Quality, Governance & Audit Overview")
    st.info(config.CLINICAL_DECISION_SUPPORT_NOTICE)

    patients = database.get_patients_for_clinician(_clinician_id())
    if not patients:
        st.info("No linked patients.")
        return

    rows = []
    for p in patients:
        entries = database.get_sleep_entries(p["user_id"])
        mood = database.get_mood_entries(p["user_id"])
        quests = database.get_questionnaires(p["user_id"], viewer_role="clinician")
        rows.append({
            "Patient": p["display_name"] or p["username"],
            "Username": p["username"],
            "Sleep Diary Entries": len(entries),
            "Mood Check-ins": len(mood),
            "Shared Questionnaires": len(quests),
            "Earliest Record": entries[0]["entry_date"] if entries else "None",
            "Latest Record": entries[-1]["entry_date"] if entries else "None",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Clinician Audit Access Log")
    st.caption("Every patient review action is recorded with clinician identity, target patient ID, action, and timestamp (GDPR Art. 30 compliance).")
    selected_p = st.selectbox("View audit log for patient", [p["display_name"] or p["username"] for p in patients])
    p_obj = next((p for p in patients if (p["display_name"] or p["username"]) == selected_p), None)
    if p_obj:
        logs = database.get_access_log(p_obj["user_id"])
        if logs:
            st.dataframe(pd.DataFrame(logs)[["username", "action", "timestamp"]], hide_index=True, use_container_width=True)
        else:
            st.info("No access events logged for this patient.")
