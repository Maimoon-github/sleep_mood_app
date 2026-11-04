"""End-to-end integration check: patient signup → data entry → analytics →
clinician review workflow, run against a temporary SQLite database.

Run with:  python tests/integration_check.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_tmpdir = tempfile.mkdtemp()
os.environ["SMA_DB_PATH"] = os.path.join(_tmpdir, "integration.db")

import database  # noqa: E402  (import after env var so it binds the temp path)
import auth  # noqa: E402
import metrics  # noqa: E402
import analytics  # noqa: E402
import instruments  # noqa: E402
from datetime import date, timedelta  # noqa: E402

database.init_db()

# 1. Create a clinician and two patients.
clinician_id = database.create_user("dr_smith", auth.hash_password("password123"), "clinician")
pat1 = database.create_user("alice", auth.hash_password("password123"), "patient")
pat2 = database.create_user("bob", auth.hash_password("password123"), "patient")
assert database.find_patient_by_username("alice")["user_id"] == pat1

# 2. Consent + link.
database.set_consent(pat1)
assert database.get_patient_profile(pat1)["consent_given"] == 1
database.link_patient_to_clinician(pat1, clinician_id)
assert [p["user_id"] for p in database.get_patients_for_clinician(clinician_id)] == [pat1]
# bob is not linked; clinician must not find them via link flows.
assert database.find_patient_by_username("bob")["clinician_id"] is None

# 3. Patient logs 30 nights of sleep + mood, with a bad patch.
today = date.today()
for i in range(30):
    d = today - timedelta(days=i)
    bad = i > 20  # a run of poor nights/mood, for flags and correlation
    database.upsert_sleep_entry(
        pat1, d.isoformat(), "23:30" if not bad else "01:30", "07:00",
        60 if bad else 15, 3 if bad else 1, 45 if bad else 10,
        2 if bad else 4, "rough patch" if bad else None,
    )
    database.add_mood_entry(pat1, d.isoformat() + "T09:00:00", d.isoformat(),
                            4 if bad else 7, 7 if bad else 3, 4 if bad else 7, None)

sleep_df = metrics.build_sleep_dataframe(database.get_sleep_entries(pat1))
mood_df = metrics.build_mood_daily_dataframe(database.get_mood_entries(pat1))
df = metrics.merge_sleep_mood(sleep_df, mood_df)
df = metrics.add_lagged_columns(df)
assert len(df) == 30
assert sleep_df["valid_sleep"].all(), "all 30 nights should compute valid sleep metrics"

# 4. Metrics behave as expected on this constructed dataset.
sub = sleep_df[sleep_df["valid_sleep"]]
assert sub["tst_hours"].min() < 6.0  # bad patch dipped below 6 h
assert sub["tst_hours"].max() > 7.0
sri = metrics.sleep_regularity_simplified(sleep_df)
assert sri is not None and 0 <= sri <= 100
c = metrics.completeness(df, today - timedelta(days=29), today)
assert c["sleep_nights"] == 30 and c["sleep_completeness_pct"] == 100.0
assert c["mood_days"] == 30

# 5. Correlation machinery: the constructed dataset drifts in stage (bad patch
# at one end, good nights at the other), so ranks align almost perfectly.
# This is deliberate: it demonstrates the shared-trend confound that the
# narrative always labels ("exploratory, affected by autocorrelation...").
res = metrics.spearman_pair(df, "tst_hours", "mood_mean")
assert res is not None and res["sufficient"] and abs(res["rho"]) > 0.9

# 6. Analytics produce complete, non-diagnostic narrative.
obs = analytics.all_observations(df, today - timedelta(days=29), today)
assert len(obs) > 15
assert any("Spearman" in o for o in obs) and any("exploratory" in o for o in obs)
assert any("95% CI" in o for o in obs)
assert any("regularity" in o.lower() for o in obs)
fl = analytics.flagged_days_report(df)
assert any("**" in o for o in fl), "flagged days should list dates"
assert not any("diagnos" in o.lower() and "not" not in o.lower() for o in obs)

# 7. Questionnaires: save private + shared PHQ-9, verify clinician scoping.
total_bad, band_bad = instruments.score_phq9({i: 2 for i in range(9)})
assert total_bad == 18 and band_bad == "Moderately severe"
database.save_questionnaire(pat1, "PHQ-9", database.now_iso(), {i: 2 for i in range(9)}, 18.0, band_bad, visible_to_clinician=False)
database.save_questionnaire(pat1, "GAD-7", database.now_iso(), {i: 1 for i in range(7)}, 7.0, "Mild", visible_to_clinician=True)
seen_by_clinician = database.get_questionnaires(pat1, viewer_role="clinician")
seen_by_patient = database.get_questionnaires(pat1, viewer_role="patient")
assert len(seen_by_clinician) == 1 and seen_by_clinician[0]["instrument"] == "GAD-7"
assert len(seen_by_patient) == 2
# A PHQ-9 with a positive item 9 is detectable from stored responses.
database.save_questionnaire(pat1, "PHQ-9", database.now_iso(), {i: 1 for i in range(9)}, 9.0, "Mild", visible_to_clinician=True)
pos = [q for q in database.get_questionnaires(pat1, "PHQ-9", viewer_role="clinician")
       if int(q["responses"].get(8, 0)) > 0]
assert len(pos) == 1

# 8. Audit log + access scoping.
database.log_access(clinician_id, pat1, "review_patient_dashboard")
logs = database.get_access_log(pat1)
assert logs and logs[0]["username"] == "dr_smith"

# 9. Export contains everything; deletion removes everything.
export = database.export_all_patient_data(pat1)
assert len(export["sleep_entries"]) == 30 and len(export["mood_entries"]) == 30
assert len(export["questionnaires"]) == 3
database.delete_account_and_data(pat2)
assert database.get_user_by_id(pat2) is None
assert database.get_sleep_entries(pat2) == []

print("INTEGRATION CHECK PASSED: patient -> analytics -> clinician workflow verified.")
