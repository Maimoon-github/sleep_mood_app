"""SQLite data-access layer for the sleep and mood application.

This module is intentionally dependency-light and uses only the standard
library plus no ORM. In production, replace SQLite with a server-side database
and add encryption at rest and in transit.

Access-control notes (research prototype):
- Patient data is only ever read by functions that take the patient's own id,
  or by clinician flows that first resolve the patient via the
  patient_profiles.clinician_id link. Questionnaire responses additionally
  require the viewer to be the patient or their linked clinician.
- Every clinician review writes to access_log.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import config


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('patient','clinician')),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS patient_profiles (
                user_id INTEGER PRIMARY KEY,
                display_name TEXT,
                clinician_id INTEGER,
                consent_given INTEGER NOT NULL DEFAULT 0,
                consent_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(clinician_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS sleep_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                entry_date TEXT NOT NULL,          -- night date (ISO date)
                bedtime TEXT,                       -- HH:MM
                wake_time TEXT,                     -- HH:MM
                sleep_onset_latency_min REAL,
                awakenings_count INTEGER,
                waso_min REAL,
                sleep_quality_1_5 INTEGER,
                notes TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(patient_id, entry_date),
                FOREIGN KEY(patient_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS mood_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                entry_timestamp TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                mood_score_1_10 INTEGER,
                anxiety_1_10 INTEGER,
                energy_1_10 INTEGER,
                notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(patient_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                obs_date TEXT NOT NULL,
                notes TEXT,
                context_tags TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(patient_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS questionnaires (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                instrument TEXT NOT NULL,
                administered_at TEXT NOT NULL,
                responses_json TEXT NOT NULL,
                total_score REAL NOT NULL,
                severity_band TEXT,
                created_at TEXT NOT NULL,
                visible_to_clinician INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY(patient_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                patient_id INTEGER,
                action TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS clinical_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clinician_id INTEGER NOT NULL,
                patient_id INTEGER NOT NULL,
                session_date TEXT NOT NULL,
                note_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(clinician_id) REFERENCES users(id),
                FOREIGN KEY(patient_id) REFERENCES users(id)
            );
            """
        )
        # Migrate schema safely if columns do not exist
        for col_def in [
            ("mood_entries", "dominant_emotion", "TEXT"),
            ("mood_entries", "context_tags", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE {col_def[0]} ADD COLUMN {col_def[1]} {col_def[2]}")
            except sqlite3.OperationalError:
                pass


# ---------- Users and profiles ----------

def create_user(username: str, password_hash: str, role: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
            (username.strip(), password_hash, role, now_iso()),
        )
        user_id = cur.lastrowid
        if role == "patient":
            conn.execute(
                "INSERT INTO patient_profiles (user_id, display_name) VALUES (?,?)",
                (user_id, username.strip()),
            )
    return user_id


def get_user_by_username(username: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_patient_profile(user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM patient_profiles WHERE user_id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def set_consent(patient_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE patient_profiles SET consent_given = 1, consent_at = ? WHERE user_id = ?",
            (now_iso(), patient_id),
        )


def link_patient_to_clinician(patient_id: int, clinician_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE patient_profiles SET clinician_id = ? WHERE user_id = ?",
            (clinician_id, patient_id),
        )


def unlink_patient(patient_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE patient_profiles SET clinician_id = NULL WHERE user_id = ?",
            (patient_id,),
        )


def get_patients_for_clinician(clinician_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.user_id, p.display_name, u.username
            FROM patient_profiles p
            JOIN users u ON u.id = p.user_id
            WHERE p.clinician_id = ?
            ORDER BY p.display_name
            """,
            (clinician_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def find_patient_by_username(username: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT u.id AS user_id, u.username, p.display_name, p.clinician_id
            FROM users u
            JOIN patient_profiles p ON p.user_id = u.id
            WHERE u.username = ? AND u.role = 'patient'
            """,
            (username.strip(),),
        ).fetchone()
    return dict(row) if row else None


# ---------- Sleep entries ----------

def upsert_sleep_entry(
    patient_id: int,
    entry_date: str,
    bedtime: str | None,
    wake_time: str | None,
    sol: float | None,
    awakenings: int | None,
    waso: float | None,
    quality: int | None,
    notes: str | None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO sleep_entries
              (patient_id, entry_date, bedtime, wake_time, sleep_onset_latency_min,
               awakenings_count, waso_min, sleep_quality_1_5, notes, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(patient_id, entry_date) DO UPDATE SET
              bedtime=excluded.bedtime,
              wake_time=excluded.wake_time,
              sleep_onset_latency_min=excluded.sleep_onset_latency_min,
              awakenings_count=excluded.awakenings_count,
              waso_min=excluded.waso_min,
              sleep_quality_1_5=excluded.sleep_quality_1_5,
              notes=excluded.notes
            """,
            (patient_id, entry_date, bedtime, wake_time, sol, awakenings, waso, quality, notes, now_iso()),
        )


def get_sleep_entries(patient_id: int, start: str | None = None, end: str | None = None) -> list[dict]:
    query = "SELECT * FROM sleep_entries WHERE patient_id = ?"
    params: list[Any] = [patient_id]
    if start:
        query += " AND entry_date >= ?"
        params.append(start)
    if end:
        query += " AND entry_date <= ?"
        params.append(end)
    query += " ORDER BY entry_date"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ---------- Mood entries ----------

def add_mood_entry(
    patient_id: int,
    entry_timestamp: str,
    entry_date: str,
    mood: int | None,
    anxiety: int | None,
    energy: int | None,
    notes: str | None,
    dominant_emotion: str | None = None,
    context_tags: str | None = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO mood_entries
              (patient_id, entry_timestamp, entry_date, mood_score_1_10,
               anxiety_1_10, energy_1_10, notes, dominant_emotion, context_tags, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (patient_id, entry_timestamp, entry_date, mood, anxiety, energy, notes, dominant_emotion, context_tags, now_iso()),
        )


def get_mood_entries(patient_id: int, start: str | None = None, end: str | None = None) -> list[dict]:
    query = "SELECT * FROM mood_entries WHERE patient_id = ?"
    params: list[Any] = [patient_id]
    if start:
        query += " AND entry_date >= ?"
        params.append(start)
    if end:
        query += " AND entry_date <= ?"
        params.append(end)
    query += " ORDER BY entry_timestamp"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ---------- Observations ----------

def add_observation(patient_id: int, obs_date: str, notes: str, tags: str | None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO observations (patient_id, obs_date, notes, context_tags, created_at) VALUES (?,?,?,?,?)",
            (patient_id, obs_date, notes, tags, now_iso()),
        )


def get_observations(patient_id: int, start: str | None = None, end: str | None = None) -> list[dict]:
    query = "SELECT * FROM observations WHERE patient_id = ?"
    params: list[Any] = [patient_id]
    if start:
        query += " AND obs_date >= ?"
        params.append(start)
    if end:
        query += " AND obs_date <= ?"
        params.append(end)
    query += " ORDER BY obs_date"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ---------- Questionnaires ----------

def save_questionnaire(
    patient_id: int,
    instrument: str,
    administered_at: str,
    responses: dict,
    total_score: float,
    severity_band: str | None,
    visible_to_clinician: bool = True,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO questionnaires
              (patient_id, instrument, administered_at, responses_json, total_score,
               severity_band, visible_to_clinician, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (patient_id, instrument, administered_at, json.dumps(responses), total_score,
             severity_band, int(visible_to_clinician), now_iso()),
        )


def get_questionnaires(
    patient_id: int,
    instrument: str | None = None,
    viewer_role: str = "clinician",
) -> list[dict]:
    """Return questionnaire rows.

    Clinician views must pass viewer_role='clinician'; the query then excludes
    entries the patient chose to keep private. Patients always see their own.
    """
    query = "SELECT * FROM questionnaires WHERE patient_id = ?"
    params: list[Any] = [patient_id]
    if viewer_role == "clinician":
        query += " AND visible_to_clinician = 1"
    if instrument:
        query += " AND instrument = ?"
        params.append(instrument)
    query += " ORDER BY administered_at"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        resp = json.loads(d["responses_json"])
        # JSON object keys are always strings; restore integer item indices so
        # safety-item detection (e.g., PHQ-9 item 9) works after storage.
        d["responses"] = {
            int(k) if isinstance(k, str) and k.isdigit() else k: v
            for k, v in resp.items()
        }
        out.append(d)
    return out


# ---------- Audit log ----------

def log_access(user_id: int, patient_id: int | None, action: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO access_log (user_id, patient_id, action, timestamp) VALUES (?,?,?,?)",
            (user_id, patient_id, action, now_iso()),
        )


def get_access_log(patient_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT a.*, u.username
            FROM access_log a
            JOIN users u ON u.id = a.user_id
            WHERE a.patient_id = ?
            ORDER BY a.timestamp DESC
            """,
            (patient_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------- Clinical Notes ----------

def add_clinical_note(
    clinician_id: int,
    patient_id: int,
    session_date: str,
    note_type: str,
    content: str,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO clinical_notes
              (clinician_id, patient_id, session_date, note_type, content, created_at)
            VALUES (?,?,?,?,?,?)
            """,
            (clinician_id, patient_id, session_date, note_type, content.strip(), now_iso()),
        )
        return cur.lastrowid


def get_clinical_notes(patient_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.*, u.username as clinician_username
            FROM clinical_notes c
            JOIN users u ON u.id = c.clinician_id
            WHERE c.patient_id = ?
            ORDER BY c.session_date DESC, c.created_at DESC
            """,
            (patient_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_clinical_note(note_id: int, clinician_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM clinical_notes WHERE id = ? AND clinician_id = ?",
            (note_id, clinician_id),
        )


# ---------- Export and deletion ----------

def export_all_patient_data(patient_id: int) -> dict[str, list[dict]]:
    """Everything the app stores about a patient, per table."""
    with get_conn() as conn:
        def _all(table: str, order: str) -> list[dict]:
            rows = conn.execute(f"SELECT * FROM {table} WHERE patient_id = ? ORDER BY {order}", (patient_id,)).fetchall()
            return [dict(r) for r in rows]

        profile = conn.execute(
            "SELECT * FROM patient_profiles WHERE user_id = ?", (patient_id,)
        ).fetchone()
        data = {
            "patient_profile": dict(profile) if profile else {},
            "sleep_entries": _all("sleep_entries", "entry_date"),
            "mood_entries": _all("mood_entries", "entry_timestamp"),
            "observations": _all("observations", "obs_date"),
            "questionnaires": _all("questionnaires", "administered_at"),
            "clinical_notes": _all("clinical_notes", "session_date"),
        }
    for q in data["questionnaires"]:
        raw = q.pop("responses_json")
        q["responses"] = {
            int(k) if isinstance(k, str) and k.isdigit() else k: v
            for k, v in (json.loads(raw) if isinstance(raw, str) else raw).items()
        }
    return data


def delete_sleep_entry(patient_id: int, entry_date: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM sleep_entries WHERE patient_id = ? AND entry_date = ?",
            (patient_id, entry_date),
        )


def delete_mood_entry(patient_id: int, entry_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM mood_entries WHERE patient_id = ? AND id = ?",
            (patient_id, entry_id),
        )


def delete_account_and_data(patient_id: int) -> None:
    """Remove all of a patient's records, profile row, and user row."""
    with get_conn() as conn:
        for table in ("sleep_entries", "mood_entries", "observations", "questionnaires", "access_log", "clinical_notes"):
            conn.execute(f"DELETE FROM {table} WHERE patient_id = ?", (patient_id,))
        conn.execute("DELETE FROM access_log WHERE user_id = ?", (patient_id,))
        conn.execute("DELETE FROM patient_profiles WHERE user_id = ?", (patient_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (patient_id,))


# ---------- Synthetic Clinical Demonstration Cohort ----------

def seed_demo_patients(clinician_id: int) -> list[str]:
    """Populate 4 realistic clinical phenotype demonstration patients.

    Generates longitudinal sleep, mood, questionnaire, and observation data
    for evidence-based clinical exploration (Day, Week, Month, Year).
    """
    import math
    from datetime import date, timedelta
    import auth

    created = []
    today = date.today()
    demo_pw = auth.hash_password("demo1234")

    # Cohort specifications
    cohort = [
        {
            "username": "alex_circadian",
            "name": "Alex Morgan (Circadian Delay & Social Jetlag)",
            "type": "delayed_phase",
        },
        {
            "username": "sarah_insomnia",
            "name": "Sarah Jenkins (Chronic Insomnia & Anxiety)",
            "type": "insomnia",
        },
        {
            "username": "david_cyclical",
            "name": "David Chen (Cyclical Affective Instability)",
            "type": "cyclical",
        },
        {
            "username": "elena_baseline",
            "name": "Elena Rostova (Stable Sleep Baseline)",
            "type": "baseline",
        },
    ]

    for p_spec in cohort:
        uname = p_spec["username"]
        existing = find_patient_by_username(uname)
        if existing:
            p_id = existing["user_id"]
        else:
            p_id = create_user(uname, demo_pw, "patient")
            created.append(p_spec["name"])

        # Update profile and link to clinician
        with get_conn() as conn:
            conn.execute(
                "UPDATE patient_profiles SET display_name = ?, clinician_id = ?, consent_given = 1, consent_at = ? WHERE user_id = ?",
                (p_spec["name"], clinician_id, now_iso(), p_id),
            )

        # Generate 90 days of rich longitudinal data
        ptype = p_spec["type"]
        for days_ago in range(89, -1, -1):
            cur_date = today - timedelta(days=days_ago)
            d_str = cur_date.isoformat()
            is_weekend = cur_date.weekday() in (4, 5) # Fri/Sat nights

            if ptype == "delayed_phase":
                # Weekdays: late bed (01:30), early wake (07:30) -> short sleep
                # Weekends: late bed (03:15), late wake (11:30) -> oversleep
                if is_weekend:
                    bed = "03:15"
                    wake = "11:30"
                    sol = 20.0
                    waso = 15.0
                    awakenings = 1
                    quality = 4
                    sleep_notes = "Weekend sleep-in to catch up"
                    mood = 7.5
                    anxiety = 3.5
                    energy = 7.0
                    emotion = "Relaxed"
                    tags = "Social engagement, Late heavy meal"
                else:
                    bed = "01:30"
                    wake = "07:30"
                    sol = 25.0
                    waso = 20.0
                    awakenings = 1
                    quality = 2 if cur_date.weekday() in (0, 1) else 3
                    sleep_notes = "Workday alarm; felt groggy"
                    # Monday/Tuesday mood dips
                    mood = 4.2 if cur_date.weekday() == 0 else 5.2 if cur_date.weekday() == 1 else 6.2
                    anxiety = 6.5 if cur_date.weekday() in (0, 1) else 4.5
                    energy = 3.5 if cur_date.weekday() in (0, 1) else 5.0
                    emotion = "Exhausted" if cur_date.weekday() == 0 else "Focused"
                    tags = "Caffeine (<6h bed), Screen in bed"

            elif ptype == "insomnia":
                # High SOL, high WASO, low efficiency, high anxiety
                var_f = math.sin(days_ago / 4.0) * 15.0
                sol = max(30.0, 50.0 + var_f)
                waso = max(25.0, 55.0 + var_f * 0.8)
                bed = "23:15"
                wake = "06:45"
                awakenings = 3
                quality = 1 if sol + waso > 100 else 2
                sleep_notes = "Tossing and turning, racing thoughts"
                mood = max(2.5, min(6.5, 4.5 - (sol + waso - 80) / 40.0))
                anxiety = max(4.0, min(9.5, 7.5 + (sol + waso - 80) / 30.0))
                energy = max(2.0, min(5.5, 3.8 - waso / 40.0))
                emotion = "Anxious" if anxiety > 7.0 else "Exhausted"
                tags = "High work stress, Screen in bed" if days_ago % 3 == 0 else "Caffeine (<6h bed)"

            elif ptype == "cyclical":
                # 30-day cyclical mood/sleep wave
                phase = math.sin(days_ago / 6.0)
                if phase > 0.4:
                    # Hypomanic / activation phase: short sleep, high energy
                    bed = "02:30"
                    wake = "06:45"
                    sol = 10.0
                    waso = 10.0
                    awakenings = 0
                    quality = 4
                    sleep_notes = "Felt wired, didn't need much sleep"
                    mood = 8.5
                    anxiety = 4.0
                    energy = 9.2
                    emotion = "Euphoric"
                    tags = "Late heavy meal, Vigorous exercise"
                elif phase < -0.4:
                    # Depressive phase: hypersomnia/long sleep, low energy, low mood
                    bed = "22:00"
                    wake = "08:45"
                    sol = 40.0
                    waso = 45.0
                    awakenings = 2
                    quality = 2
                    sleep_notes = "Slept long but woke unrefreshed"
                    mood = 3.2
                    anxiety = 6.8
                    energy = 2.2
                    emotion = "Sad"
                    tags = "Illness / physical pain"
                else:
                    # Euthymic / transition
                    bed = "23:45"
                    wake = "07:30"
                    sol = 20.0
                    waso = 20.0
                    awakenings = 1
                    quality = 3
                    sleep_notes = "Normal night"
                    mood = 6.0
                    anxiety = 4.5
                    energy = 5.8
                    emotion = "Calm"
                    tags = "Meditation / relaxation"

            else:  # baseline healthy
                bed = "23:15"
                wake = "07:15"
                sol = 12.0 + (days_ago % 5)
                waso = 10.0
                awakenings = 1
                quality = 4
                sleep_notes = "Good restful sleep"
                mood = 7.5 + (days_ago % 4) * 0.3
                anxiety = 2.5
                energy = 7.2
                emotion = "Content"
                tags = "Vigorous exercise" if days_ago % 2 == 0 else "Meditation / relaxation"

            upsert_sleep_entry(p_id, d_str, bed, wake, sol, awakenings, waso, quality, sleep_notes)

            # Mood check-in (timestamped morning + optional afternoon check-in)
            add_mood_entry(p_id, f"{d_str}T09:15:00", d_str, int(round(mood)), int(round(anxiety)), int(round(energy)), "Morning self-check", emotion, tags)
            if days_ago % 2 == 0:
                eve_mood = max(1, min(10, int(round(mood + (1 if is_weekend else -1)))))
                add_mood_entry(p_id, f"{d_str}T19:30:00", d_str, eve_mood, int(round(anxiety)), int(round(energy)), "Evening check-in", emotion, tags)

            # Observations every 7-10 days
            if days_ago % 10 == 0:
                add_observation(p_id, d_str, f"Context check for {cur_date.strftime('%B %d')}: {tags}", tags)

        # Questionnaires at 30-day intervals
        for offset, admin_d in [(60, today - timedelta(days=60)), (30, today - timedelta(days=30)), (0, today)]:
            dt_iso = admin_d.isoformat() + "T10:00:00"
            if ptype == "delayed_phase":
                save_questionnaire(p_id, "ISI", dt_iso, {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 1, 6: 1}, 9.0, "Subthreshold insomnia", visible_to_clinician=True)
                save_questionnaire(p_id, "GAD-7", dt_iso, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0}, 6.0, "Mild", visible_to_clinician=True)
            elif ptype == "insomnia":
                save_questionnaire(p_id, "ISI", dt_iso, {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 2, 6: 2}, 19.0, "Moderate clinical insomnia", visible_to_clinician=True)
                save_questionnaire(p_id, "GAD-7", dt_iso, {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1}, 13.0, "Moderate", visible_to_clinician=True)
                save_questionnaire(p_id, "PHQ-9", dt_iso, {0: 1, 1: 1, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0}, 9.0, "Mild", visible_to_clinician=True)
            elif ptype == "cyclical":
                save_questionnaire(p_id, "MDQ", dt_iso, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 0, 10: 1, 11: 1, "cooc": 1, "problem": "Serious problem"}, 10.0, "Screen positive", visible_to_clinician=True)
                depr_score = 15.0 if offset == 30 else 5.0
                depr_band = "Moderately severe" if offset == 30 else "Mild"
                save_questionnaire(p_id, "PHQ-9", dt_iso, {i: (2 if offset == 30 else 0) for i in range(9)}, depr_score, depr_band, visible_to_clinician=True)
            else:
                save_questionnaire(p_id, "ISI", dt_iso, {i: 0 for i in range(7)}, 0.0, "No clinically significant insomnia", visible_to_clinician=True)
                save_questionnaire(p_id, "PHQ-9", dt_iso, {i: 0 for i in range(9)}, 0.0, "Minimal", visible_to_clinician=True)

        # Seed an initial clinician consultation note
        add_clinical_note(
            clinician_id, p_id, (today - timedelta(days=14)).isoformat(),
            "Clinical Formulation",
            f"Baseline assessment note for {p_spec['name']}. Reviewing longitudinal self-monitoring diary. "
            "Discussing sleep scheduling, stimulus control, and correlation with daytime affect.",
        )

    return created
