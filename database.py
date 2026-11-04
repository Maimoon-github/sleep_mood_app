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
            """
        )


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
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO mood_entries
              (patient_id, entry_timestamp, entry_date, mood_score_1_10,
               anxiety_1_10, energy_1_10, notes, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (patient_id, entry_timestamp, entry_date, mood, anxiety, energy, notes, now_iso()),
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
        for table in ("sleep_entries", "mood_entries", "observations", "questionnaires", "access_log"):
            conn.execute(f"DELETE FROM {table} WHERE patient_id = ?", (patient_id,))
        conn.execute("DELETE FROM access_log WHERE user_id = ?", (patient_id,))
        conn.execute("DELETE FROM patient_profiles WHERE user_id = ?", (patient_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (patient_id,))
