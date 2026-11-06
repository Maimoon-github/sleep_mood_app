"""Unit and lifecycle tests for persistent authentication, account management,
credential updates, and account deletion.

Runnable via:
    python -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bind a temporary database isolated from production data
_tmpdir = tempfile.mkdtemp()
os.environ["SMA_DB_PATH"] = os.path.join(_tmpdir, "test_auth_lifecycle.db")

import auth
import config
import database


class TestAuthLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database.init_db()

    def test_01_persistent_authentication_case_and_whitespace_invariant(self):
        # Sign up with mixed-case username
        ok, msg = auth.signup("AlexRivera", "SecurePassword123!", "patient")
        self.assertTrue(ok, f"Signup failed: {msg}")

        # Duplicate signup with different case must be prevented
        ok2, msg2 = auth.signup("alexrivera", "AnotherPassword123!", "patient")
        self.assertFalse(ok2)
        self.assertIn("already exists", msg2.lower())

        # Login must succeed with exact case
        u1 = auth.login("AlexRivera", "SecurePassword123!")
        self.assertIsNotNone(u1)
        self.assertEqual(u1["username"], "AlexRivera")

        # Login must succeed with lowercase
        u2 = auth.login("alexrivera", "SecurePassword123!")
        self.assertIsNotNone(u2)
        self.assertEqual(u2["id"], u1["id"])

        # Login must succeed with uppercase
        u3 = auth.login("ALEXRIVERA", "SecurePassword123!")
        self.assertIsNotNone(u3)
        self.assertEqual(u3["id"], u1["id"])

        # Login must succeed with leading/trailing whitespace (e.g. mobile autofill)
        u4 = auth.login("  alexrivera  ", "SecurePassword123!")
        self.assertIsNotNone(u4)
        self.assertEqual(u4["id"], u1["id"])

        # Wrong password must fail
        self.assertIsNone(auth.login("AlexRivera", "WrongPassword999"))

    def test_02_credentials_remain_valid_across_connection_restarts(self):
        # Retrieve user through a completely new connection
        with database.get_conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE", ("alexrivera",)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["role"], "patient")

        # Login succeeds seamlessly
        u = auth.login("alexrivera", "SecurePassword123!")
        self.assertIsNotNone(u)

    def test_03_password_update_lifecycle(self):
        u = auth.login("alexrivera", "SecurePassword123!")
        uid = u["id"]

        # Wrong current password
        ok, msg = auth.change_password(uid, "WrongOldPass!", "NewPassword456!", "NewPassword456!")
        self.assertFalse(ok)
        self.assertIn("does not match", msg)

        # Mismatched confirmation
        ok, msg = auth.change_password(uid, "SecurePassword123!", "NewPassword456!", "DifferentPass789!")
        self.assertFalse(ok)
        self.assertIn("do not match", msg)

        # Too short (< 8 chars)
        ok, msg = auth.change_password(uid, "SecurePassword123!", "short", "short")
        self.assertFalse(ok)
        self.assertIn("at least 8", msg)

        # Successful password change
        ok, msg = auth.change_password(uid, "SecurePassword123!", "NewPassword456!", "NewPassword456!")
        self.assertTrue(ok)

        # Old password must no longer work
        self.assertIsNone(auth.login("alexrivera", "SecurePassword123!"))

        # New password works cleanly across all case variations
        u_new = auth.login("ALEXRIVERA", "NewPassword456!")
        self.assertIsNotNone(u_new)
        self.assertEqual(u_new["id"], uid)

    def test_04_patient_account_deletion_cascades_all_data(self):
        # Create a patient and seed realistic user records
        ok, _ = auth.signup("patient_to_delete", "Password123!", "patient")
        self.assertTrue(ok)
        u = auth.login("patient_to_delete", "Password123!")
        pid = u["id"]

        database.set_consent(pid)
        database.upsert_sleep_entry(pid, "2026-11-01", "23:00", "07:00", 15, 0, 0, 4, "restful")
        database.add_mood_entry(pid, "2026-11-01T09:00:00", "2026-11-01", 8, 2, 7, "great", "Calm", "Relaxed")
        database.save_questionnaire(pid, "ISI", "2026-11-01T10:00:00", {0: 0}, 0.0, "None", True)

        # Confirm data exists prior to deletion
        self.assertEqual(len(database.get_sleep_entries(pid)), 1)
        self.assertEqual(len(database.get_mood_entries(pid)), 1)
        self.assertEqual(len(database.get_questionnaires(pid, viewer_role="patient")), 1)

        # Attempt deletion with incorrect password
        ok, msg = auth.delete_account(pid, "WrongPassword!")
        self.assertFalse(ok)
        self.assertIn("Incorrect password", msg)
        self.assertIsNotNone(database.get_user_by_id(pid))

        # Authorize deletion with valid password
        ok, msg = auth.delete_account(pid, "Password123!")
        self.assertTrue(ok)

        # Verify total purge of user and all associated records
        self.assertIsNone(database.get_user_by_id(pid))
        self.assertIsNone(database.get_patient_profile(pid))
        self.assertEqual(len(database.get_sleep_entries(pid)), 0)
        self.assertEqual(len(database.get_mood_entries(pid)), 0)
        self.assertEqual(len(database.get_questionnaires(pid, viewer_role="patient")), 0)

        # Login must now return None
        self.assertIsNone(auth.login("patient_to_delete", "Password123!"))

    def test_05_clinician_account_deletion_unlinks_patients_safely(self):
        # Create clinician and patient, then link
        auth.signup("dr_lifecycle", "DoctorPass123!", "clinician")
        auth.signup("patient_linked", "PatientPass123!", "patient")
        doc = auth.login("dr_lifecycle", "DoctorPass123!")
        pat = auth.login("patient_linked", "PatientPass123!")

        database.link_patient_to_clinician(pat["id"], doc["id"])
        database.add_clinical_note(doc["id"], pat["id"], "2026-11-01", "Intake", "Intake consultation note.")

        # Verify link and note exist
        linked = database.get_patients_for_clinician(doc["id"])
        self.assertEqual(len(linked), 1)
        self.assertEqual(len(database.get_clinical_notes(pat["id"])), 1)

        # Delete clinician account
        ok, msg = auth.delete_account(doc["id"], "DoctorPass123!")
        self.assertTrue(ok)

        # Clinician is removed
        self.assertIsNone(database.get_user_by_id(doc["id"]))
        self.assertIsNone(auth.login("dr_lifecycle", "DoctorPass123!"))

        # Patient profile remains safe and active, but clinician_id is safely reset to NULL
        pat_profile = database.get_patient_profile(pat["id"])
        self.assertIsNotNone(pat_profile)
        self.assertIsNone(pat_profile["clinician_id"])

        # Notes authored by the deleted clinician are cleaned up
        self.assertEqual(len(database.get_clinical_notes(pat["id"])), 0)

    def test_06_zero_seed_data_generated_on_init(self):
        # Ensure that running init_db() does not re-seed or create dummy users
        database.init_db()
        dummy_names = ["alex_circadian", "sarah_insomnia", "david_cyclical", "elena_baseline", "test_doc", "dr_clarke"]
        for dummy in dummy_names:
            self.assertIsNone(database.get_user_by_username(dummy), f"Found dummy user {dummy} in database!")


if __name__ == "__main__":
    unittest.main()
