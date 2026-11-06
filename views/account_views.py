"""Account lifecycle and settings view for both patients and clinicians.

Provides profile overview, credential management (password change), and
self-service account deletion with comprehensive cascade data purging.
"""

from datetime import datetime
import streamlit as st

import auth
import database


def account_settings():
    st.subheader("⚙️ Account Settings & Lifecycle Management")
    st.caption("Manage your profile, update authentication credentials, or permanently delete your account.")

    user = st.session_state.user
    if not user:
        st.error("No active session found. Please log in.")
        return

    # Refresh user record from database to ensure fresh state
    fresh_user = database.get_user_by_id(user["id"])
    if not fresh_user:
        st.error("Account record no longer exists. Logging out.")
        st.session_state.user = None
        st.rerun()
        return
    st.session_state.user = fresh_user
    user = fresh_user

    tab_profile, tab_password, tab_delete = st.tabs([
        "👤 Profile & Credentials",
        "🔑 Change Password",
        "⚠️ Delete Account",
    ])

    with tab_profile:
        st.markdown("### Profile Overview")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.text_input("Username", value=user["username"], disabled=True,
                          help="Usernames are permanently unique and case-insensitive.")
            st.text_input("Account Role", value=user["role"].capitalize(), disabled=True)
        with p_col2:
            created_dt = user.get("created_at", "")
            try:
                formatted_created = datetime.fromisoformat(created_dt).strftime("%B %d, %Y at %I:%M %p")
            except Exception:
                formatted_created = created_dt
            st.text_input("Member Since", value=formatted_created, disabled=True)
            st.text_input("Account ID", value=f"#{user['id']}", disabled=True)

        if user["role"] == "patient":
            prof = database.get_patient_profile(user["id"])
            curr_name = prof["display_name"] if prof and prof.get("display_name") else user["username"]
            st.markdown("#### Display Preferences")
            with st.form("update_profile_form"):
                new_name = st.text_input("Display Name", value=curr_name, max_chars=60,
                                         help="Name displayed to your linked psychologist.")
                if st.form_submit_button("Save Display Name"):
                    ok, msg = auth.update_display_name(user["id"], new_name)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        else:
            patients = database.get_patients_for_clinician(user["id"])
            st.markdown("#### Clinical Practice Scope")
            st.info(f"You currently have **{len(patients)}** patient(s) actively linked to your clinician account.")

    with tab_password:
        st.markdown("### Update Authentication Credentials")
        st.caption("Choose a strong password of at least 8 characters. Your new password will be valid across all devices.")

        with st.form("change_password_form"):
            curr_pw = st.text_input("Current Password", type="password")
            new_pw = st.text_input("New Password (min 8 characters)", type="password")
            confirm_pw = st.text_input("Confirm New Password", type="password")
            submit_pw = st.form_submit_button("Update Password")

            if submit_pw:
                if not curr_pw or not new_pw or not confirm_pw:
                    st.error("Please fill in all password fields.")
                else:
                    ok, msg = auth.change_password(user["id"], curr_pw, new_pw, confirm_pw)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    with tab_delete:
        st.markdown("### Permanent Account Deletion")
        st.error(
            "⚠️ **Warning: Account deletion is immediate, permanent, and irreversible.**\n\n"
            "In strict accordance with privacy and data protection rights:\n"
            "- **Zero Retained Data:** All associated records (sleep diary entries, momentary affect ratings, "
            "standardized questionnaires, clinical consultation notes, and audit logs) are permanently purged.\n"
            "- **No Recovery:** Once deleted, your account and credentials cannot be restored."
        )

        with st.form("delete_account_form"):
            del_pw = st.text_input("Confirm your password to authorize deletion", type="password")
            confirm_ack = st.checkbox(
                "I understand that this action is permanent and all my data will be immediately and irreversibly deleted."
            )
            submit_del = st.form_submit_button("Permanently Delete My Account", type="primary")

            if submit_del:
                if not confirm_ack:
                    st.error("You must check the confirmation checkbox to authorize account deletion.")
                elif not del_pw:
                    st.error("Please enter your current password to confirm deletion.")
                else:
                    ok, msg = auth.delete_account(user["id"], del_pw)
                    if ok:
                        st.session_state.user = None
                        st.session_state["flash_message"] = "Your account and all associated data have been permanently deleted."
                        st.rerun()
                    else:
                        st.error(msg)
