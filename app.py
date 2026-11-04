"""Main Streamlit entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st

import auth
import config
import database
from views import clinician_views, patient_views

st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

database.init_db()


def _init_session():
    defaults = {"user": None, "page": "Dashboard", "review_patient_id": None}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def login_page():
    st.title("🌙 Sleep & Mood Longitudinal Assessment")
    st.caption(
        "A research-prototype application for systematic sleep and mood self-monitoring. "
        "It does not provide diagnoses or emergency care. If you are in crisis, contact local emergency services."
    )
    tab_login, tab_signup = st.tabs(["Log in", "Create account"])
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
            if submitted:
                user = auth.login(username, password)
                if user:
                    st.session_state.user = user
                    st.session_state.page = "Dashboard"
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password (min 8 characters)", type="password")
            role = st.selectbox("I am a", ["patient", "clinician"])
            submitted = st.form_submit_button("Create account")
            if submitted:
                ok, msg = auth.signup(new_username, new_password, role)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
    with st.expander("If you are in crisis"):
        st.info(config.CRISIS_RESOURCES)


def sidebar():
    st.sidebar.title("🌙 Sleep & Mood")
    user = st.session_state.user
    st.sidebar.markdown(f"**{user['username']}**  \nRole: `{user['role']}`")
    if st.sidebar.button("Log out"):
        st.session_state.user = None
        st.session_state.review_patient_id = None
        st.session_state.page = "Dashboard"
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Research prototype. Health data is stored locally in SQLite. "
        "Not a production HIPAA/GDPR-compliant system."
    )


def patient_app():
    sidebar()
    pages = {
        "Dashboard": patient_views.dashboard,
        "Log Sleep": patient_views.sleep_entry,
        "Log Mood": patient_views.mood_entry,
        "Daily Notes": patient_views.daily_notes,
        "Questionnaires": patient_views.questionnaires,
        "My Data": patient_views.my_data,
    }
    choice = st.sidebar.radio("Navigate", list(pages.keys()), key="patient_nav")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Reminder**  \nRecord sleep in the morning. Bad nights are useful data. "
        "Correlations shown are exploratory, not diagnostic."
    )
    pages[choice]()


def clinician_app():
    sidebar()
    pages = {
        "Patient List": clinician_views.patient_list,
        "Patient Review": clinician_views.patient_review,
        "Data Quality": clinician_views.data_quality,
    }
    choice = st.sidebar.radio("Navigate", list(pages.keys()), key="clinician_nav")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Clinical use note**  \nInsights are evidence-based observations and potential associations, "
        "not automated diagnoses. Always integrate with clinical interview and judgement."
    )
    pages[choice]()


_init_session()

if st.session_state.user is None:
    login_page()
elif st.session_state.user["role"] == "patient":
    patient_app()
elif st.session_state.user["role"] == "clinician":
    clinician_app()
else:
    st.error("Unknown role. Please contact support.")
