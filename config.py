"""Application configuration and single source of truth for reference values.

Every threshold used by the analytics layer is defined here and must carry a
citable evidence note. Reference values are population-level guides; they are
never applied to an individual as a diagnostic rule.
"""

import os

APP_NAME = "Sleep & Mood Longitudinal Assessment"
APP_VERSION = "0.3.0 (evidence-based research prototype)"
DB_PATH = os.getenv("SMA_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sleep_mood.db"))

# ---------------------------------------------------------------------------
# Sleep reference values
# ---------------------------------------------------------------------------
# AASM/SRS joint consensus (Watson et al., 2015): adults 18+ should sleep >= 7 h
# per night on a regular basis. This is a population recommendation, not a
# per-person prescription.
SLEEP_DURATION_REFERENCE_HOURS = 7.0
SLEEP_DEBT_BASELINE_HOURS = 7.5
# Consensus sleep diary literature commonly contrasts sleep efficiency >= 85%
# with lower values; sleep efficiency is TST / time in bed.
SLEEP_EFFICIENCY_REFERENCE_PCT = 85.0
# Sleep-onset latency above ~30 min is conventionally considered prolonged in
# the diary literature (see Carney et al., 2012 and standard CBT-I practice).
SOL_REFERENCE_MIN = 30.0
# WASO above ~30-45 min is clinically notable in adult sleep assessment.
WASO_REFERENCE_MIN = 30.0
# Social jetlag (difference between weekend and weekday mid-sleep time) >= 1.5 h
# is associated with metabolic and affective vulnerability (Roenneberg et al., 2012).
SOCIAL_JETLAG_FLAG_HOURS = 1.5

# ---------------------------------------------------------------------------
# Flags used to surface "days worth reviewing" in the diary.
# These are pre-specified, transparent cutpoints for describing the data, not
# diagnostic rules. They deliberately err toward flagging rather than ignoring.
# ---------------------------------------------------------------------------
FLAG_TST_SHORT_HOURS = 6.0
FLAG_TST_LONG_HOURS = 10.0
FLAG_SLEEP_EFFICIENCY_LOW_PCT = 75.0
FLAG_SOL_LONG_MIN = 45.0
FLAG_WASO_HIGH_MIN = 60.0
FLAG_MOOD_VERY_LOW = 3.0        # on the 1..10 patient-reported daily mood scale
FLAG_MOOD_DROP = 3.0            # day-over-day decrease of >= 3 points
FLAG_ANXIETY_VERY_HIGH = 8.0    # on the 1..10 patient-reported anxiety scale
FLAG_ENERGY_VERY_LOW = 2.0      # on the 1..10 patient-reported energy scale

# Context tags for standardized clinical logging
CONTEXT_TAG_CHOICES = [
    "Caffeine (<6h bed)",
    "Alcohol (<4h bed)",
    "Vigorous exercise",
    "Late heavy meal",
    "Screen in bed",
    "Prescribed medication",
    "High work stress",
    "Social engagement",
    "Illness / physical pain",
    "Travel / jet lag",
    "Meditation / relaxation",
]

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
MIN_PAIRS_FOR_CORRELATION = 7   # smallest n at which a Spearman rho is shown
CORRELATION_ALPHA = 0.05        # for the Fisher-z 95% confidence interval

# ---------------------------------------------------------------------------
# Questionnaire severity bands (paraphrased instruments see instruments.py)
# Sources: Kroenke et al. 2001 (PHQ-9); Spitzer et al. 2006 (GAD-7);
# Bastien et al. 2001 / Morin et al. 2011 (ISI). Bands are severity descriptors
# validated against research samples, not individual diagnostic criteria.
# ---------------------------------------------------------------------------
PHQ9_BANDS = [
    (0, 4, "Minimal"),
    (5, 9, "Mild"),
    (10, 14, "Moderate"),
    (15, 19, "Moderately severe"),
    (20, 27, "Severe"),
]
GAD7_BANDS = [
    (0, 4, "Minimal"),
    (5, 9, "Mild"),
    (10, 14, "Moderate"),
    (15, 21, "Severe"),
]
ISI_BANDS = [
    (0, 7, "No clinically significant insomnia"),
    (8, 14, "Subthreshold insomnia"),
    (15, 21, "Moderate clinical insomnia"),
    (22, 28, "Severe clinical insomnia"),
]

# MDQ (Hirschfeld et al., 2000): positive screen requires >= 7 symptoms AND
# co-occurrence AND self-reported moderate/severe problems. Screening only.
MDQ_MIN_SYMPTOMS = 7

# The PHQ-9 item index (0-based) asking about thoughts of death or self-harm.
# Any non-zero response is surfaced with safety framing. See README.
PHQ9_ITEM9_INDEX = 8

# ---------------------------------------------------------------------------
# Safety and Clinical Governance
# ---------------------------------------------------------------------------
CRISIS_RESOURCES = (
    "If you are in crisis or need to talk with someone right now: contact your "
    "local emergency number or nearest emergency department; reach a person you "
    "trust and, where possible, stay with them. Crisis lines differ by country "
    "(examples: US/Canada 988; UK & Ireland, Samaritans 116 123; Australia, "
    "Lifeline 13 11 14; EU-wide emergency number 112). If you are unsure, search "
    "for a local crisis line."
)

SAFETY_DISCLAIMER = (
    "This application is a research prototype. It records and organises "
    "self-reported information; it does not detect emergencies, provide "
    "diagnoses, or replace care from a qualified clinician."
)

CLINICAL_DECISION_SUPPORT_NOTICE = (
    "CLINICAL DECISION SUPPORT NOTICE: All visualizations, metrics, and flagged patterns represent descriptive "
    "observations and statistical associations of self-reported data. They are not automated diagnoses, clinical "
    "predictions, or prescriptive treatment directives. Diagnostic formulation and risk assessment remain the "
    "sole clinical responsibility of the licensed practitioner."
)
