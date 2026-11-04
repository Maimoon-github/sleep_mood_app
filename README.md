# 🌙 Sleep & Mood Longitudinal Assessment

A comprehensive, patient-centred, evidence-based Streamlit application for systematic sleep and mood self-monitoring, featuring an interactive longitudinal review dashboard for psychologists and mental health researchers.

It records patient-reported data, renders validated clinical visualizations, calculates evidence-based derivations (Sleep Efficiency, Regularity, Social Jetlag, Sleep Debt, Affective Instability MSSD), and produces **descriptive, uncertainty-qualified observations — never automated diagnoses**.

> **Clinical Decision Support Notice & Safety.** This application does not provide diagnoses, prescriptive treatments, or emergency care. Diagnostic formulation and risk assessment remain the sole clinical responsibility of the licensed practitioner. If you are in crisis, contact your local emergency services or a crisis line. Evidence-informed resources are shown in-app (`config.py` → `CRISIS_RESOURCES`).

---

## 1. Evidence Base & Scientific Literature

All design decisions, cutoffs, and calculations were derived from peer-reviewed scientific literature and clinical practice guidelines:

### Sleep Measurement & Chronobiology
| Metric / Practice | Evidence & Guideline | Clinical Relevance |
|---|---|---|
| **Consensus Sleep Diary (CSD-Core)**: Bedtime, SOL, awakenings count, WASO, wake time, sleep quality | **Carney et al., *SLEEP* 2012;35(2):287–302** | Gold-standard standardized prospective sleep diary across clinical insomnia research and AASM guidelines. |
| **Sleep Derivations**: `TIB = wake - bed`; `TST = TIB - SOL - WASO`; `SE = (TST/TIB) * 100` | Standard sleep medicine definitions (AASM, CBT-I protocol) | Sleep efficiency < 85% is clinically indicative of sleep maintenance difficulties or excessive time in bed. |
| **Duration Benchmark (≥ 7 h)** | **AASM & SRS joint consensus** — Watson et al., *SLEEP* 2015;38(6):843–844 | Adults should regularly sleep ≥ 7 h; used strictly as a population reference line, never an individual diagnostic rule. |
| **Subjective Diary Framing** | Kearns et al. 2022 (*JCSM*); Curtis et al. 2018 | Diary-vs-actigraphy disagreement is well documented; insomnia samples tend to underestimate sleep (paradoxical insomnia). Data is explicitly labelled diary-derived. |
| **Sleep Regularity Index (SRI)** | **Phillips et al., *Sci Rep* 2017**; **Windred et al., *SLEEP* 2024;47(1):zsad253** | Day-to-day concordance in sleep timing predicts cardiometabolic health and mortality independently of duration. |
| **Social Jetlag ($\Delta \text{MSF}_{\text{sc}}$)** | **Wittmann et al., 2006; Roenneberg et al., 2012** | Discrepancy between workday and free-day mid-sleep timing (≥ 1.5 h flag); reflects circadian misalignment. |
| **Sleep Debt Accumulation** | Standard sleep physiology (Van Dongen et al., 2003) | Cumulative nightly deficit against a baseline 7.5h reference. |

### Mood Measurement & Affective Dynamics
| Metric / Practice | Evidence | Clinical Relevance |
|---|---|---|
| **Ecological Momentary Assessment (EMA)** | **Shiffman, Stone & Hufford 2008; Moore et al. 2016** | Repeated real-time ratings (Valence, Tension/Anxiety, Energy) reduce retrospective recall bias. |
| **Circumplex Model of Affect** | **Russell 1980 (*JPSP*)** | 2D affective space: Valence (Displeasure ↔ Pleasure) $\times$ Arousal (Deactivation ↔ Activation). |
| **Affective Instability (MSSD)** | **Ebner-Priemer et al., 2007; Jahng et al., 2008; Trull et al., 2008** | Mean Squared Successive Difference: $\text{MSSD} = \frac{1}{N-1}\sum (x_{i+1} - x_i)^2$. Distinguishes sharp emotional swings from gradual trends. |

### Validated Standardized Questionnaires
| Instrument | Severity Bands / Scoring | Reference |
|---|---|---|
| **PHQ-9** (0–27) | 0–4: Minimal, 5–9: Mild, 10–14: Moderate, 15–19: Moderately Severe, 20–27: Severe (Cutoff ≥ 10) | Kroenke, Spitzer & Williams, *J Gen Intern Med* 2001 |
| **GAD-7** (0–21) | 0–4: Minimal, 5–9: Mild, 10–14: Moderate, 15–21: Severe (Cutoff ≥ 10) | Spitzer et al., *Arch Intern Med* 2006 |
| **ISI** (0–28) | 0–7: None, 8–14: Subthreshold, 15–21: Moderate insomnia, 22–28: Severe clinical insomnia | Bastien et al. 2001; Morin et al., *SLEEP* 2011 |
| **MDQ** (0–13) | Screen positive requires ≥ 7 symptoms **AND** co-occurrence **AND** moderate/serious impact | Hirschfeld et al., *Am J Psychiatry* 2000 |

> **Safety-Sensitive Item: PHQ-9 Item 9.** Any non-zero response to item 9 (thoughts of death or self-harm) is immediately surfaced with emergency and crisis helpline resources, prompting a clinical safety interview.

### Statistical & Methodological Discipline
- **Spearman Rank Correlation ($\rho$):** Used for non-normal, ordinal diary scales. Minimum threshold: $n \ge 7$ paired observations required before reporting.
- **Fisher's $z$ 95% Confidence Intervals:** Every reported correlation includes sample size $n$, exact $p$-value, and 95% CI to prevent small-sample overinterpretation.
- **Exploratory Labeling:** Output explicitly notes autocorrelation, shared seasonal/weekly trends, and unmeasured confounders. Causal claims are prohibited.

---

## 2. Architecture & Data Model

```text
sleep_mood_app/
├── app.py                 # Streamlit entry point, role-based auth & navigation, demo login
├── config.py              # Single source of truth: clinical cutoffs, bands, crisis notices
├── auth.py                # PBKDF2-SHA256 password hashing (260,000 rounds), signup/login
├── database.py            # SQLite data-access layer, clinical notes, demo cohort generator
├── instruments.py         # Standardized questionnaire definitions and validated scorers
├── metrics.py             # Sleep derivations, SRI, Social Jetlag, Sleep Debt, MSSD, Spearman + CIs
├── analytics.py           # Narrative observations, clinical report generation, data quality
├── visualizations.py      # Altair charts (Circumplex, timelines, heatmaps, day-of-week, debt)
├── tests/
│   ├── test_metrics.py    # 22 comprehensive unit tests (unittest & pytest compatible)
│   └── integration_check.py # End-to-end clinical workflow integration verification
└── views/
    ├── patient_views.py   # Consent gate, multi-scale dashboard, CSD sleep log, EMA mood log
    └── clinician_views.py # Patient list, longitudinal review (Day, Week, Month, Year), report generator
```

### Relational Schema (`sleep_mood.db`)
- `users`: `id`, `username`, `password_hash`, `role` (`patient` / `clinician`), `created_at`.
- `patient_profiles`: `user_id` (PK), `display_name`, `clinician_id`, `consent_given`, `consent_at`.
- `sleep_entries`: `id`, `patient_id`, `entry_date` (UNIQUE), `bedtime`, `wake_time`, `sol_min`, `awakenings_count`, `waso_min`, `sleep_quality_1_5`, `notes`, `created_at`.
- `mood_entries`: `id`, `patient_id`, `entry_timestamp`, `entry_date`, `mood_score_1_10`, `anxiety_1_10`, `energy_1_10`, `dominant_emotion`, `context_tags`, `notes`, `created_at`.
- `observations`: `id`, `patient_id`, `obs_date`, `notes`, `context_tags`, `created_at`.
- `questionnaires`: `id`, `patient_id`, `instrument`, `administered_at`, `responses_json`, `total_score`, `severity_band`, `visible_to_clinician`, `created_at`.
- `clinical_notes`: `id`, `clinician_id`, `patient_id`, `session_date`, `note_type`, `content`, `created_at`.
- `access_log`: `id`, `user_id`, `patient_id`, `action`, `timestamp`.

---

## 3. Setup & Execution Guide

### Using the `stream_app` Conda Environment
```bash
# 1. Activate the environment
conda activate stream_app

# 2. Navigate to directory
cd "/home/maimoon-amin/Antigravity code/streamlit apps/sleep_mood_app"

# 3. Run unit tests
python -m unittest discover tests

# 4. Run integration verification check
python tests/integration_check.py

# 5. Launch the Streamlit application
streamlit run app.py
```

### Demonstration Accounts & Pre-Loaded Clinical Phenotypes
The application features a 1-click **"⚡ Quick Demo Access"** tab on the login screen, allowing psychologists, researchers, and reviewers to instantly explore the longitudinal system with 4 realistic clinical phenotypes:
1. **Alex Morgan (`alex_circadian`)** — Delayed Sleep-Wake Phase & Social Jetlag (late bedtimes on weekdays, weekend oversleeping, Monday mood dips).
2. **Sarah Jenkins (`sarah_insomnia`)** — Chronic Insomnia & Daytime Anxiety (elevated SOL > 45m, frequent WASO, moderate ISI = 19, high anxiety).
3. **David Chen (`david_cyclical`)** — Cyclical Affective Instability & Bipolar Spectrum (alternating periods of short sleep with high energy vs depressive dips, MDQ screen positive).
4. **Elena Rostova (`elena_baseline`)** — Stable Sleep & Affect Baseline (regular sleep schedule, high efficiency SE = 94%, stable positive affect).

---

## 4. Multi-Scale Assessment & Visualization Strategy

The application provides dedicated interactive views across four distinct time scales:

1. **📅 Daily Assessment**:
   - Single-day 24h deep-dive.
   - Sleep Architecture breakdown (Bedtime, Wake time, TIB, TST, SOL, WASO, Awakenings, Efficiency %, Quality rating).
   - Intraday Momentary Affect Timeline (`intraday_mood_chart`): Morning, afternoon, and evening check-ins.
   - Russell Circumplex Model Plot (`circumplex_affect_chart`): Valence $\times$ Arousal quadrants with crosshairs.
   - Contextual factors (caffeine, exercise, screen time, medication) and patient notes.
2. **📊 Weekly Assessment**:
   - Day-of-week profile (Monday to Sunday) for sleep duration and mood.
   - Weekday vs. Weekend contrast bar chart (`weekday_vs_weekend_comparison_chart`).
   - Social Jetlag calculation ($|\text{MSF} - \text{MSW}|$ with ≥ 1.5 h clinical threshold alert).
   - Cumulative sleep debt accumulation curve (`sleep_debt_chart`).
3. **🗓️ Monthly Assessment**:
   - 30-day synchronized sleep duration and mood trajectory (`sleep_mood_timeline`).
   - Calendar Heatmaps for Daily Mood and Daily Sleep Duration.
   - Mood distribution histogram and floating bedtime/waketime drift chart.
   - Cross-lagged scatter plots (Sleep tonight $\to$ Next-day mood, and Mood today $\to$ Sleep quality tonight).
4. **📈 Yearly & Longitudinal Assessment**:
   - 12-month macro trajectory (`annual_monthly_trajectory`) displaying monthly average sleep and mood.
   - Standardized Questionnaire longitudinal progression (PHQ-9, GAD-7, ISI) with clinical cutoff bands.
   - Annual Sleep Regularity Index (SRI) trend and data completeness audit.
   - **One-Click Clinical Longitudinal Report Generator**: Generates formatted, non-diagnostic documentation ready for export or EMR charting.

---

## 5. Verification & Testing Results

- **Unit Tests (`tests/test_metrics.py`):** 22 tests passing (100%). Covers sleep metrics, overnight and after-midnight times, zero-clamping, regularity, social jetlag, sleep debt, MSSD, Spearman rank correlation, Fisher-z 95% CIs, seasonal confound detection, and questionnaire scoring bands.
- **Integration Check (`tests/integration_check.py`):** End-to-end verified. Tests patient signup, consent gate, data logging, analytics narrative generation, questionnaire clinician-scoping privacy, PHQ-9 item 9 detection, audit logging, JSON export, and hard deletion.
- **Browser & UI Verification:** Tested under Streamlit 1.66.1 in the `stream_app` environment.

---

## 6. Privacy, Ethics & Clinical Boundaries

- **Consent Gate:** Explicit informed consent is mandatory prior to recording any health data.
- **Patient Privacy Switches:** Patients can selectively toggle whether standardized questionnaire scores are shared with their linked clinician or kept private.
- **Right to Portability & Erasure:** 1-click comprehensive JSON data export and permanent account deletion (GDPR Art. 17 right to erasure).
- **Audit Logging:** Every clinician patient review is timestamped and recorded in `access_log`.
- **Strictly Non-Diagnostic Posture:** The application displays exploratory statistical associations and descriptive metrics to empower clinical reasoning, adhering to NHS DCB0129 and FDA SaMD clinical decision support principles.
