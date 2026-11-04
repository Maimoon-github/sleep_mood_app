# 🌙 Sleep & Mood Longitudinal Assessment

A patient-centred, research-prototype Streamlit application for systematic
sleep and mood self-monitoring, with a longitudinal review dashboard for
clinicians. It records what patients report, renders evidence-based
visualisations, and produces **descriptive, uncertainty-qualified
observations — never diagnoses**.

> **Safety.** This application does not provide diagnoses, treatment, or
> emergency care. If you are in crisis, contact your local emergency number or
> a crisis line. Evidence-informed resources shown in-app (config.py → `CRISIS_RESOURCES`).

---

## 1. Evidence base (research findings with references)

Design decisions were driven by verified sources; the logic of each is carried
into the code and disclaimers.

### Sleep measurement
| Practice adopted | Evidence |
|---|---|
| Diary fields: bedtime, sleep-onset latency, awakenings, WASO, final wake time, quality rating | **Consensus Sleep Diary** — Carney et al., *SLEEP* 2012;35(2):287–302 (the standardised prospective sleep self-monitoring instrument). |
| TST = TIB − SOL − WASO; SE = TST/TIB × 100 | Standard sleep-medicine derivations used with the consensus diary. |
| "≥ 7 h/night" used only as a **population reference line** | **AASM & SRS joint consensus** — Watson et al., *SLEEP* 2015;38(6):843–844: adults should sleep ≥ 7 h regularly; 6 h or less inappropriate to sustain health. |
| DIARY DATA ALWAYS LABELLED SUBJECTIVE | **Diary-vs-actigraphy disagreement is well documented**, largest for WASO; insomnia samples tend to *underestimate* sleep (paradoxical insomnia) — e.g., Kearns et al. 2022 (*JCSM*), Curtis et al. 2018; shepherded in-app as "diary-derived". |
| Sleep-timing regularity as a first-class metric (simplified 0–100 index) | Windred et al., *SLEEP* 2024;47(1):zsad253 (sleep regularity predicted mortality risk *beyond duration*); Fischer et al., *SLEEP* 2021 (metric properties). The full SRI requires epoch data; the app's proxy is computed from diary bed/wake concordance and labelled as such. |

### Mood measurement
| Practice adopted | Evidence |
|---|---|
| Multiple mood ratings per day with timestamps (mood, anxiety, energy; 1–10), within-day variability computed | **Ecological Momentary Assessment (EMA)** literature: momentary/repeated real-time ratings reduce recall bias vs. retrospective reports (Shiffman, Stone & Hufford 2008; Moore et al. 2016). |
| Daily aggregation = mean; variability kept and displayed | Within-person fluctuation is preserved and always shown with SDs, never collapsed to a single impression. |

### Questionnaires (paraphrased items; use licensed originals in production)
| Instrument | Bands/scores used | Source |
|---|---|---|
| **PHQ-9** (0–27) | 0/5/10/15/20 → minimal / mild / moderate / moderately severe / severe; ≥ 10 standard screening cutoff (88%/88%) | Kroenke, Spitzer & Williams, *J Gen Intern Med* 2001;16(9):606–13. |
| **GAD-7** (0–21) | 0/5/10/15 → minimal / mild / moderate / severe; ≥ 10 cutoff for further evaluation | Spitzer et al., *Arch Intern Med* 2006;166(10):1092–7. |
| **ISI** (0–28) | 0–7 / 8–14 / 15–21 / 22–28 | Bastien et al., *Sleep Med* 2001;2(4):297–307; Morin et al., *SLEEP* 2011;34(5):601–8. |
| **MDQ** (0–13) | Screen positive requires ≥ 7 symptoms **AND** co-occurrence **AND** moderate/serious impact | Hirschfeld et al., *Am J Psychiatry* 2000;157(11):1873–5. |

**Safety-sensitive item.** PHQ-9 **item 9** (thoughts of death or self-harm):
any non-zero response is always surfaced with crisis resources — never hidden
inside a total. It is presented as a prompt for a clinical conversation, not an
automated risk assessment (consistent with guidance, e.g., Simon et al. 2018 on
PHQ-9 item-9 responses and subsequent behaviour).

### Statistics
- **Spearman ρ** for monotonic sleep–mood associations (ordinal scales, non-normal diary data); a minimum of **7 paired observations** is required before any value is displayed.
- Every correlation is displayed with **n, exact p, and 95% Fisher-z confidence interval** — small-sample correlations without CIs are systematically over-interpreted.
- All association output is labelled **exploratory**: autocorrelation of repeated within-person measures, shared seasonal/work trends, and third variables (e.g., illness, stressors) confound them. No causal language is generated anywhere.

### Privacy & governance
- Health data is **special-category data** (GDPR Art. 9). Design follows the special-category posture: explicit consent *before* any health-data entry; data minimisation (only assessment-relevant fields); patient-initiated **export** and **deletion**; questionnaire entries can be kept **private from the linked clinician**; clinician access is **audit-logged** (who, what, when).
- The app is honest about its own limits: it labels itself a local SQLite research prototype in the sidebar and README; production deployment requires a server-side database, encryption at rest/in transit, and recognised compliance work.

---

## 2. Architecture & data model

### Module structure

```text
app.py                 # entry point: auth routing, role-based navigation
├── config.py          # single source of truth: thresholds, bands, safety text
├── auth.py            # PBKDF2-SHA256 password hashing/verification, signup/login
├── database.py        # SQLite data-access layer (std lib only, no ORM)
├── instruments.py     # PHQ-9/GAD-7/ISI/MDQ definitions + scorers
├── metrics.py         # sleep metrics, regularity, merges, Spearman + Fisher-z CI,
│                      # completeness, pre-specified flagged days
├── analytics.py       # transparent narrative summaries (no diagnostic language)
├── visualizations.py  # Altair charts (reference lines, calendars, heatmap)
└── views/
    ├── patient_views.py     # consent gate, dashboard, entries, notes, questionnaires,
    │                        # my-data (export/delete)
    └── clinician_views.py   # patient list/linking, longitudinal review, data quality
```

- **Patient UI flow:** consent → record sleep/mood/notes → optional questionnaires → dashboard → export/delete under *My Data*.
- **Clinician UI flow:** link patient by username → review timeline/sleep/mood/associations/questionnaires/notes → data-quality overview. All access logged.

### Data model (SQLite `sleep_mood.db`)

| Table | Key columns | Notes |
|---|---|---|
| `users` | id, username, password_hash, role (`patient`/`clinician`) | PBKDF2-SHA256, 260 000 iterations, per-user salt |
| `patient_profiles` | user_id (PK), display_name, clinician_id, consent_given, consent_at | consent required before any data entry |
| `sleep_entries` | patient_id, entry_date (unique per night), bedtime, wake_time, SOL, awakenings, WASO, quality 1–5, notes | one row per night, updatable |
| `mood_entries` | patient_id, entry_timestamp, entry_date, mood/anxiety/energy 1–10, notes | multiple per day allowed (EMA-style) |
| `observations` | patient_id, obs_date, notes, context_tags | free-text context |
| `questionnaires` | patient_id, instrument, administered_at, responses_json, total_score, severity_band, **visible_to_clinician** | patient privacy switch |
| `access_log` | user_id, patient_id, action, timestamp | every clinician review |

**Calculated metrics** (never raw): `TIB = wake − bed`; `TST = TIB − SOL − WASO` (floored at 0); `SE = TST/TIB`; sleep-timing **regularity** (0–100, diary bed/wake concordance across adjacent nights); **completeness** = recorded vs expected calendar days.

---

## 3. Setup & execution

```bash
# 1. Python 3.10+ recommended
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

- Database location defaults to `sleep_mood.db` next to the code; override with
  `SMA_DB_PATH=/path/to/app.db` (recommended outside any shared directory).
- Optional dev dependency: `pip install pytest` for the test suite.
- No external services are contacted; no data ever leaves the local machine.

---

## 4. Validation results

**Automated (pytest, 19 passing):**
- Sleep-metric derivations: overnight wrap-around bedtimes, after-midnight bedtimes, invalid inputs, TST floor at zero (5 tests).
- Regularity: 100.0 for identical nights; `None` when < 2 adjacent nights (2).
- Associations: insufficient-n suppression, perfect-correlation handling, missing-column safety, Fisher-z CI bracketing and symmetry, shared-seasonal-trend demonstration (4).
- Merges preserve all dates with outer joins; flag thresholds trip/hide correctly (3).
- Questionnaire scorers: PHQ-9/GAD-7/ISI band boundaries; MDQ full 3-condition screen rule (5).

**Integration (tests/integration_check.py, passing):** creates clinician + 2 patients; consents, links; writes 30 nights sleep+30 days mood with a constructed worsening patch; verifies completeness (30/30), regularity bounds, correlation machinery, narrative output completeness (Spearman + CI + regularity + flags present, no diagnostic language), questionnaire **privacy scoping** (clinician sees only shared entries), **PHQ-9 item-9 detection** from stored responses, audit log, full export, and hard deletion.

**Manual:** app boots headless (HTTP 200, no exceptions); consent gate blocks data entry until acknowledged; every dashboard figure renders only when sufficient data exists.

**Bugs found and fixed during verification:**
1. JSON round-trip turned integer item keys into strings → PHQ-9 item-9 safety detection silently failed; now normalised in the data layer (both views + export).
2. Export queries ran after the DB context closed (`Cannot operate on a closed database`); now executed inside the connection block.

---

## 5. Known limitations & future improvements

**Limitations (documented, deliberate):**
- **Not a diagnostic tool.** All insights are descriptive; no triage decisions are made by the app.
- **Diary subjectivity.** Diary-vs-actigraphy disagreement is real (especially WASO; underestimation in insomnia). No actigraphy/wearable ingestion yet.
- **Regularity is a proxy.** The published SRI needs epoch-level sleep/wake data; the diary version captures bed/wake timing only — no naps, no intraday fragmentation.
- **Correlations are confounded.** Within-person autocorrelation and shared trends are not removed; CIs are approximate Fisher-z, unsuited to very small n or extreme ρ.
- **Security posture is prototype-grade.** Local SQLite plaintext storage; PBKDF2 (not Argon2); no TLS, no rate limiting, no MFA.
- **Item wording is paraphrased** for copyright reasons — official licensed PHQ-9/GAD-7/ISI/MDQ versions required for production or research use.

**Recommended next steps (priority order):**
1. **Actigraphy/wearable ingestion** (e.g., FITbit/Apple Health exports) → true SRI, WASO objectivity, diary-vs-device discrepancy as an explicit, clinically interesting metric.
2. **Minimally important change** tracking on PHQ-9/GAD-7/ISI over time for clinician review of treatment response (validated change thresholds, e.g., ISI ≥ 6).
3. Age- and population-appropriate reference sets (adolescents, shift workers, perinatal), chosen per patient profile.
4. **Server-side deployment hardening**: Postgres + encryption at rest, Argon2id, TLS, rate limiting, MFA, session cookies with `HttpOnly`/`Secure`; formal privacy review (DPIA) and, where applicable, HIPAA/GDPR compliance programme.
5. Automated cron-style reminder nudges for morning diary completion (compliance is the biggest threat to longitudinal value).
6. Multilingual UI + validated translated instruments; accessibility audit (WCAG 2.1 AA) on colour encodings.
