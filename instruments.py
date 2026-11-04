"""Validated questionnaire definitions and scoring.

Safety and licensing notes:
- Item wording for PHQ-9, GAD-7, ISI and MDQ is paraphrased to avoid copyright
  concerns. In a production clinical deployment, use the official, licensed
  versions of each instrument and follow the publisher's scoring instructions.
- Severity bands describe research-sample distributions (sources in
  config.py); they are not diagnostic criteria or treatment rules.
- PHQ-9 item 9 (index 8) is treated as a safety-sensitive item whose positive
  response is always surfaced, never hidden in a total score.
"""

import config
from typing import Callable

PHQ9_ITEMS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself, or that you are a failure, or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed, or the opposite — being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead or of hurting yourself in some way",
]

GAD7_ITEMS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid, as if something awful might happen",
]

ISI_ITEMS = [
    "Difficulty falling asleep",
    "Difficulty staying asleep",
    "Problem waking up too early",
    "Dissatisfaction with current sleep pattern",
    "Interference with daily functioning",
    "Noticeability of sleep problem to others",
    "Worry or distress about current sleep problem",
]

# MDQ symptom questions (paraphrased; binary yes/no), per Hirschfeld et al. 2000.
MDQ_ITEMS = [
    "Suddenly more self-confident than usual",
    "Much less need for sleep than usual, yet feeling rested",
    "Much more talkative than usual, or speaking fast and hard to interrupt",
    "Thoughts racing or jumping quickly",
    "Easily distracted that it interfered with things",
    "Marked increase in activity at work, school, or socially",
    "Marked increase in energy or activity that pleased you",
    "Marked increase in physical or social activity (e.g., shopping, calls, projects)",
    "Doing things unusual for you or that others noted as excessive or foolish",
    "Doing pleasurable or risky activities that could get you into trouble (e.g., spending sprees, impulsive sex)",
    "Much more irritable than unusual, or starting conflicts",
    "Feeling unusually high or 'wired', so that others thought you were not your normal self",
]
MDQ_COOC = "Have several of these happened during the same period of time?"
MDQ_PROBLEM = "How much of a problem did any of this cause (e.g., unable to work, family or money trouble)?"
MDQ_PROBLEM_OPTIONS = ["No problem", "Minor problem", "Moderate problem", "Serious problem"]

LIKERT_0_3 = [
    (0, "Not at all"),
    (1, "Several days"),
    (2, "More than half the days"),
    (3, "Nearly every day"),
]

ISI_0_4 = [
    (0, "None"),
    (1, "Mild"),
    (2, "Moderate"),
    (3, "Severe"),
    (4, "Very severe"),
]

YES_NO = [(0, "No"), (1, "Yes")]


def _band(total: float, bands: list[tuple[float, float, str]]) -> str:
    for low, high, label in bands:
        if low <= total <= high:
            return label
    return "Outside defined range"


def score_phq9(responses: dict[int, int]) -> tuple[int, str]:
    total = sum(int(responses.get(i, 0)) for i in range(9))
    return total, _band(total, config.PHQ9_BANDS)


def score_gad7(responses: dict[int, int]) -> tuple[int, str]:
    total = sum(int(responses.get(i, 0)) for i in range(7))
    return total, _band(total, config.GAD7_BANDS)


def score_isi(responses: dict[int, int]) -> tuple[int, str]:
    total = sum(int(responses.get(i, 0)) for i in range(7))
    return total, _band(total, config.ISI_BANDS)


def score_mdq(responses: dict) -> tuple[int, str]:
    """MDQ screen.

    Total = number of symptoms answered "yes" (0..13).
    Band  = "Screen positive" if symptoms >= MDQ_MIN_SYMPTOMS AND
            co-occurrence is Yes AND problem is Moderate or Serious,
            else "Screen negative".
    """
    n_items = len(MDQ_ITEMS)
    total = sum(int(responses.get(i, 0)) for i in range(n_items))
    cooc = int(responses.get("cooc", 0))
    problem = str(responses.get("problem", MDQ_PROBLEM_OPTIONS[0]))
    if total >= config.MDQ_MIN_SYMPTOMS and cooc == 1 and problem in ("Moderate problem", "Serious problem"):
        band = "Screen positive"
    else:
        band = "Screen negative"
    return total, band


INSTRUMENTS: dict[str, dict] = {
    "PHQ-9": {
        "name": "Patient Health Questionnaire-9 (PHQ-9)",
        "description": "Nine items assessing depressive symptoms over the past two weeks. A screening and severity measure, not a diagnosis.",
        "items": PHQ9_ITEMS,
        "options": LIKERT_0_3,
        "scorer": score_phq9,
        "max_score": 27,
        "safety_items": {config.PHQ9_ITEM9_INDEX: "thoughts of death or self-harm"},
    },
    "GAD-7": {
        "name": "Generalized Anxiety Disorder-7 (GAD-7)",
        "description": "Seven items assessing anxiety symptoms over the past two weeks. A screening and severity measure, not a diagnosis.",
        "items": GAD7_ITEMS,
        "options": LIKERT_0_3,
        "scorer": score_gad7,
        "max_score": 21,
        "safety_items": {},
    },
    "ISI": {
        "name": "Insomnia Severity Index (ISI)",
        "description": "Seven items assessing insomnia severity and daytime impact. Not a stand-alone diagnostic tool.",
        "items": ISI_ITEMS,
        "options": ISI_0_4,
        "scorer": score_isi,
        "max_score": 28,
        "safety_items": {},
    },
    "MDQ": {
        "name": "Mood Disorder Questionnaire (MDQ)",
        "description": "Thirteen yes/no items describing periods of unusually elevated or irritable mood, plus co-occurrence and impact. A screening aid only; a positive screen is not a diagnosis and does not mean treatment is required.",
        "items": MDQ_ITEMS,
        "options": YES_NO,
        "scorer": score_mdq,
        "max_score": 13,
        "extra_questions": {"cooc": MDQ_COOC, "problem": MDQ_PROBLEM},
        "extra_options": {"problem": MDQ_PROBLEM_OPTIONS},
        "safety_items": {},
    },
}
