"""Erzeugt zwei synthetische Tabellen einer CKD-Kohorte – so, wie sie aus zwei Systemen kommen.

patients (Studien-CRF-Export):
- birth_date, enrol_date als Text im deutschen Format "TT.MM.JJJJ"
- sex uneinheitlich kodiert ("w", "W", " m", "weiblich", "männlich", ...)
- ckd_stage mit Groß-/Kleinschreibung und Leerzeichen ("g3a", "G3B ", ...)

labs (Laborsystem, Long-Format):
- eine Zeile pro Kreatinin-Abnahme, drawn_at als ISO-Text "JJJJ-MM-TT HH:MM"
- Abnahmen vor UND nach Einschluss, unsortiert; einige Patient:innen ohne Wert im Baseline-Fenster
"""
import numpy as np
import pandas as pd

STAGES = ["G1", "G2", "G3a", "G3b", "G4", "G5"]
# typisches Serum-Kreatinin [mg/dl] je Stadium (grob, nur für die Simulation)
_CREA = {"G1": 0.8, "G2": 1.0, "G3a": 1.4, "G3b": 1.8, "G4": 2.9}
_SEX_F = ["w", "W", "weiblich", " w", "f"]
_SEX_M = ["m", "M", " M", "männlich", "m "]


def _messy_stage(stage: str, rng) -> str:
    s = stage.lower() if rng.random() < 0.3 else stage
    if rng.random() < 0.2:
        s = s.upper()
    if rng.random() < 0.15:
        s = rng.choice([" ", ""]) + s + rng.choice([" ", "  "])
    return s


def make_patients(n: int = 300, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    birth = pd.Timestamp("1940-01-01") + pd.to_timedelta(rng.integers(0, 60 * 365, n), unit="D")
    enrol = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    stage = rng.choice(STAGES[:5], n, p=[0.10, 0.25, 0.25, 0.20, 0.20])   # kein G5 in dieser Kohorte
    female = rng.random(n) < 0.5
    return pd.DataFrame({
        "patient_id": np.arange(4001, 4001 + n),
        "birth_date": birth.strftime("%d.%m.%Y"),
        "enrol_date": enrol.strftime("%d.%m.%Y"),
        "sex": [rng.choice(_SEX_F) if f else rng.choice(_SEX_M) for f in female],
        "ckd_stage": [_messy_stage(s, rng) for s in stage],
    })


def make_labs(patients: pd.DataFrame | None = None, seed: int = 44) -> pd.DataFrame:
    """Kreatinin-Abnahmen zu den Patient:innen aus make_patients()."""
    if patients is None:
        patients = make_patients()
    rng = np.random.default_rng(seed)
    rows = []
    for pid, enrol_txt, stage_txt in patients[["patient_id", "enrol_date", "ckd_stage"]].itertuples(index=False):
        enrol = pd.to_datetime(enrol_txt, format="%d.%m.%Y")
        base = _CREA[stage_txt.strip().upper().replace("A", "a").replace("B", "b")]
        # Abnahmen vor Einschluss: bei ~15 % nur außerhalb des 30-Tage-Fensters
        if rng.random() < 0.15:
            pre_days = rng.integers(35, 120, rng.integers(0, 3))
        else:
            pre_days = rng.integers(0, 40, rng.integers(1, 4))
        post_days = -rng.integers(1, 366, rng.integers(2, 7))
        for d in np.concatenate([pre_days, post_days]):
            t = enrol - pd.Timedelta(days=int(d)) + pd.Timedelta(hours=int(rng.integers(7, 16)),
                                                                minutes=int(rng.integers(0, 60)))
            crea = base * np.exp(rng.normal(0, 0.12))
            rows.append((pid, t.strftime("%Y-%m-%d %H:%M"), round(float(crea), 2)))
    labs = pd.DataFrame(rows, columns=["patient_id", "drawn_at", "creatinine"])
    return labs.sample(frac=1, random_state=seed).reset_index(drop=True)   # bewusst unsortiert
