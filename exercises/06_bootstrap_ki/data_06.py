"""Erzeugt eine synthetische Kohorte mit Phosphatstörungen (keine echten Patientendaten).

make_cohort() – eine Zeile pro Person:
- fgf23   [pg/ml]  rechtsschief (lognormal), bei Personen mit Ereignis im Mittel höher (starker Marker)
- alp     [U/l]    alkalische Phosphatase, nur schwach mit dem Ereignis verknüpft (schwacher Marker)
- event   0/1      Nephrokalzinose innerhalb von 2 Jahren (rein synthetisch), ca. 30 %

make_rare() – kleine Studie mit nur 3 Ereignissen bei n = 40 (Stolperfalle für den Bootstrap).
"""
import numpy as np
import pandas as pd


def make_cohort(n: int = 160, seed: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    event = (rng.random(n) < 0.30).astype(int)
    fgf23 = np.exp(rng.normal(np.log(60) + 0.9 * event, 0.75, n)).round(1)    # AUC ≈ 0,8
    alp = rng.normal(180 + 12 * event, 45, n).round(0)                       # AUC ≈ 0,6
    return pd.DataFrame({
        "patient_id": np.arange(6001, 6001 + n),
        "fgf23": fgf23,
        "alp": alp,
        "event": event,
    })


def make_rare(n: int = 40, n_pos: int = 3, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    event = np.zeros(n, dtype=int)
    event[rng.choice(n, n_pos, replace=False)] = 1
    score = rng.normal(0.0 + 1.0 * event, 1.0, n).round(3)
    return pd.DataFrame({"event": event, "score": score})
