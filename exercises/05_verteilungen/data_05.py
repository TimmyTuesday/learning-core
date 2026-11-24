"""Erzeugt synthetische Laborwerte einer Kohorte mit Phosphatstörungen (keine echten Patientendaten).

labs (eine Zeile pro Person):
- phosphate  [mg/dl]  annähernd normalverteilt
- fgf23      [pg/ml]  stark rechtsschief (lognormal), ca. 5 % fehlend
- crp        [mg/l]   rechtsschief, auf 0,1 gerundet → einige Werte sind exakt 0.0
"""
import numpy as np
import pandas as pd


def make_labs(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    phosphate = rng.normal(3.4, 0.55, n).round(2)
    fgf23 = np.exp(rng.normal(np.log(60), 0.9, n)).round(1)      # Median ≈ 60 pg/ml
    fgf23[rng.random(n) < 0.05] = np.nan
    crp = np.exp(rng.normal(np.log(1.2), 1.3, n)).round(1)        # Rundung erzeugt Nullen
    return pd.DataFrame({
        "patient_id": np.arange(5001, 5001 + n),
        "phosphate": phosphate,
        "fgf23": fgf23,
        "crp": crp,
    })
