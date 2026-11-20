"""Erzeugt einen synthetischen Labor-Export im Wide-Format – so, wie er oft aus dem KIS/LIS kommt.

Typische Probleme:
- Werte als Text mit deutschem Dezimalkomma ("0,82")
- Werte unter der Nachweisgrenze (LOD) als "<0,3"
- fehlende Messungen als "n.b." (nicht bestimmt), "-" oder leerer String
- eine Spalte pro Parameter UND Zeitpunkt (nicht tidy)
"""
import numpy as np
import pandas as pd

PARAMETER = ["phosphat", "alp"]   # Phosphat [mmol/l], alkalische Phosphatase [U/l]
MONATE = [0, 6, 12]


def _fmt_de(x: float, nachkomma: int) -> str:
    return f"{x:.{nachkomma}f}".replace(".", ",")


def make_lab_export(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.choice(["XLH", "Kontrolle"], size=n, p=[0.5, 0.5])
    xlh = group == "XLH"
    df = pd.DataFrame({"patient_id": np.arange(1001, 1001 + n), "group": group})

    for m in MONATE:
        # XLH: niedriges Phosphat, unter Therapie langsam steigend; Kontrollen stabil
        phos = np.where(xlh, 0.60 + 0.015 * m, 1.10) + rng.normal(0, 0.10, n)
        alp = np.where(xlh, 420 - 8 * m, 180) + rng.normal(0, 40, n)
        df[f"phosphat_m{m}"] = [_fmt_de(v, 2) for v in phos.clip(0.2, None)]
        df[f"alp_m{m}"] = [_fmt_de(v, 0) for v in alp.clip(30, None)]

    # Werte unter der Nachweisgrenze (nur Phosphat, LOD = 0,3 mmol/l)
    for idx in rng.choice(n, 4, replace=False):
        df.loc[idx, f"phosphat_m{rng.choice(MONATE)}"] = "<0,3"

    # fehlende Werte in verschiedenen Schreibweisen
    werte_spalten = [c for c in df.columns if c.startswith(tuple(PARAMETER))]
    for marker in ["n.b.", "-", ""]:
        for _ in range(8):
            df.loc[rng.integers(n), rng.choice(werte_spalten)] = marker

    # Ein paar Patient:innen ohne Follow-up nach Monat 6 (Drop-out)
    for idx in rng.choice(n, 6, replace=False):
        df.loc[idx, ["phosphat_m12", "alp_m12"]] = "n.b."

    return df
