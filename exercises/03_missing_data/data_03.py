"""Erzeugt eine synthetische Nephrologie-Kohorte mit fehlenden Werten – bewusst NICHT zufällig verteilt.

Mechanismen (so, wie sie in echten Registern vorkommen):
- ldl:       MCAR – fehlt rein zufällig (z. B. Probe hämolysiert, ~10 %)
- bmi:       MCAR – Gewicht nicht dokumentiert (~6 %)
- egfr_12m:  MAR  – Follow-up nach 12 Monaten fehlt häufiger bei Älteren (Drop-out hängt vom Alter ab)

Mit return_truth=True bekommst du zusätzlich die "wahren" eGFR-Werte aller Personen –
das gibt es in der Realität nie, hier aber zum Nachprüfen des Bias.
"""
import numpy as np
import pandas as pd

AGE_GROUPS = ["<50", "50-69", ">=70"]


def make_cohort(n: int = 600, seed: int = 11, return_truth: bool = False):
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 90, n)
    sex = rng.choice(["f", "m"], n)
    diabetes = rng.random(n) < 0.15 + 0.004 * (age - 25)
    bmi = rng.normal(27, 4, n).round(1)
    ldl = rng.normal(3.2, 0.8, n).round(2)                      # mmol/l
    # eGFR sinkt mit dem Alter und bei Diabetes
    egfr = (125 - 0.9 * age - 8 * diabetes + rng.normal(0, 10, n)).clip(8, None).round(0)

    df = pd.DataFrame({
        "patient_id": np.arange(5001, 5001 + n),
        "age": age,
        "age_group": pd.cut(age, [0, 49, 69, 200], labels=AGE_GROUPS).astype(str),
        "sex": sex,
        "diabetes": diabetes,
        "bmi": bmi,
        "ldl": ldl,
        "egfr_12m": egfr.astype(float),
    })
    truth = df["egfr_12m"].copy()

    # MCAR
    df.loc[rng.random(n) < 0.10, "ldl"] = np.nan
    df.loc[rng.random(n) < 0.06, "bmi"] = np.nan
    # MAR: Drop-out-Wahrscheinlichkeit steigt mit dem Alter (≈ 5 % bei 25 J., ≈ 60 % bei 89 J.)
    p_drop = 1 / (1 + np.exp(-(age - 75) / 8))
    df.loc[rng.random(n) < p_drop, "egfr_12m"] = np.nan

    return (df, truth) if return_truth else df
