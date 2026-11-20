"""Erzeugt eine synthetische Kohorte mit realistischen Datenproblemen."""
import numpy as np
import pandas as pd


def make_cohort(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "group": rng.choice(["XLH", "Kontrolle"], size=n, p=[0.4, 0.6]),
        "age": rng.normal(34, 12, n).round(1).clip(4, 85),
        "sex": rng.choice(["m", "M", "male", "w", "f", "F", "female"], size=n),
        "egfr": rng.normal(95, 20, n).round(0),
    })
    df.loc[df["group"] == "XLH", "egfr"] -= 12
    df.loc[rng.choice(n, 8, replace=False), "egfr"] = rng.choice([-1, 999, 450], 8)
    df.loc[rng.choice(n, 15, replace=False), "egfr"] = np.nan
    df.loc[rng.choice(n, 6, replace=False), "age"] = np.nan
    dups = df.sample(10, random_state=seed)
    return pd.concat([df, dups], ignore_index=True)
