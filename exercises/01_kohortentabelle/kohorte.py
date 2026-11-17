"""Übung 01 – Musterlösung."""
import numpy as np
import pandas as pd

SEX_MAP = {"m": "m", "M": "m", "male": "m", "w": "f", "f": "f", "F": "f", "female": "f"}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates(subset="patient_id", keep="first").copy()
    out["sex"] = out["sex"].map(SEX_MAP)
    out.loc[~out["egfr"].between(0, 200), "egfr"] = np.nan  # NaN bleibt NaN
    return out.reset_index(drop=True)


def missing_report(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().sort_values(ascending=False)


def table_one(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("group")
    return pd.DataFrame({
        "n": g.size(),
        "age_mean": g["age"].mean(),
        "age_sd": g["age"].std(),
        "female_pct": g["sex"].apply(lambda s: (s == "f").mean() * 100),
        "egfr_median": g["egfr"].median(),
        "egfr_q1": g["egfr"].quantile(0.25),
        "egfr_q3": g["egfr"].quantile(0.75),
    }).round(1)
