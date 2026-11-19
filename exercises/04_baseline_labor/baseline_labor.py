"""Übung 04 – Musterlösung. Tests: pytest exercises/04_baseline_labor"""
import numpy as np
import pandas as pd

STAGES = ["G1", "G2", "G3a", "G3b", "G4", "G5"]
_SEX_MAP = {"w": "f", "weiblich": "f", "f": "f", "m": "m", "männlich": "m"}


def clean_patients(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Explizites Format: sonst wird "03.04.1960" ggf. als 4. März gelesen
    for col in ["birth_date", "enrol_date"]:
        out[col] = pd.to_datetime(out[col], format="%d.%m.%Y")

    sex = out["sex"].str.strip().str.lower().map(_SEX_MAP)
    out["sex"] = pd.Categorical(sex, categories=["f", "m"])

    # "g3a" / "G3B " -> "G3a" / "G3b"
    st = out["ckd_stage"].str.strip().str.upper().str.replace(r"([AB])$", lambda m: m.group(1).lower(), regex=True)
    out["ckd_stage"] = pd.Categorical(st, categories=STAGES, ordered=True)

    # vollendete Lebensjahre: Jahresdifferenz minus 1, falls Geburtstag im Einschlussjahr noch nicht erreicht
    b, e = out["birth_date"], out["enrol_date"]
    before_bday = (e.dt.month < b.dt.month) | ((e.dt.month == b.dt.month) & (e.dt.day < b.dt.day))
    out["age"] = (e.dt.year - b.dt.year - before_bday.astype(int)).astype("int64")
    return out


def attach_baseline_lab(patients: pd.DataFrame, labs: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    lab = labs.copy()
    lab["creatinine_bl_at"] = pd.to_datetime(lab["drawn_at"])
    lab["day"] = lab["creatinine_bl_at"].dt.normalize()             # Vergleich auf Kalendertag-Ebene
    lab = lab.sort_values(["day", "creatinine_bl_at"])               # merge_asof braucht sortierten Schlüssel
    lab = lab[["patient_id", "day", "creatinine", "creatinine_bl_at"]].rename(columns={"creatinine": "creatinine_bl"})

    left = patients.copy()
    left["_order"] = np.arange(len(left))
    left["day"] = left["enrol_date"].dt.normalize()
    lab["day"] = lab["day"].astype(left["day"].dtype)                # gleiche Zeitauflösung
    merged = pd.merge_asof(
        left.sort_values("day"), lab,
        on="day", by="patient_id", direction="backward",
        tolerance=pd.Timedelta(days=window_days),
    )
    merged["days_before"] = (merged["day"] - merged["creatinine_bl_at"].dt.normalize()).dt.days.astype(float)
    merged = merged.sort_values("_order").drop(columns=["_order", "day"])
    merged.index = patients.index
    return merged


def stage_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ckd_stage", observed=False)   # pandas 3: Default observed=True würde G5 verschlucken
    res = pd.DataFrame({
        "n": g.size(),
        "median_age": g["age"].median(),
        "median_creatinine_bl": g["creatinine_bl"].median(),
    })
    res.index = pd.Index(res.index.astype(str), name="ckd_stage")
    return res
