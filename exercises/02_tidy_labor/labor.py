"""Übung 02 – Musterlösung. Tests: pytest exercises/02_tidy_labor"""
import numpy as np
import pandas as pd

FEHLEND = {"n.b.", "-", ""}


def to_numeric_de(s: pd.Series) -> pd.Series:
    # Text normalisieren: Leerzeichen weg, fehlende Marker -> NaN
    t = s.astype("string").str.strip()
    t = t.mask(t.isin(FEHLEND))
    # Werte unter Nachweisgrenze merken, "<" entfernen
    unter_lod = t.str.startswith("<", na=False)
    t = t.str.lstrip("<").str.replace(",", ".", regex=False)
    out = pd.to_numeric(t, errors="coerce").astype(np.float64)
    # LOD/2-Konvention
    out[unter_lod] = out[unter_lod] / 2
    return out


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    long = df.melt(id_vars=["patient_id", "group"], var_name="variable", value_name="raw")
    # "phosphat_m12" -> parameter="phosphat", month=12
    parts = long["variable"].str.extract(r"^(?P<parameter>.+)_m(?P<month>\d+)$")
    long["parameter"] = parts["parameter"]
    long["month"] = parts["month"].astype(int)
    long["value"] = to_numeric_de(long["raw"])
    return (long[["patient_id", "group", "parameter", "month", "value"]]
            .sort_values(["patient_id", "parameter", "month"])
            .reset_index(drop=True))


def change_from_baseline(long: pd.DataFrame) -> pd.DataFrame:
    keys = ["patient_id", "group", "parameter"]
    base = (long[long["month"] == 0]
            .set_index(keys)["value"].rename("baseline"))
    fu = long[(long["month"] > 0) & long["value"].notna()].sort_values("month")
    last = fu.groupby(keys).agg(last=("value", "last"), last_month=("month", "last"))
    out = base.to_frame().join(last, how="outer").reset_index()
    out["delta"] = out["last"] - out["baseline"]   # NaN, wenn eins fehlt
    return out
