import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from baseline_labor import STAGES, attach_baseline_lab, clean_patients, stage_summary  # noqa: E402
from data_04 import make_labs, make_patients  # noqa: E402


@pytest.fixture
def raw_mini():
    return pd.DataFrame({
        "patient_id": [1, 2, 3, 4],
        "birth_date": ["29.02.2000", "29.02.2000", "03.04.1960", "12.11.1975"],
        "enrol_date": ["28.02.2026", "01.03.2026", "03.04.2025", "11.11.2025"],
        "sex": ["w", " M", "weiblich", "männlich"],
        "ckd_stage": ["g3a", "G3B ", "G1", " G4"],
    }, index=[7, 8, 9, 10])


@pytest.fixture
def mini_patients():
    # bewusst NICHT nach patient_id sortiert, eigener Index
    return pd.DataFrame({
        "patient_id": [3, 1, 2, 4, 6],
        "enrol_date": pd.to_datetime(["2025-03-10", "2025-01-15", "2025-02-01", "2025-05-01", "2025-06-01"]),
    }, index=[10, 11, 12, 13, 14])


@pytest.fixture
def mini_labs():
    return pd.DataFrame({
        "patient_id": [1, 3, 2, 1, 3, 2, 1, 5, 6, 2],
        "drawn_at": ["2025-01-15 08:15",   # 1: am Einschlusstag (morgens!) -> Baseline
                     "2025-02-25 11:00",   # 3: 13 Tage vorher -> Baseline
                     "2024-12-01 09:00",   # 2: 62 Tage vorher -> zu alt
                     "2025-01-10 09:00",   # 1: älter als der Wert vom Einschlusstag
                     "2025-02-20 07:30",   # 3: 18 Tage vorher, aber nicht der letzte
                     "2025-01-02 09:00",   # 2: genau 30 Tage vorher -> gerade noch drin
                     "2025-02-01 10:00",   # 1: nach Einschluss -> ignorieren
                     "2025-01-01 09:00",   # 5: gibt es in patients nicht
                     "2025-05-01 12:00",   # 6: 31 Tage vorher -> zu alt
                     "2025-03-01 08:00"],  # 2: nach Einschluss -> ignorieren
        "creatinine": [1.3, 0.95, 2.0, 1.1, 0.9, 2.1, 1.5, 9.9, 3.0, 2.2],
    })


# ---------- Aufgabe 1 ----------
def test_clean_dates_and_age(raw_mini):
    out = clean_patients(raw_mini)
    assert pd.api.types.is_datetime64_any_dtype(out["birth_date"])
    assert pd.api.types.is_datetime64_any_dtype(out["enrol_date"])
    assert out.loc[9, "birth_date"] == pd.Timestamp("1960-04-03")      # 3. April, nicht 4. März
    assert pd.api.types.is_integer_dtype(out["age"])
    assert out["age"].tolist() == [25, 26, 65, 49]                       # Schaltjahr + Geburtstag morgen


def test_clean_categories(raw_mini):
    out = clean_patients(raw_mini)
    assert isinstance(out["sex"].dtype, pd.CategoricalDtype)
    assert list(out["sex"].cat.categories) == ["f", "m"]
    assert out["sex"].tolist() == ["f", "m", "f", "m"]
    st = out["ckd_stage"]
    assert isinstance(st.dtype, pd.CategoricalDtype) and st.cat.ordered
    assert list(st.cat.categories) == STAGES
    assert st.tolist() == ["G3a", "G3b", "G1", "G4"]
    assert (st >= "G3a").tolist() == [True, True, False, True]           # geordnet -> vergleichbar


def test_clean_keeps_input_and_index(raw_mini):
    before = raw_mini.copy()
    out = clean_patients(raw_mini)
    assert raw_mini.equals(before)
    assert list(out.index) == [7, 8, 9, 10]
    assert out["patient_id"].tolist() == [1, 2, 3, 4]


def test_clean_cohort():
    out = clean_patients(make_patients())
    assert out["sex"].notna().all() and out["ckd_stage"].notna().all()
    assert out["age"].between(23, 86).all()


# ---------- Aufgabe 2 ----------
def test_baseline_mini(mini_patients, mini_labs):
    out = attach_baseline_lab(mini_patients, mini_labs)
    assert list(out.index) == [10, 11, 12, 13, 14]
    assert out["patient_id"].tolist() == [3, 1, 2, 4, 6]
    bl = dict(zip(out["patient_id"], out["creatinine_bl"]))
    assert bl[3] == pytest.approx(0.95)
    assert bl[1] == pytest.approx(1.3)
    assert bl[2] == pytest.approx(2.1)
    assert np.isnan(bl[4]) and np.isnan(bl[6])
    days = dict(zip(out["patient_id"], out["days_before"]))
    assert (days[3], days[1], days[2]) == (13, 0, 30)
    assert pd.api.types.is_datetime64_any_dtype(out["creatinine_bl_at"])
    assert out.loc[11, "creatinine_bl_at"] == pd.Timestamp("2025-01-15 08:15")
    assert pd.isna(out.loc[13, "creatinine_bl_at"])


def test_baseline_window_and_input(mini_patients, mini_labs):
    labs_before = mini_labs.copy()
    out = attach_baseline_lab(mini_patients, mini_labs, window_days=60)
    assert mini_labs.equals(labs_before)
    assert len(out) == len(mini_patients)
    assert out.loc[14, "creatinine_bl"] == pytest.approx(3.0)            # 31 Tage: jetzt drin


def test_baseline_cohort():
    pat = clean_patients(make_patients())
    out = attach_baseline_lab(pat, make_labs())
    assert len(out) == len(pat)
    has = out["creatinine_bl"].notna()
    assert 0.7 < has.mean() < 0.95
    assert out.loc[has, "days_before"].between(0, 30).all()
    assert (out.loc[has, "creatinine_bl_at"].dt.normalize() <= out.loc[has, "enrol_date"]).all()


# ---------- Aufgabe 3 ----------
def test_stage_summary_mini():
    df = pd.DataFrame({
        "ckd_stage": pd.Categorical(["G2", "G4", "G2"], categories=STAGES, ordered=True),
        "age": [40, 70, 50],
        "creatinine_bl": [1.0, np.nan, 1.2],
    })
    res = stage_summary(df)
    assert list(res.index) == STAGES
    assert list(res.columns) == ["n", "median_age", "median_creatinine_bl"]
    assert pd.api.types.is_integer_dtype(res["n"])
    assert res["n"].tolist() == [0, 2, 0, 0, 1, 0]
    assert res.loc["G2", "median_age"] == pytest.approx(45.0)
    assert res.loc["G2", "median_creatinine_bl"] == pytest.approx(1.1)
    assert np.isnan(res.loc["G4", "median_creatinine_bl"])
    assert np.isnan(res.loc["G5", "median_age"])


def test_stage_summary_cohort():
    pat = clean_patients(make_patients())
    res = stage_summary(attach_baseline_lab(pat, make_labs()))
    assert res["n"].sum() == len(pat)
    assert res.loc["G5", "n"] == 0
    med = res["median_creatinine_bl"].dropna()
    assert list(med.index) == ["G1", "G2", "G3a", "G3b", "G4"]
    assert med.is_monotonic_increasing                                   # schlechtere GFR -> höheres Kreatinin
