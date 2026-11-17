import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from data import make_lab_export  # noqa: E402
from labor import change_from_baseline, to_long, to_numeric_de  # noqa: E402


@pytest.fixture
def raw():
    return make_lab_export()


# ---------- Aufgabe 1 ----------
def test_to_numeric_de_basic():
    s = pd.Series(["0,82", "<0,3", "n.b.", "-", "", "412"], index=list("abcdef"))
    out = to_numeric_de(s)
    assert out.dtype == np.float64
    assert list(out.index) == list("abcdef")
    assert out["a"] == pytest.approx(0.82)
    assert out["b"] == pytest.approx(0.15)
    assert out[["c", "d", "e"]].isna().all()
    assert out["f"] == pytest.approx(412.0)


def test_to_numeric_de_does_not_mutate():
    s = pd.Series(["1,5", "<0,3"])
    before = s.copy()
    to_numeric_de(s)
    assert s.equals(before)


def test_to_numeric_de_on_export(raw):
    out = to_numeric_de(raw["phosphat_m0"])
    assert out.dtype == np.float64
    assert out.dropna().between(0, 3).all()


# ---------- Aufgabe 2 ----------
def test_to_long_shape_and_columns(raw):
    long = to_long(raw)
    assert list(long.columns) == ["patient_id", "group", "parameter", "month", "value"]
    assert len(long) == len(raw) * 2 * 3          # Person × Parameter × Zeitpunkt
    assert set(long["parameter"]) == {"phosphat", "alp"}
    assert sorted(long["month"].unique()) == [0, 6, 12]
    assert pd.api.types.is_integer_dtype(long["month"])
    assert long["value"].dtype == np.float64
    assert list(long.index) == list(range(len(long)))


def test_to_long_sorted_and_values(raw):
    long = to_long(raw)
    key = long[["patient_id", "parameter", "month"]]
    assert key.equals(key.sort_values(["patient_id", "parameter", "month"]).reset_index(drop=True))
    first = raw.iloc[0]
    row = long[(long["patient_id"] == first["patient_id"])
               & (long["parameter"] == "alp") & (long["month"] == 0)]
    assert row["value"].iloc[0] == pytest.approx(float(first["alp_m0"].replace(",", ".")))
    assert long["value"].isna().sum() > 0          # Missings bleiben erhalten


# ---------- Aufgabe 3 ----------
@pytest.fixture
def mini_long():
    return pd.DataFrame({
        "patient_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "group": ["XLH"] * 3 + ["Kontrolle"] * 3 + ["XLH"] * 3,
        "parameter": ["phosphat"] * 9,
        "month": [0, 6, 12] * 3,
        "value": [0.6, 0.7, 0.8,      # vollständig
                  1.1, 1.2, np.nan,   # Drop-out nach Monat 6
                  np.nan, 0.7, 0.9],  # keine Baseline
    })


def test_change_from_baseline_mini(mini_long):
    cfb = change_from_baseline(mini_long).set_index("patient_id")
    assert {"group", "parameter", "baseline", "last", "last_month", "delta"} <= set(cfb.columns)
    assert cfb.loc[1, "delta"] == pytest.approx(0.2)
    assert cfb.loc[1, "last_month"] == 12
    assert cfb.loc[2, "last"] == pytest.approx(1.2)
    assert cfb.loc[2, "last_month"] == 6
    assert cfb.loc[2, "delta"] == pytest.approx(0.1)
    assert np.isnan(cfb.loc[3, "delta"])          # nicht imputieren


def test_change_from_baseline_export(raw):
    cfb = change_from_baseline(to_long(raw))
    assert len(cfb) == len(raw) * 2
    assert not cfb.duplicated(["patient_id", "parameter"]).any()
    phos = cfb[cfb["parameter"] == "phosphat"].groupby("group")["delta"].median()
    assert phos["XLH"] > phos["Kontrolle"]        # Phosphat steigt unter Therapie
    assert (cfb["last_month"].dropna() > 0).all()
