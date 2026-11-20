import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from data_03 import make_cohort  # noqa: E402
from missing import impute_group_mean, missing_patterns, missing_vs_observed  # noqa: E402

COLS = ["bmi", "ldl", "egfr_12m"]


@pytest.fixture
def cohort():
    return make_cohort(return_truth=True)


@pytest.fixture
def mini():
    return pd.DataFrame({
        "age":   [40, 50, 60, 70, 80, 90],
        "grp":   ["a", "a", "a", "b", "b", "b"],
        "x":     [1.0, np.nan, 3.0, np.nan, 10.0, 20.0],
        "y":     [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0],
    })


# ---------- Aufgabe 1 ----------
def test_patterns_mini(mini):
    pat = missing_patterns(mini, ["x", "y"])
    assert list(pat.columns) == ["x", "y", "n", "pct"]
    assert pat["n"].sum() == len(mini)
    assert list(pat.index) == list(range(len(pat)))
    assert pat["n"].is_monotonic_decreasing
    assert pd.api.types.is_bool_dtype(pat["x"])
    top = pat.iloc[0]
    assert (top["x"], top["y"], top["n"]) == (False, False, 3)
    assert top["pct"] == pytest.approx(50.0)
    both = pat[pat["x"] & pat["y"]]
    assert len(both) == 1 and both["n"].iloc[0] == 1


def test_patterns_cohort(cohort):
    df, _ = cohort
    pat = missing_patterns(df, COLS)
    assert pat["n"].sum() == len(df)
    assert pat["pct"].sum() == pytest.approx(100.0)
    assert not pat.duplicated(COLS).any()
    complete = pat[~pat[COLS].any(axis=1)]
    assert complete["n"].iloc[0] == df[COLS].notna().all(axis=1).sum()


# ---------- Aufgabe 2 ----------
def test_missing_vs_observed_mini(mini):
    res = missing_vs_observed(mini, "x", ["age"])
    assert list(res.columns) == ["mean_observed", "mean_missing", "diff", "smd"]
    assert res.loc["age", "mean_missing"] == pytest.approx(60.0)     # 50, 70
    assert res.loc["age", "mean_observed"] == pytest.approx(67.5)    # 40, 60, 80, 90
    assert res.loc["age", "diff"] == pytest.approx(-7.5)
    var_m, var_o = np.var([50, 70], ddof=1), np.var([40, 60, 80, 90], ddof=1)
    assert res.loc["age", "smd"] == pytest.approx(-7.5 / np.sqrt((var_m + var_o) / 2))


def test_missing_vs_observed_detects_mar(cohort):
    df, _ = cohort
    res = missing_vs_observed(df, "egfr_12m", ["age", "bmi"])
    assert list(res.index) == ["age", "bmi"]
    assert res.loc["age", "smd"] > 0.5           # Drop-outs sind deutlich älter → MAR
    assert abs(res.loc["bmi", "smd"]) < 0.25     # bmi hängt nicht am Drop-out


# ---------- Aufgabe 3 ----------
def test_impute_group_mean_mini(mini):
    before = mini.copy()
    out = impute_group_mean(mini, "x", "grp")
    assert mini.equals(before)                              # Eingabe unverändert
    assert out["x"].notna().all()
    assert out.loc[1, "x"] == pytest.approx(2.0)            # Mittel aus 1 und 3
    assert out.loc[3, "x"] == pytest.approx(15.0)           # Mittel aus 10 und 20
    assert out.loc[0, "x"] == 1.0                           # beobachtete Werte bleiben
    assert out["x_imputed"].tolist() == [False, True, False, True, False, False]
    assert list(out.index) == list(mini.index)


def test_impute_reduces_bias(cohort):
    df, truth = cohort
    true_mean = truth.mean()
    cc_mean = df["egfr_12m"].mean()                          # Complete-Case
    imp = impute_group_mean(df, "egfr_12m", "age_group")
    assert imp["egfr_12m"].notna().all()
    assert imp["egfr_12m_imputed"].sum() == df["egfr_12m"].isna().sum()
    # Complete-Case überschätzt die eGFR (die Älteren fehlen) ...
    assert cc_mean - true_mean > 2
    # ... gruppenweise Imputation korrigiert einen Großteil davon
    assert abs(imp["egfr_12m"].mean() - true_mean) < 0.5 * abs(cc_mean - true_mean)
    # Preis der Single Imputation: Streuung wird unterschätzt
    assert imp["egfr_12m"].std() < truth.std()
