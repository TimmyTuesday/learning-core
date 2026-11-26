import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from bootstrap_ki import auc, bootstrap_auc_ci, bootstrap_ci, paired_auc_diff_ci  # noqa: E402
from data_06 import make_cohort, make_rare  # noqa: E402

KEYS = ["estimate", "lower", "upper", "se"]


@pytest.fixture
def cohort():
    return make_cohort()


def width(ci):
    return ci["upper"] - ci["lower"]


# ---------- Aufgabe 1: bootstrap_ci ----------
def test_ci_keys_and_order(cohort):
    ci = bootstrap_ci(cohort["alp"])
    assert list(ci.index) == KEYS
    assert ci["estimate"] == pytest.approx(cohort["alp"].mean())
    assert ci["lower"] < ci["estimate"] < ci["upper"]
    assert ci["se"] > 0


def test_ci_reproducible_with_seed(cohort):
    a = bootstrap_ci(cohort["alp"], seed=1)
    b = bootstrap_ci(cohort["alp"], seed=1)
    c = bootstrap_ci(cohort["alp"], seed=2)
    pd.testing.assert_series_equal(a, b)
    assert not a.equals(c), "anderer Seed, identisches Ergebnis – wird der Seed benutzt?"
    # bei B = 2000 darf der Seed die Grenzen nur wenig verschieben
    assert abs(a["lower"] - c["lower"]) < 0.1 * width(a)


def test_ci_close_to_normal_theory(cohort):
    x = cohort["alp"].to_numpy()
    se = x.std(ddof=1) / np.sqrt(len(x))
    ci = bootstrap_ci(x, B=4000)
    assert ci["se"] == pytest.approx(se, rel=0.1)
    assert width(ci) == pytest.approx(2 * 1.96 * se, rel=0.15)


def test_ci_level_and_n(cohort):
    x = cohort["alp"]
    assert width(bootstrap_ci(x, level=0.90)) < width(bootstrap_ci(x, level=0.99))
    assert width(bootstrap_ci(x.iloc[:30])) > width(bootstrap_ci(x))


def test_ci_custom_stat(cohort):
    ci = bootstrap_ci(cohort["fgf23"], stat=np.median)
    assert ci["estimate"] == pytest.approx(cohort["fgf23"].median())
    assert ci["lower"] <= ci["estimate"] <= ci["upper"]


def test_ci_rejects_nan_and_tiny():
    with pytest.raises(ValueError):
        bootstrap_ci(pd.Series([1.0, 2.0, np.nan, 4.0]))
    with pytest.raises(ValueError):
        bootstrap_ci([3.0])


# ---------- Aufgabe 2: auc + bootstrap_auc_ci ----------
def test_auc_small_examples():
    assert auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)          # Gleichstand = 0.5
    assert auc([0, 0, 1], [1, 2, 3]) == pytest.approx(1.0)
    assert auc([0, 0, 1], [3, 2, 1]) == pytest.approx(0.0)


def test_auc_rejects_single_class_and_nan():
    with pytest.raises(ValueError):
        auc([1, 1, 1], [0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        auc([0, 1, 1], [0.2, np.nan, 0.4])


def test_auc_cohort(cohort):
    a = auc(cohort["event"], cohort["fgf23"])
    assert 0.7 < a < 0.95
    # rangbasiert: monotone Transformation (log) ändert die AUC nicht
    assert auc(cohort["event"], np.log(cohort["fgf23"])) == pytest.approx(a)


def test_bootstrap_auc_ci_cohort(cohort):
    ci = bootstrap_auc_ci(cohort["event"], cohort["fgf23"])
    assert list(ci.index) == KEYS
    assert ci["estimate"] == pytest.approx(auc(cohort["event"], cohort["fgf23"]))
    assert 0 <= ci["lower"] < ci["estimate"] < ci["upper"] <= 1
    pd.testing.assert_series_equal(ci, bootstrap_auc_ci(cohort["event"], cohort["fgf23"]))


def test_bootstrap_auc_ci_rare_events_stratified():
    rare = make_rare()
    assert rare["event"].sum() == 3
    # Ohne Stratifizierung hätten ~4 % der Replikate kein einziges Ereignis -> auc() wirft ValueError
    ci = bootstrap_auc_ci(rare["event"], rare["score"], B=2000, seed=0)
    assert 0 <= ci["lower"] <= ci["estimate"] <= ci["upper"] <= 1
    assert width(ci) > 0.2, "3 Ereignisse – das KI muss sehr breit sein"


# ---------- Aufgabe 3: paired_auc_diff_ci ----------
def test_paired_identical_scores_zero(cohort):
    d = paired_auc_diff_ci(cohort["event"], cohort["fgf23"], cohort["fgf23"])
    assert list(d.index) == KEYS
    assert d["estimate"] == 0 and d["lower"] == 0 and d["upper"] == 0


def test_paired_sign_and_strong_vs_weak(cohort):
    y = cohort["event"]
    d = paired_auc_diff_ci(y, cohort["fgf23"], cohort["alp"])
    assert d["estimate"] == pytest.approx(auc(y, cohort["fgf23"]) - auc(y, cohort["alp"]))
    assert d["estimate"] > 0
    assert d["lower"] < d["estimate"] < d["upper"]


def test_paired_is_much_narrower_than_single(cohort):
    # Zweitmessung mit kleinem Messfehler: einzelne AUC-KIs sind breit, die gepaarte Differenz eng
    rng = np.random.default_rng(99)
    y = cohort["event"]
    a = np.log(cohort["fgf23"].to_numpy())
    b = a + rng.normal(0, 0.05, len(a))
    d = paired_auc_diff_ci(y, a, b)
    single = bootstrap_auc_ci(y, a)
    assert width(d) < 0.5 * width(single), "Differenz-KI zu breit – gleiche Personen für beide Scores?"
