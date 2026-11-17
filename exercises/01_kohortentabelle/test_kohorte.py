import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from data import make_cohort  # noqa: E402
from kohorte import clean, missing_report, table_one  # noqa: E402


@pytest.fixture
def raw():
    return make_cohort()


def test_clean_removes_duplicates(raw):
    out = clean(raw)
    assert out["patient_id"].is_unique
    assert len(out) == 300
    assert list(out.index) == list(range(len(out)))


def test_clean_harmonises_sex(raw):
    assert set(clean(raw)["sex"].unique()) == {"m", "f"}


def test_clean_implausible_egfr(raw):
    egfr = clean(raw)["egfr"].dropna()
    assert egfr.between(0, 200).all()


def test_clean_does_not_mutate_input(raw):
    before = raw.copy()
    clean(raw)
    assert raw.equals(before)


def test_missing_report(raw):
    rep = missing_report(clean(raw))
    assert rep.index[0] == "egfr"
    assert rep.is_monotonic_decreasing
    assert rep.between(0, 1).all()


def test_table_one(raw):
    t1 = table_one(clean(raw))
    cols = {"n", "age_mean", "age_sd", "female_pct", "egfr_median", "egfr_q1", "egfr_q3"}
    assert cols <= set(t1.columns)
    assert set(t1.index) == {"XLH", "Kontrolle"}
    assert t1["n"].sum() == 300
    assert t1.loc["XLH", "egfr_median"] < t1.loc["Kontrolle", "egfr_median"]
    assert (t1["egfr_q1"] <= t1["egfr_median"]).all() and (t1["egfr_median"] <= t1["egfr_q3"]).all()
    assert t1["female_pct"].between(0, 100).all()
    assert np.isfinite(t1["age_sd"]).all()
