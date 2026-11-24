import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import noir  # noqa: E402
from data_05 import make_labs  # noqa: E402
from verteilungen import describe_distribution, log_summary, plot_distribution, qq_points  # noqa: E402


@pytest.fixture
def labs():
    return make_labs()


# ---------- Aufgabe 1 ----------
def test_describe_keys_and_counts(labs):
    d = describe_distribution(labs["fgf23"])
    assert list(d.index) == ["n", "n_missing", "mean", "median", "sd", "skew", "mean_minus_median"]
    assert d["n"] == labs["fgf23"].notna().sum()
    assert d["n_missing"] == labs["fgf23"].isna().sum()
    assert not d.isna().any(), "NaN in der Ausgabe – fehlende Werte vorher entfernt?"


def test_describe_small_example():
    d = describe_distribution(pd.Series([1.0, 2.0, 3.0, 10.0, np.nan]))
    assert d["mean"] == pytest.approx(4.0)
    assert d["median"] == pytest.approx(2.5)
    assert d["sd"] == pytest.approx(np.std([1, 2, 3, 10], ddof=1))
    assert d["mean_minus_median"] == pytest.approx(1.5)
    # g1 (Fisher-Pearson) – NICHT pandas' bereinigtes G1 (Series.skew() gäbe ≈ 1.76)
    assert d["skew"] == pytest.approx(1.0182, abs=1e-3)


def test_describe_symmetric_vs_skewed(labs):
    assert abs(describe_distribution(labs["phosphate"])["skew"]) < 0.5
    f = describe_distribution(labs["fgf23"])
    assert f["skew"] > 2 and f["mean"] > f["median"]


# ---------- Aufgabe 2 ----------
def test_log_summary_geo_mean_small():
    out = log_summary(pd.Series([1.0, 10.0, 100.0, np.nan]))
    assert list(out.index) == ["geo_mean", "gsd", "skew_log"]
    assert out["geo_mean"] == pytest.approx(10.0)
    assert out["gsd"] == pytest.approx(np.exp(np.std(np.log([1, 10, 100]), ddof=1)))
    assert out["skew_log"] == pytest.approx(0.0, abs=1e-9)


def test_log_summary_fgf23(labs):
    out = log_summary(labs["fgf23"])
    assert out["geo_mean"] < describe_distribution(labs["fgf23"])["mean"]
    assert abs(out["skew_log"]) < 0.5, "nach log sollte FGF23 annähernd symmetrisch sein"
    assert out["gsd"] > 1


def test_log_summary_rejects_zero(labs):
    assert (labs["crp"] == 0).any()
    with pytest.raises(ValueError):
        log_summary(labs["crp"])
    with pytest.raises(ValueError):
        log_summary(pd.Series([2.0, -1.0, 3.0]))


# ---------- Aufgabe 3 ----------
def test_qq_points_shape():
    s = pd.Series([3.0, np.nan, 1.0, 2.0, 4.0])
    qq = qq_points(s)
    assert list(qq.columns) == ["theoretical", "sample"]
    assert len(qq) == 4 and list(qq.index) == [0, 1, 2, 3]
    assert qq["theoretical"].iloc[0] == pytest.approx(-1.150349, abs=1e-5)   # Φ⁻¹(0.125)
    assert qq["theoretical"].sum() == pytest.approx(0.0, abs=1e-9)
    assert qq["sample"].is_monotonic_increasing
    assert qq["sample"].mean() == pytest.approx(0.0, abs=1e-9)
    assert qq["sample"].std(ddof=1) == pytest.approx(1.0)


def test_qq_normal_vs_skewed(labs):
    qp = qq_points(labs["phosphate"])
    assert np.corrcoef(qp["theoretical"], qp["sample"])[0, 1] > 0.99
    qf = qq_points(labs["fgf23"])
    assert qf["sample"].iloc[-1] > qf["theoretical"].iloc[-1] + 1, "rechter Rand sollte nach oben abknicken"


def _n_points(ax):
    counts = [len(line.get_xdata()) for line in ax.lines]
    counts += [len(c.get_offsets()) for c in ax.collections]
    return counts


@pytest.mark.parametrize("log", [False, True])
def test_plot_distribution(labs, log):
    s = labs["fgf23"]
    fig = plot_distribution(s, log=log)
    try:
        assert len(fig.axes) == 2
        assert to_hex(fig.get_facecolor()) == noir.BG.lower(), "noir.use_noir() aufgerufen?"
        ax_h, ax_q = fig.axes
        assert len(ax_h.patches) > 0, "links fehlt das Histogramm"
        med = np.log10(s.dropna()).median() if log else s.median()
        accent = [ln for ln in ax_h.lines if to_hex(ln.get_color()) == noir.ACCENT.lower()]
        assert accent, "Median-Linie in noir.ACCENT fehlt"
        assert np.allclose(accent[0].get_xdata(), med), "Akzent-Linie liegt nicht beim Median (der geplotteten Skala)"
        assert s.notna().sum() in _n_points(ax_q), "rechts sollte der QQ-Plot alle Nicht-NaN-Punkte zeigen"
    finally:
        plt.close(fig)


def test_plot_log_rejects_zero(labs):
    with pytest.raises(ValueError):
        plot_distribution(labs["crp"], log=True)
    plt.close("all")
