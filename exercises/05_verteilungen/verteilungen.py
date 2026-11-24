"""Übung 05 – Musterlösung."""
import sys
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # Repo-Wurzel für noir.py
import noir  # noqa: E402


def describe_distribution(s: pd.Series) -> pd.Series:
    x = s.dropna().to_numpy(dtype=float)
    mean, median = x.mean(), np.median(x)
    d = x - mean
    m2, m3 = np.mean(d**2), np.mean(d**3)
    g1 = m3 / m2**1.5                      # Fisher-Pearson-Schiefe (nicht bias-korrigiert)
    return pd.Series({
        "n": len(x),
        "n_missing": int(s.isna().sum()),
        "mean": mean,
        "median": median,
        "sd": x.std(ddof=1),
        "skew": g1,
        "mean_minus_median": mean - median,
    })


def log_summary(s: pd.Series) -> pd.Series:
    x = s.dropna().to_numpy(dtype=float)
    if (x <= 0).any():
        raise ValueError(f"{(x <= 0).sum()} Werte <= 0 – log nicht definiert")
    lx = np.log(x)
    return pd.Series({
        "geo_mean": np.exp(lx.mean()),
        "gsd": np.exp(lx.std(ddof=1)),
        "skew_log": describe_distribution(pd.Series(lx))["skew"],
    })


def qq_points(s: pd.Series) -> pd.DataFrame:
    x = np.sort(s.dropna().to_numpy(dtype=float))
    n = len(x)
    p = (np.arange(1, n + 1) - 0.5) / n
    nd = NormalDist()
    theo = np.array([nd.inv_cdf(pi) for pi in p])
    return pd.DataFrame({"theoretical": theo, "sample": (x - x.mean()) / x.std(ddof=1)})


def plot_distribution(s: pd.Series, log: bool = False, title: str | None = None):
    noir.use_noir()
    x = s.dropna().astype(float)
    if log:
        if (x <= 0).any():
            raise ValueError("Werte <= 0 – log10 nicht definiert")
        x = np.log10(x)
    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax_h.hist(x, bins=30, color=noir.CREAM, alpha=0.8)
    ax_h.axvline(x.median(), color=noir.ACCENT, lw=1.5, label="Median")
    ax_h.axvline(x.mean(), color=noir.COOL, ls="--", label="Mittelwert")
    ax_h.set_xlabel(f"log10({s.name})" if log else str(s.name))
    ax_h.legend(frameon=False)
    qq = qq_points(x)
    ax_q.plot(qq["theoretical"], qq["sample"], "o", ms=2.5, color=noir.CREAM)
    lim = [qq["theoretical"].min(), qq["theoretical"].max()]
    ax_q.plot(lim, lim, color=noir.INK_DIM, lw=0.8)
    ax_q.set_xlabel("theoretische Quantile N(0,1)")
    ax_q.set_ylabel("standardisierte Stichprobe")
    fig.suptitle(title or f"{s.name}{' (log10)' if log else ''}")
    return fig
