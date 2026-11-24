"""Übung 05 – fülle die vier Funktionen aus. Tests: pytest exercises/05_verteilungen"""
import sys
from pathlib import Path
from statistics import NormalDist  # noqa: F401  (Tipp für Aufgabe 3)

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # Repo-Wurzel für noir.py
import noir  # noqa: E402,F401


def describe_distribution(s: pd.Series) -> pd.Series:
    # TODO: NaN entfernen, dann eine Series mit genau diesen Einträgen (in dieser Reihenfolge):
    #   n, n_missing, mean, median, sd (ddof=1), skew (g1, Fisher-Pearson – selbst rechnen!),
    #   mean_minus_median
    raise NotImplementedError


def log_summary(s: pd.Series) -> pd.Series:
    # TODO: ValueError, falls ein (Nicht-NaN-)Wert <= 0 ist.
    #   Sonst Series: geo_mean, gsd (geometrischer SD-Faktor), skew_log (g1 der log-Werte)
    raise NotImplementedError


def qq_points(s: pd.Series) -> pd.DataFrame:
    # TODO: DataFrame mit Spalten theoretical, sample (RangeIndex, ohne NaN):
    #   sample      = sortierte, standardisierte Werte (Mittelwert 0, SD 1 mit ddof=1)
    #   theoretical = Normal-Quantile an den Positionen (i - 0.5) / n, i = 1..n
    raise NotImplementedError


def plot_distribution(s: pd.Series, log: bool = False, title: str | None = None):
    # TODO: noir.use_noir(); Figure mit 2 Achsen zurückgeben:
    #   links  Histogramm + senkrechte Median-Linie in noir.ACCENT (+ Mittelwert gestrichelt, andere Farbe)
    #   rechts QQ-Plot aus qq_points() + Referenzlinie y = x
    #   log=True -> alles auf log10-Skala der Werte; Werte <= 0 -> ValueError
    raise NotImplementedError
