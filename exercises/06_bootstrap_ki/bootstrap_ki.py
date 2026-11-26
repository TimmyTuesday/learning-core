"""Übung 06 – Musterlösung. Tests: pytest exercises/06_bootstrap_ki"""
from typing import Callable

import numpy as np
import pandas as pd

KEYS = ["estimate", "lower", "upper", "se"]


def _percentile_ci(estimate: float, reps: np.ndarray, level: float) -> pd.Series:
    alpha = (1 - level) / 2
    lower, upper = np.quantile(reps, [alpha, 1 - alpha])
    return pd.Series([estimate, lower, upper, reps.std(ddof=1)], index=KEYS, dtype=float)


def bootstrap_ci(x, stat: Callable = np.mean, B: int = 2000, level: float = 0.95,
                 seed: int = 42) -> pd.Series:
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        raise ValueError("x enthält NaN – fehlende Werte bewusst behandeln, nicht still mitschleppen")
    if len(x) < 2:
        raise ValueError("Bootstrap braucht mindestens 2 Werte")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))          # B Stichproben mit Zurücklegen
    reps = np.array([stat(x[i]) for i in idx])
    return _percentile_ci(float(stat(x)), reps, level)


def auc(y, score) -> float:
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    if np.isnan(score).any():
        raise ValueError("score enthält NaN")
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC braucht beide Klassen")
    # Mann-Whitney: alle Paare (Ereignis, kein Ereignis) vergleichen
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_auc_ci(y, score, B: int = 2000, level: float = 0.95, seed: int = 42) -> pd.Series:
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    est = auc(y, score)
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        i = _stratified_indices(y, rng)
        reps[b] = auc(y[i], score[i])
    return _percentile_ci(est, reps, level)


def paired_auc_diff_ci(y, score_a, score_b, B: int = 2000, level: float = 0.95,
                       seed: int = 42) -> pd.Series:
    y = np.asarray(y)
    a = np.asarray(score_a, dtype=float)
    b_ = np.asarray(score_b, dtype=float)
    est = auc(y, a) - auc(y, b_)
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        i = _stratified_indices(y, rng)                          # dieselben Personen für beide Scores
        reps[b] = auc(y[i], a[i]) - auc(y[i], b_[i])
    return _percentile_ci(est, reps, level)
