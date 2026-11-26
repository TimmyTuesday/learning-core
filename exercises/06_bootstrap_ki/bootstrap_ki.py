"""Übung 06 – fülle die vier Funktionen aus. Tests: pytest exercises/06_bootstrap_ki"""
from typing import Callable

import numpy as np
import pandas as pd  # noqa: F401


def bootstrap_ci(x, stat: Callable = np.mean, B: int = 2000, level: float = 0.95,
                 seed: int = 42) -> pd.Series:
    # TODO: Perzentil-Bootstrap-KI für eine beliebige Kennzahl stat(x).
    #   - NaN in x oder weniger als 2 Werte -> ValueError (keine stillen NaN!)
    #   - rng = np.random.default_rng(seed); B Stichproben mit Zurücklegen, jeweils Länge n
    #   - Rückgabe: Series mit genau estimate, lower, upper, se
    #       estimate = stat auf den Originaldaten, se = SD der Replikate (ddof=1)
    raise NotImplementedError


def auc(y, score) -> float:
    # TODO: AUC als Wahrscheinlichkeit P(score_Ereignis > score_kein_Ereignis), Gleichstand zählt 0.5.
    #   - nur eine Klasse vorhanden oder NaN -> ValueError
    raise NotImplementedError


def bootstrap_auc_ci(y, score, B: int = 2000, level: float = 0.95, seed: int = 42) -> pd.Series:
    # TODO: Perzentil-KI der AUC mit STRATIFIZIERTEM Bootstrap:
    #   Ereignisse und Nicht-Ereignisse getrennt mit Zurücklegen ziehen (Anzahl je Klasse bleibt gleich).
    #   Rückgabe: Series estimate, lower, upper, se
    raise NotImplementedError


def paired_auc_diff_ci(y, score_a, score_b, B: int = 2000, level: float = 0.95,
                       seed: int = 42) -> pd.Series:
    # TODO: KI für ΔAUC = auc(y, score_a) - auc(y, score_b), GEPAART:
    #   in jedem Replikat dieselben gezogenen Personen für beide Scores (stratifiziert wie oben).
    #   Rückgabe: Series estimate, lower, upper, se
    raise NotImplementedError
