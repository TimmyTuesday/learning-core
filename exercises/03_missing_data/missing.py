"""Übung 03 – fülle die drei Funktionen aus. Tests: pytest exercises/03_missing_data"""
import pandas as pd


def missing_patterns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # TODO: je Missingness-Muster über `cols` eine Zeile:
    #   eine bool-Spalte pro col (True = fehlt), dazu `n` (int) und `pct` (0–100)
    #   absteigend nach n sortiert, Index 0..k-1
    raise NotImplementedError


def missing_vs_observed(df: pd.DataFrame, target: str, covariates: list[str]) -> pd.DataFrame:
    # TODO: Index = covariates; Spalten mean_observed, mean_missing, diff, smd
    #   mean_missing = Mittelwert der Kovariate dort, wo `target` fehlt
    #   diff = mean_missing - mean_observed
    #   smd  = diff / sqrt((var_missing + var_observed) / 2)   (var mit ddof=1)
    raise NotImplementedError


def impute_group_mean(df: pd.DataFrame, target: str, by: str) -> pd.DataFrame:
    # TODO: Kopie zurückgeben, in der fehlende `target`-Werte durch den Mittelwert
    #   der beobachteten Werte derselben `by`-Gruppe ersetzt sind,
    #   plus bool-Spalte f"{target}_imputed". Eingabe nicht verändern.
    raise NotImplementedError
