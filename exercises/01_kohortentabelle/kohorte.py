"""Übung 01 – fülle die drei Funktionen aus. Tests: pytest exercises/01_kohortentabelle"""
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Duplikate entfernen, sex vereinheitlichen, unplausible egfr -> NaN, Index zurücksetzen
    raise NotImplementedError


def missing_report(df: pd.DataFrame) -> pd.Series:
    # TODO: Anteil fehlender Werte je Spalte, absteigend sortiert
    raise NotImplementedError


def table_one(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: je group: n, age_mean, age_sd, female_pct, egfr_median, egfr_q1, egfr_q3
    raise NotImplementedError
