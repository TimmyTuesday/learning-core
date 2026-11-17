"""Übung 02 – fülle die drei Funktionen aus. Tests: pytest exercises/02_tidy_labor"""
import pandas as pd


def to_numeric_de(s: pd.Series) -> pd.Series:
    # TODO: Text mit Dezimalkomma -> float64
    #   "0,82" -> 0.82 | "<0,3" -> 0.15 (LOD/2) | "n.b.", "-", "" -> NaN
    raise NotImplementedError


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Wide -> Long (tidy): Spalten patient_id, group, parameter, month (int), value (float)
    #   sortiert nach patient_id, parameter, month; Index 0..n-1
    raise NotImplementedError


def change_from_baseline(long: pd.DataFrame) -> pd.DataFrame:
    # TODO: je (patient_id, parameter): baseline (month 0), last, last_month, delta = last - baseline
    raise NotImplementedError
