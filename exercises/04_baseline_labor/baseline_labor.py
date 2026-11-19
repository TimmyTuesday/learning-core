"""Übung 04 – fülle die drei Funktionen aus. Tests: pytest exercises/04_baseline_labor"""
import pandas as pd

STAGES = ["G1", "G2", "G3a", "G3b", "G4", "G5"]


def clean_patients(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Kopie zurückgeben, in der
    #   birth_date, enrol_date -> datetime64 (Format "TT.MM.JJJJ"!)
    #   sex       -> category mit genau den Kategorien ["f", "m"]
    #   ckd_stage -> geordnete category mit allen STAGES (G1 < ... < G5)
    #   age       -> neue int-Spalte: vollendete Lebensjahre am enrol_date
    raise NotImplementedError


def attach_baseline_lab(patients: pd.DataFrame, labs: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    # TODO: je Patient:in die letzte Kreatinin-Abnahme am oder vor dem Einschlusstag,
    #   höchstens window_days Kalendertage vorher.
    #   Neue Spalten: creatinine_bl (float), creatinine_bl_at (datetime), days_before (float)
    #   Kein Wert im Fenster -> NaN/NaT. Reihenfolge und Index wie `patients`.
    raise NotImplementedError


def stage_summary(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: eine Zeile pro ckd_stage (ALLE Kategorien, in Reihenfolge, auch leere)
    #   Spalten n (int), median_age, median_creatinine_bl
    raise NotImplementedError
