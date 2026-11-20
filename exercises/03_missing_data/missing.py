"""Übung 03 – Musterlösung (Branch `solutions`)."""
import numpy as np
import pandas as pd


def missing_patterns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # isna() liefert pro Zeile das Muster; value_counts zählt identische Muster
    pat = (
        df[cols].isna()
        .value_counts(sort=False)          # Series mit MultiIndex aus bools
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    pat["n"] = pat["n"].astype(int)
    pat["pct"] = 100 * pat["n"] / len(df)
    return pat[cols + ["n", "pct"]]


def missing_vs_observed(df: pd.DataFrame, target: str, covariates: list[str]) -> pd.DataFrame:
    miss = df[target].isna()
    rows = {}
    for c in covariates:
        m, o = df.loc[miss, c].dropna(), df.loc[~miss, c].dropna()
        diff = m.mean() - o.mean()
        smd = diff / np.sqrt((m.var(ddof=1) + o.var(ddof=1)) / 2)
        rows[c] = {"mean_observed": o.mean(), "mean_missing": m.mean(), "diff": diff, "smd": smd}
    return pd.DataFrame.from_dict(rows, orient="index")[["mean_observed", "mean_missing", "diff", "smd"]]


def impute_group_mean(df: pd.DataFrame, target: str, by: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{target}_imputed"] = out[target].isna()
    # transform("mean") ignoriert NaN und gibt einen Wert pro Zeile zurück (Index bleibt)
    out[target] = out[target].fillna(out.groupby(by)[target].transform("mean"))
    return out
