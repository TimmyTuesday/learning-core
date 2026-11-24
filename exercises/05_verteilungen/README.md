# Übung 05 – Verteilungen beschreiben und plotten (Scientific Noir)

**Woche 2 · ca. 30 min · Datei:** `verteilungen.py` · **Tests:** `pytest exercises/05_verteilungen`

Synthetische Laborwerte einer Kohorte mit Phosphatstörungen (n = 400): Serum-Phosphat (fast normalverteilt),
FGF23 (stark rechtsschief, ~5 % fehlend) und CRP (rechtsschief, auf 0,1 mg/l gerundet → einige Werte sind
exakt `0.0`). Ziel: eine Verteilung mit Zahlen **und** Bild beschreiben – und erkennen, wann der Mittelwert lügt.
Baut direkt auf Lektion 04 (Schiefe, log-Transformation, QQ-Plot) auf.

```python
from data_05 import make_labs
labs = make_labs()
print(labs.head()); print(labs.describe())
print((labs["crp"] == 0).sum(), labs["fgf23"].isna().sum())
```

## Aufgaben
1. **`describe_distribution(s)`** – NaN ignorieren, `pd.Series` mit genau den Einträgen
   `n, n_missing, mean, median, sd, skew, mean_minus_median` (in dieser Reihenfolge).
   - `sd` mit `ddof=1`.
   - `skew` = **g₁ (Fisher-Pearson)** wie in L04: `m3 / m2**1.5`, mit `m_k` = Mittelwert von `(x − x̄)^k`.
     Achtung: `Series.skew()` liefert die bias-bereinigte Variante G₁ – bei kleinem n ein deutlich anderer Wert.
2. **`log_summary(s)`** – für positive, rechtsschiefe Werte: `pd.Series` mit `geo_mean`
   (geometrisches Mittel), `gsd` (geometrischer SD-*Faktor* = exp der SD der ln-Werte, ddof=1) und `skew_log`
   (g₁ der ln-Werte). Enthält `s` einen Wert ≤ 0 → **`ValueError`** statt stiller `-inf`/`NaN`.
3. **QQ-Plot und Abbildung**
   - a) **`qq_points(s)`** – DataFrame mit Spalten `theoretical`, `sample` (ohne NaN, `RangeIndex`):
     `sample` = sortierte, standardisierte Werte; `theoretical` = Standardnormal-Quantile an den
     Plotting-Positionen `(i − 0.5) / n`. Ohne scipy machbar.
   - b) **`plot_distribution(s, log=False, title=None)`** – ruft `noir.use_noir()` auf und gibt eine
     `Figure` mit **zwei** Achsen zurück: links Histogramm mit senkrechter Median-Linie in `noir.ACCENT`
     (Mittelwert gestrichelt in einer anderen Farbe), rechts QQ-Plot (alle Punkte) plus Referenzlinie y = x.
     Mit `log=True` wird alles auf der **log10**-Skala gezeigt (Median-Linie also bei `median(log10(x))`);
     Werte ≤ 0 → `ValueError`.

## Bonus (ohne Test)
- Speichere `plot_distribution(labs["fgf23"])` und `plot_distribution(labs["fgf23"], log=True)` als PNG
  und vergleiche: Wo knickt der QQ-Plot ab, und was passiert damit nach der Transformation?
- CRP hat Nullen. Vergleiche drei Auswege: Nullen weglassen, `np.log1p`, Nullen durch `0.05` (halbe
  Rundungsgrenze, analog LOD/2 aus Ü02) ersetzen. Wie ändern sich `geo_mean` und `skew_log` – und welche
  Variante würdest du im Methodenteil begründen?
- Wie würdest du FGF23 in einer Baseline-Tabelle (Ü01) berichten: Mittelwert ± SD, Median [IQR] oder
  geometrisches Mittel × / ÷ GSD?

## Konzept dahinter
Mittelwert und SD beschreiben eine Verteilung nur dann gut, wenn sie annähernd symmetrisch ist.
Bei rechtsschiefen Biomarkern (FGF23, CRP, Ferritin) zieht der lange rechte Rand den Mittelwert nach oben
(`mean_minus_median > 0`, g₁ ≫ 0). Auf der log-Skala werden solche Werte oft symmetrisch; zurücktransformiert
ist der Mittelwert der log-Werte das **geometrische Mittel**, und Streuung wird **multiplikativ** (× / ÷ GSD).
Der QQ-Plot macht Abweichungen von der Normalverteilung sichtbar, die ein Histogramm mit ungünstigen Bins
verschleiert. Und: log ist für Werte ≤ 0 nicht definiert – eine Transformation, die das stillschweigend
in `-inf` oder `NaN` verwandelt, ist derselbe Fehlertyp wie `errors="coerce"` bei LOD-Werten.
