# Übung 03 – Missing Data: Muster, Mechanismus, Bias

**Woche 1 · ca. 30 min · Datei:** `missing.py` · **Tests:** `pytest exercises/03_missing_data`

Synthetische Nephrologie-Kohorte (n = 600) mit Baseline-Daten und eGFR nach 12 Monaten.
Werte fehlen hier **nicht zufällig**: Ein Teil ist MCAR, der Drop-out beim Follow-up hängt vom Alter ab (MAR).
Deine Aufgabe: Missingness erst beschreiben, dann den Mechanismus prüfen, dann den Bias korrigieren.

```python
from data_03 import make_cohort
df, truth = make_cohort(return_truth=True)   # truth = "wahre" eGFR aller 600 (gibt's real nie)
print(df.head()); print(df.isna().mean())
```

## Aufgaben
1. **`missing_patterns(df, cols)`** – Missingness-Muster zählen (wie `md.pattern()` in R/mice).
   Ergebnis: eine bool-Spalte pro `col` (True = fehlt), dazu `n` (int) und `pct` (0–100);
   eine Zeile pro vorkommendem Muster, absteigend nach `n`, Index `0..k-1`.
2. **`missing_vs_observed(df, target, covariates)`** – Unterscheiden sich Personen mit fehlendem
   `target` von denen mit beobachtetem? Index = `covariates`, Spalten
   `mean_observed`, `mean_missing`, `diff` (= missing − observed) und
   `smd` = `diff / sqrt((var_missing + var_observed) / 2)` (Varianz mit `ddof=1`).
3. **`impute_group_mean(df, target, by)`** – fehlende `target`-Werte durch den Mittelwert der
   beobachteten Werte in derselben `by`-Gruppe ersetzen (z. B. `age_group`), plus bool-Spalte
   `f"{target}_imputed"`. Kopie zurückgeben, Eingabe und Index unverändert.

Der letzte Test vergleicht drei Zahlen: wahrer eGFR-Mittelwert, Complete-Case-Mittelwert und
dein imputierter Mittelwert. Schau sie dir an, bevor du weitergehst.

## Bonus (ohne Test)
- Plotte mit `noir.use_noir()` den Anteil fehlender `egfr_12m` gegen das Alter (z. B. 5-Jahres-Bins),
  Linie im Akzent #FF4A1C. Sieht das nach MCAR aus?
- Warum ist `std()` nach Imputation kleiner als die wahre Streuung – und was macht
  **Multiple Imputation** (`sklearn.impute.IterativeImputer(sample_posterior=True)` mehrfach, Rubin's Rules) anders?
- Kann man aus den Daten allein MAR von MNAR (Drop-out wegen schlechter eGFR selbst) unterscheiden?

## Konzept dahinter
Rubin unterscheidet **MCAR** (Fehlen hängt von nichts ab), **MAR** (hängt nur von *beobachteten*
Variablen ab, hier Alter) und **MNAR** (hängt vom fehlenden Wert selbst ab). Complete-Case-Analysen
sind nur unter MCAR unverzerrt. Eine SMD > 0,1–0,2 zwischen „fehlt“ und „beobachtet“ ist ein
deutliches Warnsignal gegen MCAR. Bedingte Imputation korrigiert MAR-Bias im Mittelwert, aber
Single Imputation tut so, als wären imputierte Werte gemessen → Standardfehler zu klein.
Für Publikationen: Anteil Missing pro Variable berichten und den angenommenen Mechanismus begründen (STROBE, TRIPOD+AI).
