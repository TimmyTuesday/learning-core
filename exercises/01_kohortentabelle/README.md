# Übung 01 – Kohortentabelle („Table 1“) mit pandas

**Woche 1 · ca. 30 min · Datei:** `kohorte.py` · **Tests:** `pytest exercises/01_kohortentabelle`

Du bekommst eine synthetische Kohorte mit typischen Problemen echter Klinikexporte
(Duplikate, unplausible Werte, uneinheitliche Kodierung, fehlende Werte).

## Aufgaben
1. **`clean(df)`**
   - doppelte `patient_id` entfernen (erste Zeile behalten)
   - `sex` vereinheitlichen: `"m"`, `"M"`, `"male"` → `"m"`; `"w"`, `"f"`, `"F"`, `"female"` → `"f"`
   - unplausible `egfr` (< 0 oder > 200) auf `NaN` setzen
   - Index anschließend zurücksetzen
2. **`missing_report(df)`** – Anteil fehlender Werte je Spalte (0–1), absteigend sortiert, als `pd.Series`.
3. **`table_one(df)`** – je `group` eine Zeile mit den Spalten
   `n`, `age_mean`, `age_sd`, `female_pct` (0–100), `egfr_median`, `egfr_q1`, `egfr_q3`.

## Bonus (ohne Test)
Plotte die eGFR-Verteilung je Gruppe als Strip-Plot mit `noir.use_noir()`. Welche Gruppe würdest du als Erstes nachuntersuchen – und warum?

## Konzept dahinter
Eine saubere Table 1 ist der erste Plausibilitätscheck jeder Studie. Achte darauf, *wann* du bereinigst: Duplikate vor jeder Statistik entfernen, sonst sind n und Prozentwerte verzerrt.
