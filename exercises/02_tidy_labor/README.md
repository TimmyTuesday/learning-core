# Übung 02 – Tidy Labordaten: Wide → Long, Datentypen, Missing Data

**Woche 1 · ca. 30 min · Datei:** `labor.py` · **Tests:** `pytest exercises/02_tidy_labor`

Du bekommst einen synthetischen Laborexport einer XLH-Verlaufsstudie (Phosphat, alkalische Phosphatase
zu Monat 0, 6 und 12) – im Wide-Format und mit allen Eigenheiten deutscher Laborsysteme.
Schau dir die Daten zuerst an:

```python
from data_02 import make_lab_export
raw = make_lab_export()
print(raw.head()); print(raw.dtypes)
```

## Aufgaben
1. **`to_numeric_de(s)`** – eine Text-Spalte in `float64` umwandeln:
   - Dezimalkomma: `"0,82"` → `0.82`
   - unter Nachweisgrenze: `"<0,3"` → `0.15` (Konvention **LOD/2**)
   - `"n.b."`, `"-"`, `""` → `NaN`
   - Eingabe nicht verändern; Index beibehalten
2. **`to_long(df)`** – Wide → Long (tidy). Ergebnis hat genau die Spalten
   `patient_id`, `group`, `parameter` (`"phosphat"`/`"alp"`), `month` (int: 0, 6, 12), `value` (float, via Aufgabe 1).
   Sortiert nach `patient_id`, `parameter`, `month`; Index `0..n-1`. Fehlende Werte bleiben als Zeile mit `NaN` erhalten.
3. **`change_from_baseline(long)`** – je (`patient_id`, `parameter`) eine Zeile mit den Spalten
   `patient_id`, `group`, `parameter`, `baseline` (Wert in Monat 0), `last` (letzter **vorhandener** Wert nach Monat 0),
   `last_month`, `delta = last − baseline`. Fehlt die Baseline oder jeder Follow-up-Wert → `delta` ist `NaN`
   (nicht imputieren!).

## Bonus (ohne Test)
Plotte mit `noir.use_noir()` einen Spaghetti-Plot Phosphat über die Zeit (eine dünne Linie pro Person,
Gruppenmedian im Akzent #FF4A1C). Wie viele XLH-Patient:innen haben ein `last_month` von 6 statt 12 –
und warum ist „letzter vorhandener Wert“ (≈ LOCF) bei Drop-outs methodisch heikel?

## Konzept dahinter
*Tidy Data* heißt: eine Zeile pro Beobachtung (Person × Parameter × Zeitpunkt), eine Spalte pro Variable.
Erst im Long-Format funktionieren `groupby`, Plots und gemischte Modelle sauber. Und: Wie du `"<0,3"` und
`"n.b."` kodierst, ist eine **methodische Entscheidung** (LOD/2, Zensierung, Missingness-Mechanismus) –
sie gehört in den Methodenteil, nicht nur in den Code.
