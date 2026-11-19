# Übung 04 – Baseline-Labor: Datumsformate, Kategorien, zeitnahe Joins

**Woche 1 · ca. 30 min · Datei:** `baseline_labor.py` · **Tests:** `pytest exercises/04_baseline_labor`

Zwei synthetische Tabellen einer CKD-Kohorte (n = 300) aus zwei Systemen: der Studien-CRF-Export
(`patients`, Datumsangaben als `"TT.MM.JJJJ"`, Geschlecht und CKD-Stadium uneinheitlich kodiert) und der
Laborexport (`labs`, eine Zeile pro Kreatinin-Abnahme, vor *und* nach Einschluss, unsortiert).
Ziel: pro Person den **Baseline-Kreatininwert** finden – das klassische „letzter Wert vor Einschluss“-Problem.

```python
from data_04 import make_patients, make_labs
pat, labs = make_patients(), make_labs()
print(pat.head()); print(pat.dtypes)
print(pat["sex"].value_counts()); print(pat["ckd_stage"].unique())
print(labs.head())
```

## Aufgaben
1. **`clean_patients(df)`** – Datentypen reparieren, Kopie zurückgeben (Index unverändert):
   - `birth_date`, `enrol_date` → `datetime64` (deutsches Format: `"03.04.1960"` ist der **3. April**)
   - `sex` → `category` mit genau den Kategorien `["f", "m"]` (`"w"`, `" M"`, `"weiblich"`, `"männlich"` …)
   - `ckd_stage` → **geordnete** `category` mit allen Stadien `G1 < G2 < G3a < G3b < G4 < G5`
     (`"g3a"`, `"G3B "` → `"G3a"`, `"G3b"`)
   - neue Spalte `age` (int): **vollendete** Lebensjahre am Einschlusstag – Vorsicht bei Geburtstagen,
     die im Einschlussjahr noch bevorstehen (und beim 29. Februar). Nicht `days / 365.25` runden!
2. **`attach_baseline_lab(patients, labs, window_days=30)`** – an jede Zeile von `patients` (Ausgabe von 1)
   die **letzte** Kreatinin-Abnahme hängen, die **am oder vor dem Einschlusstag** liegt und höchstens
   `window_days` Kalendertage zurückreicht. Eine Abnahme am Einschlusstag um 08:15 zählt (Kalendertag-Vergleich).
   Neue Spalten: `creatinine_bl` (float), `creatinine_bl_at` (Zeitstempel der Abnahme), `days_before`
   (float, Kalendertage). Ohne Wert im Fenster → `NaN`/`NaT`, die Zeile bleibt erhalten.
   Reihenfolge und Index wie `patients`; `labs` nicht verändern.
3. **`stage_summary(df)`** – eine Zeile pro CKD-Stadium, **alle** Kategorien in ihrer Reihenfolge,
   auch solche ohne Patient:innen (in dieser Kohorte: `G5`). Index = Stadien als Text,
   Spalten `n` (int), `median_age`, `median_creatinine_bl`.

## Bonus (ohne Test)
- Plotte mit `noir.use_noir()` Kreatinin-Baseline je Stadium (Strip-Plot, Median als Strich im Akzent #FF4A1C).
- Wie viele Personen verlierst du, wenn du das Fenster auf 7 Tage verkürzt – und wie würdest du das im
  Methodenteil berichten? Was ändert sich, wenn du statt „letzter Wert“ den **nächstgelegenen** Wert
  (auch kurz *nach* Einschluss) erlaubst – und warum ist das bei einer Interventionsstudie heikel?
- Rechne aus Kreatinin, Alter und Geschlecht die eGFR nach CKD-EPI 2021 – passt sie zu `ckd_stage`?

## Konzept dahinter
Viele Fehler in klinischen Analysen sind **Typ-Fehler**: ein Datum, das als Text sortiert wird, ein
`"TT.MM."`/`"MM.TT."`-Verwechsler, ein Stadium, das alphabetisch statt klinisch geordnet ist.
Explizite `dtypes` (`datetime64`, geordnete `category`) machen solche Annahmen sichtbar und testbar.
Der Baseline-Join ist ein **Zeitfenster-Join** (`pd.merge_asof`): „letzter Wert vor t₀ innerhalb von Δt“.
Fenstergröße und Richtung (nur vorher vs. nächstgelegen) sind methodische Entscheidungen – Werte nach
Einschluss als Baseline zu nutzen, ist eine Form von **Data Leakage** bzw. Immortal-Time-Problemen.
