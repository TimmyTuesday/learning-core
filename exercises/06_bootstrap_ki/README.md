# Übung 06 – Bootstrap-KI für Mittelwert und AUC

**Woche 2 · ca. 30 min · Datei:** `bootstrap_ki.py` · **Tests:** `pytest exercises/06_bootstrap_ki`

Synthetische Kohorte mit Phosphatstörungen (n = 160, ~30 % Ereignisse „Nephrokalzinose in 2 Jahren“, rein erfunden):
FGF23 (starker, rechtsschiefer Marker) und alkalische Phosphatase `alp` (schwacher Marker). Dazu eine Mini-Studie
mit nur **3 Ereignissen bei n = 40**. Ziel: Unsicherheit einer Kennzahl selbst per Bootstrap schätzen – reproduzierbar
über den Seed – und zwei Marker **fair** vergleichen. Baut auf Lektion 05 (KI, Perzentil-Bootstrap, gepaarter
Bootstrap) und Paper 01 (Bootstrap-KI bei n = 104) auf; L06 (ROC/AUC) am Freitag vertieft die AUC.

```python
from data_06 import make_cohort, make_rare
cohort = make_cohort()
print(cohort.head()); print(cohort["event"].mean())
rare = make_rare(); print(rare["event"].sum())
```

## Aufgaben
1. **`bootstrap_ci(x, stat=np.mean, B=2000, level=0.95, seed=42)`** – Perzentil-Bootstrap für eine beliebige
   Kennzahl. Rückgabe: `pd.Series` mit genau `estimate, lower, upper, se` (in dieser Reihenfolge).
   - `estimate` = `stat` auf den Originaldaten, `se` = SD der B Replikate (`ddof=1`).
   - Zufall nur über `np.random.default_rng(seed)` – gleicher Seed → identisches Ergebnis.
   - NaN in `x` oder weniger als 2 Werte → **`ValueError`** (wie in Ü05: keine stillen NaN).
2. **AUC und ihr KI**
   - a) **`auc(y, score)`** – AUC als Wahrscheinlichkeit, dass eine Person *mit* Ereignis einen höheren Score hat
     als eine Person *ohne* (Mann-Whitney); Gleichstand zählt 0,5. Nur eine Klasse oder NaN → `ValueError`.
     Ohne scikit-learn – aber du darfst dein Ergebnis mit `sklearn.metrics.roc_auc_score` gegenprüfen.
   - b) **`bootstrap_auc_ci(y, score, B, level, seed)`** – **stratifiziert**: Ereignisse und Nicht-Ereignisse
     getrennt mit Zurücklegen ziehen, sodass jede Klasse ihre Größe behält. Rückgabe wie in Aufgabe 1.
3. **`paired_auc_diff_ci(y, score_a, score_b, B, level, seed)`** – KI für ΔAUC = AUC(a) − AUC(b). **Gepaart:**
   in jedem Replikat werden dieselben Personen für beide Scores verwendet (stratifiziert wie in 2b).

## Bonus (ohne Test)
- **Seed-Shopping-Check (L02):** Berechne `bootstrap_auc_ci` für FGF23 mit 20 verschiedenen Seeds, einmal mit
  `B=200`, einmal mit `B=2000`. Wie stark wandert die untere Grenze? Was gehört deshalb in den Methodenteil?
- **Abdeckung simulieren (L05):** Ziehe 1000-mal n = 10 Werte aus `3 * LogNormal(0, 1)` (wahres μ = 3·e^0,5) und
  zähle, wie oft dein 95-%-Bootstrap-KI μ enthält. Wiederhole mit n = 100. Passt das zu den ~80 % bzw. ~92 % aus L05?
- **Brücke zu Ü05:** `bootstrap_ci(fgf23, stat=geo_mean)` – ist das KI um den Schätzwert symmetrisch? Warum nicht?
- Warum sieht man in `make_rare()` ein riesiges AUC-KI, und was würdest du einem Reviewer zu „AUC 0,70“ bei
  3 Ereignissen antworten?

## Konzept dahinter
Der Bootstrap ersetzt die unbekannte Population durch die Stichprobe selbst: Wiederholtes Ziehen mit Zurücklegen
zeigt, wie stark eine Kennzahl allein durch die Zusammensetzung der Stichprobe schwanken würde. Das Perzentil-KI
liest diese Schwankung direkt ab – ohne Formel, auch für Median, geometrisches Mittel oder AUC. Zwei Dinge
entscheiden über die Qualität: **die Resampling-Einheit** muss die Studienstruktur abbilden (bei seltenen Ereignissen
stratifiziert, sonst entstehen Replikate ohne Ereignis; bei mehreren Messungen pro Person: Person, nicht Messung),
und **Vergleiche zweier Marker an denselben Personen müssen gepaart sein**. Weil beide Marker dieselbe
Stichprobenschwankung teilen, hebt sie sich in der Differenz größtenteils auf – zwei getrennte KIs, die sich
überlappen, sagen deshalb fast nichts darüber, ob sich die AUCs unterscheiden.
