# Lektion 02 – Ein Seed, eine Wahrheit? Reproduzierbarkeit mit venv, git und Seeds

**Woche 1 · 30 min · Python & pandas · Reproduzierbarkeit** · Interaktive Version: https://claude.ai/artifact/BqjHapawx9E5M8mvzRRYQA

In der interaktiven Version ist jede Linie dieselbe Analyse auf derselben synthetischen Kohorte: zufälliger 70/30-Split, dann Bootstrap des Test-AUC. Nur der Seed ändert sich. Die Liniengarbe zeigt, wie weit ein einzelner berichteter AUC allein durch Zufall daneben liegen kann (Populations-AUC 0,75). Mit Reglern änderst du die Zahl der Seeds und die Kohortengröße; im Modus „Seed fixiert (42)“ fällt die Garbe zu einer Linie zusammen.

> **Merksatz:** Seed fixieren macht ein Ergebnis wiederholbar. Über Seeds berichten macht es ehrlich.

## 01 · Konzept in drei Schritten

1. **Code: git hält den Code fest.** Ein Ergebnis gehört zu genau einem Commit. Schreib `git rev-parse HEAD` ins Log jeder Auswertung. Ohne Commit-Hash weißt du in zwei Jahren nicht, welche Version die Tabelle 2 erzeugt hat.
2. **Umgebung: venv hält die Pakete fest.** `python -m venv .venv` isoliert das Projekt, `pip freeze > requirements.txt` pinnt die Versionen. Gleicher Code mit anderer numpy- oder scikit-learn-Version kann andere Zahlen liefern.
3. **Zufall: Seeds halten den Zufall fest.** `np.random.default_rng(42)` und `random_state=42` machen Splits und Bootstrap wiederholbar. Der Seed ist aber kein Gütesiegel: Berichte die Streuung über mehrere Seeds statt eines Glückstreffers.

## 02 · Mini-Code (10 min)

`setup.sh`
```bash
# einmal pro Projekt (Linux/macOS)
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scikit-learn matplotlib
pip freeze > requirements.txt       # Versionen pinnen
git add requirements.txt repro_demo.py && git commit -m "Seed-Demo"
git rev-parse --short HEAD           # diesen Hash ins Ergebnis-Log
```

`repro_demo.py`
```python
# Python 3.11+, numpy, scikit-learn, matplotlib – nur synthetische Daten
import sys
import numpy as np, sklearn, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1) Umgebung protokollieren – gehört in jedes Ergebnisprotokoll
print(f"python {sys.version.split()[0]} | numpy {np.__version__} | sklearn {sklearn.__version__}")

# 2) Synthetische Kohorte: fester Seed -> immer dieselben Daten
X, y = make_classification(n_samples=400, n_features=8, n_informative=4,
                           weights=[0.7], random_state=0)

def auc_fuer_seed(seed: int) -> float:
    """Ein kompletter Lauf: Split -> Fit -> AUC, gesteuert durch EINEN Seed."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)
    m = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])

aucs = np.array([auc_fuer_seed(s) for s in range(50)])
print(f"AUC über 50 Seeds: {aucs.mean():.3f} ± {aucs.std(ddof=1):.3f} "
      f"(min {aucs.min():.3f}, max {aucs.max():.3f})")
print("Seed 42 zweimal identisch:", auc_fuer_seed(42) == auc_fuer_seed(42))

# 3) Plot im Scientific-Noir-Stil (Leader-Label statt Legende)
plt.rcParams.update({"figure.facecolor": "#05070A", "axes.facecolor": "#05070A",
    "axes.edgecolor": "#1E2A33", "axes.labelcolor": "#8A97A3",
    "xtick.color": "#8A97A3", "ytick.color": "#8A97A3", "text.color": "#E8ECEF",
    "font.family": "monospace", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6, 3))
ax.scatter(range(50), aucs, s=14, color="#E8ECEF", lw=0)
ax.axhline(aucs.mean(), color="#FF4A1C", lw=1)
b = aucs.argmax()
ax.annotate("bester Seed", (b, aucs[b]), xytext=(b + 6, aucs[b]), fontsize=8,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#8A97A3"))
ax.set(xlabel="Seed", ylabel="Test-AUC")
fig.tight_layout(); fig.savefig("seed_streuung.png", dpi=200)
```

**Aufgabe:** Lass das Skript laufen (laut Lektion: AUC ≈ 0,74 ± 0,05, Spannweite ≈ 0,60–0,86). Setze dann `n_samples=2000`. Was passiert mit der SD über Seeds? Prüfe dasselbe in der interaktiven Version mit dem Regler.

## 03 · Quiz (5 min)

**1. Oben im Skript steht `np.random.seed(42)`. Später rufst du `train_test_split(X, y)` ohne `random_state` auf. Ist der Split reproduzierbar?**
- A) Nein, scikit-learn ignoriert numpy-Seeds komplett
- B) Ja, aber nur solange sich Anzahl und Reihenfolge der Zufallsaufrufe davor nicht ändern
- C) Ja, immer, egal welcher Code davor läuft
- D) Nur wenn PYTHONHASHSEED gesetzt ist

<details><summary>Antwort</summary>

**B.** Mit `random_state=None` nutzt scikit-learn die globale numpy-RandomState-Instanz, die `np.random.seed` setzt. Der Split ist also wiederholbar, aber fragil: Fügst du davor eine weitere Zufallsoperation ein, verschiebt sich der Strom. Besser: `random_state` explizit übergeben oder ein eigenes `default_rng(42)`-Objekt durchreichen.
</details>

**2. Eine Kollegin führt dein Skript auf demselben Commit mit demselben Seed aus und bekommt einen leicht anderen AUC. Was ist die wahrscheinlichste Ursache?**
- A) Seeds gelten nur auf dem Rechner, auf dem sie gesetzt wurden
- B) Unterschiedliche Paketversionen, weil keine requirements.txt bzw. kein Lockfile gepinnt war
- C) git verändert beim Klonen Gleitkommazahlen
- D) Python rechnet grundsätzlich nicht deterministisch

<details><summary>Antwort</summary>

**B.** Gleicher Code und gleicher Seed reichen nicht, wenn sich Bibliotheken unterscheiden. Neue Versionen ändern Defaults, Solver oder sogar Zufallsströme: NumPy garantiert für den modernen `Generator` keine identischen Streams über Versionen hinweg. Deshalb: venv plus gepinnte Versionen, und die Versionen ins Ergebnis-Log schreiben.
</details>

**3. Du trainierst mit 20 Seeds und berichtest im Abstract den besten Test-AUC (0,86). Der Mittelwert liegt bei 0,74. Was ist das Problem?**
- A) Keins, der beste Seed zeigt das Potenzial des Modells
- B) Der Wert ist optimistisch verzerrt; ehrlich wäre Mittelwert ± SD über Seeds oder ein vorab festgelegter Seed
- C) Man darf grundsätzlich nur einen Seed verwenden
- D) Das Problem ist nur, dass 20 Seeds zu wenige sind

<details><summary>Antwort</summary>

**B.** Seed-Auswahl nach dem Ergebnis ist eine Form von Selektion auf den Testdaten. Der beste Seed misst vor allem Glück beim Split, wie die Liniengarbe zeigt. Berichte die Verteilung über Seeds (oder Cross-Validation) und lege den Analyse-Seed vor dem Blick auf die Ergebnisse fest.
</details>

## 04 · Reflexion (5 min)

Antworte in der Claude-App in zwei bis drei Sätzen: **Nimm deine Auswertung aus Übung 03 (Missing Data). Was müsstest du festhalten, damit ein Kollege in zwei Jahren exakt dieselben Zahlen bekommt, und welcher dieser drei Bausteine (Code, Umgebung, Zufall) fehlt dir heute noch?** Schick gern auch dein Quiz-Ergebnis mit.

---
Weiter: Woche 2 · Bootstrap & Konfidenzintervalle
