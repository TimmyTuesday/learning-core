# Lektion 01 – Missing Data: warum ein Wert fehlt, entscheidet über den Bias

**Woche 1 · 30 min · Python & pandas · klinische Daten** · Interaktive Version: https://claude.ai/artifact/JRLfHD9YpVcdYAdZvuB1Xp

In der interaktiven Version ist jeder Punkt eine synthetische Person (n = 600): Alter gegen eGFR. Ein Teil der eGFR-Werte fehlt, und du siehst, was eine Complete-Case-Analyse nie sieht: wo die fehlenden Werte wirklich liegen. Du wechselst Anteil fehlend, Mechanismus (MCAR/MAR/MNAR) und Analyse (Complete Case, Mittelwert-Imputation, Regression auf das Alter) und beobachtest, wie weit der geschätzte Mittelwert vom wahren wegdriftet und wie sich die SD verändert.

> **Merksatz:** MCAR: weniger Power. MAR: korrigierbar. MNAR: nur Sensitivitätsanalyse.

Was die Simulation zeigt:
- **MCAR + Complete Case:** Mittelwert stimmt, nur die Fallzahl schrumpft.
- **MAR + Complete Case:** Jüngere (hohe eGFR) fehlen öfter, also wird μ unterschätzt. Mittelwert-Imputation repariert nichts. Regression auf das Alter nutzt genau die Variable, die das Fehlen erklärt: Der Bias verschwindet fast.
- **MNAR:** Hohe eGFR-Werte fehlen, weil sie hoch sind. Regression auf das Alter hilft nur teilweise; der Rest-Bias ist aus den Daten nicht sichtbar.
- **Mittelwert-Imputation** lässt die SD in jedem Fall künstlich schrumpfen.

## 01 · Konzept in drei Schritten

1. **MCAR – rein zufällig.** Missing Completely At Random: Das Fehlen hängt von nichts ab, etwa ein verlorenes Röhrchen. Complete Case bleibt **unverzerrt**, du verlierst nur Fallzahl und Power. In klinischen Daten selten.
2. **MAR – erklärbar durch Beobachtetes.** Missing At Random: Das Fehlen hängt von **beobachteten** Variablen ab, z. B. bei jüngeren Ambulanzpatienten wird seltener Labor abgenommen. Complete Case ist verzerrt, aber Imputation oder Modelle, die das Alter einbeziehen, korrigieren das (Multiple Imputation, MICE).
3. **MNAR – der Wert selbst entscheidet.** Missing Not At Random: Das Fehlen hängt vom **fehlenden Wert** ab, z. B. Nierengesunde bekommen keine eGFR. Keine Methode, die nur beobachtete Daten nutzt, behebt das sicher. Ob MAR oder MNAR vorliegt, ist aus den Daten nicht testbar: Sensitivitätsanalysen (δ-Adjustierung) und klinisches Wissen.

## 02 · Mini-Code (10 min)

Simuliert dieselben drei Mechanismen mit n = 2000 und vergleicht Complete Case mit einer Regressions-Imputation auf das Alter. Läuft mit numpy, pandas, matplotlib. Passt zu Übung 03 im Repo.

```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ohne Bildschirm; in Jupyter weglassen
import matplotlib.pyplot as plt

# Noir-Plotstil (Scientific Noir)
plt.rcParams.update({"figure.facecolor": "#05070A", "axes.facecolor": "#05070A",
    "axes.edgecolor": "#1E2A33", "axes.labelcolor": "#8A97A3", "text.color": "#E8ECEF",
    "xtick.color": "#8A97A3", "ytick.color": "#8A97A3", "font.family": "monospace"})

rng = np.random.default_rng(42)
n = 2000
alter = rng.uniform(25, 85, n)
egfr = 120 - 1.0 * (alter - 25) + rng.normal(0, 12, n)   # synthetisch, keine Patientendaten
df = pd.DataFrame({"alter": alter, "egfr": egfr})

def maske(score, anteil=0.4, staerke=2.0):
    # logistisches Fehlen: P(fehlt) = sigmoid(a + staerke*z)
    z = (score - score.mean()) / (score.std() + 1e-12)
    lo, hi = -10, 10
    for _ in range(50):                       # Intercept a per Bisektion: mittlere P = anteil
        a = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(a + staerke * z)))
        lo, hi = (a, hi) if p.mean() < anteil else (lo, a)
    return rng.uniform(size=len(z)) < p

mechanismen = {"MCAR": maske(np.zeros(n)),               # unabhängig von allem
               "MAR":  maske(-df["alter"].to_numpy()),   # Jüngere fehlen öfter (Alter beobachtet)
               "MNAR": maske(df["egfr"].to_numpy())}     # hohe eGFR fehlt öfter (Wert selbst)

wahr = df["egfr"].mean()
for name, fehlt in mechanismen.items():
    obs = df.loc[~fehlt]
    cc = obs["egfr"].mean()                               # Complete Case
    b = np.polyfit(obs["alter"], obs["egfr"], 1)          # Regression egfr ~ alter
    imp = df["egfr"].where(~fehlt, np.polyval(b, df["alter"]))
    print(f"{name}: fehlt {fehlt.mean():.0%} | wahr {wahr:5.1f} | "
          f"CC {cc:5.1f} | Regr.-Imputation {imp.mean():5.1f}")

# Aufgabe: Plotte eGFR gegen Alter, fehlende Werte grau, beobachtete hell.
fehlt = mechanismen["MNAR"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df.alter[fehlt], df.egfr[fehlt], s=4, c="#1E2A33")
ax.scatter(df.alter[~fehlt], df.egfr[~fehlt], s=4, c="#E8ECEF")
ax.axhline(wahr, c="#8A97A3", lw=0.5); ax.axhline(df.egfr[~fehlt].mean(), c="#FF4A1C", lw=1)
ax.set(xlabel="Alter [J]", ylabel="eGFR [ml/min/1,73 m²]")
fig.savefig("missing_mnar.png", dpi=150)
```

Erwartete Ausgabe (seed 42):
```
MCAR: fehlt 41% | wahr  90.1 | CC  90.3 | Regr.-Imputation  90.1
MAR: fehlt 40% | wahr  90.1 | CC  81.0 | Regr.-Imputation  90.5
MNAR: fehlt 40% | wahr  90.1 | CC  79.9 | Regr.-Imputation  86.1
```

**Aufgabe:** Setze `staerke=4` beim MNAR-Fall. Wie stark wächst der Bias der Regressions-Imputation? Und warum hilft das Alter hier nur teilweise?

## 03 · Quiz (5 min)

**1. In einem Register fehlt das Labor bei jüngeren ambulanten Patienten häufiger. Alter und Versorgungssetting sind vollständig erfasst; innerhalb gleicher Alters- und Setting-Gruppen fehlt das Labor zufällig. Welcher Mechanismus?**
- A) MCAR
- B) MAR
- C) MNAR
- D) Keiner, das ist Selection Bias

<details><summary>Antwort</summary>

**B.** Das Fehlen hängt nur von beobachteten Variablen (Alter, Setting) ab: MAR. Complete Case ist für den Gesamtmittelwert verzerrt, ein Imputationsmodell mit Alter und Setting korrigiert das.
</details>

**2. Mittelwert-Imputation bei 40 % fehlenden Werten unter MCAR. Was passiert?**
- A) Mittelwert und SD bleiben korrekt
- B) Mittelwert korrekt, SD und Korrelationen werden künstlich kleiner
- C) Mittelwert verzerrt, SD korrekt
- D) Die Power steigt, weil n wieder vollständig ist

<details><summary>Antwort</summary>

**B.** Alle Imputate liegen exakt auf dem Mittelwert: μ bleibt, aber die Streuung schrumpft, Zusammenhänge mit anderen Variablen werden abgeschwächt und Konfidenzintervalle zu schmal. Die scheinbar größere Fallzahl ist keine echte Information. Multiple Imputation behebt das, indem sie Unsicherheit mitschätzt (Rubin’s Rules).
</details>

**3. Wie kannst du aus den beobachteten Daten zwischen MAR und MNAR unterscheiden?**
- A) Mit Little’s MCAR-Test
- B) Durch Vergleich der Mittelwerte von Fällen mit und ohne Fehlwert
- C) Gar nicht sicher; nur mit Sensitivitätsanalysen und Fachwissen
- D) Mit einer logistischen Regression des Fehlindikators

<details><summary>Antwort</summary>

**C.** Little’s Test und Vergleiche beobachteter Variablen können höchstens MCAR widerlegen. Ob das Fehlen zusätzlich vom unbeobachteten Wert selbst abhängt, steckt definitionsgemäß nicht in den Daten. Deshalb: plausible MNAR-Szenarien durchspielen (z. B. δ-Adjustierung) und berichten, wie stabil das Ergebnis ist.
</details>

## 04 · Reflexion (5 min)

Antworte in der Claude-App in zwei bis drei Sätzen: **Nenne eine Variable aus deinem Forschungsalltag, die vermutlich nicht zufällig fehlt. Welchen Mechanismus (MAR oder MNAR) vermutest du, und welche beobachtete Variable könntest du in ein Imputationsmodell aufnehmen, um den Bias zu verkleinern?** Bitte keine Patientendaten, nur die Variable und deine Überlegung.

---
Passt zu: Übung 03 (`exercises/03_missing_data/`) · Nächste Lektion: Reproduzierbarkeit (venv, git, Seeds)
