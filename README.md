# learning-core

Übungs-Repo des **Clinician-Scientist Learning Core** (Data Science · Medical AI · MRI).
Curriculum und Fortschritt liegen im claude.ai-Projekt unter `learning-core/`.

## Setup (einmalig)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Ablauf einer Coding-Challenge (Di/Do, 30 min)
1. `git pull` – der Scheduler legt neue Übungen unter `exercises/NN_name/` ab.
2. `README.md` der Übung lesen, Funktionen in der `.py`-Datei ausfüllen.
3. `pytest exercises/NN_name` bis alles grün ist.
4. In Claude Code: `claude` starten und z. B. „Gib mir einen Hinweis zu Aufgabe 2" – Claude ist als Tutor konfiguriert (siehe `CLAUDE.md`).
5. Lösung vergleichen: `git diff main solutions -- exercises/NN_name`
6. Commit + push; in der Claude-App kurz Ergebnis melden („01 grün, Aufgabe 3 schwer") → wird in `progress.md` übernommen.

## Plot-Stil
`from noir import use_noir; use_noir()` stellt matplotlib auf das Design-System „Scientific Noir" um.
