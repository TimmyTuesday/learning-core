# Lektion 00 – k-Space: wo das MRT-Bild wirklich entsteht

**Einstieg · 30 min · Track C · MRI** · Interaktive Version: https://claude.ai/artifact/HNPHtf7sSerCcPkijq1nFL

Der Scanner misst kein Bild, sondern dessen Frequenzanteile. Erst die inverse Fourier-Transformation macht daraus Anatomie. In der interaktiven Version ist der k-Space eines Shepp-Logan-Phantoms (64 × 64) als drehbares Punktwolken-Gebirge dargestellt: Höhe = Signalstärke (log). Ein Regler maskiert Zentrum oder Peripherie, daneben siehst du Original, maskierten k-Space und Rekonstruktion sowie Signalenergie und RMSE live.

> **Merksatz:** Zentrum = Kontrast. Peripherie = Kanten.

## 01 · Konzept in drei Schritten

1. **Messen im Frequenzraum.** Gradientenfelder kodieren den Ort als Phase und Frequenz. Jede Messzeile füllt eine Zeile k-Space, nicht eine Bildzeile.
2. **Das Zentrum trägt die Energie.** Niedrige Raumfrequenzen (Mitte) bestimmen Helligkeit und Kontrast. Mit wenigen Prozent der Koeffizienten steckt dort der Großteil der Signalenergie.
3. **Die Peripherie trägt die Schärfe.** Hohe Frequenzen erzeugen Kanten und feine Details. Weniger Peripherie heißt geringere Auflösung, kürzere Messzeit und Gibbs-Ringing.

## 02 · Mini-Code (10 min)

```python
import numpy as np

# img: 2D-Array, z. B. eine Schicht aus nibabel.load(...).get_fdata()
K = np.fft.fftshift(np.fft.fft2(img))          # Bild -> k-Space, Zentrum in die Mitte
ky, kx = np.indices(K.shape)
ky, kx = ky - K.shape[0]//2, kx - K.shape[1]//2
mask = np.hypot(kx, ky) <= r                    # Low-Pass: nur das Zentrum behalten
rec = np.abs(np.fft.ifft2(np.fft.ifftshift(K * mask)))

energie = (np.abs(K*mask)**2).sum() / (np.abs(K)**2).sum()
# Aufgabe: Plotte energie gegen r (1..32) mit noir.use_noir().
# Ab welchem r liegen 95 % der Energie im Zentrum?
```

## 03 · Quiz (5 min)

**1. Du behältst nur die innersten 5 % der k-Space-Koeffizienten. Wie sieht das rekonstruierte Bild aus?**
- A) Scharf, aber kontrastarm
- B) Unscharf, aber mit erhaltenem Grobkontrast
- C) Komplett schwarz
- D) Nur Kanten sichtbar

<details><summary>Antwort</summary>

**B.** Das Zentrum enthält die niedrigen Raumfrequenzen und damit den Großteil der Signalenergie. Der Kontrast bleibt erhalten, die Kanten fehlen.
</details>

**2. Was zeigt ein Bild, das nur aus der k-Space-Peripherie rekonstruiert wurde?**
- A) Die Gewebekontraste T1/T2
- B) Kanten und feine Strukturen
- C) Rauschen ohne Information
- D) Dasselbe wie das Original

<details><summary>Antwort</summary>

**B.** Hohe Frequenzen kodieren schnelle Intensitätswechsel, also Kanten. Ohne Zentrum fehlen die flächigen Helligkeiten fast vollständig.
</details>

**3. Für ein CNN willst du MRT-Daten künstlich „verschlechtern“, um es robuster zu machen. Welche Augmentation ist physikalisch am realistischsten?**
- A) Gaußsches Weichzeichnen im Bildraum
- B) k-Space-Peripherie maskieren (Low-Pass, ggf. mit Ringing)
- C) Zufällige Farbverschiebung
- D) Horizontal spiegeln

<details><summary>Antwort</summary>

**B.** Eine reduzierte k-Space-Abdeckung bildet nach, was der Scanner bei kürzerer Messzeit tatsächlich tut, einschließlich Gibbs-Ringing. Bibliotheken wie TorchIO bieten k-Space-basierte Transforms (z. B. RandomMotion, RandomGhosting) an.
</details>

## 04 · Reflexion (5 min)

Antworte in der Claude-App in zwei Sätzen: **Warum verliert eine schnelle Sequenz mit wenigen Phasenkodierschritten zuerst Schärfe und nicht Kontrast?**

---
Vertiefung: Woche 5 · MRI-Physik für Data Scientists
