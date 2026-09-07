# crownseg — Einzelbaumkronen aus echten Annotationen

Eigener Ordner, damit nichts aus der bestehenden Pipeline angefasst wird. Die
Skripte hier importieren nur untereinander, nicht aus dem Repo-Wurzelverzeichnis.

## Ja, der Datensatz hat Ground-Truth-Segmentierungen

**BAMFORESTS** (Troles et al. 2024, *Remote Sensing* 16(11), 1935; CC BY-NC-SA 4.0)
liegt unter `/scratch/shared/$USER/data/bamforests`. Es sind **Instanz**-Polygone,
nicht nur Boxen und nicht nur eine Vordergrundmaske — handdigitalisierte
Kronenumrisse im COCO-Format, eine Klasse `tree`, 2048x2048-Kacheln bei
**1.70 cm/px**:

| Split | Kacheln | Kronen | Gebiete |
|---|---|---|---|
| train | 1439 | 58 228 | Stadtwald, Tretzendorf |
| val | 382 | 15 177 | Stadtwald, Tretzendorf |
| **test1** | 313 | 6 720 | **Hain** — in Training und Val nicht enthalten |
| test2 | 322 | 12 320 | Stadtwald, Tretzendorf |

Im Mittel 40 Kronen je Kachel, Median-Kronenbreite 258 px (≈ 4.4 m), p95 530 px.
Kein Multipolygon im Satz, jede Krone ist genau ein Ring.

**test1 ist die einzige ehrliche Zahl.** Hain kommt nirgends im Training vor;
test2 misst nur, wie gut die bereits gesehenen Gebiete sitzen. Beide immer
getrennt berichten.

Das vierte TIFF-Band ist der Alphakanal des Orthomosaiks und wird verworfen.

## Warum nicht einfach `crownnet.py` weiterdrehen

Der bestehende Ansatz sagt drei Karten vorher (Inneres, Rand, Zentrum) und
schneidet die Instanzen hinterher per Watershed heraus. Gemessen auf BAMFORESTS:

| Gebiet | Präzision | Trefferquote | F1 @ IoU 0.5 |
|---|---|---|---|
| Hain | 0.074 | 0.056 | **0.063** |
| Stadtwald | 0.176 | 0.090 | **0.119** |
| Tretzendorf | 0.076 | 0.030 | **0.043** |

Zwei strukturelle Gründe, warum das so nicht besser wird:

1. **Kronen überlappen sich.** Eine einzige Labelkarte kann das nicht abbilden —
   beim Rastern überschreibt die letzte Krone die vorherige. Bei
   ineinandergewachsenen Laubbäumen ist das die Regel, nicht die Ausnahme, und
   das Ziel selbst ist damit falsch.
2. **Das Netz sagt nie eine Instanz vorher**, nur wo ein Inneres aufhört. Die
   eigentliche Trennentscheidung trifft der Watershed auf einer Karte, die dafür
   nicht optimiert wurde. Es gibt auch keine Konfidenz je Krone, also keinen
   Regler zwischen Präzision und Trefferquote.

Dazu kam der eingefrorene Backbone: trainiert wurden ~2 M Parameter eines Kopfes
auf 0.34-fach herunterskalierten Bildern.

## Der Plan

**Stufe 1 — Aufbereitung.** TIFF nach JPEG in Originalauflösung, Polygone je
Kachel als `annotations.json`. Kein Verschmelzen zu Labelkarten, kein
Herunterskalieren. `bamforests.py`

**Stufe 2 — Mask R-CNN (`maskrcnn.py`).** Jede Krone eine eigene Maske mit
eigener Konfidenz, Überlappung erlaubt. ResNet50-FPN-v2, COCO-vortrainiert, voll
feingetunt. Der Maskenkopf läuft mit 56x56 statt der üblichen 28x28
(`--mask-pool 28`), weil die Kronen für COCO-Verhältnisse riesig sind.
Trainiert wird auf 1024er Ausschnitten mit Maßstabsjitter 0.5–1.0; angewendet
wird im gleitenden Fenster, wobei am Fensterrand angeschnittene Kronen verworfen
werden und der Überlapp sie im Nachbarfenster vollständig einfängt.

**Stufe 3 — Messen (`metrics.py`).** F1 bei IoU 0.5 (die Zahl, die zählt, wenn
später je Krone ein Ausschnitt an den Artklassifikator geht) und AP@0.5
(schwellenunabhängig, vergleichbar mit veröffentlichten Zahlen). Getrennt nach
Gebiet. Kronen unter 400 px Fläche fallen auf beiden Seiten raus — im GT sind
Polygone bis herunter zu 3 px, die kein Verfahren treffen kann.

**Stufe 4 — Übertragung auf die eigenen Frames.** Hier sitzt das eigentliche
Risiko, nicht im Modell: BAMFORESTS hat 1.70 cm/px, ein 1920-px-Frame aus 100 m
Höhe bei 73.7° Bildwinkel hat 7.8 cm/px — Faktor 4.6. Eine Krone, die im
Training 258 px breit war, ist im Frame 56 px breit. `--mode predict` skaliert
die Frames deshalb aus Flughöhe und Bildwinkel auf den BAMFORESTS-Maßstab hoch,
bevor das Modell sie sieht. Für die Ordner ohne bekannte Höhe bleibt das eine
Schätzung — derselbe offene Punkt wie bei der Artbestimmung.

**Stufe 5, offen.** Wenn Mask R-CNN steht und die Zahl auf test1 belastbar ist:
die 56x56-Maske ist bei 300-px-Kronen immer noch grob (≈ 5 px je Maskenpixel).
Ein Nachschärfen der Umrisse — SAM3 mit der vorhergesagten Box als Prompt, oder
ein Randkopf — ist der nächste Hebel, aber erst messen, dann bauen.

## Benutzung

```bash
sbatch crownseg/sbatch/run_prepare.sbatch              # einmalig, ~2456 Kacheln
sbatch crownseg/sbatch/run_maskrcnn.sbatch             # Training
MODE=eval    sbatch crownseg/sbatch/run_maskrcnn.sbatch
MODE=inspect sbatch crownseg/sbatch/run_maskrcnn.sbatch
MODE=predict sbatch crownseg/sbatch/run_maskrcnn.sbatch
EXTRA="--epochs 40 --scale-jitter 0.3 1.0" sbatch crownseg/sbatch/run_maskrcnn.sbatch
```

| Wo | Was |
|---|---|
| `/scratch/shared/$USER/data/bamforests/crownseg/` | aufbereitete Kacheln |
| `/scratch/shared/$USER/data/treeclf/checkpoints/crownseg_maskrcnn.pth` | Gewichte |
| `results_crownseg/` | Auswertung, Überlagerungen (gitignored) |

## Gemessen

Erster Durchlauf (Job 666/667, Auswahl nach Validierungsverlust, also Epoche 2;
40 Kacheln je Split, Score >= 0.5, IoU >= 0.5):

| Split | Gebiet | GT | Vorhersagen | Präzision | Trefferquote | F1 | mittl. IoU | AP50 |
|---|---|---|---|---|---|---|---|---|
| test2 | Stadtwald | 2126 | 3180 | 0.523 | 0.783 | **0.627** | 0.761 | 0.661 |
| test1 | Hain | 780 | 2628 | 0.094 | 0.315 | **0.144** | 0.675 | 0.128 |

Zum Vergleich `crownnet.py`: 0.119 (Stadtwald) und 0.063 (Hain).

Auf bekanntem Gebiet funktioniert es also, auf dem fremden nicht — und zwar aus
einem messbaren Grund, nicht aus einem vagen:

| | Kronen/Kachel | annotierte Fläche | Median-Ø | p95-Ø |
|---|---|---|---|---|
| Stadtwald (train) | 47.5 | 0.57 | 281 px | 503 px |
| Tretzendorf (train) | 32.3 | 0.56 | 324 px | 664 px |
| **Hain (test1)** | 21.5 | 0.57 | **392 px** | **842 px** |

Gleiche Flächenabdeckung, halb so viele Kronen: die Bäume in Hain sind
großflächiger. Das Modell zerlegt sie in Stücke (2628 Vorhersagen für 780
Kronen), die getroffenen sitzen dabei ordentlich (mittlere IoU 0.675). Drei
Ursachen, alle behoben:

1. **Anker.** Mask R-CNN schlägt standardmäßig 32–512 px vor. Eine 842-px-Krone
   kann die RPN damit nicht vorschlagen, egal wie lange trainiert wird.
   `--anchor-scale 2.0` verschiebt die Leiter auf 64–1024 px.
2. **Maßstabsjitter ging nur nach unten** (0.5–1.0), das Netz hat nie größere
   Kronen gesehen als die im Training. Jetzt 0.6–1.8.
3. **Modellauswahl nach Validierungsverlust.** Der steigt bei Mask R-CNN
   routinemäßig weiter, während die Genauigkeit noch zunimmt — ausgewählt wurde
   deshalb Epoche 2 von 25. Jetzt entscheidet die Instanz-F1 auf ganzen
   Validierungskacheln.

Zwei weitere Fehler dabei gefunden und behoben: das gleitende Fenster verwarf
alles, was den Fensterrand berührt — eine Krone breiter als der Überlapp berührt
in *jedem* Fenster einen Rand und verschwand komplett (jetzt Zuordnung über den
Kronenmittelpunkt); und die Kachelstichprobe der Auswertung nahm die ersten N
statt gleichmäßig zu greifen, wodurch Tretzendorf aus test2 komplett herausfiel.

## Verfahrensvergleich auf gemeinsamer Grundlage

Alle Verfahren auf denselben 40 Kacheln aus test1 (Hain, GT 780 Kronen),
IoU >= 0.5, gemessen mit `eval_labels.py` über Labelkarten. Damit sind auch die
Verfahren aus dem Repo-Wurzelverzeichnis vergleichbar, ohne sie anzufassen.

| Verfahren | Vorhersagen | Präzision | Trefferquote | F1 | mittl. IoU |
|---|---|---|---|---|---|
| **EoMT (DINOv3), trainiert** | 814 | 0.478 | 0.499 | **0.488** | 0.755 |
| Mask R-CNN, trainiert | 424 | 0.583 | 0.317 | 0.410 | 0.714 |
| SAM 3 + Depth Pro, teilen + säen | 932 | 0.320 | 0.382 | 0.348 | 0.725 |
| SAM 3 + Depth Pro, nur teilen | 570 | 0.405 | 0.296 | 0.342 | 0.764 |
| SAM 3 `tree`, korrigierte Kronengröße | 509 | 0.395 | 0.258 | 0.312 | 0.780 |
| Hybrid: SAM 1 + Tiefen-Watershed | 812 | 0.241 | 0.251 | 0.246 | 0.776 |
| Tiefe + SAM 1, promptbar | 304 | 0.405 | 0.158 | 0.227 | 0.759 |
| `crownnet.py` (Ausgangspunkt) | – | 0.074 | 0.056 | 0.063 | 0.682 |

### Architekturvergleich, volle Splits

40 gleichmäßig gegriffene Kacheln je Split, direkte Instanzen statt Labelkarten:

| Architektur | test2 Stadtwald | test2 Tretzendorf | **test1 Hain** | IoU Hain | AP50 Hain |
|---|---|---|---|---|---|
| **EoMT (DINOv3)** | 0.738 | 0.723 | **0.568** | 0.769 | 0.491 |
| Mask2Former (Swin) | 0.565 | 0.562 | 0.454 | 0.708 | 0.372 |
| Mask R-CNN | 0.629 | 0.544 | 0.367 | 0.706 | 0.241 |

Identischer Datenpfad, identischer Bodenausschnitt (1024 px Kachel), identische
Fensterlogik, identische Metrik — verglichen wird die Architektur, nicht die
Umgebung. Beide query-basierten Modelle starten von COCO-Instanz-Gewichten.

EoMT hält auf dem fremden Gebiet eine mittlere IoU von **0.769** und damit
SAM-Niveau (0.76–0.79), was kein anderes trainiertes Modell erreicht. Der
naheliegende Plan, gute Auswahl mit SAM-Rändern zu kombinieren, erübrigt sich
damit — EoMT liefert beides.

### Was die Zahlen nicht zeigen

Der Bildvergleich (`show_gt.py --labels`) legt zwei Fehlerarten offen:

1. **Verschmelzen im dichten Bestand.** Wo das GT drei bis vier Bäume trennt,
   liegt oft eine EoMT-Maske. Passt zum Bild aus Trefferquote 0.499 bei guter
   IoU der Treffer: was getroffen wird, sitzt gut.
2. **Nicht-Bäume.** Hausdächer werden als Krone segmentiert. BAMFORESTS kennt
   nur die Klasse `tree`; alles andere ist unmarkierter Hintergrund, aus dem
   das Modell nie gelernt hat, dass es kein Baum ist.

### Was die Tiefe beiträgt

Zerlegt nach Herkunft, gemessen statt vermutet:

| Rolle | Wirkung |
|---|---|
| **trennen** — SAM-Maske über mehreren Wipfeln aufbrechen | 0.312 → 0.342, mit Depth Pro besser als mit DA-V2 (0.342 vs 0.315) |
| **säen** — freie Wipfel als Punkt-Prompt | 170 freie Wipfel → 113 Kandidaten → 101 zu überlappend → **12 Kronen**. Kaum Wirkung |
| **ergänzen** — Watershed auf der Restfläche | **0 Treffer aus 291 Kronen** bei IoU 0.5, 2.7 % bei IoU 0.1 |

Die Wipfel*positionen* sind gut (64–67 % liegen in einer echten Krone), aber sie
zeigen überwiegend auf Bäume, die SAM 3 bereits gefunden hat. Für die verpassten
Kronen liefert auch die Tiefe keinen eigenen Wipfel. Die vorab gerechnete
Obergrenze von F1 0.606 war deshalb zu optimistisch: sie unterstellte, jeder
freie Wipfel werde eine eigenständige neue Krone.

### Ein Deadlock, der Rechenzeit gekostet hat

Beide Trainingsläufe blieben nach 17 bzw. 16 Epochen stehen — Haupt- und
Workerprozesse in `do_poll`, GPU im Leerlauf, SLURM meldete weiter RUNNING.
Ursache ist der OpenCV-Threadpool in geforkten DataLoader-Workern; behoben mit
`cv2.setNumThreads(0)` in `bamforests.py`. Die berichteten Werte stammen aus den
besten Epochen (13 bzw. 12), die beide vor dem Hänger lagen und seit vier
Epochen rückläufig waren.

## Stand

- [x] Datenschicht, Modell, Metrik, Job-Skripte
- [x] Aufbereitung: 2456 Kacheln, 5.2 GB, GSD 1.6998 cm/px (GeoTIFF-Tag)
- [x] Erster Trainingslauf + Auswertung + Fehlerdiagnose
- [x] Zweiter Lauf mit Ankern, Jitter und F1-Auswahl: test1 0.367, test2 0.629 / 0.544
- [x] Verfahrensvergleich gegen SAM 3, Tiefen-Prompt, Hybrid, Depth Pro
- [x] EoMT und Mask2Former trainiert und verglichen: EoMT gewinnt klar (test1 0.568)
- [ ] Tiefe als vierter Eingabekanal, gegen den EoMT-RGB-Lauf gemessen
- [ ] EoMT sauber durchtrainieren (die Läufe waren nach dem Deadlock abgebrochen)
- [ ] Anwendung auf die eigenen Frames
