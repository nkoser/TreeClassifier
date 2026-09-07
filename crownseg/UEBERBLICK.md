# Instanzsegmentierung — Überblick in einer Seite

Kurzfassung dessen, was in `BERICHT.md` ausführlich steht. Chronologisch,
mit dem jeweiligen Grund für den nächsten Schritt.

## Der Ausgangspunkt

Ziel: jeden Baum einzeln erfassen, damit Stufe 2 je Krone einen Ausschnitt
bekommt. Problem: die eigenen Drohnenframes haben **keine Annotationen**. Ohne
Wahrheit lässt sich eine Segmentierung nur anschauen, nicht beurteilen — und
genau daran war die frühere Arbeit hängengeblieben. Bewertet wurde damals nach
Kronenzahl, Flächenabdeckung und dem Anteil „verdächtiger Trennungen"; letzterer
ist über die Sattelprominenz in der geschätzten Tiefe definiert, also über
dieselbe Tiefe, die die Trennung erzeugt hat. Eine Größe, die sich selbst bewertet.

Deshalb zuerst ein fremder Datensatz mit echten Kronenpolygonen: **BAMFORESTS**,
2456 Kacheln, 92 445 handdigitalisierte Kronen, 1.70 cm/px. Entscheidend ist die
Aufteilung: **Hain (test1) kommt weder im Training noch in der Validierung vor**
und ist der einzige echte Übertragungstest. test2 misst nur, wie gut bereits
gesehene Gebiete sitzen.

## Phase 1 — Messgerüst vor den Versuchen

Damit nicht jede Variante ihre eigene Erfolgsdefinition mitbringt:

- **F1 bei IoU 0.5** — Anteil getroffener Kronen bei der real gefahrenen Schwelle
- **mittlere IoU der Treffer** — trennt „findet wenig, aber sauber" von „findet viel, aber ungenau"
- **AP50** — schwellenunabhängig, nur für Verfahren mit Konfidenz je Instanz
- **je Gebiet getrennt**, nie gemittelt

Alle Verfahren schreiben Labelkarten im selben Format (`eval_labels.py`), damit
sich auch ältere Läufe nachträglich einordnen lassen. Die Fensterlogik liegt in
einem gemeinsamen Modul (`tiling.py`) — ein Vergleich, bei dem die Verfahren
unterschiedlich kacheln, misst die Kachelung mit.

## Phase 2 — Vorhandene Werkzeuge verschalten

Gemeinsam ist allen: **keine Kronenannotation als Trainingsziel**. Entweder
vortrainierte Allzweckmodelle plus handgebaute Regeln, oder — bei crownnet —
ein Training auf Flaechenkarten statt auf Instanzen.

| Verfahren | F1 Hain | Anmerkung |
|---|---:|---|
| crownnet (Innen/Rand/Zentrum + Watershed) | 0.063 | Ausgangsstand, auf Flaechenkarten trainiert |
| Tiefe + SAM 1, promptbar | 0.227 | |
| Hybrid SAM 1 + Watershed | 0.246 | |
| SAM 3, Textprompt `tree` | 0.281 → **0.312** | Größen-Prior war falsch gesetzt |
| + Depth Pro zum Teilen | 0.342 | Depth-Anything nur 0.315 |
| + Wipfel als Prompts säen | 0.348 | Zugewinn kommt fast nur vom Teilen |

**Schluss dieser Phase:** alles Kombinieren der vorhandenen Werkzeuge bleibt
zwischen 0.23 und 0.35 stecken. Zwei Sackgassen wurden dabei sauber
ausgeschlossen: Kronen in der von SAM verworfenen Restfläche zu suchen trifft in
291 Fällen **kein einziges Mal** (heißt: gegen die Entscheidung eines Modells
arbeiten, das mit IoU 0.78 gut abgrenzt), und Wipfel-Prompts liefern statt der
erhofften 170 nur 12 neue Kronen.

## Phase 3 — Training auf echten Annotationen

| Architektur | test2 Stadtwald | test2 Tretzendorf | **test1 Hain** | IoU | AP50 |
|---|---:|---:|---:|---:|---:|
| **EoMT (DINOv3)** | 0.688 | 0.698 | **0.624** | 0.773 | 0.507 |
| EoMT, kurzer Zeitplan | 0.721 | 0.687 | 0.554 | 0.773 | 0.406 |
| Mask2Former (Swin) | 0.565 | 0.562 | 0.454 | 0.708 | 0.372 |
| Mask R-CNN | 0.629 | 0.544 | 0.367 | 0.706 | 0.241 |

Mask R-CNN startete bei 0.144 mit 2628 Vorhersagen für 780 Kronen. Die Ursache
war messbar: Hain hat 21.5 Kronen je Kachel gegen 47.5 im Stadtwald, Median-
Durchmesser 392 px gegen 281 px — die Bäume sind großflächiger, das Modell
zerlegte sie. Drei Korrekturen brachten 0.367: Ankerleiter 32–512 → 64–1024 px
(eine 842-px-Krone konnte die RPN vorher gar nicht vorschlagen), Maßstabsjitter
auch nach oben, Modellauswahl nach Instanz-F1 statt nach Validierungsverlust.

Die query-basierten Architekturen haben das Ankerproblem gar nicht erst: feste
Anfragen, jede mit eigener Maske und eigenem Score, keine Anker, kein NMS.

**Bemerkenswert ist nicht die Rangfolge, sondern der Einbruch aufs fremde
Gebiet:** EoMT fällt von 0.72 auf 0.62, Mask R-CNN von 0.59 auf 0.37. EoMT
überträgt besser, nicht nur absolut besser.

Streuung zwischen zwei Läufen derselben Konfiguration: 0.015 bis 0.035.
Unterschiede in dieser Größenordnung sind nicht interpretierbar; der Abstand zu
Mask2Former (0.100) liegt darüber.

## Phase 4 — Fehler im Messen selbst

Vier davon haben Zahlen verändert, nicht nur Code:

1. **Fenster verwarf große Kronen.** „Alles verwerfen, was eine Fensterkante
   berührt" tötet jede Krone, die breiter als die Überlappung ist — in *jedem*
   Fenster. Ersetzt durch Zuordnung über den Schwerpunkt, Überlappung 768 px.
2. **Kachelauswahl nahm die ersten N alphabetisch.** Tretzendorf fiel dadurch aus
   test2 heraus; die frühen test2-Zahlen waren reiner Stadtwald.
3. **Modellauswahl nach Validierungsverlust** griff Epoche 2 von 25 — bei
   Detektion steigt der Verlust, während die Genauigkeit weiter zunimmt.
4. **AP je Kachel gemittelt** überschätzt: kurze Kurven erreichen leicht 1.0.
   Gepoolt liegt test1 bei 0.406 statt 0.458.

## Phase 5 — Widerlegte eigene Hypothesen

Auf den eigenen Frames wirken die Kronen zu grob. Geprüft und **alle verworfen**:
Inferenzmaßstab, Multiskalen-Stufen, Konfidenzschwelle, feinerer Datensatz
(Quebec), variables Bildfeld im Training. Der vorhergesagte Kronendurchmesser
bleibt bei 2.3–2.45 m, auch bei Maßstab ×1.2 gegen ×2.8. Ob das *falsch* ist,
lässt sich ohne eine unabhängige Messung der echten Kronengrößen nicht sagen.

Ebenfalls geschlossen: **Höhe hilft der Segmentierung nicht.** Vier
Fusionswege (Tiefe als vierter Kanal, eigener Zweig mit Gate, nur Tiefe,
Tiefe zum Nachteilen) plus Ruschhaupt et al. mit einem echten photogrammetrischen
CHM kommen unabhängig zum selben Ergebnis.

## Phase 6 — Dichte DINOv3-Merkmale

Statt Krone ausschneiden → ein Vektor: je 16×16-Bildfeld ein eigener Vektor,
direkt geclustert.

- **Arten:** NMI 0.43–0.49, Reinheit bis **79 %** bei 20 Clustern, ohne
  Segmentierung und ohne Labels. Kontrollen: reine Farbe 0.15, reine Position
  0.06 — beides schließt die naheliegenden Scheinerklärungen aus.
- **Einzelkronen:** F1 0.045–0.142 gegen 0.624. Trennt Arten und Bestände, nicht
  Nachbarbäume derselben Art.

**Ersetzt die Segmentierung also nicht**, könnte aber die Artstufe tragen.

## Stand

Segmentierung: **EoMT, F1 0.624** auf nie gesehenem Gebiet, mittlere IoU 0.773.
Zum Vergleich die Baseline der BAMFORESTS-Autoren: AP50 69.05/68.89 gegen unsere
62/56 — auf anderen Testteilmengen, also nicht direkt vergleichbar, aber wir
liegen darunter.
