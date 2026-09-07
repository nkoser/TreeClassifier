# Einzelkronen auf BAMFORESTS — Versuchsbericht

Stand: 20.08.2026 · Datensatz: BAMFORESTS, 92 445 Kronenpolygone · Prüfgebiet: Hain (test1), im Training nie gesehen

## Kurzfassung

Ziel ist eine Segmentierung, die jeden Baum einzeln erfasst — Voraussetzung dafür,
später je Krone einen Ausschnitt an die Artbestimmung zu geben. Der Ausgangsstand
(`crownnet.py`) traf 6 % der Kronen. Der heutige Stand trifft 55 %.

Der Sprung kam **nicht** aus dem geschickteren Kombinieren der vorhandenen
Werkzeuge. SAM, SAM 3, Depth Pro und Tiefen-Watershed sind in jeder geprüften
Verschaltung zwischen 23 % und 35 % steckengeblieben. Er kam aus dem Training auf
echten Kronenannotationen mit einer query-basierten Architektur.

| | |
|---|---|
| F1 zu Beginn (crownnet) | 0.063 |
| bestes Werkzeug-Gespann | 0.348 |
| **EoMT, trainiert** | **0.554** |
| mittlere IoU der Treffer | 0.773 |

## Warum überhaupt ein fremder Datensatz

Die eigenen Drohnenframes haben keine Annotationen. Ohne Wahrheit lässt sich jede
Segmentierung nur anschauen, nicht beurteilen — und genau daran ist die frühere
Arbeit hängengeblieben: bewertet wurde nach Kronenzahl, Flächenabdeckung und dem
Anteil „verdächtiger Trennungen". Letzterer ist über die Sattelprominenz in der
geschätzten Tiefe definiert, also über dieselbe Tiefe, die die Trennung erzeugt
hat. Eine Größe, die sich selbst bewertet.

**BAMFORESTS** (Troles et al. 2024) löst das. 2456 Kacheln zu 2048 × 2048 px aus
vier Wäldern um Bamberg, 1.70 cm Bodenauflösung, 92 445 handdigitalisierte
Kronenpolygone im COCO-Format — echte Instanzgrenzen, nicht nur Boxen.

| Split | Kacheln | Kronen | Gebiete | Rolle |
|---|---|---|---|---|
| train | 1439 | 58 228 | Stadtwald, Tretzendorf | Training |
| val | 382 | 15 177 | Stadtwald, Tretzendorf | Modellauswahl |
| **test1** | 313 | 6 720 | **Hain** | einziger echter Übertragungstest |
| test2 | 322 | 12 320 | Stadtwald, Tretzendorf | bekannte Gebiete |

**Nur test1 zählt.** Hain kommt weder im Training noch in der Validierung vor.
test2 misst lediglich, wie gut die bereits gesehenen Gebiete sitzen — wer beide
Zahlen zusammenwirft, berichtet ein besseres Ergebnis, als er hat.

> **Eigenart der Annotation.** Nur rund 57 % der Kachelfläche gehört überhaupt zu
> einer Krone. Der Rest ist Schatten, Unterwuchs, Lücke — unmarkiert. Alles, was
> ein Verfahren dort findet, zählt als Fehlalarm, selbst wenn tatsächlich ein Baum
> steht. Und es gibt nur die eine Klasse `tree`: dass ein Hausdach kein Baum ist,
> kann kein Modell aus diesen Daten lernen.

## Das Messgerüst

Vor den Versuchen stand eine verfahrensunabhängige Messung, damit alles Weitere
vergleichbar bleibt und nicht jede Variante ihre eigene Erfolgsdefinition
mitbringt.

| Größe | Was sie beantwortet |
|---|---|
| F1 @ IoU 0.5 | Anteil der Kronen, die bei der tatsächlich gefahrenen Konfidenzschwelle getroffen werden. Gierige Zuordnung nach fallender Konfidenz, jede wahre Krone höchstens einmal belegt |
| mittlere IoU | Wie gut die **getroffenen** Kronen sitzen — trennt „findet wenig, aber sauber" von „findet viel, aber ungenau" |
| AP50 | Schwellenunabhängige Güte der Rangfolge. Nur für Verfahren mit Konfidenz je Instanz sinnvoll |
| je Gebiet | Getrennt berichtet, nie gemittelt. Hain verhält sich anders als Stadtwald, und genau dieser Unterschied ist die interessante Größe |

Alle Verfahren schreiben Labelkarten im selben Format, unabhängig davon, mit
welchem Code sie entstanden sind (`eval_labels.py`). Damit ließen sich auch die
älteren Läufe nachträglich einordnen, ohne sie anzufassen. Ebenso wurde die
Fensterlogik in ein gemeinsames Modul gezogen (`tiling.py`) — ein Vergleich, bei
dem die Verfahren unterschiedlich kacheln, misst die Kachelung mit.

## Die Versuche

### 01 · Der Ausgangsstand: drei Karten plus Watershed — *verworfen*

`crownnet.py` sagt Inneres, Rand und Zentrum vorher und schneidet die Instanzen
hinterher per Watershed heraus. Gemessen auf Hain: **F1 0.063**.

Zwei strukturelle Gründe, die kein längeres Training behebt. Erstens überlappen
sich Kronen — eine einzige Labelkarte kann das nicht abbilden, beim Rastern
überschreibt die letzte Krone die vorherige. Das Trainingsziel selbst ist damit
falsch. Zweitens sagt das Netz nie eine Instanz vorher, nur wo ein Inneres
aufhört; die Trennentscheidung trifft der Watershed auf einer Karte, die dafür
nicht optimiert wurde. Eine Konfidenz je Krone gibt es ebenfalls nicht.

### 02 · Mask R-CNN: jede Krone eine eigene Maske — *teilweise*

Erster Trainingslauf: auf bekanntem Gebiet **0.627**, auf Hain **0.144** — bei
2628 Vorhersagen für 780 Kronen. Massive Überproduktion.

Die Ursache war messbar, nicht vage: bei gleicher annotierter Flächenabdeckung hat
Hain nur 21.5 Kronen je Kachel gegen 47.5 im Stadtwald. Die Bäume dort sind
großflächiger — Median-Durchmesser 392 px gegen 281 px, p95 842 px gegen 503 px.
Das Modell zerlegte sie.

Drei Korrekturen: Ankerleiter von 32–512 px auf 64–1024 px gestreckt (eine
842-px-Krone konnte die RPN vorher gar nicht vorschlagen), Maßstabsjitter auch
nach oben statt nur nach unten, und Modellauswahl nach Instanz-F1 statt nach
Validierungsverlust. Ergebnis: Hain **0.367**.

### 03 · SAM 3 mit Textprompt `tree` — *teilweise*

Ohne jedes Training **0.281** — mehr als der erste Mask-R-CNN-Stand. Die mittlere
IoU der Treffer lag bei 0.772, die besten Ränder im ganzen Feld.

Dabei war der Lauf durch einen falschen Größen-Prior ausgebremst: `--crown-px 100`
setzt die erwartete Kronenfläche auf 7854 px², und mit dem Faktor 5 flog alles über
39 270 px² heraus. Eine echte Hain-Krone hat rund 59 000 px² — korrekt gefundene
große Kronen wurden als „zu groß" verworfen. Dazu die 2×2-Kachelung, die bei
274-px-Kronen in 717-px-Fenstern zwangsläufig zuschlägt. Korrigiert: **0.312**.

### 04 · Tiefe zum Trennen — *trägt*

Eine SAM-Maske, die über mehreren Wipfeln liegt, wird an den Wipfeln aufgebrochen
— Watershed im Inneren der Maske, Prominenz relativ zur jeweiligen Instanz
gemessen.

Mit Depth Pro: **0.312 → 0.342**. Mit Depth-Anything-V2 nur 0.315. Für das Trennen
zweier benachbarter Wipfel zählt die Kantenschärfe der Tiefe, nicht ihre metrische
Richtigkeit — die Vermutung hat sich bestätigt.

### 05 · Tiefe als Detektor auf der Restfläche — *verworfen*

Wipfel in der von SAM nicht erfassten Fläche bekommen ein Watershed-Becken. Von
**291 so ergänzten Kronen trifft keine einzige** eine echte Krone bei IoU 0.5; bei
IoU 0.1 sind es 2.7 %. Kein Schwellenartefakt.

Der Grund ist strukturell: die Restfläche ist per Konstruktion das, was SAM 3 für
keinen Baum gehalten hat. Dort noch Kronen zu suchen heißt, gegen die Entscheidung
eines Modells zu arbeiten, das auf dieser Aufgabe eine mittlere IoU von 0.78
erreicht. Derselbe Schritt steckt unverändert in `segment_hybrid.py` und erklärt
dessen 0.246.

### 06 · Wipfel als zusätzliche Prompts — *kaum Wirkung*

Naheliegender Einwand: bei der Ergänzung kam die *Form* aus dem Watershed — ein
Punkt-Prompt nimmt aus der Tiefe nur das *Wo* und lässt die Grenze beim
Bildmodell. Die Vorabmessung sprach dafür: 64–67 % der freien Wipfel liegen in
einer Krone, die SAM 3 verpasst hat, Obergrenze F1 0.606.

Gebaut und gemessen: 170 freie Wipfel → 113 Kandidaten → 101 zu stark überlappend
→ **12 Kronen**. Gesamtergebnis 0.348 statt 0.342, und dieser Zugewinn kommt fast
vollständig vom aggressiveren Teilen, nicht von den 12 gesäten Kronen.

Die Obergrenze war zu optimistisch, weil sie unterstellte, jeder freie Wipfel werde
eine eigenständige *neue* Krone. Tatsächlich zeigen die Wipfel überwiegend auf
Bäume, die SAM 3 bereits hat. Für die verpassten Kronen liefert auch die Tiefe
keinen eigenen Wipfel.

### 07 · Query-basierte Architekturen — *trägt deutlich*

EoMT mit DINOv3-Backbone und Mask2Former mit Swin, beide über denselben Datenpfad,
denselben Bodenausschnitt, dieselbe Fensterlogik und dieselbe Metrik. Beide sagen
feste Anfragen vorher, jede mit eigener Maske und eigenem Score — keine Anker, kein
NMS, keine Nachbearbeitung. Das Ankerproblem aus Versuch 02 kann dort nicht
auftreten.

Auf Hain: **EoMT 0.554**, Mask2Former 0.454, Mask R-CNN 0.367. EoMT hält dabei eine
mittlere IoU von 0.773 — SAM-Niveau, das kein anderes trainiertes Modell erreicht.

Damit erübrigt sich ein zwischenzeitlich erwogener Plan, gute Auswahl mit
SAM-Rändern zu kombinieren: EoMT liefert beides zugleich.

## Gesamtergebnis

Alle Verfahren auf denselben 40 Kacheln aus test1, 780 wahre Kronen, IoU ≥ 0.5,
gemessen über Labelkarten.

| Verfahren | Vorhersagen | Präzision | Trefferquote | F1 | IoU |
|---|---|---|---|---|---|
| **EoMT (DINOv3), trainiert** | 814 | 0.478 | 0.499 | **0.488** | 0.755 |
| Mask R-CNN, trainiert | 424 | 0.583 | 0.317 | 0.410 | 0.714 |
| SAM 3 + Depth Pro, teilen + säen | 932 | 0.320 | 0.382 | 0.348 | 0.725 |
| SAM 3 + Depth Pro, nur teilen | 570 | 0.405 | 0.296 | 0.342 | 0.764 |
| SAM 3 `tree`, korrigiert | 509 | 0.395 | 0.258 | 0.312 | 0.780 |
| Hybrid: SAM 1 + Watershed | 812 | 0.241 | 0.251 | 0.246 | 0.776 |
| Tiefe + SAM 1, promptbar | 304 | 0.405 | 0.158 | 0.227 | 0.759 |
| crownnet (Ausgangspunkt) | — | 0.074 | 0.056 | 0.063 | 0.682 |

Architekturvergleich über die vollen Splits, 40 gleichmäßig gegriffene Kacheln je
Split, direkte Instanzen statt Labelkarten:

| Architektur | test2 Stadtwald | test2 Tretzendorf | **test1 Hain** | IoU Hain | AP50 Hain |
|---|---|---|---|---|---|
| **EoMT (DINOv3)** | 0.688 | 0.698 | **0.624** | 0.773 | 0.507 |
| EoMT, kurzer Zeitplan | 0.721 | 0.687 | 0.554 | 0.773 | 0.406 |
| Mask2Former (Swin) | 0.565 | 0.562 | 0.454 | 0.708 | 0.372 |
| Mask R-CNN | 0.629 | 0.544 | 0.367 | 0.706 | 0.241 |

Bemerkenswert ist nicht nur die Rangfolge, sondern der kleinere Einbruch vom
bekannten auf das fremde Gebiet: EoMT fällt von 0.72 auf 0.55, Mask R-CNN von 0.59
auf 0.37. Es überträgt besser, nicht nur absolut besser.

Die EoMT-Zahlen stammen aus dem vollständigen 30-Epochen-Lauf. Der frühere,
abgebrochene Lauf lag bei 0.568 / 0.738 / 0.723 — die Streuung zwischen zwei
Läufen derselben Konfiguration liegt also bei rund 0.015 bis 0.035. Unterschiede
in dieser Größenordnung sind nicht interpretierbar; der Abstand zu Mask2Former
(0.100) und Mask R-CNN (0.187) liegt deutlich darüber.

## Was die Zahlen nicht zeigen

Der Bildvergleich gegen die Wahrheit (`show_gt.py --labels`) legt zwei Fehlerarten
offen, die in keiner Kennzahl auftauchen.

**Verschmelzen im dichten Bestand.** Wo die Annotation drei bis vier Bäume trennt,
liegt oft eine einzige EoMT-Maske. Das passt zum Bild aus Trefferquote 0.499 bei
gleichzeitig guter IoU der Treffer: was getroffen wird, sitzt gut, aber im Gedränge
wird zusammengefasst.

**Nicht-Bäume.** Hausdächer werden als Krone segmentiert. Der Datensatz kennt nur
die Klasse `tree`, alles andere ist unmarkierter Hintergrund — dass ein Dach kein
Baum ist, steht in diesen Daten nirgends.

## Fehler auf dem Weg

Vier davon haben Ergebnisse verfälscht, bevor sie auffielen. Sie stehen hier, weil
sie erklären, warum einzelne Zwischenstände so aussahen, wie sie aussahen.

- **Gleitendes Fenster verwarf große Kronen vollständig.** Die Regel „verwirf
  alles, was den Fensterrand berührt" trifft eine Krone, die breiter als der
  Überlapp ist, in *jedem* Fenster — sie verschwand komplett. Ersetzt durch
  Zuordnung über den Kronenmittelpunkt bei Überlapp größer als die größte
  erwartete Krone.
- **Kachelstichprobe nahm die ersten N alphabetisch.** Damit fiel Tretzendorf aus
  test2 vollständig heraus; die frühen test2-Zahlen waren reiner Stadtwald.
  Ersetzt durch gleichmäßiges Greifen über den Split.
- **Modellauswahl nach Validierungsverlust.** Bei Detektionsmodellen steigt der
  Verlust routinemäßig weiter, während die Genauigkeit noch zunimmt — ausgewählt
  wurde deshalb Epoche 2 von 25. Ersetzt durch Instanz-F1 auf ganzen
  Validierungskacheln.
- **Obergrenze gegen den falschen Lauf gerechnet.** Die 0.606 für den Saatschritt
  bezogen sich auf die Masken einer anderen Konfiguration als der, die sie
  beurteilen sollten.

> **Betriebsstörung.** Beide Trainingsläufe blieben nach 17 bzw. 16 Epochen stehen
> — Haupt- und Workerprozesse in `do_poll`, GPU im Leerlauf, während SLURM sie
> weiter als laufend führte. Ursache ist der OpenCV-Threadpool in geforkten
> DataLoader-Workern; behoben mit `cv2.setNumThreads(0)`. Der Wiederholungslauf
> über volle 30 Epochen bestätigt den Stand (beste Validierungs-F1 0.707 gegen
> 0.718 im abgebrochenen Lauf, auf test1 0.554 gegen 0.568 — Streuung, kein
> Unterschied). Berichtet werden hier durchgehend die Zahlen des vollständigen
> Laufs.

## Was gerade läuft

- **Tiefenkanal: erledigt, ohne nachweisbaren Nutzen.** Depth Pro über alle 2456
  Kacheln vorgerechnet, EoMT mit der Tiefe als viertem Eingabekanal trainiert
  (Nullgewichte im neuen Kanal, damit der Vergleich keinen
  Initialisierungssprung mitmisst).

  | | test1 Hain | test2 Stadtwald | test2 Tretzendorf |
  |---|---|---|---|
  | RGB | 0.554 | 0.721 | 0.687 |
  | RGB + Tiefe | 0.559 | 0.733 | 0.698 |

  Alle drei Differenzen sind positiv, liegen mit +0.005 bis +0.012 aber
  **unterhalb der Streuung zwischen zwei identischen Läufen** (0.015–0.035).
  Ein Nutzen lässt sich daraus nicht ableiten; dafür bräuchte es mehrere Läufe
  je Variante.

  **Nachtrag: der Lauf nur auf der Tiefe.** Derselbe Aufbau ohne jedes
  Farbpixel (`--depth-only`), lange nur auf der Validierung gemessen und
  deshalb hier nie berichtet. Auf test1 nachgeholt:

  | | test1 Hain | test2 Stadtwald | test2 Tretzendorf | IoU Hain | AP50 |
  |---|---|---|---|---|---|
  | RGB | 0.554 | 0.721 | 0.687 | 0.773 | 0.406 |
  | **nur Tiefe** | **0.514** | 0.646 | 0.666 | 0.748 | 0.379 |
  | RGB + Tiefe | 0.559 | 0.733 | 0.698 | — | — |

  **93 % der RGB-Güte ohne Bild**, bei fast gleicher Randqualität. Das ordnet
  den Tiefen-Strang neu ein: nicht *wenig Information über Kronen* ist der
  Grund, warum Fusion nichts bringt, sondern **dieselbe** Information. Depth
  Pro schätzt monokular aus genau diesem Bild — die Tiefenkarte ist keine
  zweite Messung, sondern eine gelernte Umformung der ersten. Dass sie allein
  fast trägt, zeigt, dass die Kronenstruktur die Umformung übersteht; dass sie
  beim Dazufügen nichts bringt, zeigt, dass sie nichts Neues mitbringt.

  Das erklärt zugleich, warum Ruschhaupt et al. mit einem *echten*
  photogrammetrischen CHM zum selben Ergebnis kommen: bei geschlossenem
  Kronendach fehlt die Trenninformation auch der gemessenen Höhe.

  **Zu den feinabgestimmten Karten aus `depthft/`:** dort wurde Depth Pro auf
  FORTRESS nachtrainiert und der Skalenfehler behoben (AbsRel 0.98 -> 0.120).
  Für die Segmentierung ist das ohne Belang, weil `depthcache.py` die Tiefe
  **je Kachel normiert** ablegt -- der absolute Maßstab, den das Feinabstimmen
  repariert, wird verworfen, bevor das Modell die Karte sieht. In der relativen
  Struktur, die wir tatsächlich benutzen, liegt das pure Modell mit
  geschenktem Skalenfaktor sogar vorn (AbsRel 0.074 gegen 0.120). Offen bleibt
  die Kantenschärfe an Kronengrenzen, die AbsRel nicht misst und die schon
  einmal den Unterschied machte (Depth Pro 0.342 gegen Depth-Anything 0.315).
- **Auswertung.** Erledigt: test1 0.554, test2 0.721 / 0.687. Die Zahlen oben
  sind bereits die des vollständigen Laufs.
- **Offen: Nachtrainieren auf eigenen Annotationen.** Dort steht der Maßstab im Weg:
  BAMFORESTS hat 1.70 cm/px, ein 1920-px-Frame aus 100 m Höhe rund 7.8 cm/px —
  Faktor 4.6. Die Frames werden entsprechend hochskaliert, aber **prüfen lässt
  sich das nicht**, solange keine Annotationen auf eigenen Aufnahmen existieren.
  Zehn markierte Frames würden reichen, um aus dem Eindruck eine Zahl zu machen.

## Anwendung auf die eigenen Frames

Alle Modelle laufen über `crownseg/sbatch/run_frames.sbatch` auf
`/cold/Mahfuz/chosen_frames` und schreiben Labelkarten nach
`results_frames_<modell>/` sowie die vier Diagnoseansichten nach
`results_views_<modell>/`.

### Der Bildmaßstab war falsch angenommen

Die Rechnung „100 m Flughöhe → 7.8 cm/px → Faktor 4.6" ist formal richtig, aber
die 100 m waren nie gemessen, sondern ein Vorgabewert. Mit Faktor 4.6 fand Mask
R-CNN auf allen 33 Frames **exakt null** Kronen, auch bei Konfidenzschwelle 0.05.

`scale_probe.py` misst den Maßstab stattdessen aus der Modellantwort: derselbe
Bodenausschnitt in verschiedenen Auflösungen, und gesucht wird das Maximum aus
Instanzzahl und Konfidenz. Mask R-CNN eignet sich dafür gerade wegen seiner
festen Ankergrößen — bei falschem Maßstab bricht seine Konfidenz ein, während
ein query-basiertes Modell bei jedem Maßstab etwas liefert.

| Ordner | Maximum bei | Kronen | Konfidenz | → GSD | → Flughöhe |
|---|---|---|---|---|---|
| pines | ×1.0–1.4 | 39 | 0.95 | ~1.2–1.7 cm/px | 16–22 m |
| dense | ×1.0–1.4 | 32 | 0.95 | ~1.2–1.7 cm/px | 16–22 m |
| dense1 | ×1.0–1.4 | 32 | 0.93 | ~1.2–1.7 cm/px | 16–22 m |
| mixed | ×1.0 | 20 | 0.94 | 1.70 cm/px | 22 m |
| 100, 80m, mixed1, urban | kein Maximum im Inneren | ≤ 5 | — | unbestimmt | unbestimmt |

Die Sonde hat eine Reichweite, die zu beachten ist: sie nutzt Mask R-CNN als
Messgerät, und das reagiert auf urbanem Material praktisch nicht (maximal 1.2
Instanzen bei Konfidenz 0.54). Für diese vier Ordner ist sie blind. Die daraufhin
gesetzte Vorgabe ×1.0 ist **keine neutrale Annahme** — sie behauptet, der
Bildmaßstab entspreche zufällig genau dem von BAMFORESTS.

Für `urban` ergab ein Sichtvergleich über ×0.4 bis ×3.0 ein deutliches Optimum
bei **×1.5** (11 → 26 → 40 Kronen bei ×0.8 / ×1.0 / ×1.5, darüber wieder
schlechter). Bei ×2.0 und höher — als einzelner Maßstab gefahren — füllt das
Modell ganze Suchfenster mit Masken, wo Rasen liegt: Masken mit geraden Kanten,
die keine Krone sein können.

Die naheliegende Folgerung, dass die obere Multiskala-Stufe für die Maske über
der Parkwiese verantwortlich sei, ist jedoch **geprüft und falsch**: mit Stufen
0.7/1.0/1.4 (bis ×2.1) und mit 0.7/1.0 (bis ×1.5) entsteht sie gleichermaßen,
41 gegen 39 Kronen bei sonst nahezu identischem Ergebnis. Zusammen mit dem
gescheiterten Texturfilter ist damit doppelt belegt, was der Datensatz schon
nahelegte: es ist kein Parameterproblem.

Zwei Einschränkungen: ab Faktor ×3 findet kein Modell mehr etwas, weil kubisch
hochgerechnete JPEG-Frames ihre Textur verlieren — dort ist „falscher Maßstab"
nicht von „Bild zerstört" zu unterscheiden. Und die Ordnernamen `100` und `80m`
widersprechen der Messung; das lässt sich nur mit den echten Flugdaten klären.

### Multiskala und Schwelle

Der visuelle Vergleich gegen die früheren SAM-3-Läufe zeigte eine deutlich
geringere Abdeckung des Kronendachs. Zwei Ursachen, beide behoben:

| Variante (pines/frame_000006) | Kronen | Abdeckung |
|---|---|---|
| SAM 3 multiskala | 125 | 77 % |
| EoMT, ein Maßstab, Schwelle 0.5 | 131 | 63 % |
| EoMT multiskala, Schwelle 0.5 | 158 | 68 % |
| **EoMT multiskala, Schwelle 0.25** | **181** | **69 %** |

**Multiskala** (`--scale-steps 0.7 1.0 1.4`) ist Vorsorge gegen den unbekannten
Maßstab, genau wie die Kachelstufen in `segment_sam3.py`. Auf BAMFORESTS, wo der
Maßstab bekannt ist, ändert es nichts (0.469 gegen 0.488, innerhalb der
Streuung); auf den Frames bringt es 63 % → 68 % Abdeckung.

**Die Schwelle 0.5 war ein ungeprüfter Vorgabewert.** Gemessen auf test1:

| Schwelle | Vorhersagen | Präzision | Trefferquote | F1 | mittl. IoU |
|---|---|---|---|---|---|
| 0.50 | 818 | 0.578 | 0.532 | 0.554 | 0.773 |
| 0.35 | 896 | 0.557 | 0.561 | **0.559** | 0.770 |
| 0.25 | 957 | 0.533 | **0.574** | 0.553 | 0.769 |
| 0.15 | 1091 | 0.481 | 0.591 | 0.530 | 0.767 |

Von 0.50 auf 0.25 steigt die Trefferquote um 8 % relativ, während die F1
unverändert bleibt. Entscheidend: die mittlere IoU bleibt bei 0.77 — die
zusätzlichen Kronen sind genauso sauber abgegrenzt, es kommt kein
Bruchstück-Müll dazu.

### Nicht-Bäume: nachgelagert nicht reparierbar

Auf den urbanen Frames segmentiert EoMT Parkwiesen, Wegränder und Dachkanten.
Ursache ist der Datensatz: BAMFORESTS ist reiner Wald mit der einzigen Klasse
`tree`, alles andere ist unmarkierter Hintergrund und **kein Gegenbeispiel**. Das
Modell hat nie gelernt, dass Rasen kein Baum ist, weil es nie einen gesehen hat.

Der Versuch, das über einen Filter auf fertigen Masken zu beheben (`reject.py`,
Textur über den Laplace-Betrag und Relief gegen einen Ring im Ersatz-CHM), ist
gemessen gescheitert:

| Texturschwelle | Vorhersagen | Präzision | Trefferquote | F1 |
|---|---|---|---|---|
| ohne | 789 | 0.466 | 0.472 | 0.469 |
| 0.8 | 767 | 0.473 | 0.465 | 0.469 |
| 1.2 | 740 | 0.482 | 0.458 | 0.470 |
| 1.4 | 716 | 0.486 | 0.446 | 0.465 |

Die Präzision steigt, die Trefferquote fällt im gleichen Maß, die F1 bleibt über
den ganzen Bereich flach. Der Filter entfernt keine Nicht-Kronen, er wirkt wie
eine strengere Konfidenzschwelle und schneidet Grenzfälle unterschiedslos weg.

Die Verteilungen erklären warum: die urbanen Frames haben *höhere* Textur als die
Waldframes (Median 1.9–2.9 gegen 1.7–1.9) — es sind kontrastreiche
Satellitenaufnahmen mit scharfen Gebäudekanten. Textur und Relief messen
Eigenschaften, die dichte Kronen und dichte Wiesen teilen.

Auch der naheliegende Ausweg trägt nicht: unannotierte BAMFORESTS-Flächen als
Gegenbeispiele zu verwenden geht schief, weil dort 43 % der Fläche unmarkiert
sind und ein großer Teil davon sehr wohl Bäume enthält.

Was hilft, sind echte Gegenbeispiele — annotierte Kacheln, in denen Wiese, Weg
und Dach als Nicht-Baum markiert sind.

### Was bleibt

Die Abdeckung sättigt bei rund 69 % und erreicht SAM 3s 77 % nicht. Der
Unterschied ist aber nicht der, nach dem es aussieht: EoMT findet **mehr** Kronen
auf **weniger** Fläche (181 gegen 125), es zerlegt das Kronendach feiner. Welche
Auffassung richtig ist, hängt an der tatsächlichen Baumgröße im Bestand.

Die verbleibende Lücke ist der Annotationspolitik von BAMFORESTS zuzuschreiben —
57 % annotierte Fläche, Schatten und Lücken unmarkiert — und über Parameter nicht
weiter zu schließen. Auf deinen Frames zählt jede Kachel als Wald, im Datensatz
nicht.

Abdeckung je Ordner im Endstand: 58 % (`urban`) bis 76 % (`dense1`), im Mittel
rund 68 %.

## Einordnung gegen die Literatur

Erst spät geprüft, und der Vergleich korrigiert mehrere eigene Annahmen.

**Der Maßstab war richtig gemessen.** Das BAMFORESTS-Paper nennt GSD 1.61–1.82 cm
(Stadtwald 1.70 cm) — genau der aus dem GeoTIFF-Tag gelesene Wert.

**Test-Set-1 ist absichtlich der harte Fall**, aber aus einem anderen Grund als
angenommen. Nicht nur die Kronengröße unterscheidet sich: die Artenverteilung ist
radikal verschieden (Pinus im Training 36.2 %, in Hain 1.1 %; „Other" 6.8 %
gegen 52.9 %), und Hain wurde mit **einer anderen Drohne und einem anderen
Sensor** aufgenommen (DJI Phantom 4, 85 m, 84° gegen Trinity F90+ / Sony RX1 RII,
120 m, 63°).

**Die Tiefen-Befunde sind unabhängig bestätigt — mit gemessener Höhe.**
Ruschhaupt, Troles & Schmid (2025) haben dieselbe Frage untersucht, mit einem
photogrammetrischen Kronenhöhenmodell (DSM minus amtliches Geländemodell) im
Alphakanal, also demselben vierten Kanal. Ergebnis: RGB schlägt RGBA um 0.87 %
(Mask R-CNN) und 1.18 % (Mask2Former); Höheninformation hat „einen negativen
Einfluss". Damit ist auch die Frage beantwortet, die hier als unbeantwortbar
galt: *gemessene* Höhe hilft ebenso wenig wie geschätzte.

**Ein Feintuning von SAM 3 lohnt nicht.** Gemessene Obergrenze auf test1: von 428
GT-Kronen wird nur bei 271 überhaupt eine der 3743 Rohmasken bei IoU ≥ 0.5
fündig — **63.3 %**. Ein perfekter Auswähler auf diesen Vorschlägen käme also
kaum über EoMTs heutige Trefferquote von 0.651 hinaus. Dazu kommt, dass
`Sam3Model` in transformers keinen Trainingspfad hat; die ungarische Zuordnung
und vier Verlustterme müssten selbst geschrieben werden.

**Eigener Metrikfehler.** Die AP wurde je Kachel gerechnet und gemittelt. Bei
rund 20 Kronen je Kachel kann eine kurze Kurve leicht auf 1.0 laufen, was den
Mittelwert schönt — die gepoolte, COCO-übliche Rechnung liegt niedriger
(test1 0.406 statt 0.458). Nur die gepoolte ist mit veröffentlichten Zahlen
vergleichbar.

**Wo wir stehen.** Ruschhaupt et al. berichten auf Stadtwald+Tretzendorf 69.05 %
AP50 (Mask R-CNN) und 68.89 % (Mask2Former); wir kommen dort auf 59.9 % und
58.8 %. Wir liegen also **unter** den veröffentlichten Grundlinien — die frühere
Formulierung „EoMT ist der klare Sieger" galt nur gegen die eigene, schwächere
Mask-R-CNN-Umsetzung. Die Testmengen sind allerdings nicht identisch (ihr Split
aus 1621 Kacheln zu 1024 px gegen unsere 40 Kacheln zu 2048 px).

### Was der Lernratenzeitplan ausmacht

Der Versuch, die Lücke über längeres Training zu schließen, hat mehr gezeigt als
erwartet. Bei 80 Epochen à 400 Schritten fiel der beste Stand auf **Epoche 5** —
also auf 8000 Ausschnitte, weniger als der alte Lauf insgesamt sah. Trotzdem ist
das Ergebnis auf test1 deutlich besser: F1 0.624 statt 0.554, AP50 0.507 statt
0.406.

Nicht die Zahl der Schritte ist die Ursache, sondern der Zeitplan: OneCycleLR
verteilt Aufwärmen und Abklingen über die *geplante* Gesamtlänge. Bei 32 000
geplanten Schritten liegt Epoche 5 noch im Aufwärmen bei niedriger Rate; im alten
Lauf mit 6000 Schritten war sie dort längst über dem Höhepunkt. Die Lernrate 1e-4
war schlicht zu hoch.

**Und die Validierung hat dabei in die Irre geführt.** Sie läuft auf 8 Kacheln aus
Stadtwald und Tretzendorf — Gebieten, auf denen sich wenig ändert. Während sie
stagnierte, verbesserte sich test1 um 0.07. Eine Modellauswahl nach dieser
Validierung optimiert nicht das, worauf es ankommt.

## Zweiter Datensatz und variables Bildfeld

Auf den Kiefernframes fasst das Modell mehrere Bäume zu einer Maske zusammen.
Naheliegende Erklärung: BAMFORESTS annotiert Kronen mit 4.78 m (Stadtwald) bis
6.66 m (Hain) Median, Nik's Kiefern sind rund halb so groß.

**Quebec Trees** (Cloutier et al. 2023, CC-BY-4.0) deckt den fehlenden Bereich
ab. An den Polygonen selbst gemessen, nicht aus der Publikation übernommen:

| | Median-⌀ | p5 | p95 | GSD |
|---|---|---|---|---|
| Quebec Trees, 22 933 Kronen | 4.09 m | **1.82 m** | 8.54 m | 1.64 cm/px |
| BAMFORESTS Stadtwald | 4.78 m | – | – | 1.70 cm/px |
| BAMFORESTS Hain | 6.66 m | – | – | 1.82 cm/px |

Nach Art: Abies balsamea 2.78 m (n=2895), Thuja occidentalis 2.97 m (n=1510),
Picea 3.02 m (n=599) — der Bereich der Kiefern in `pines`.

Aufbereitet mit `quebec.py`: Polygone aus UTM in Pixelkoordinaten, Kacheln
2048 px wie bei BAMFORESTS, Zone 3 komplett als Testgebiet zurückgehalten.
543 / 459 / 214 Kacheln, 112 Kronen je Kachel gegen 40 bei BAMFORESTS.

**Variables Bildfeld** (`quebec_cog.py`): Ausschnitte direkt aus dem
Orthomosaik statt aus vorgeschnittenen Kacheln. Aus einer 2048er Kachel lässt
sich der Maßstab nur bis 5.4 cm/px aufweiten; aus dem 40 000 × 42 000 px großen
Mosaik beliebig weit. Die COGs haben Übersichtsstufen bis 1/128, ein 7500-px-
Fenster auf 640 heruntergelesen kostet 7 ms — weniger als eine JPEG-Kachel.

Die Grenze setzt nicht die Datenlage, sondern das Modell: **EoMT hat 200
Anfragen.** Bei 360–440 Kronen je Hektar passen rund 150 Kronen in ein Bildfeld
von 0.31 ha, das entspricht 8.7 cm/px. Zu volle Ausschnitte werden verworfen,
statt dem Netz ein unlösbares Ziel zu geben (gemessen: 0.5 % leere Ausschnitte,
Median 31 Kronen).

### Ergebnis

| | test1 Hain | test2 Stadtwald | test2 Tretzendorf | Quebec Zone 3 | Kronen-⌀ `pines` |
|---|---|---|---|---|---|
| nur BAMFORESTS | **0.624** | 0.688 | 0.698 | – | 2.45 m |
| + Quebec | 0.583 | **0.730** | **0.717** | 0.631 | 2.32 m |
| + variables Bildfeld | 0.558 | 0.721 | 0.714 | **0.638** | 2.37 m |

**Quebec bringt** auf den BAMFORESTS-Kerngebieten +0.04 und liefert erstmals
einen Wert für feinkronige Bestände (Zone 3: F1 0.631, Trefferquote 0.777 — die
höchste gemessene). Es kostet auf dem harten Übertragungstest Hain 0.04.

**Das variable Bildfeld bringt nichts:** +0.007 auf Quebec, −0.009 bis −0.025 auf
BAMFORESTS. Der Sampler arbeitet nachweislich korrekt, das Ergebnis ist also
kein Messfehler.

**Und die eigentliche Frage bleibt offen.** Der vorhergesagte Kronendurchmesser
auf `pines` liegt über alle drei Modelle bei 2.32 bis 2.45 m. Fünf Erklärungen
für die grobe Segmentierung wurden geprüft und verworfen: Maßstab bei der
Anwendung, Multiskala-Stufen, Konfidenzschwelle, feinerer Datensatz, variables
Bildfeld. Ob 2.4 m zu groß ist, lässt sich nicht entscheiden — die Zahl stammt
aus der Modellausgabe selbst.

### Der Maßstab der urbanen Aufnahmen

Aus den Fehlern selbst bestimmt: die fälschlich als Kronen segmentierten Objekte
an der Parkreihe sind 18–28 px lang bei einer Streckung von 1.5–2.0, also PKW von
oben. Bei 4.5 m Fahrzeuglänge folgt **17–25 cm/px**.

| | GSD | Faktor zu BAMFORESTS |
|---|---|---|
| Trainingsdaten | 1.6–1.8 cm/px | 1 |
| Drohnenframes | ~2.0 cm/px | 1.2 |
| **urbane Screenshots** | **~20 cm/px** | **12** |

Die Modelle sind für `urban` um den Faktor 12 falsch skaliert. Das erklärt
fehlende Bäume und Autos als Kronen zugleich. Erreichbar wäre das mit rund 800
statt 200 Anfragen, oder mit einem Datensatz in dieser Auflösung —
[OAM-TCD](https://huggingface.co/datasets/restor/tcd) bietet 10 cm/px,
280 000 Einzelbäume, 5072 Kacheln, CC-BY-4.0, 3.55 GB.

## Stufe 2: Artbestimmung auf den Instanzen

Die Instanzen aus `crownseg` gehen an den DINOvTree-Kopf (`classify.py`), mit
dem Schwerpunkt der Maske statt der Boxmitte und dem Maßstab aus der Messung
statt aus einer angenommenen Flughöhe.

### Der Klassifikator funktioniert — auf seiner Domäne

Auf Nik's Frames liefert er Unsinn: der Kiefernbestand wird zu 384 von 158
Kronen als Gelb-Birke geführt. Zwei Erklärungen kamen in Frage — Übertragung
oder eigene Verdrahtung —, und Quebec Zone 3 mit Artlabels trennt sie.

**90.1 % Genauigkeit** (1303 von 1446 Kronen), gegen 27.2 % für die häufigste
Klasse und 7.1 % Zufall. Je Art zwischen 48 % (Acer saccharum) und 100 %
(Pinus strobus, Tsuga canadensis). Die Verdrahtung ist also korrekt; die
unsinnigen Antworten auf den eigenen Frames sind ein reines Übertragungsproblem.
Der Label-Satz kennt *Pinus strobus*, nicht *Pinus sylvestris* — das Modell muss
antworten und wählt das Ähnlichste.

Beunruhigend dabei: mittlere Konfidenz **0.99 bei richtigen und 0.90 bei
falschen** Vorhersagen. Die Konfidenz taugt außerhalb der Domäne nicht als
Warnsignal.

### Clustern löst das Label-Problem

Statt in kanadische Klassen zu zwingen: die Merkmalsvektoren gruppieren, ohne
Labels. Geprüft auf Quebec Zone 3, geclustert ohne die Labels, verglichen danach:

| Cluster | ARI | NMI | Reinheit |
|---|---|---|---|
| 8 | **0.764** | 0.728 | 82.9 % |
| 12 | 0.686 | 0.725 | 86.8 % |
| 14 | 0.595 | 0.720 | **89.0 %** |
| 20 | 0.410 | 0.666 | 89.8 % |
| *beschrifteter Klassenkopf* | *0.762* | *0.773* | *90.1 %* |
| *zufällige Gruppen* | *−0.001* | *0.022* | – |

**Das Clustern ohne Labels erreicht denselben ARI wie der beschriftete
Klassifikator.** Die Merkmale tragen die Artinformation vollständig; was fehlte,
waren nur die Namen. Praktisch heißt das: zwölf bis zwanzig Cluster benennen
statt Tausende Bäume, bei rund neun von zehn richtig zugeordneten Kronen.

### Die Ausschnittsgröße war eine geerbte Annahme

DINOvTree wurde auf 9.73 m Ausschnitten trainiert, und diese Zahl wurde
unbesehen übernommen. Gemessen auf Quebec, Ausschnitt als Vielfaches des
*jeweiligen* Kronendurchmessers:

| Faktor | Kronenanteil | Genauigkeit | ARI | Reinheit |
|---|---|---|---|---|
| 1.5 | 44 % | 80.6 % | 0.496 | 83.2 % |
| **2.4** | **17 %** | **86.2 %** | **0.509** | **86.1 %** |
| 3.5 | 8 % | 84.9 % | 0.489 | 82.8 % |
| 5.0 | 4 % | 79.8 % | 0.430 | 80.3 % |
| 8.0 | 1.6 % | 69.2 % | 0.344 | 71.2 % |

Optimum bei Faktor 2.4. Zu eng ist ebenfalls schlechter — etwas Umgebung trägt
bei. Die 9.73 m entsprechen bei Quebecs 3.41-m-Kronen Faktor 2.9 und liegen
damit nahe am Optimum; bei Nik's 2.4-m-Kiefern bedeuten sie Faktor 4 und rund
vier Punkte Verlust. `classify.py` bestimmt den Ausschnitt jetzt je Krone.

Das behebt einen Parameterfehler, nicht das Übertragungsproblem: mit
angepasstem Ausschnitt bleibt die Kiefer eine Birke.

## Stufe 2 mit europäischen Arten: FORTRESS

Die vorigen beiden Abschnitte enden am selben Punkt: die Merkmale tragen die
Artinformation, aber der Kopf zeigt auf 14 kanadische Klassen. Der einzige Weg,
der die Kiefer zur Kiefer macht, ist ein Label-Satz mit europäischen Arten.

**FORTRESS** (Schiefer, Frey & Kattenborn 2022, DOI 10.35097/538, CC BY 4.0):
47 Drohnenbefliegungen im Südschwarzwald, 79 ha, GSD 0.65–1.87 cm, 9389
Artpolygone in 16 Klassen, dazu ein nDSM je Gebiet. Der Maßstab liegt im selben
Bereich wie BAMFORESTS (1.70 cm) und wie Nik's Aufnahmen; die Arten sind die,
die in seinen Beständen tatsächlich stehen.

### Beschriftung durch Verschneidung

FORTRESS liefert Artpolygone, keine Einzelkronen — die Polygone umfassen oft
mehrere Bäume derselben Art. `fortress.py` kachelt die Orthomosaike auf
1.70 cm/px um, segmentiert mit unserem EoMT, rastert die Artpolygone und
verschneidet beides: jede vorhergesagte Krone bekommt die Art, die ihre Maske
mehrheitlich überdeckt, plus `abdeckung` (welcher Anteil der Maske überhaupt
eine Klasse trägt) und `reinheit` (Anteil der Mehrheitsart).

Der erste Test davon war wertlos: ich baute die Semantikkarte aus denselben
Polygonen, die auch als Wahrheit dienten — 100 % ist dann keine Messung. Auf
vorhergesagten Kronen wiederholt: **98.8 % richtig zugeordnet** über 607 Kronen,
100 % bei den 71 %, die die Abdeckungs- und Reinheitsschwelle überstehen.

Ergebnis über alle 47 Gebiete: **9373 beschriftete Kronenausschnitte**.

| Art | n | | Art | n |
|---|---:|---|---|---:|
| Picea abies | 4669 | | Pseudotsuga menziesii | 157 |
| Fagus sylvatica | 1710 | | Larix decidua | 74 |
| Abies alba | 1018 | | Quercus spec. | 32 |
| *forest floor* | 843 | | Betula pendula | 29 |
| **Pinus sylvestris** | **458** | | *other* | 12 |
| *deadwood* | 189 | | Fraxinus excelsior | 8 |
| Acer pseudoplatanus | 173 | | | |

Waldboden und Totholz sind hier nicht Beifang, sondern das erste Mal, dass ein
Datensatz überhaupt Gegenbeispiele liefert — bisher musste jede vorhergesagte
Krone ein Baum sein.

Auffällig: die Esche fällt von 175 Polygonen auf 8 Ausschnitte. Erklärung
vermutlich Verschneidung — schmale Polygone im Kronenrandbereich verlieren die
Mehrheit an den Nachbarn.

### Der neue Kopf

`train_head.py`: derselbe eingefrorene DINOv3-Backbone, ein neuer Kopf auf die
acht Klassen mit mindestens 100 Beispielen. Eiche, Birke, Esche und Lärche
fallen raus — mit 8 bis 74 Beispielen lassen sie sich weder lernen noch messen.

Zwei Entscheidungen bestimmen, ob die Zahl etwas wert ist:

**Aufteilung nach Gebiet, nicht nach Ausschnitt.** Ausschnitte desselben
Gebiets teilen Beleuchtung, Aufnahmetag und teils denselben Baum. Elf der 47
Gebiete sind zurückgehalten; die Zahl ist damit eine Übertragungszahl, so wie
Hain bei BAMFORESTS und Zone 3 bei Quebec.

**Ausgewogene Genauigkeit statt roher.** Die Fichte stellt 59 % des Testsatzes.
Eine rohe Genauigkeit wäre vor allem ein Maß dafür, wie oft die Fichte richtig
erkannt wird.

**72.8 % roh, 65.8 % ausgewogen** über acht Klassen — gegen 59.2 % für
immer-Fichte und 12.5 % für Raten.

| Klasse | n | Trefferquote | meist vorhergesagt |
|---|---:|---:|---|
| **Pinus sylvestris** | 98 | **77.6 %** | Pinus sylvestris |
| Picea abies | 1431 | 72.4 % | Picea abies |
| Abies alba | 287 | ~70 % | Abies alba |
| Fagus sylvatica | 277 | ~67 % | Fagus sylvatica |
| Pseudotsuga menziesii | 20 | ~45 % | Pseudotsuga menziesii |
| Acer pseudoplatanus | 48 | ~35 % | **Fagus sylvatica** |
| *forest floor* | 177 | ~86 % | forest floor |
| *deadwood* | 78 | ~86 % | deadwood |

Drei Beobachtungen, die die Zahlen selbst hergeben:

*Der Bergahorn wird zur Buche.* Die Verwechslung geht fast vollständig in eine
Richtung. Beide sind Laubbäume mit ähnlicher Kronentextur, und die Buche hat
zehnmal so viele Trainingsbeispiele — bei Zweifel gewinnt die häufigere.

*Langes Training kauft fast nur Fichte.* Die rohe Genauigkeit steigt über 60
Epochen von 78 % auf 80 %, die ausgewogene bleibt bei rund 64 % stehen. Die
Auswahl nach ausgewogener Genauigkeit behält deshalb einen frühen Stand.

*Zwei Punkte Streuung zwischen Läufen.* Derselbe Aufbau mit anderem
Zufallsstartwert ergab 78.2 % / 68.2 % statt 72.8 % / 65.8 %. Der gespeicherte
Kopf ist der schwächere der beiden; ihn gegen den besseren zu tauschen hieße,
auf dem Testsatz auszuwählen. Der Startwert ist seither fest.

### Einordnung

Der Quebec-Kopf erreicht 90.1 % — aber auf handgezeichneten Einzelkronen. Unsere
Beschriftung kommt aus der Verschneidung unserer eigenen Segmentierung mit den
Artpolygonen; die Fehler der Segmentierung stecken im Label mit drin. Der
Abstand misst also nicht nur das Modell, sondern auch die Beschriftungsquelle.

Entscheidend ist nicht der Vergleich, sondern dass der Kopf *Pinus sylvestris*
antworten **kann**. Das war der ganze Grund, FORTRESS zu nehmen.

### Spielt die Instanzquelle eine Rolle?

Der Kopf oben lernte auf Kronen, die unser EoMT gefunden hat. Naheliegende
Gegenprobe: dieselbe Kette mit SAM 3 als Instanzquelle. `fortress.py` bekam
dafür einen Schalter `--segmenter sam3` — Prompt "tree", Kachelstufen 2/3/4,
Schwelle 0.15, angeschnittene Instanzen verworfen, nach Score zusammengeführt,
also genau die Einstellungen des Laufs auf den eigenen Frames. Alles danach
bleibt gleich: Verschneidung, Ausschnitt Faktor 2.4, derselbe eingefrorene
Backbone, dieselben 47 Gebiete, dieselbe Aufteilung, derselbe Startwert.

SAM 3 liefert 8840 Ausschnitte gegen 9373, aber besser verteilt: Kiefer 555
statt 458, Tanne 1336 statt 1018, Douglasie 219 statt 157, Fichte 3881 statt
4669. Die Kronen sind kleiner (Median 4.05 m gegen 4.56 m).

| Klasse | aus SAM-3-Instanzen | aus EoMT-Instanzen |
|---|---:|---:|
| **ausgewogen gesamt** | **66.8 %** | **65.8 %** |
| Abies alba | 73.4 % | ~70 % |
| Pinus sylvestris | 72.2 % | 77.6 % |
| Acer pseudoplatanus | 65.0 % | ~35 % |
| Picea abies | 59.7 % | 72.4 % |
| Fagus sylvatica | 56.5 % | ~67 % |
| Pseudotsuga menziesii | 35.7 % | ~45 % |
| *deadwood* / *forest floor* | 91.2 / 80.7 % | ~86 / ~86 % |

**Kein Unterschied.** Ein Punkt liegt innerhalb der zwei Punkte, die schon
zwischen zwei Läufen desselben Aufbaus schwanken. Die rohe Genauigkeit fällt
von 72.8 % auf 64.8 %, aber nur weil der Testsatz anders zusammengesetzt ist —
die Fichte stellt hier 50.7 % statt 59.2 %. Der Sprung beim Bergahorn steht auf
20 Testkronen und ist nicht belastbar.

Dasselbe zeigt die Anwendung auf die eigenen Frames: EoMT-Kopf und SAM-3-Kopf
kommen je Ordner auf dieselbe Artverteilung (Kiefernanteil in `dense` 56 %
gegen 50 %, in `pines` 3 % gegen 4 %). Für die Artbestimmung ist es also
gleichgültig, wer die Krone ausgeschnitten hat — solange der Ausschnitt den
Baum enthält, trägt die Textur die Information. Das schließt eine der offenen
Fragen ab und spart künftig den Vergleich.

Ein Nebenbefund: der SAM-3-Kopf ist weniger überzeugt von sich (mittlere
Sicherheit 0.65 gegen 0.79 in `pines`). Nach dem, was der Quebec-Kopf gezeigt
hat — Konfidenz 0.90 bei *falschen* Antworten außerhalb der Domäne —, ist die
niedrigere Zahl eher ein gutes Zeichen als ein schlechtes.

### Dichte DINOv3-Merkmale: trennen die Flaechenmerkmale selbst?

Bisher lief alles ueber Ausschnitte -- eine Krone wird ausgeschnitten, der
Backbone liefert *einen* Vektor. DINOv3 liefert aber je 16x16-Bildfeld einen
eigenen Vektor, und diese Feldmerkmale tragen bekanntlich eine emergente
Segmentierung (LOST, TokenCut, STEGO holen daraus Objektgrenzen ohne jede
Maske). `dinocluster.py` misst, was davon fuer uns brauchbar ist. Backbone ist
der auf Quebec feingetunte DINOv3, k-Means auf L2-normierten Vektoren.

**Flaeche -> Art.** Gegen die Artpolygone von FORTRESS, 40 752 beschriftete
Bildfelder aus 16 Kacheln in vier Gebieten, ohne Segmentierung und ohne Labels:

| Cluster | DINOv3 NMI / Reinheit | nur RGB-Farbe | nur Ort im Bild |
|---|---|---|---|
| 4 | **0.485** / 66.7 % | 0.116 / 42.6 % | 0.032 / 33.5 % |
| 8 | 0.445 / 66.1 % | 0.143 / 44.7 % | 0.037 / 33.9 % |
| 12 | 0.430 / 70.8 % | 0.142 / 45.4 % | 0.051 / 36.6 % |
| 20 | 0.476 / **79.0 %** | 0.147 / 48.0 % | 0.063 / 38.9 % |

Haeufigste Art allein: 33.5 %.

Beide Kontrollen sind noetig und beide entlasten das Ergebnis. **Farbe** erklaert
nur ein Drittel der Transinformation -- der Verdacht, hier wuerden Herbstfarben
statt Arten sortiert, bekommt damit eine Gegenzahl. **Position** war der
ernstere Einwand: Artpolygone sind grosse zusammenhaengende Flaechen, und die
Clusterkarten sehen blockig aus, also koennte ein guter NMI schlicht daher
kommen, dass beide raeumlich glatt sind. Nur den Ort geclustert ergibt 0.03 bis
0.06 und Reinheit auf Rateniveau. Der Einwand war falsch.

**Flaeche -> Einzelkrone.** Gegen die Kronenpolygone von BAMFORESTS test1,
Zusammenhangskomponenten der Cluster als Instanzen:

| Cluster | Instanzen | echte | Treffer | F1 | mittlere IoU |
|---|---:|---:|---:|---:|---:|
| 4 | 266 | 273 | 12 | 0.045 | 0.652 |
| 8 | 416 | 273 | 25 | 0.073 | 0.648 |
| 12 | 484 | 273 | 43 | 0.114 | 0.656 |
| 20 | 711 | 273 | 70 | **0.142** | 0.660 |

Gegen 0.624 des trainierten EoMT. Die emergente Segmentierung trennt Arten und
Bestaende, nicht Nachbarbaeume derselben Art -- zwei nebeneinanderstehende
Fichten sind in den Merkmalen dasselbe Ding, und genau deren Trennung ist der
schwere Teil der Kronenabgrenzung. Die mittlere IoU von 0.66 sagt: was
getroffen wird, ist sauber getroffen; es wird nur fast nichts getroffen.

Bilder in `results_views_dinov3/` -- je Frame das Original neben der Clusterkarte
bei 6 und bei 20 Clustern.

Praktischer Schluss: die Feldmerkmale ersetzen die Segmentierung nicht, koennten
aber die Artstufe tragen -- als Artkarte ueber die Flaeche, die man mit unseren
Instanzen verschneidet. Das ist derselbe Weg, den `fortress.py` geht, nur mit
gelernten Clustern statt mit beschrifteten Polygonen.

### Depth Pro auf FORTRESS feinabstimmen?

Naheliegend, weil FORTRESS ein gemessenes nDSM je Gebiet mitliefert: RGB rein,
Höhe in Metern raus, keine Maßstabsmehrdeutigkeit, 79 ha bei rund 1 cm. Für die
Segmentierung lohnt es trotzdem nicht — vier Fusionswege in diesem Bericht und
Ruschhaupt et al. mit einem echten photogrammetrischen CHM kommen unabhängig
zum selben Ergebnis, dass Höhe die Kronensegmentierung nicht verbessert. Nur
wenn die Baumhöhe selbst ein Ziel ist, ist der Aufwand gerechtfertigt.

---

Alle Zahlen gemessen auf BAMFORESTS · Troles, Schmid, Fan & Tian (2024),
*Remote Sensing* 16(11), 1935 · CC BY-NC-SA 4.0
