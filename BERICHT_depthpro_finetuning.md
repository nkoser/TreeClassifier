# Depth Pro auf FORTRESS feinabgestimmt — metrische Baumhöhen aus Nadirbildern

> **Hinweis zum Trainingsstand:** Die in diesem Bericht ausgewerteten Gewichte
> unter `/scratch/shared/$USER/runs/depthft` entstanden mit dem damaligen
> Log-Tiefen-Loss. Der Trainingscode verwendet inzwischen einen korrigierten
> Huber-Loss auf dem metrischen Höhenfehler und schreibt neue Läufe nach
> `/scratch/shared/$USER/runs/depthft_huber_v2`. Die hier dokumentierten alten
> Messwerte ändern sich dadurch nicht; für den neuen Loss ist ein Retraining
> erforderlich.

Alles zu diesem Bericht liegt in [`depthft/`](depthft/); die technische
Dokumentation der einzelnen Skripte steht in [`depthft/README.md`](depthft/README.md).

---

## Kurzfassung

Depth Pro liefert zu unseren Drohnenframes Tiefenkarten, deren Höhe nicht
stimmt. Gemessen auf FORTRESS-Nadirbildern sagt es **0,7 bis 2,6 m Tiefe**
voraus, wo tatsächlich 24 bis 85 m sind — ein Skalenfaktor von **0,02**, und
zwar unabhängig von der Flughöhe. Zu kleine Tiefe heißt: alles sitzt zu nah an
der Kamera, die Bäume erscheinen zu hoch. Genau das beobachtete Symptom.

Feinabgestimmt auf FORTRESS sinkt der relative Fehler auf Testgebieten, die im
Training nie vorkamen, von **AbsRel 0,98 auf 0,120**; der Anteil der Bildpunkte
innerhalb von 25 % der wahren Tiefe steigt von **0,000 auf 0,843**.

Auf unseren eigenen Frames macht pures Depth Pro aus jedem Bestand **0,3 bis
2,0 m hohe Büsche**. Das feinabgestimmte Modell liefert 13 bis 47 m — plausible
Baumhöhen — und trifft das Bodenniveau auf **0,6 m** genau.

Zwei Ergebnisse waren nicht erwartet und sind wichtiger als die Zahlen selbst:

1. **Der Bildwinkel unserer Kamera ist nicht 73,7°, sondern rund 48°.** Der Wert
   im Code war ein Vorgabewert, keine Kameraangabe. Er geht linear in jede Tiefe
   ein — alle bisherigen Höhen aus diesen Frames sind um Faktor 1,7 zu klein.
2. **Depth Pros relative Struktur war nie das Problem.** Mit geschenktem
   Skalenfaktor erreicht das pure Modell AbsRel 0,074 und schlägt damit das
   feinabgestimmte. Es fehlte ausschließlich der Maßstab.

---

## 1. Warum Depth Pro hier scheitert

Depth Pro ist auf Bodenperspektiven trainiert — Straßen, Innenräume, Portraits.
Eine Nadiraufnahme aus 80 m kommt darin nicht vor. Aus dem Prüflauf über zehn
Ausschnitte bei 73,7° Bildwinkel:

| | Wahrheit | pures Depth Pro |
|---|---|---|
| Tiefe bei 27 m Flughöhe | 3,4–27,5 m | 1,0–1,9 m |
| Tiefe bei 64 m Flughöhe | 22,7–64,0 m | 1,1–1,6 m |
| Skalenfaktor | 1,00 | **0,06** |
| Bildwinkel (eigener Kopf) | 73,7° | 18–41° |

Zwei Dinge fallen auf. Der Skalenfehler ist **flughöhenunabhängig** — das Modell
gibt immer ungefähr dasselbe aus, es liest die Höhe gar nicht aus dem Bild. Und
sein eingebauter Bildwinkelkopf, der die metrische Skala mitbestimmt, liegt um
Faktor zwei daneben.

---

## 2. Woher die Wahrheit kommt

**FORTRESS** (Schiefer, Frey & Kattenborn 2022, CC BY 4.0): 47 UAV-Gebiete im
Südschwarzwald zu je 1,7 ha, Orthomosaik bei 0,77–1,57 cm/px, dazu je Gebiet ein
normalisiertes Höhenmodell (nDSM) bei 5 cm — Meter über Boden, für jeden
Bildpunkt.

Direkt trainieren lässt sich damit nicht: ein Orthomosaik hat keine Kamera,
keinen Bildwinkel, keine Tiefe. Die entsteht erst durch eine Annahme — hänge
eine Nadirkamera in Höhe `H` über den Bestand:

```
Tiefe        d    = H − nDSM
Bodenauflös. GSD  = H / f_px
Bodenbreite       = 2 · H · tan(HFOV / 2)
```

Aus einem Gebiet werden so beliebig viele virtuelle Frames **mit exakter
metrischer Tiefenkarte**, in beliebiger Flughöhe. Flughöhe, Bildwinkel und
Position werden je Ausschnitt gewürfelt.

### Trainiert wird im Heimatraum des Modells

Depth Pro gibt keine Meter aus, sondern kanonische inverse Tiefe. Metrisch wird
daraus erst im Nachlauf:

```
d = (f_px / Bildbreite) / D_roh  =  k / D_roh
```

`k` hängt allein am Bildwinkel. Der alte, hier ausgewertete Lauf wurde direkt
gegen `D_gt = k / d_gt` trainiert. Der Checkpoint ist mit den normalen
Hugging-Face-Klassen ladbar; für metrische Tiefe muss `k` dennoch wie in
`inferenz.py` vorgegeben werden, weil der unveränderte Bildwinkelkopf auf
Nadirbildern unzuverlässig ist.

Der Verlust ist L1 auf der Log-Tiefe (bestraft relativen statt absoluten Fehler)
plus mehrskalige Gradientenanpassung (macht Kronengrenzen scharf).

---

## 3. Drei Fallen, die das Training still verdorben hätten

### Exakte Nullen im nDSM sind Füllung, nicht Gelände

In den Höhenmodellen ist `0.00` der mit Abstand häufigste Einzelwert — im Median
6 % der Fläche, im schlimmsten Gebiet **43 %**, in großen zusammenhängenden
Blöcken, unter denen im Orthomosaik geschlossener Wald steht. Echter Boden
streut um null herum; er trifft ihn nicht zehntausendfach exakt. Es sind die
Stellen, an denen die Photogrammetrie keine Höhe rekonstruieren konnte.

Diese Flächen als Boden zu lernen hieße: Kronen auf Höhe null. Sie werden
verworfen. Der gültige Anteil liegt danach zwischen 51 % und 97 %, im Median bei
82 %.

### Der Gradient explodiert beim Skalensprung

Der Verlust lebt auf `log d = log k − log D`, sein Gradient bezüglich der
Modellausgabe ist also `1/D`. Während das Modell `D` um Faktor 50 nach unten
treibt, wächst dieser Gradient selbst um Faktor 50 mit — ein sich verstärkender
Abstieg, der über null hinausschießt. Hinter der ReLU des Kopfes ist er dann
tot, und die Tiefe läuft ins Hunderttausendfache. Genau so beobachtet: sauberer
Abstieg bis Schritt 250, danach AbsRel 7000 und MAE 400 000 m.

**Lösung: den Sprung gar nicht erst verlangen.** Der Skalenfehler wird auf ein
paar Stapeln gemessen (0,0315) und damit die letzte 1×1-Faltung des Kopfes
skaliert. Weil sie vor der abschließenden ReLU sitzt und der Faktor positiv ist,
ist das exakt äquivalent zu `D → faktor · D` — aber als echte Gewichtsänderung,
nicht als Beipackzettel. Die Lernrate dieser Schicht wird mitskaliert, sonst
rissen Adam-Schritte in gewohnter Größe die nun winzigen Gewichte auseinander.

Wirkung, noch vor dem ersten Lernschritt: Verlust 3,67 → 0,43, AbsRel 0,97 → 0,37.

### Die Kamera muss über den Wipfeln hängen

Ohne Schranke entstünden Ausschnitte, in denen 30-m-Bäume bei 27 m Flughöhe fast
bis zur Linse reichen. Die Flughöhe wird je Gebiet auf mindestens *höchster
Wipfel + 20 m* gesetzt.

---

## 4. Ergebnisse auf den Testgebieten

Fünf Gebiete, 200 Ausschnitte, weder im Training noch in der Validierung gesehen.

| Variante | AbsRel | MAE | δ<1,25 | Skalenfehler |
|---|---|---|---|---|
| `pur_skalenangleich` *(Orakel)* | 0,074 | 4,41 m | 0,941 | 1,000 |
| **`feinabgestimmt_kamera`** | **0,120** | **7,36 m** | **0,843** | 0,945 |
| `feinabgestimmt_hoehenanker` | 0,167 | 8,96 m | 0,797 | 1,170 |
| `pur_hoehenanker` | 0,226 | 11,98 m | 0,632 | 1,231 |
| `pur_fovkopf` | 0,939 | 56,11 m | 0,000 | 0,061 |
| `pur_kamera` | 0,976 | 58,14 m | 0,000 | 0,024 |

`mae_m` ist zugleich der Fehler der **Höhe über Boden** — die ist Flughöhe minus
Tiefe, und die Flughöhe kürzt sich in der Differenz heraus.

### Das Feintuning wirkt

AbsRel von 0,98 auf 0,120, δ<1,25 von 0,000 auf 0,843, Skalenfehler von 0,024
auf 0,945 — im Median noch 5,5 % daneben.

### Aber die Struktur war nie das Problem

`pur_skalenangleich` ist pures Depth Pro, global so skaliert, dass der Median
exakt stimmt. Es erreicht **0,074 und schlägt damit das feinabgestimmte Modell.**

Das ist kein anwendbares Verfahren — der Faktor kommt aus der Wahrheit, die man
im Einsatz nicht hat. Die Zeile ist eine obere Schranke, und richtig gelesen
sagt sie zweierlei: Depth Pros *relative* Tiefenstruktur ist auf Nadirbildern
ausgezeichnet, und was ihm fehlt, ist ausschließlich der Maßstab. Das
Feintuning kommt ohne jede Hilfe (0,120) nahe an diese Schranke heran, überholt
sie aber nicht.

> Ich hatte zwischenzeitlich das Gegenteil vermutet — weil die vorhergesagte
> Tiefenspanne *innerhalb* eines Bildes bei gut einem Meter lag, wo in
> Wirklichkeit 40 m liegen, schien auch der Kontrast kaputt. Das war falsch: die
> kleine Spanne ist eine Folge des Skalenfehlers, kein eigener Mangel.

### Der naheliegende Anker funktioniert nicht

Eine Drohne kennt ihre Flughöhe aus Barometer und GPS. Es liegt nahe, damit zu
skalieren, statt ein Modell zu trainieren: skaliere, bis die tiefste Stelle im
Bild der Flughöhe entspricht. Gemessen ist das **schlechter** — 0,226 statt
0,976 für pur, aber eben auch schlechter als die 0,120 des Feintunings. Selbst
auf das feinabgestimmte Modell angewandt verschlechtert der Anker (0,167).

Der Grund steht im Skalenfehler von 1,23: **im geschlossenen Kronendach ist die
tiefste sichtbare Stelle nicht der Boden.** Aus den nDSM-Daten gemessen liegt sie
im Median bei **0,917 der Flughöhe** (5.–95. Perzentil: 0,82–0,99). Die gelernte
Skala ist verlässlicher als diese geometrische Annahme.

### Bildschärfe spielt kaum eine Rolle

Mit Weichzeichnung, Rauschen und JPEG-Artefakten auf Videobildqualität gebracht:
AbsRel 0,135 statt 0,120. Das Modell überträgt sich.

---

## 5. Der Bildwinkel — der größte Einzelfehler im Projekt

Die 73,7° stammen aus einem Vorgabewert im Code, nicht aus einer Kameraangabe.
Der Wert geht **linear** in jede Tiefe ein, und man sieht es der Tiefenkarte
nicht an: sie sieht richtig aus und ist es nicht.

Ohne EXIF lässt er sich eingrenzen, wenn die Flughöhe bekannt ist:
`k = Bodenfaktor · H / p95(1/D)`. Das hat eine eingebaute Probe — **verschiedene
Flughöhen müssen denselben Bildwinkel ergeben.**

| Ordner | bekannte Höhe | rückgerechneter Bildwinkel |
|---|---|---|
| `80m` | 80 m | 46,9° (Streuung 1,0°) |
| `100` | 100 m | 50,4° (Streuung 1,1°) |

Spanne 3,4° — die Probe besteht. **Empfohlener Wert: 48,0°**, Unsicherheitsband
44,8°–52,9° (dominiert von der Frage, wie gut der Boden sichtbar ist). Die
bisherigen Tiefen sind damit um **Faktor 1,68** zu korrigieren.

Zum Kontrast: pures Depth Pro ergibt rückgerechnet 1,4–1,6° und ist dabei
*konsistenter* (Spanne 0,3°). Das ist kein Qualitätsmerkmal, sondern das
Gegenteil — es gibt unabhängig von der Flughöhe immer dasselbe aus.

**Diese Rückrechnung ist kein Ersatz für eine echte Kalibrierung.** Sie setzt
voraus, dass das Tiefenmodell stimmt, und ist damit teilweise zirkulär. Der
belastbare Teil ist die Konsistenz: dass *ein* Bildwinkel beide bekannten Höhen
trifft, könnte das Modell nicht leisten, wenn es die Flughöhe nicht tatsächlich
aus dem Bild läse. **AnyCam** auf den Originalvideos wäre der unabhängige Weg —
es schätzt die Intrinsics aus der Bildbewegung, also über einen völlig anderen
Informationsweg. Dafür werden die Videos gebraucht, die Einzelframes reichen nicht.

---

## 6. Anwendung auf unsere Frames

Mit kalibriertem Bildwinkel (48°), über alle 33 Frames:

| | pur | feinabgestimmt |
|---|---|---|
| geschätzte Flughöhe | 2,98 m | **83,0 m** |
| **Bodenfehler** | −80,7 m | **−0,61 m** |
| Kronenhöhe | 0,89 m | 28,2 m |
| Punkte unter dem Boden | 0 % | 6 % |

Je Ordner, Kronenhöhe (annahmefrei — eine Differenz braucht keinen Bezugspunkt):

| Ordner | Flughöhe | Quelle | pur | feinabgestimmt |
|---|---|---|---|---|
| `100` | 100 m | Ordnername | 1,0 m | 35,4 m |
| `80m` | 80 m | Ordnername | 0,6 m | 18,7 m |
| `dense` | 51 m | geschätzt | 0,6 m | 13,1 m |
| `dense1` | 69 m | geschätzt | 0,5 m | 16,7 m |
| `mixed` | 92 m | geschätzt | 1,1 m | 45,9 m |
| `mixed1` | 103 m | geschätzt | 2,0 m | 47,1 m |
| `pines` | 60 m | geschätzt | 0,6 m | 14,5 m |
| `urban` | 120 m | geschätzt | 0,8 m | 37,4 m |

Die Flughöhen der Ordner ohne Zahl im Namen sind vom Modell geschätzt, nicht
gemessen.

**Der Bodenfehler von 0,6 m ist teilweise eingebaut**, weil der Bildwinkel aus
genau diesen Frames zurückgerechnet wurde. Nicht eingebaut ist die Konsistenz:
ein einziger Bildwinkel bringt alle acht Ordner gleichzeitig auf plausible Werte,
und nur 6 % der Bildpunkte landen unter dem Boden.

**Vorbehalt zur Kronenhöhe:** das ist die Tiefenspanne im Bild, nicht zwingend
die Baumhöhe. Bei geneigtem Gelände steckt das Relief mit drin — die 46–47 m bei
`mixed`/`mixed1` sind für mitteleuropäische Bäume unrealistisch und dürften
daher rühren.

> **`urban` ist gar keine Drohnenaufnahme.** Der Ordner enthält vier
> **Bildschirmfotos** in wechselnden Auflösungen (1463×705, 1378×709, …), keine
> Videoframes. Für die gilt der Bildwinkel von 48° nicht: ein Screenshot zeigt
> einen Ausschnitt, also einen engeren Bildwinkel, und um wie viel ist unbekannt.
> Sämtliche `urban`-Werte in diesem Bericht — 37,4 m Kronenhöhe, 120 m
> geschätzte Flughöhe — sind damit um einen unbekannten Faktor falsch und
> gehören nicht in die Auswertung. `karten_export.py` und `punktwolke.py` warnen
> seitdem, wenn ein Bild nicht 1920×1080 im Verhältnis 16:9 ist.

---

## 7. Grenzen

- **Flughöhe 25–120 m, Nadirblick, Wald.** Schräge Aufnahmen kamen im Training
  nicht vor.
- **100 m ist eine leichte Extrapolation.** Die Gebiete sind 130 m breit; aus
  80 m deckt eine Aufnahme bei 73,7° genau 120 m ab — gerade noch drin, aus
  100 m wären es 150 m. Die effektive Auflösung endet im Training bei 8,3 cm,
  unsere 100-m-Frames bräuchten 9,8 cm. Faktor 1,18 darüber hinaus.
- **Die Wahrheit stammt aus Orthomosaiken**, nicht aus echten Einzelaufnahmen.
  Ein Ortho zeigt jeden Baum von genau oben, ein Foto zeigt Kronenflanken zum
  Bildrand hin. Für die Höhe eines Baumes kaum relevant, für die genaue Lage
  seiner Kante etwas mehr. `--strahl-tiefe` schaltet auf die Tiefe entlang des
  Sehstrahls um.
- **Niedrige Höhen sind leicht unterrepräsentiert.** Die verworfenen
  Füllflächen liegen bevorzugt in Kronenlücken und Schatten — dort, wo Boden
  sichtbar wäre.
- **Nur der Decoder wurde trainiert** (40 M von 952 M Parametern). Ob ein voller
  Durchlauf mehr bringt, ist ungetestet.

---

## 7b. Punktwolken

`depthft/punktwolke.py` macht aus den Tiefenkarten 3D-Wolken, als `.ply`
(CloudCompare, MeshLab, Blender) und `.las` 1.2 (lidR, LAStools). Beide Formate
werden direkt geschrieben, im Container liegt keine Punktwolkenbibliothek.

**Der Boden kommt aus den Daten, nicht aus der Flughöhe — und das ist nicht der
Notbehelf, sondern der bessere Weg.** Gegen das nDSM der Testgebiete gemessen
(`hoehe_pruefen.py`, 100 Ausschnitte):

| Weg zur Höhe über Boden | MAE | Bias | braucht |
|---|---|---|---|
| **Geländemodell** (35 m, Faktor 0,917) | **5,64 m** | **+0,30 m** | nichts |
| aus bekannter Flughöhe, `Z = H − d` | 7,46 m | +4,67 m | die Flughöhe |

`Z = H − d` kann Hangneigung nicht abbilden; ein Geländemodell schon. Der
Bodenfaktor 0,917 wurde zuvor aus den nDSM-Daten geschätzt und kommt hier
unabhängig noch einmal heraus, wenn man ihn gegen die Wahrheit optimiert.

**Zwei Bedingungen, ohne die es schiefgeht.** Das Modell darf **nie über der
beobachteten Oberfläche liegen** — sonst landen Punkte unter dem Boden, gemessen
bis 29 m tief in einem Bestand mit 62 m Geländeabfall, von dem die zu starke
Glättung nur 25 m nachbildete. Und die Glättung darf die Hangneigung nicht
wegbügeln. Nach beiden Korrekturen: 0,0 % Punkte unter dem Boden statt 6,7 %.

**Die Höhe ist ungenauer als die Tiefe, und zwar aus Arithmetik.** Auf der Tiefe
liegt der relative Fehler bei 12 %; die Höhe ist aber eine Differenz zweier
großer Zahlen (80 m Flughöhe − 60 m Tiefe = 20 m Baum). Ein Tiefenfehler von 5 m
wandert unvermindert in die Höhe, wo er relativ viermal so schwer wiegt — rund
19 % bei Baumhöhen um 30 m. Die Korrelation zwischen geschätzter und wahrer Höhe
liegt bei 0,69. Für „welcher Baum ist höher als sein Nachbar" reicht das, für
„dieser Baum ist 24,3 m hoch" nicht.

**Der Bildwinkel verzerrt anders als erwartet.** In der Rückprojektion
`X = (u−cx)·Z/f` kürzt er sich in X und Y heraus, weil `Z` und `f_px` beide
proportional zu ihm sind. Ein falscher Bildwinkel lässt Kronendurchmesser also
korrekt und streckt allein die Höhe — Bäume werden zu spitz oder zu flach, nicht
zu breit.

Die Geometrie ist unabhängig bestätigt: der Frame aus 80 m ergibt eine Wolke von
70,3 m Breite, rechnerisch erwartet sind 2·80·tan(24°) = 71,2 m; aus 100 m sind
es 88,9 m gegenüber 89,1 m erwartet.

| Frame | Boden (Z p02) | Wipfel (Z p95) | Punkte unter Boden |
|---|---|---|---|
| `100` | 6,2 m | 41,1 m | 0,1 % |
| `80m` | 3,0 m | 23,4 m | 0,2 % |
| `dense` | 3,2 m | 17,5 m | 0,6 % |
| `dense1` | 1,4 m | 23,7 m | 0,8 % |
| `mixed` | 0,4 m | 43,0 m | 1,2 % |
| `mixed1` | 0,4 m | 24,8 m | 0,0 % |
| `pines` | 1,7 m | 21,9 m | 1,4 % |

Dass Z nicht bei null anfängt, ist richtig: im geschlossenen Kronendach ist der
Boden nicht sichtbar, das Geländemodell schätzt ihn rund 9 % der Flughöhe unter
dem tiefsten sichtbaren Punkt. Die Wolke schwebt also nicht, sie beginnt dort,
wo die Sicht endet.

## 7c. Ein Versuch, der nicht aufging: direkt auf Höhe trainieren

Der Umweg über die Tiefe hat einen offensichtlichen Makel — er optimiert den
Tiefenfehler, während uns der Höhenfehler interessiert, und die Höhe ist eine
Differenz zweier großer Zahlen, was den relativen Fehler vervierfacht. Ein
zweites Modell (`finetune_hoehe.py`) sagt deshalb die Höhe über Boden direkt
vorher, mit dem nDSM als Ziel.

Auf den ersten Blick ein Erfolg: MAE 5,34 m gegenüber 5,64 m, Korrelation über
alle Bildpunkte 0,77 gegenüber 0,69, und es braucht weder Bildwinkel noch
Bodenbezug.

**Auf den zweiten Blick nicht.** Auf unseren Frames lieferte es für jeden
Bestand 22–29 m — auffällig uniform. Die Prüfung bestätigt den Verdacht:

| Weg | r **zwischen** Beständen | vorhergesagte Streuung | wahre Streuung |
|---|---|---|---|
| direktes Höhenmodell | **−0,16** | ± 1,1 m | ± 6,0 m |
| Geländemodell aus der Tiefe | 0,10 | ± 5,4 m | ± 6,0 m |
| aus bekannter Flughöhe | **0,64** | ± 8,6 m | ± 6,0 m |

Das Modell rät den Trainingsmittelwert. Es differenziert *innerhalb* eines
Bildes gut — Wipfel gegen Lücke — und ist *zwischen* Beständen blind.

Der Grund ist grundsätzlicher Natur und trifft auch das Geländemodell: **ohne
Maßstabsreferenz ist ein 30-m-Baum aus 90 m Flughöhe von einem 15-m-Baum aus
45 m nicht zu unterscheiden.** Gleiche scheinbare Größe, gleicher Detailgrad.
Die Information muss von außen kommen — und die einzige Quelle dafür ist die
Flughöhe. Meine Begründung, das Höhenmodell mache sich vom unsicheren
Bildwinkel unabhängig, stimmt; nur macht es sich damit auch von der einzigen
Skalenreferenz unabhängig, die es gibt.

Warum der MAE das nicht zeigte: er mittelt über alle Bildpunkte, wo die
Variation innerhalb eines Bildes dominiert. Und er *belohnt* das Raten des
Mittelwerts — der Mittelwert ist definitionsgemäß der beste Schätzer, wenn man
nichts weiß. Eine Metrik, die die Anwendung bestraft.

**Konsequenz für die Praxis:**

| Was gebraucht wird | Weg |
|---|---|
| Kronen segmentieren, relative Struktur | Geländemodell — lokal genau, ohne Flughöhe |
| Bestände vergleichen, absolute Höhe | `Z = H − d` mit bekannter Flughöhe |
| Punktwolken zum Anschauen | Geländemodell |

Der Checkpoint liegt unter `/scratch/shared/$USER/runs/depthft_hoehe/bestes` und
ist über `--modellart hoehe` wählbar, ist aber nicht die Vorgabe.

## 8. Was als Nächstes lohnt

1. **Den Bildwinkel unabhängig bestätigen** — EXIF der Originaldateien oder
   AnyCam auf den Videos. Das ist der größte Hebel: Faktor 1,68 auf jede Höhe.
2. **`pur + gelernter Skalenschätzer`.** Die Orakel-Zeile zeigt, dass ein
   winziges Netz, das nur *einen* Skalenfaktor je Bild schätzt, theoretisch
   0,074 erreichen könnte — besser als das vollständig feinabgestimmte Modell
   und um Größenordnungen billiger zu trainieren. Nach dem Befund aus 7c wäre
   allerdings zu erwarten, dass auch dieser Schätzer den Mittelwert rät, solange
   er die Flughöhe nicht als Eingabe bekommt. **Die Flughöhe als zusätzliche
   Eingabe ins Modell** zu geben, wäre der eigentlich vielversprechende Weg: sie
   ist bei einer Drohne bekannt, und das Modell müsste den Maßstab dann nicht
   mehr erraten.
3. **Boden-Detektion statt Perzentil.** Das Geländemodell schätzt den Boden aus
   den tiefsten sichtbaren Stellen. Eine echte Bodenmaske — Pixel erkennen, an
   denen wirklich Boden zu sehen ist — käme ohne den pauschalen Faktor 0,917 aus
   und wäre in offenen Beständen deutlich genauer.
4. **`--trainable all`** mit Gradient-Checkpointing, um zu sehen, ob der Encoder
   noch etwas beiträgt.

---

## 9. Das Versandpaket

```
/scratch/shared/$USER/runs/depthft/versand/
  depthpro-fortress-nadir/          3,81 GB
  depthpro-fortress-nadir.tar.gz    2,30 GB
```

Enthält Gewichte im HuggingFace-Format, den passenden Bildprozessor, das
schlanke `inferenz.py`, ein `beispiel.py` und eine Modellkarte. Laden lässt sich
das Modell mit `DepthProForDepthEstimation.from_pretrained(pfad)`. Die metrische
Nachrechnung läuft anschließend über `inferenz.py`, damit der Bildwinkel
vorgegeben und nicht vom unveränderten Kopf geschätzt wird.

Wer Ergebnisse veröffentlicht, sollte FORTRESS zitieren (Schiefer, Frey &
Kattenborn 2022, CC BY 4.0) — das gilt auch für Kollegen, die nur die Gewichte
bekommen. Steht so in der Modellkarte.
