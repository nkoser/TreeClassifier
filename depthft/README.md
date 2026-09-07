# depthft — Depth Pro auf FORTRESS feinabstimmen

> **Stand der Gewichte:** Der vorhandene Lauf unter
> `/scratch/shared/$USER/runs/depthft` wurde noch mit dem alten
> Log-Tiefen-Loss trainiert. Der korrigierte Code schreibt neue Läufe
> standardmäßig nach `/scratch/shared/$USER/runs/depthft_huber_v2`; die alten
> Gewichte werden nicht überschrieben.

Eigener Ordner, damit an der bestehenden Pipeline nichts angefasst wird. Die
Skripte hier importieren nur untereinander, nicht aus dem
Repo-Wurzelverzeichnis.

## Das Problem

Depth Pro liefert zu unseren Frames Tiefenkarten, aber die Höhe stimmt nicht:
Kronen sitzen zu hoch, Baumhöhen kommen zu klein heraus. Das ist kein Zufall.
Depth Pro ist auf Bodenperspektiven trainiert — Straßen, Innenräume, Portraits.
Eine Nadiraufnahme aus 80 m kommt darin nicht vor. Was das Modell gelernt hat,
ist die **Struktur** einer Szene; was es nicht gelernt hat, ist der **Maßstab**
eines Aufnahmefalls, den es nie gesehen hat.

Genau das ist reparierbar, sobald Wahrheit vorliegt.

## Woher die Wahrheit kommt

**FORTRESS** (Schiefer, Frey & Kattenborn 2022, CC BY 4.0) liegt unter
`/scratch/shared/$USER/data/fortress`: 47 UAV-Gebiete im Südschwarzwald zu je
1,7 ha, Orthomosaik bei 0,77–1,57 cm/px, dazu je Gebiet ein **normalisiertes
Höhenmodell** (nDSM) — Meter über Boden, für jeden Bildpunkt.

Direkt trainieren lässt sich damit nicht. Ein Orthomosaik ist kein Foto: es hat
keine Kamera, keinen Bildwinkel, keine Tiefe. Die entsteht erst durch eine
Annahme — hänge eine Nadirkamera in Höhe `H` über den Bestand:

```
Tiefe        d    = H - nDSM
Bodenauflös. GSD  = H / f_px
Bodenbreite       = 2 * H * tan(HFOV / 2)
```

Aus einem Gebiet werden damit beliebig viele **virtuelle Frames mit exakter
metrischer Tiefenkarte**, in beliebiger Flughöhe. Flughöhe, Bildwinkel und
Position werden je Ausschnitt gewürfelt.

Der Preis dieser Annahme: ein Ortho zeigt jeden Baum von genau oben, ein echtes
Foto zeigt Kronenflanken zum Bildrand hin. Für die Frage *wie hoch ist dieser
Baum* spielt das kaum eine Rolle, für die Frage *wo genau ist seine Kante* etwas
mehr. Wer es genau nehmen will, schaltet mit `--strahl-tiefe` auf die Tiefe
entlang des Sehstrahls statt entlang der optischen Achse.

## Der Raum, in dem trainiert wird

Depth Pro gibt keine Meter aus, sondern **kanonische inverse Tiefe**. Metrisch
wird daraus erst im Nachlauf des Prozessors
([`image_processing_depth_pro.py:108`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/depth_pro/image_processing_depth_pro.py)):

```
d = (f_px / Bildbreite) / D_roh  =  k / D_roh
```

`k` hängt allein am Bildwinkel und nicht an der Auflösung. Das hat drei Folgen,
die den ganzen Aufbau bestimmen:

1. **`k` wird vorgegeben, nicht geschätzt.** Depth Pro hat einen Bildwinkelkopf,
   der `k` mitschätzt. Bei bekannter Drohnenkamera ist das die schlechtere Wahl —
   der Kopf ist auf Bodenperspektiven trainiert, und ein Fehler in `k` geht
   *linear* in jede Tiefe ein. Der Kopf bleibt eingefroren und im Checkpoint
   erhalten, benutzt wird er nicht.
2. **Die Ausgabe bleibt kanonische inverse Tiefe.** Trainiert wird über die
   daraus berechnete metrische Höhe `h = H - k/D`. Der Checkpoint bleibt mit
   den normalen Hugging-Face-Klassen ladbar. Weil der eingefrorene
   Bildwinkelkopf bei Nadirbildern unzuverlässig ist, muss `k` bei der
   metrischen Nachrechnung weiterhin vorgegeben werden.
3. **Die Auflösung der Ausschnitte ist frei.** Sie stehen auf 1536 px Breite,
   damit zwischen Ausschnitt und Modelleingang gar nicht erst umskaliert wird.

### Der Zuschnitt ist 16:9, nicht quadratisch

Depth Pro quetscht jedes Bild auf 1536×1536, ohne Rücksicht auf das
Seitenverhältnis. Unsere Frames sind 1920×1080, werden in dieser Kette also um
Faktor 1,78 in der Höhe gestaucht. Wer auf Quadraten trainiert und auf
gestauchten Bildern anwendet, hat sich den Fehler selbst gebaut. Die Ausschnitte
kommen deshalb im Seitenverhältnis der Zielframes (`--seitenverhaeltnis`,
Vorgabe 16/9) und laufen anschließend durch dieselbe Stauchung.

Aus demselben Grund gibt es als Augmentierung nur Spiegelungen und keine
Vierteldrehung — die würde das Seitenverhältnis kippen.

Der Verlust hat zwei Teile:

| Teil | Wirkung |
|---|---|
| Huber auf der metrischen Höhe | optimiert direkt den Höhenfehler in Metern und ist gegen einzelne nDSM-Ausreißer robust. |
| Gradientenanpassung über 4 Skalen | macht Kronengrenzen scharf. Ein reiner Pixelverlust belohnt weichere Übergänge. |

### Der Kopf wird vorgespannt, bevor trainiert wird

Pures Depth Pro liegt in diesem Aufnahmefall um etwa Faktor 50 daneben. Wegen
`d = k/D` ist die Abbildung nahe null sehr steil. Die Vorspannung setzt die
Ausgabe vor dem ersten Optimizer-Schritt in den physikalisch relevanten Bereich
und vermeidet damit einen unnötig instabilen Skalenwechsel.

Der Ausweg ist, den Sprung gar nicht erst zu verlangen. `--vorspannen auto`
misst den Skalenfehler auf ein paar Stapeln und skaliert damit die letzte
Faltung des Kopfes. Weil sie eine 1×1-Faltung vor der abschließenden ReLU ist
und der Faktor positiv, ist das **exakt** äquivalent zu `D → faktor · D` — aber
als echte Gewichtsänderung, nicht als Sonderweg beim Anwenden. Der ausgelieferte
Checkpoint bleibt dadurch ohne Beipackzettel brauchbar.

Die Lernrate dieser einen Schicht wird mit demselben Faktor skaliert. Sonst
rissen Adam-Schritte in gewohnter Größe die nun um Größenordnungen kleineren
Gewichte sofort auseinander — Adam normiert die Schrittweite weg, sie hängt
allein an der Lernrate.

Der metrische Fehler wird vorwärts exakt ausgewertet. Für den Rückwärtslauf
wird der Jacobian von `k/D` am jeweiligen Zielwert linearisiert. So bleibt die
Optimierungsrichtung am Ziel exakt, kann nahe `D=0` aber nicht mehr singulär
werden. Zusätzlich laufen Trainingssamples in kurzen, gemischten Gebietsblöcken
statt 200 Ausschnitte desselben Bestands unmittelbar hintereinander.

Trainiert wird standardmäßig `--trainable decoder`: Nacken, Fusionsstufe und
Kopf, rund 60 M Parameter. Der Encoder läuft eingefroren unter `no_grad` — das
spart den Großteil des Speichers und reicht, denn der Maßstab sitzt im Kopf,
nicht in den Merkmalen. `--trainable all` stimmt alles mit ab und braucht
deutlich mehr GPU.

## Reihenfolge

```bash
sbatch depthft/sbatch/run_prepare.sbatch        # 47 Orthos -> Bodenraster, ~1 h
sbatch depthft/sbatch/run_check.sbatch          # PFLICHT, siehe unten
sbatch depthft/sbatch/run_finetune.sbatch       # Feinabstimmung
sbatch depthft/sbatch/run_evaluate.sbatch       # metrisch, auf Testgebieten
sbatch depthft/sbatch/run_apply_frames.sbatch   # auf unseren eigenen Frames
sbatch depthft/sbatch/run_export.sbatch         # Versandpaket für Kollegen
```

Alle Skripte nehmen Vorgaben über Umgebungsvariablen, z. B.

```bash
EPOCHS=12 TRAINABLE=all BATCH=1 ACCUM=16 sbatch depthft/sbatch/run_finetune.sbatch
SITES="CFB014 CFB019" sbatch depthft/sbatch/run_prepare.sbatch
ALTITUDES="pines=35 dense=60 urban=50" sbatch depthft/sbatch/run_apply_frames.sbatch
```

### Zwei Eigenheiten der FORTRESS-Höhenmodelle

Beide fielen erst im Prüflauf auf, und beide hätten das Training still verdorben.

**Exakte Nullen sind Füllung, nicht Gelände.** In den nDSM-Dateien ist `0.00` der
mit Abstand häufigste Einzelwert — 9 bis 27 % der Fläche, in großen
zusammenhängenden Blöcken, unter denen im Orthomosaik geschlossener Wald steht.
Echter Boden streut um null herum, er trifft ihn nicht zehntausendfach exakt.
Diese Flächen als Boden zu lernen hieße: Kronen auf Höhe null. `prepare.py`
verwirft sie deshalb (`--nullen-behalten` schaltet es zum Vergleichen ab). Über
alle 47 Gebiete: Füllung im Median 6 %, im schlimmsten Gebiet 43 %; der gültige
Anteil liegt danach zwischen 51 % und 97 %, im Median bei 82 % (steht je Gebiet
in `index.json`).

Die verworfenen Flächen liegen bevorzugt in Kronenlücken und Schatten — dort, wo
die Photogrammetrie keine Höhe rekonstruieren konnte. Das heißt: **niedrige
Höhen sind im Training leicht unterrepräsentiert**, gerade der Bodenbezug. Bei
Gebieten mit 0 % Füllung — das ist die Mehrheit — bleibt die Verteilung
vollständig, deshalb ist es tragbar. Beim Auswerten des Bodenniveaus lohnt der
Blick trotzdem.

**Die Kamera muss über den Wipfeln hängen.** Ohne Schranke entstünden
Ausschnitte, in denen 30-m-Bäume bei 27 m Flughöhe fast bis zur Linse reichen —
ein Aufnahmefall, den es bei uns nicht gibt. `--abstand-min` (Vorgabe 20 m)
setzt die Flughöhe je Gebiet auf mindestens *höchster Wipfel + 20 m*.

### Was pures Depth Pro hier leistet — die Messlatte

Aus dem Prüflauf über zehn Ausschnitte bei 73,7° Bildwinkel:

| | Wahrheit | pures Depth Pro |
|---|---|---|
| Tiefe bei 27 m Flughöhe | 3,4–27,5 m | 1,0–1,9 m |
| Tiefe bei 64 m Flughöhe | 22,7–64,0 m | 1,1–1,6 m |
| Skalenfaktor | 1,00 | **0,06** |
| AbsRel | — | **0,94** |
| Bildwinkel (Kopf) | 73,7° | 18–41° |

Depth Pro kollabiert bei Nadir-Waldbildern auf rund einen Meter Tiefe, und zwar
unabhängig von der Flughöhe. Zu kleine Tiefe heißt: alles sitzt zu nah an der
Kamera, die Bäume erscheinen zu hoch — genau das beobachtete Symptom.

Auffällig ist die zweite Zeile: die vorhergesagte **Spanne** innerhalb eines
Bildes beträgt gut einen halben Meter, wo in Wirklichkeit 40 m liegen. Daraus
ließe sich schließen, dass nicht nur der Maßstab falsch ist, sondern auch der
Kontrast — und ein globaler Skalenfaktor deshalb nicht helfen kann.

**Dieser Schluss ist falsch, und die Auswertung weist ihn nach.** Global auf den
richtigen Median skaliert, erreicht pures Depth Pro AbsRel 0,074 und schlägt
damit das feinabgestimmte Modell. Die relative Struktur ist ausgezeichnet; die
kleine absolute Spanne ist nur die Folge des Skalenfehlers, kein eigener Mangel.
Was fehlt, ist ausschließlich der Maßstab.

Nur: den richtigen Faktor kennt man im Einsatz nicht. Siehe unten.

### `run_check.sbatch` ist nicht optional

Der teuerste Fehler in dieser Kette wäre ein Versatz zwischen Orthomosaik und
Höhenmodell: die Georeferenzierung stimmt nicht, die Kronen im nDSM sitzen zwei
Meter neben denen im Bild, und das Training lernt geduldig Unsinn — 48 Stunden
lang, mit sinkendem Verlust. `check.py` legt beides nebeneinander und prüft die
Geometrie gegen die Formeln oben. **Die Vergleichsstreifen unter
`results_depthft/check/` müssen angesehen werden**, bevor das Training startet:
das Relief des nDSM muss auf den Kronen im Bild liegen.

Nebenbei liefert `check.py` die Ausgangslage (Tabelle oben) und weist je
Ausschnitt aus, wie viel Fläche überhaupt Wahrheit trägt. Im Vergleichsstreifen
ist fehlende Wahrheit schwarz — sie darf nicht als Boden durchgehen.

## Was gemessen wird

`evaluate.py` vergleicht auf Gebieten, die im Training nie vorkamen, vier
Varianten. Die dritte ist die aufschlussreichste:

| Variante | Was sie beantwortet |
|---|---|
| `pur_fovkopf` | Depth Pro so, wie man es von der Stange nimmt. |
| `pur_kamera` | Derselbe Lauf, `k` vorgegeben. Trennt Fehler im Bildwinkel von Fehlern in der Tiefe. |
| `pur_skalenangleich` | Global so skaliert, dass der Median exakt stimmt. **Kein anwendbares Verfahren, sondern ein Orakel** — der Faktor kommt aus der Wahrheit. Misst, wie gut die *relative* Struktur ist. |
| `*_hoehenanker` | Skaliert, bis die tiefste Stelle im Bild der bekannten Flughöhe entspricht. Anwendbar, denn eine Drohne kennt ihre Höhe. |
| `feinabgestimmt_kamera` | Das Ergebnis: die Skala kommt aus dem Bild selbst. |

### Gemessen, 200 Ausschnitte aus fünf Testgebieten

| Variante | AbsRel | MAE | δ<1,25 | Skalenfehler |
|---|---|---|---|---|
| `pur_skalenangleich` *(Orakel)* | 0,074 | 4,41 m | 0,941 | 1,000 |
| **`feinabgestimmt_kamera`** | **0,120** | **7,36 m** | **0,843** | 0,945 |
| `feinabgestimmt_hoehenanker` | 0,167 | 8,96 m | 0,797 | 1,170 |
| `pur_hoehenanker` | 0,226 | 11,98 m | 0,632 | 1,231 |
| `pur_fovkopf` | 0,939 | 56,11 m | 0,000 | 0,061 |
| `pur_kamera` | 0,976 | 58,14 m | 0,000 | 0,024 |

Drei Dinge stehen darin.

**Das Feintuning wirkt.** Von AbsRel 0,98 auf 0,120, von δ<1,25 = 0,000 auf 0,843.
Der Skalenfehler geht von 0,024 auf 0,945 — im Median noch 5,5 % daneben.

**Die Struktur war nie das Problem.** Mit geschenktem Skalenfaktor erreicht das
pure Modell 0,074. Das Feintuning kommt ohne jede Hilfe auf 0,120 und damit nahe
an diese Schranke heran, aber es überholt sie nicht.

**Der naheliegende Anker funktioniert nicht.** Eine Drohne kennt ihre Flughöhe —
es liegt nahe, damit zu skalieren statt ein Modell zu trainieren. Gemessen ist
das *schlechter* (0,226 gegenüber 0,976 für pur, aber auch schlechter als
0,120), und der Grund steht im Skalenfehler von 1,23: **im geschlossenen
Kronendach ist die tiefste sichtbare Stelle nicht der Boden.** Selbst beim
feinabgestimmten Modell verschlechtert der Anker das Ergebnis (0,167 statt
0,120) — die gelernte Skala ist verlässlicher als die geometrische Annahme.

Die Bildschärfe spielt kaum eine Rolle: mit `--videolook` (Weichzeichnung,
Rauschen, JPEG) ergibt sich 0,135 statt 0,120. Das Modell überträgt sich also
auf Videobildqualität.

`mae_m` ist zugleich der Fehler der **Höhe über Boden**: die ist Flughöhe minus
Tiefe, und die Flughöhe kürzt sich in der Differenz heraus.

## Messen ohne Wahrheit — auf unseren eigenen Frames

Für `/cold/Mahfuz/chosen_frames` gibt es kein nDSM. Trotzdem lässt sich messen,
und zwar an etwas, das gar keine Höhenkarte braucht: **die Tiefe zum Boden ist
die Flughöhe.**

```
Flughöhe geschätzt = 95. Perzentil der Tiefe
Kronenhöhe         = 95. Perzentil - 2. Perzentil der Tiefe
```

Die zweite Zahl ist die wichtigere, weil sie ohne jede Annahme auskommt: die
Spanne zwischen Boden und Wipfel ist die Baumhöhe, ganz gleich wie hoch die
Drohne wirklich hing. Ein Modell, das den Bestand auf 6 m zusammendrückt, fällt
hier sofort auf — auch dann, wenn seine Tiefenkarte hübsch aussieht.

Ist die Flughöhe bekannt (Zahl im Ordnernamen wie `80m`, oder über
`--altitudes`), kommen zwei Prüfungen dazu: das Bodenniveau muss bei 0 m liegen,
und kein Bildpunkt darf unter dem Boden sitzen.

> **Offene Stelle: 100 m sind eine kleine Extrapolation.** Die Gebiete sind
> 130 m breit. Bei 73,7° Bildwinkel deckt eine Aufnahme aus 80 m genau 120 m ab —
> gerade noch drin. Aus 100 m wären es 150 m, mehr als das Gebiet hergibt. Im
> Training endet die Bodenbreite deshalb bei rund 127 m, also bei einer
> effektiven Auflösung von 8,3 cm auf dem 1536er Eingang, während unsere
> 100-m-Frames 9,8 cm brauchen. Faktor 1,18 darüber hinaus — vertretbar, aber
> beim Auswerten des Ordners `100` im Blick zu behalten. Der Fall 80 m ist voll
> abgedeckt.

> **`urban` enthält Screenshots, keine Drohnenframes** — vier Bildschirmfotos in
> wechselnden Auflösungen. Der vorgegebene Bildwinkel gilt dort nicht, alle
> Werte sind um einen unbekannten Faktor falsch. `karten_export.py` und
> `punktwolke.py` warnen bei Bildern, die nicht 1920×1080 sind.

> **Offene Stelle.** Der Bildwinkel 73,7° ist ein Schätzwert im Code, keine
> gemessene Kameraangabe (siehe `BERICHT_hoehe_aus_bildern.md`). Er geht linear
> in jede Tiefe ein. Liegen EXIF-Daten vor, gehört der Wert dorther —
> `--hfov-deg` nimmt ihn entgegen. Ebenso sind die Flughöhen der Ordner ohne
> Zahl im Namen (`dense`, `mixed`, `pines`, `urban`) unbekannt und fallen auf
> 100 m zurück; das verfälscht dort das Bodenniveau, **nicht** aber die
> Kronenhöhe.

## Das Versandpaket

`export.py` baut einen Ordner, den jemand ohne dieses Repository benutzen kann:
Gewichte im Hugging-Face-Format, passender Bildprozessor, das schlanke
`inferenz.py`, ein `beispiel.py` und eine Modellkarte, in der die eine Sache
steht, an der sonst alles scheitert — dass der Bildwinkel vorgegeben und nicht
geschätzt gehört. Mit `--tar` liegt ein `tar.gz` daneben.

```bash
sbatch depthft/sbatch/run_export.sbatch
# -> /scratch/shared/$USER/runs/depthft/versand/depthpro-fortress-nadir[.tar.gz]
```

## Dateien

| Datei | Aufgabe |
|---|---|
| `prepare.py` | 47 Orthos + nDSM → gemeinsames Bodenraster bei 2 cm/px, ~1,2 GB. Der teure Teil, einmalig. |
| `dataset.py` | schneidet daraus virtuelle Nadirframes; Geometrie und Augmentierung. |
| `check.py` | Wahrheit gegen Bild, Geometrie gegen Formel, Ausgangslage des puren Modells. |
| `finetune.py` | die Feinabstimmung. |
| `inferenz.py` | Anwenden mit vorgegebener Kamera. Ohne Projektabhängigkeiten, wird mitgeliefert. |
| `evaluate.py` | pur gegen feinabgestimmt, metrisch, auf Testgebieten. |
| `apply_frames.py` | pur gegen feinabgestimmt auf unseren eigenen Frames. |
| `export.py` | Versandpaket. |
| `karten_export.py` | Tiefen- und Höhenkarten als npy/png/jpg zum Weiterverarbeiten. |
| `punktwolke.py` | 3D-Wolken als `.ply` und `.las`, am Boden verankert statt an der Kamera. |
| `kalibrieren.py` | Bildwinkel rückwärts aus Frames mit bekannter Flughöhe. |
| `vergleichsbild.py` | Abbildungen pur gegen feinabgestimmt, lesbar statt gesättigt. |
| `bilder.py` | Farbskala, Reliefschattierung, Beschriftung, Balkendiagramm. |

## Herkunft der Daten

FORTRESS: Schiefer, F., Frey, J. & Kattenborn, T. (2022), CC BY 4.0. Wer
Ergebnisse dieses Modells veröffentlicht, sollte den Datensatz zitieren — auch
der Kollege, der die Gewichte bekommt. Steht so in der Modellkarte.
