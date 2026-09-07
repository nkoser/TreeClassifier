# TreeClassifier — Baumarten-Inferenz auf eigenen Drohnen-Frames

Zweistufige Pipeline, die den veröffentlichten **DINOvTree-B**-Checkpoint (ECCV 2026,
[RolnickLab/DINOvTree](https://github.com/RolnickLab/DINOvTree)) auf eigene, nicht
georeferenzierte Drohnen-Frames anwendet.

```
Frame (1920x1080 JPG)
  └─ Instanzen                                → ein Baum pro Segment
      ├─ segment_hybrid.py (bestes Ergebnis)   SAM + Watershed für die Restfläche
      ├─ segment_sam.py                       SAM → Kronenpolygone an echten Bildkanten
      ├─ segment_trees.py                     Tiefe + Watershed → Kronenpolygone
      └─ infer_species.py --detector deepforest   DeepForest → Boxen (schwächste Variante)
          └─ baum-zentrierter Crop, auf 512x512   → ein Crop pro Baum
              └─ DINOvTree-B (Quebec Trees)       → Artwahrscheinlichkeit + Höhe
```

> **Stand:** `segment_trees.py` liefert deutlich bessere Instanzen als DeepForest,
> ist aber noch nicht in `infer_species.py` verdrahtet — die Ergebnisse in
> `results/` beruhen weiterhin auf DeepForest-Boxen.

DINOvTree selbst detektiert **nichts** — es klassifiziert immer den Baum in der
Bildmitte eines 512×512-Ausschnitts. Die Instanzen müssen deshalb von außen kommen;
im Original-Paper sind das manuell annotierte Kronenpolygone, hier ersetzt sie
DeepForest.

## Setup

Alles läuft über SLURM im vorhandenen `vesselgpt_7.sif`, nichts direkt auf dem
Login-Knoten:

```bash
sbatch sbatch/setup_env.sbatch      # env auf /scratch + Checkpoint, einmalig
squeue -u $USER                     # Status
```

Der Container bringt Python 3.11.8 und **torch 2.10.0+cu128** mit — genau das, was die
RTX PRO 6000 Blackwell des Knotens braucht (`sm_120`, Treiber 580.x / CUDA 13.0; ältere
CUDA-Builds importieren sauber und sterben dann beim ersten Kernel). Statt einer zweiten
Torch-Installation legt `setup_env.sbatch` ein venv **mit `--system-site-packages`** unter
`/scratch/shared/$USER/envs/treeclf` an: es erbt Torch aus dem Container und ergänzt nur
das Fehlende (deepforest, opencv, omegaconf). Das spart 7 GB im Home und hält die
Torch-Version konsistent mit deinen anderen Jobs.

Das venv ist nur **innerhalb** des Containers benutzbar — sein Interpreter zeigt auf
`/miniconda3`. Alle Läufe gehen deshalb durch `apptainer exec --nv`.

Ablage:

| Was | Wo |
|---|---|
| Env | `/scratch/shared/$USER/envs/treeclf` |
| Checkpoint | `/scratch/shared/$USER/data/treeclf/checkpoints/` |
| HF-/Torch-Cache | `/scratch/shared/$USER/hf_cache` |
| Job-Logs | `/scratch/shared/$USER/runs/logs/%x_%j.{out,err}` |
| Ergebnisse | `results/` im Repo (gitignored) |

`third_party/DINOvTree/` ist eine Kopie des Original-Repos (nur der Modellcode wird
importiert). **Metas DINOv3-Gewichte werden nicht gebraucht** — der Checkpoint enthält
den kompletten feingetunten Backbone. Das Original-Repo lädt sie trotzdem stur über eine
URL aus `dinov3_urls.json`, deshalb biegt `build_dinovtree()` `torch.hub.load` kurz um
und baut die Architektur uninitialisiert, bevor der Checkpoint mit `strict=True`
darübergelegt wird.

## Benutzung

```bash
sbatch sbatch/run_infer_species.sbatch                       # Defaults
ALTITUDES="pines=35 dense=60" sbatch sbatch/run_infer_species.sbatch
OUT=results_v2 MAX_TREES=200 sbatch sbatch/run_infer_species.sbatch
FRAMES="/cold/Mahfuz/chosen_frames/dense/frame_000073.jpg" sbatch sbatch/run_scale_sweep.sbatch
```

Ein voller Durchlauf über alle 33 Frames mit 150 Bäumen/Frame dauert **50 Sekunden**
(zum Vergleich: 27 Minuten auf CPU für nur 40 Bäume/Frame). Job-Parameter als
Umgebungsvariablen vor `sbatch`: `INPUT`, `OUT`, `MAX_TREES`, `BATCH`, `ALTITUDES`,
`CKPT`, `SIF`, `ENV_DIR`, `EXTRA` (beliebige weitere Flags).

Wichtige Parameter des Skripts selbst:

| Flag | Default | Bedeutung |
|---|---|---|
| `--altitudes ORDNER=HÖHE` | – | Flughöhe pro Ordner, z.B. `--altitudes pines=35 dense=60` |
| `--altitude` | 100 | Fallback-Flughöhe in m |
| `--hfov-deg` | 73.7 | Horizontaler Bildwinkel (DJI 24-mm-äquiv., 16:9) |
| `--crop-mode` | `gsd` | `gsd`: Ausschnitt deckt 9.73 m ab wie im Training. `relative`: Vielfaches der Kronenbox, ohne Kamerawissen |
| `--footprint-m` | 9.73 | Kantenlänge des Ausschnitts am Boden |
| `--min-score` | 0.35 | DeepForest-Konfidenzschwelle |
| `--max-trees-per-frame` | 40 | Nur die N sichersten Detektionen (GPU ≈ 0.015 s/Baum, CPU ≈ 0.85 s/Baum) |
| `--device` | `auto` | `cuda` wenn verfügbar, sonst `cpu` |

Ausgabe je Frame: `results/<ordner>/<frame>_trees.csv` (Boxen, Top-3-Arten mit
Wahrscheinlichkeiten, vorhergesagte Höhe, Entropie) und `<frame>_overlay.jpg`.
Dazu `results/all_trees.csv` und `results/summary_by_folder.csv`.

## Maßstab: der kritische Parameter

Der Checkpoint hat **ausschließlich** Ausschnitte von 9.73 m Kantenlänge gesehen
(512 px × 1.9 cm/px, ohne Resampling). Ein Frame ohne Georeferenz hat aber keinen
bekannten Maßstab. Der GSD wird deshalb aus Flughöhe und Bildwinkel geschätzt:

```
GSD = 2 · Höhe · tan(HFOV/2) / Bildbreite
Crop-Kantenlänge in Quellpixeln = 9.73 m / GSD
```

`scale_sweep.py` zeigt, wie empfindlich das ist — dieselben Detektionen bei
verschiedenen Ausschnittsgrößen:

```bash
FRAMES=/cold/Mahfuz/chosen_frames/pines/frame_000006.jpg \
  CROP_SIZES="100 150 250 400 600" N_TREES=6 sbatch sbatch/run_scale_sweep.sbatch
```

Bei den Testframes kippt die Top-1-Art zwischen laub- und nadelbaumdominiert, je
nachdem ob 150 px oder 400 px als 9.73 m interpretiert werden. Ohne belastbare
Flughöhe pro Ordner sind die Artvorhersagen entsprechend wackelig.

## Instanzen: drei Ansätze im Vergleich

| Verfahren | Kronen (33 Frames) | Ø Durchmesser | Flächendeckung | Charakter |
|---|---|---|---|---|
| DeepForest | 3617 (aus 20563 roh) | 27 px | – | Kronenfragmente, keine Bäume |
| Tiefe + Watershed | 4697 | 94–117 px | 71–99 % | kachelt lückenlos, auch wo kein Baum ist |
| SAM (vit-large) | 4848 | 65–103 px | 50–74 % | echte Kronenränder, lässt Unsicheres weg |
| **Hybrid (SAM + Watershed)** | **5897** | 63–97 px | 65–76 % | SAM-Präzision, Watershed füllt die Lücken |

Der entscheidende Unterschied zwischen den letzten beiden: Watershed ist eine
*Partition* — es zerlegt die Maske in genau so viele Teile, wie Marker hineingehen,
unabhängig davon ob dort Bäume stehen. Die hohe Flächendeckung ist deshalb kein
Qualitätsmerkmal, sondern ein Artefakt. SAM segmentiert entlang tatsächlicher
Bildkanten und lässt Bereiche weg, für die es keine Evidenz hat.

### Ablation der SAM-Varianten

`ablate_sam.py` vergleicht die Checkpoints unter identischen Filterregeln
(4 repräsentative Frames, Mittelwerte):

| Modell | Rohmasken | Kronen | Abdeckung | s/Frame |
|---|---|---|---|---|
| **sam_vit_large** | 283 | 140 | **0.61** | 3.4 |
| sam_vit_base | 275 | 140 | 0.60 | 2.9 |
| sam_vit_huge | 263 | 133 | 0.56 | 3.4 |
| sam2.1_large | 140 | 81 | 0.37 | 2.0 |
| sam2_large | 130 | 82 | 0.37 | 2.1 |
| sam2.1_base | 132 | 79 | 0.34 | 2.1 |

Zwei unerwartete Befunde: **SAM 1 schlägt SAM 2 deutlich**, und innerhalb von SAM 1 ist
`vit-huge` schlechter als `vit-large` *und* `vit-base` — größer ist hier nicht besser.

Der naheliegende Einwand, SAM 2 werde durch gemeinsame Schwellen benachteiligt (seine
IoU-/Stabilitätsscores sind anders kalibriert), wurde geprüft: selbst mit komplett
offenen Schwellen (`--pred-iou-thresh 0.0 --stability-score-thresh 0.5`) kommt
SAM 2.1 nur auf 41–49 % Abdeckung. Der Unterschied ist echt, kein Artefakt.

### SAM 3 — Ergebnis

Läuft (Zugang vorausgesetzt) und ist auf den schwierigen Beständen die beste Variante,
aber kein Selbstläufer. Drei Befunde:

1. **Nur der Prompt `"tree"` funktioniert.** `"tree crown"` liefert 0–8 Instanzen pro
   Frame, `"treetop"` gar nichts. SAM 3 kennt den Begriff, nicht die Umschreibung.
2. **Der Formfilter muss aus** (`--no-shape-filter`) und die Schwelle runter auf 0.15.
   Mit den von SAM 1 übernommenen Defaults verwirft man 44 % der Instanzen — SAM 3
   liefert bereits Instanzen statt eines Skalenstapels, der Filter ist überflüssig.
3. **Kachelränder müssen behandelt werden.** Ohne das entstehen schnurgerade Schnitte
   quer durch Kronen: Instanzen, die über eine Kachelgrenze laufen, werden zerteilt und
   beide Hälften behalten. `--drop-cut` (Default an) verwirft angeschnittene Instanzen —
   dank Überlappung ist dasselbe Objekt in der Nachbarkachel vollständig enthalten.

| Ordner | Hybrid | SAM 3 | Δ |
|---|---|---|---|
| `mixed` | 0.65 | **0.76** | +0.11 |
| `dense` | 0.75 | **0.83** | +0.08 |
| `mixed1` | 0.66 | **0.70** | +0.04 |
| `pines` | 0.68 | **0.72** | +0.04 |
| `80m` | 0.76 | 0.77 | +0.01 |
| `dense1` | 0.74 | 0.74 | 0.00 |
| `100` | **0.76** | 0.72 | −0.04 |
| `urban` | **0.74** | 0.47 | −0.27 |

SAM 3 gewinnt auf fünf von acht Beständen, mit **weniger und größeren** Instanzen
(4587 statt 5897, Median 91 statt 81 px) — also weniger Zersplitterung. Es verliert
deutlich auf `urban`; das sind die Screenshots mit abweichenden Bildgrößen, bei denen
die feste 2×2-Kachelung nicht passt.

Der Anteil verdächtiger Trennungen liegt bei SAM 3 mit 51 % höher als beim Hybrid
(40 %) — teils real, teils ein Artefakt der Kennzahl, die gegen dieselbe schwache
monokulare Tiefe misst und großflächigere Kronen härter bestraft.

### Zugang

`facebook/sam3` ist zugangsbeschränkt. Zugang auf der Modellseite anfordern, dann ein
Read-Token unter [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
erzeugen und hinterlegen — `huggingface_hub` liest diesen Pfad unter `HF_HOME` von
selbst, alle Skripte funktionieren danach ohne weitere Änderung:

```bash
printf 'hf_DEIN_TOKEN' > /scratch/shared/$USER/hf_cache/token
chmod 600 /scratch/shared/$USER/hf_cache/token
sbatch sbatch/run_segment_sam3.sbatch
```

SAM 3 arbeitet grundlegend anders als SAM 1/2 und braucht deshalb einen eigenen
Codepfad (`segment_sam3.py`): es bekommt den Begriff als **Text** (`--prompt tree`) und
liefert Instanzen mit Score, statt ein Punktraster abzutasten und einen Skalenstapel aus
Blatt/Ast/Krone zurückzugeben. Der Formfilter ist damit optional (`--no-shape-filter`).

Weil eine 100-px-Krone beim internen Resize auf gut 50 px schrumpft, verarbeitet das
Skript den Frame standardmäßig in 2×2 überlappenden Kacheln und führt die Instanzen
danach über eine Überlappungsauflösung zusammen (`--tiles`, `--tile-overlap`).

## Tiefe als Prompt-Quelle für SAM

`segment_prompted.py` kombiniert beide Informationsquellen an der Stelle, an der sie
sich am besten ergänzen:

```
Tiefe  →  WO ist ein Baum      (CHM-Wipfel als Prompt-Punkt, mit Prominenzprüfung)
SAM    →  WO ist seine Grenze  (promptbare Segmentierung, ein Punkt pro Krone)
```

Vorher musste SAM selbst herausfinden, wo ein Baum anfängt — per blindem Punktraster
(SAM 1/2) oder per Textbegriff (SAM 3). Das weiß die Tiefe besser: ein Wipfel ist ein
lokales Maximum im Ersatz-CHM. Umgekehrt zieht SAM die Grenze aus echten Bildkanten
statt aus der geglätteten Tiefenoberfläche, wie es `segment_trees.py` tut.

SAM liefert je Punkt drei Kandidaten unterschiedlicher Ausdehnung (Teil, Objekt,
Kontext). Ausgewählt wird **nicht der mit dem höchsten Score** — der zielt oft auf das
ganze Kronendach —, sondern der, dessen Fläche am besten zur erwarteten Kronengröße
passt.

```bash
sbatch sbatch/run_segment_prompted.sbatch
PROMINENCE=0.08 sbatch sbatch/run_segment_prompted.sbatch
```

| Verfahren | Kronen | Kompaktheit | Abdeckung | verdächtige Trennungen |
|---|---|---|---|---|
| SAM 3 Multiskala | 5957 | 0.53–0.65 | 0.51–0.84 | 57 % |
| Hybrid | 5897 | 0.55–0.62 | 0.65–0.76 | 40 % |
| **Tiefenprompt** | 3502 | **0.65–0.74** | 0.37–0.67 | **44 %** |

Der Tiefenprompt liefert deutlich **weniger, dafür sauberere** Instanzen: die Kompaktheit
liegt durchgängig 0.1 über den anderen Verfahren, die Formen sind erkennbar kronenrund
statt ausgefranst. Der Preis ist Abdeckung — von 247 gefundenen Wipfeln überstehen nur
154 die Flächen- und Formprüfung.

## Bild und Tiefe verbinden: teilen und verschmelzen

`refine_crowns.py` ist die Zusammenführung beider Informationsquellen. SAM 3 liefert die
Instanzen aus dem **Bild**, die monokulare **Tiefe** korrigiert sie in beide Richtungen:

| Korrektur | Auslöser |
|---|---|
| **Teilen** | Eine Instanz enthält zwei prominente Wipfel mit einer Kerbe dazwischen → zwei Bäume wurden zusammengefasst. Getrennt wird am Sattel, per Watershed innerhalb der Instanz. |
| **Verschmelzen** | Zwei Nachbarinstanzen haben keinen Sattel zwischen sich *und* dieselbe Farbe → ein Baum wurde zerschnitten. |

Erst teilen, dann verschmelzen: falsch zusammengefasste Blobs werden aufgebrochen, danach
die Bruchstücke korrekt gruppiert. Entscheidend beim Teilen ist, dass die Prominenz
**innerhalb der jeweiligen Instanz** gemessen wird — eine niedrige Krone hat einen
kleineren Höhenumfang als eine hohe, ein global gesetzter Schwellwert würde bei ihr nie
auslösen.

```bash
sbatch sbatch/run_refine_crowns.sbatch
SPLITPROM=0.25 sbatch sbatch/run_refine_crowns.sbatch
EXTRA="--no-merge" sbatch sbatch/run_refine_crowns.sbatch
```

| Variante | Instanzen | Teilungen | Verschmelzungen | verdächtige Trennungen |
|---|---|---|---|---|
| SAM 3 Multiskala (roh) | 5957 | – | – | 2361 (57 %) |
| nur verschmelzen | 5223 | – | 734 | 1691 (50 %) |
| `SPLITPROM=0.50` | 5676 | 103 | 368 | – |
| **`SPLITPROM=0.35`** | **5807** | 211 | 375 | **1954 (48 %)** |
| `SPLITPROM=0.25` | 5974 | 350 | 402 | – |
| nur teilen | 6167 | 196 | – | – |

Die **Flächenabdeckung bleibt bei 80 %** — die Korrektur verändert nur, *welche*
Instanzen es gibt, nicht wie viel Kronendach erfasst wird. Genau das war das Ziel: die
Abdeckung von SAM 3 behalten und nur die Grenzen reparieren, die es falsch gesetzt hat.

### Laufzeit

Die erste Fassung brauchte 10 Minuten für 4 Frames, weil für jede Instanz eine Maske
über das ganze Bild gebildet wurde (250 Instanzen × 2 Megapixel je Runde). Über
`regionprops`-Bounding-Boxen läuft dasselbe in **2,5 Minuten für alle 33 Frames**.

## Falsch getrennte Kronen zusammenführen

`merge_crowns.py` führt Nachbarinstanzen zusammen, wenn **zwei unabhängige Kriterien**
dafür sprechen, dass sie zum selben Baum gehören:

- **Sattelprominenz** — zwischen den Wipfeln zweier echter Nachbarbäume liegt eine
  Kerbe. Läuft die Grenze über eine durchgehende Kuppel, ist der Sattel flach.
- **Farbabstand** im Lab-Raum — zwei Teile derselben Krone sind farblich nahezu
  identisch, zwei verschiedene Bäume unterscheiden sich meist messbar.

Beide müssen zustimmen. Das ist wichtig, weil die Sattelprominenz gegen die geschätzte
monokulare Tiefe misst — die schwächste Stelle der Pipeline. Der Farbabstand ist davon
völlig unabhängig und fängt deren Fehler teilweise ab. Eine Flächenobergrenze verhindert
das Aufschaukeln zu Großblobs, und weil jede Verschmelzung Wipfel und Sättel verändert,
läuft das Ganze in mehreren Runden.

```bash
sbatch sbatch/run_merge_crowns.sbatch
SPLIT=0.10 COLOR=16 sbatch sbatch/run_merge_crowns.sbatch
```

| Schwelle | Kronen | verdächtige Trennungen |
|---|---|---|
| ohne | 5957 | 2361 (57 %) |
| `SPLIT=0.06 COLOR=12` | 5553 | 1926 (52 %) |
| `SPLIT=0.10 COLOR=16` | 5223 | 1691 (50 %) |
| `SPLIT=0.15 COLOR=20` | 4815 | 1451 (50 %) |

## Diagnoseansichten

`visualize_crowns.py` rendert vier Ansichten aus der gespeicherten Labelkarte —
ohne Modellinferenz, in Sekunden, beliebig oft neu:

| Ansicht | Beantwortet |
|---|---|
| `instanzen` | Jede Krone in eigener Farbe, halbtransparent. Zwei Farben auf einer optisch durchgehenden Krone = Falschtrennung. Bild bleibt überall in voller Helligkeit. |
| `luecken` | Nicht erfasste Fläche wird **schraffiert statt abgedunkelt** — die Textur bleibt sichtbar, man kann beurteilen wie viel Struktur dort noch ist. |
| `relief` | Kronengrenzen auf der Reliefschattierung. Folgt die Linie einem Höhenrücken oder schneidet sie eine Kuppel? |
| `trennungen` | Jede Grenze zwischen zwei Kronen nach **Sattelprominenz** eingefärbt: rot = flacher Sattel, die beiden gehören vermutlich zusammen; grün = tiefe Kerbe, die Trennung ist durch das Relief gedeckt. |

```bash
OUT=results_views_<name> SEGMENTS=<segmentordner> sbatch sbatch/run_visualize.sbatch
FRAMES="dense/frame_000073.jpg" VIEWS="trennungen" sbatch sbatch/run_visualize.sbatch
```

**Konvention:** je Änderung ein eigener Ordner `results_views_<name>`, damit die
Varianten nebeneinander bestehen bleiben. Rendern dauert rund zwei Minuten für alle
33 Frames und braucht kein Modell — die Labelkarten reichen.

| Ordner | Segmentierung |
|---|---|
| `results_views_verfeinert` | SAM 3 + Tiefe, teilen & verschmelzen (aktueller Stand) |
| `results_views_verfeinert_p025` / `_p050` | dasselbe, aggressiver bzw. vorsichtiger geteilt |
| `results_views_sam3_multiskala` | SAM 3 roh, Kachelstufen 2+3+4 |
| `results_views_verschmolzen` | SAM 3 + nur verschmelzen |
| `results_views_hybrid` | SAM 1 + Watershed |
| `results_views_tiefenprompt` | CHM-Wipfel → SAM (nur 3 Testframes) |

Die Sattelprominenz ist die objektive Version der Frage „wurden hier zwei Bäume
zerschnitten?": wie weit fällt die Oberfläche vom niedrigeren der beiden Wipfel bis zum
höchsten Punkt der gemeinsamen Grenze ab, relativ zur Spannweite des Bildes. Über alle
33 Frames sind **1222 von 3037 Trennungen (40 %) verdächtig** — mit deutlichen
Unterschieden zwischen den Beständen (`dense` 13 %, `100` 49 %).

## Hybrid: SAM + Watershed für die Restfläche

`segment_hybrid.py` kombiniert beide Verfahren dort, wo sie jeweils stark sind: SAM
legt zuerst die Kronen mit klarer Kante fest, danach läuft das Tiefen-Watershed
**ausschließlich auf der von SAM nicht erfassten Restfläche**. Damit verliert es seine
Hauptschwäche — es kann das Bild nicht mehr flächig zukacheln, sondern nur noch Lücken
schließen.

```bash
sbatch sbatch/run_segment_hybrid.sbatch
```

Ein Detail, das nötig war: die Prominenzschwelle der Wipfelsuche wird auf der
*Restfläche* berechnet, nicht auf dem ganzen Bild. Sonst dominiert das Relief der
bereits gefundenen SAM-Kronen die Statistik und der Rest fällt pauschal durch.

Der Zugewinn konzentriert sich genau dort, wo SAM schwach war:

| Ordner | SAM allein | Hybrid | Watershed-Anteil |
|---|---|---|---|
| `dense` | 53 % | **75 %** | 188 von 554 |
| `pines` | 50 % | **68 %** | 281 von 852 |
| `mixed1` | 57 % | 66 % | 103 von 638 |
| `80m` | 74 % | 76 % | 96 von 868 |

Die Herkunft steht in der CSV-Spalte `quelle`. Das ist wichtig für die Interpretation,
denn die Formgüte der beiden Anteile unterscheidet sich deutlich: SAM-Kronen haben eine
Median-Kompaktheit von 0.71 und Solidity 0.94, die Watershed-Ergänzungen nur 0.47 und
0.78. Die Ergänzungen sind also erkennbar unregelmäßiger — was zu erwarten ist, weil sie
per Konstruktion die Reste zwischen bereits gesetzten Kronen sind. Wer nur saubere
Instanzen braucht, filtert auf `quelle == "sam"`.

## Kronenabgrenzung mit SAM

```bash
sbatch sbatch/run_segment_sam.sbatch
FRAMES="100/frame_000537.jpg" EXTRA="--pred-iou-thresh 0.6" sbatch sbatch/run_segment_sam.sbatch
```

SAM erzeugt Masken auf allen Skalen gleichzeitig (Blatt, Ast, Krone, Bestand), die
Arbeit steckt in der Auswahl: Flächenfenster um die erwartete Kronengröße,
Kompaktheit gegen Schattenbänder, und eine gierige Überlappungsauflösung nach Score,
die den Skalenstapel auf eine Maske pro Krone reduziert.

Die wirksamsten Parameter sind **nicht** die eigenen Filter, sondern SAMs interne
Güteschwellen `--pred-iou-thresh` (Default hier 0.70 statt 0.88) und
`--stability-score-thresh` (0.85 statt 0.95). Mit den Bibliotheks-Defaults sortiert SAM
den Großteil der Kronen aus, bevor sie überhaupt herauskommen — im Kronendach sind
Grenzen objektiv unscharf, und die Defaults sind für alltagsübliche Objekte gedacht.
Absenken hob die Flächendeckung von 53 % auf 67 %.

## Kronenabgrenzung über monokulare Tiefe

`segment_trees.py` ersetzt die DeepForest-Boxen durch echte Kronenpolygone. Der Trick:
Einzelbaumabgrenzung braucht Höheninformation — zwei benachbarte grüne Kronen haben im
RGB oft keine sichtbare Grenze. Die Forstpraxis löst das über ein CHM, das hier fehlt.
Ein monokulares Tiefenmodell (`Depth-Anything-V2-Metric-Outdoor`, lag bereits im
`hf_cache`) liefert eine Ersatzoberfläche mit derselben Struktur:

1. Tiefe schätzen, invertieren (näher an der Kamera = höher).
2. **Detrend**: großskaligen Anteil abziehen — entfernt Kameraschräglage und den
   Bodenebenen-Prior des Modells. Analog zu DSM minus DTM.
3. Glätten, damit Blattwerktextur keine Scheinwipfel erzeugt.
4. Lokale Maxima als Wipfelmarker, Mindestabstand 0.3 × Kronendurchmesser.
5. Markerbasiertes Watershed auf der invertierten Oberfläche.
6. Segmente nach Fläche und Achsenverhältnis filtern.

```bash
sbatch sbatch/run_segment_trees.sbatch
CROWN_PX=120 EXTRA="--save-chm" sbatch sbatch/run_segment_trees.sbatch
```

Die Tiefenkarten werden unter `/scratch/shared/$USER/data/treeclf/depth_cache` gecacht,
Parametertuning läuft danach ohne Modellinferenz. Die drei wirksamen Stellschrauben sind
`--crown-px` (setzt alle Skalen), `--smooth-factor` und `--min-distance-factor`; die
letzten beiden bestimmen, wie viele Wipfel überleben, und damit die Flächendeckung.

Ergebnis über alle 33 Frames: **4697 Kronen**, 69–190 pro Frame, Mediandurchmesser
94–117 px, Flächendeckung des Kronendachs 71–99 %. Zum Vergleich DeepForest:
20563 Rohdetektionen mit 27 px Median, die Kronenfragmente statt Bäume treffen.

## Grenzen — bitte lesen, bevor Ergebnisse interpretiert werden

1. **Der Klassenraum ist kanadisch.** Der Checkpoint kennt exakt 14 Klassen aus dem
   Quebec-Trees-Datensatz und kann nichts anderes ausgeben:
   `dead`, *Thuja occidentalis*, *Abies balsamea*, *Larix laricina*, *Tsuga canadensis*,
   *Fagus grandifolia*, *Populus* (Gattung), *Acer pensylvanicum*, *Acer saccharum*,
   *Acer rubrum*, *Pinus strobus*, *Betula alleghaniensis*, *Betula papyrifera*,
   *Picea* (Gattung).
   Eine mitteleuropäische *Fagus sylvatica* landet zwangsläufig auf *Fagus grandifolia*,
   eine *Picea abies* auf *Picea* — auf **Gattungsebene** ist das oft brauchbar, auf
   Artebene nicht. Arten ohne nordamerikanisches Pendant (Douglasie, Eiche, Esche,
   Linde, Hainbuche) haben keine korrekte Ausgabemöglichkeit.
2. **Domain-Shift.** Trainiert auf photogrammetrischen Orthomosaiken bei 1.9 cm/px,
   angewendet auf komprimierte Video-Einzelframes bei ~5–8 cm/px. Die Softmax-Werte
   sind dadurch systematisch zu selbstsicher; `entropy` in der CSV ist der ehrlichere
   Indikator.
3. **Die Höhe ist nicht kalibriert.** `height_m_pred` stammt aus dem Höhen-Head und ist
   auf den Quebec-Höhenbereich (Mittel 14.2 m) und den dortigen Maßstab trainiert. Auf
   fremden Daten ohne DSM ist der Wert bestenfalls eine Rangordnung, keine Messung.
4. **Die Instanzen sind ein eigener Fehlerpfad.** DeepForest liefert in dichten
   Kronendächern Kronenfragmente statt Bäume — falsche Zentren verschieben den Crop und
   damit die Klassifikation, ohne dass es an DINOvTree liegt. `segment_trees.py`
   behebt das weitgehend, hat aber eigene Grenzen: die monokulare Tiefe ist auf
   Bodenperspektiven trainiert, nicht auf Nadir aus 80 m, und in strukturarmen
   Beständen verschmelzen benachbarte Kronen weiterhin (sichtbar an der Flächendeckung
   von nur 71–72 % in `urban` und `dense`).

Für belastbare Artansprache auf mitteleuropäischen Beständen führt kein Weg an
eigenen Labels vorbei: Der Classification Head (`query_token_cls`, `cross_attn_cls`,
`norm_cls`, `classifier` — zusammen ~3 M Parameter) lässt sich mit eingefrorenem
Backbone auf wenigen hundert annotierten Kronen neu trainieren.

## Job-Skripte

| Skript | Zweck |
|---|---|
| `sbatch/setup_env.sbatch` | venv auf /scratch (erbt Container-Torch) + Checkpoint, einmalig |
| `sbatch/run_infer_species.sbatch` | Voller Inferenzlauf über alle Ordner |
| `sbatch/run_scale_sweep.sbatch` | Maßstabs-Diagnose auf einzelnen Frames |
| `sbatch/run_segment_hybrid.sbatch` | Hybrid SAM + Watershed (bestes Ergebnis) |
| `sbatch/run_visualize.sbatch` | Diagnoseansichten aus den Labelkarten |
| `sbatch/run_refine_crowns.sbatch` | Teilen + Verschmelzen über die Tiefe |
| `sbatch/run_segment_sam.sbatch` | Kronenabgrenzung nur mit SAM |
| `sbatch/run_ablate_sam.sbatch` | Ablation über SAM-Varianten |
| `sbatch/run_segment_trees.sbatch` | Kronenabgrenzung über Tiefe + Watershed |
| `sbatch/run_detect_only.sbatch` | Nur DeepForest-Detektionen, zur Diagnose |
| `sbatch/run_depth_probe.sbatch` | Tiefenkarten/Hillshade zur Sichtprüfung |

Alles läuft über SLURM — nichts direkt auf dem Login-Knoten starten.
