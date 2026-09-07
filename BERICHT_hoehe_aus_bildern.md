# Höhe aus Bildern — Grundlagen, Verfahren und was davon in dieser Pipeline steckt

Ein Bericht von Grund auf, ohne Vorkenntnisse in Kameratechnik, Stereo oder
Photogrammetrie. Alle Beispiele rechnen mit den tatsächlichen Zahlen dieses
Projekts.

---

## Inhalt

1. [Das Problem: warum Farbe nicht reicht](#1-das-problem-warum-farbe-nicht-reicht)
2. [Wie eine Kamera aus einer 3D-Welt ein 2D-Bild macht](#2-wie-eine-kamera-aus-einer-3d-welt-ein-2d-bild-macht)
3. [Warum ein einzelnes Bild keine Höhe enthält](#3-warum-ein-einzelnes-bild-keine-höhe-enthält)
4. [Weg A — Höhe raten: monokulare Tiefenschätzung](#4-weg-a--höhe-raten-monokulare-tiefenschätzung)
5. [Weg B — Höhe messen: Parallaxe](#5-weg-b--höhe-messen-parallaxe)
6. [Das konkrete Verfahren: Plane + Parallax](#6-das-konkrete-verfahren-plane--parallax)
7. [Von einem Paar zur Karte](#7-von-einem-paar-zur-karte)
8. [Destillation: die Messung in ein Netz überführen](#8-destillation-die-messung-in-ein-netz-überführen)
9. [Was mit der Höhenkarte passiert: CHM, Wipfel, Watershed](#9-was-mit-der-höhenkarte-passiert-chm-wipfel-watershed)
10. [Woher die Modelle kommen: Trainingsdaten und Domänenlücke](#10-woher-die-modelle-kommen-trainingsdaten-und-domänenlücke)
11. [Fehlerbilder und wo die Pipeline steht](#11-fehlerbilder-und-wo-die-pipeline-steht)
12. [Glossar](#12-glossar)
13. [Literatur](#13-literatur)

---

## 1. Das Problem: warum Farbe nicht reicht

Ziel des Projekts ist, in Drohnenaufnahmen **einzelne Baumkronen** abzugrenzen und
anschließend deren Art zu bestimmen. Der zweite Teil ist gelöst (DINOvTree). Der
erste Teil ist das Problem.

Stell dir ein geschlossenes Kronendach von oben vor. Zwei benachbarte Buchen
berühren sich. Im RGB-Bild siehst du dort:

- links grün, mittig grün, rechts grün
- überall dieselbe Blattstruktur
- keine Kante, keine Farbdifferenz, kein Kontrast

Es gibt **im Bild** schlicht keine Information, wo der eine Baum aufhört. Ein
Mensch tut sich damit genauso schwer wie ein Algorithmus. Kein Segmentierungs-
modell der Welt kann eine Grenze finden, die nicht abgebildet ist.

In drei Dimensionen ist die Grenze dagegen offensichtlich: jeder Baum hat einen
Wipfel, und zwischen zwei Wipfeln liegt eine **Senke**. Das Kronendach sieht von
der Seite aus wie eine Hügelkette, nicht wie eine Platte. Die Baumgrenze ist der
Talboden zwischen zwei Hügeln.

```
        Wipfel A          Wipfel B
           /\                /\
          /  \    Senke     /  \
         /    \    \/      /    \
        /      \______\___/      \
       /                          \
    ===============================  Boden
       |<-- Baum A -->|<- Baum B ->|
              Grenze liegt hier ^
```

Genau so arbeitet die Forstfernerkundung seit Jahrzehnten: auf einem **CHM**
(Canopy Height Model, Kronenhöhenmodell). Das ist eine Rasterkarte, die für jeden
Bodenpunkt sagt, wie hoch die Vegetation dort ist. Wipfel = lokale Maxima,
Grenzen = Wasserscheiden dazwischen. Erzeugt wird ein CHM normalerweise mit
**LiDAR** — ein Laserscanner am Flugzeug misst Millionen von Entfernungen.

Wir haben kein LiDAR. Wir haben Videoframes einer Drohne.

Die gesamte Pipeline dreht sich deshalb um eine Frage:

> **Wie bekommt man aus gewöhnlichen RGB-Bildern eine brauchbare Höhenkarte?**

Es gibt dafür zwei grundsätzlich verschiedene Antworten — raten und messen. Beide
sind in diesem Repo implementiert, und der Vergleich der beiden ist der Kern der
Arbeit.

---

## 2. Wie eine Kamera aus einer 3D-Welt ein 2D-Bild macht

Um zu verstehen, warum Höhe schwer ist, muss man erst verstehen, was eine Kamera
überhaupt tut. Das ist einfacher als es klingt.

### Das Lochkameramodell

Denk dir eine geschlossene Schachtel mit einem winzigen Loch vorne und einem Film
hinten. Licht von einem Punkt in der Welt fällt durch das Loch und trifft genau
eine Stelle auf dem Film. Das ist das gesamte Modell — moderne Objektive sind
komplizierter, aber geometrisch verhalten sie sich so.

```
   Welt                    Loch            Sensor
                             |
   Punkt P  ----             |             ----
   (X, Y, Z)    ----         |         ----
                    ----     |     ----
                        -----o-----  <-- Bildpunkt (u, v)
                    ----     |     ----
                ----         |         ----
            ----             |             ----
                          <- f ->
                        Brennweite
```

Die Abbildungsgleichung dazu ist eine simple Strahlensatz-Rechnung:

```
u = f * X / Z
v = f * Y / Z
```

- `X, Y, Z` sind die Weltkoordinaten des Punkts, gemessen von der Kamera aus.
  `Z` ist die **Tiefe** — der Abstand entlang der Blickrichtung.
- `u, v` sind die Bildkoordinaten in Pixeln.
- `f` ist die **Brennweite, ausgedrückt in Pixeln**. Das ist keine physikalische
  Länge, sondern eine Umrechnungskonstante, die von Objektiv *und* Sensorauflösung
  abhängt.

Die entscheidende Stelle ist die **Division durch Z**. Alles wird kleiner, je
weiter es weg ist. Das ist Perspektive, in einer Zeile.

### Brennweite und Bildwinkel bei unseren Frames

Man gibt Objektive meist über den **Bildwinkel** (Field of View) an statt über
`f`. Der Zusammenhang:

```
f = (Bildbreite_px / 2) / tan(HFOV / 2)
```

Für dieses Projekt ([infer_species.py:350](infer_species.py#L350)):

- Bildbreite: 1920 px (die Frames sind 1920×1080)
- horizontaler Bildwinkel: 73,7°  ← **Achtung: das ist ein Default-Schätzwert im
  Code, keine gemessene Kameraangabe.** Falls die echten EXIF-Daten verfügbar
  sind, sollte der Wert dort herkommen.

```
f = 960 / tan(36,85°) = 960 / 0,750 = 1280 px
```

### Bodenauflösung (GSD)

Bei einer Nadiraufnahme — Kamera zeigt senkrecht nach unten — ist `Z` für den
Boden gleich der Flughöhe `H`. Damit lässt sich ausrechnen, wie viele Meter ein
Pixel abdeckt. Das nennt man **GSD** (Ground Sample Distance),
[infer_species.py:133](infer_species.py#L133):

```
Bodenbreite = 2 * H * tan(HFOV / 2)
GSD         = Bodenbreite / Bildbreite_px
```

Für den Ordner `80m`:

```
Bodenbreite = 2 * 80 m * 0,750 = 120 m
GSD         = 120 m / 1920 px  = 0,0625 m/px  =  6,25 cm pro Pixel
```

**Was das praktisch bedeutet:**

| Objekt | reale Größe | im Bild |
|---|---|---|
| Baumkrone, mittel | 8 m Durchmesser | 128 px |
| Baumkrone, groß | 15 m | 240 px |
| einzelner Ast | 20 cm | 3 px |
| Parkbank | 1,5 m | 24 px |

Der Default `--crown-px 100` in [segment_trees.py](segment_trees.py) entspricht
also einer Krone von gut 6 m Durchmesser. Plausibel.

> **Randnotiz zum Maßstabsproblem:** Der DINOvTree-Checkpoint wurde auf Kacheln
> mit 1,9 cm/px trainiert ([infer_species.py:41](infer_species.py#L41)). Unsere
> Frames haben 6,25 cm/px — Faktor 3,3 gröber. Ein Baum, den das Netz im Training
> mit 400 px Breite gesehen hat, ist bei uns 120 px breit. Deshalb gibt es
> [scale_sweep.py](scale_sweep.py) und die GSD-Umrechnung beim Zuschneiden. Das
> ist ein eigenes Thema, aber es zeigt: Maßstab ist in dieser Domäne überall das
> Kernproblem.

---

## 3. Warum ein einzelnes Bild keine Höhe enthält

Jetzt kommt der springende Punkt. Schau nochmal auf die Abbildungsgleichung:

```
u = f * X / Z
```

Wir kennen `u` (das ist das Bild) und `f`. Wir wollen `X` und `Z` wissen. Das ist
**eine Gleichung mit zwei Unbekannten**. Sie hat unendlich viele Lösungen.

Konkret: alle diese Punkte landen auf demselben Pixel.

```
Kamera
   o
   |\
   | \      * kleiner Baum, 30 m entfernt
   |  \
   |   \
   |    \
   |     \     * mittlerer Baum, 60 m entfernt
   |      \
   |       \
   |        \
   |         \      * riesiger Baum, 90 m entfernt
   |          \
```

Ein 3 m hoher Strauch nah an der Kamera und eine 30 m hohe Eiche weit weg sind im
Bild **exakt ununterscheidbar**, wenn sie auf demselben Sehstrahl liegen. Das ist
kein Mangel der Technik, sondern eine mathematische Eigenschaft der Projektion:
Information geht verloren, und zwar unwiederbringlich.

Diese Mehrdeutigkeit heißt **Tiefenmehrdeutigkeit** (depth ambiguity). Sie ist der
Grund, warum es die beiden folgenden Kapitel überhaupt gibt.

Es gibt genau zwei Auswege:

- **Raten.** Zusätzliches Wissen über die Welt einbringen — "Bäume sind meistens
  zwischen 5 und 40 m hoch", "Blattwerk sieht aus dieser Entfernung so aus". Das
  macht ein neuronales Netz. → Kapitel 4
- **Messen.** Eine zweite Beobachtung von einem anderen Ort hinzunehmen. Dann sind
  es zwei Gleichungen für zwei Unbekannte, und die Sache ist eindeutig lösbar.
  → Kapitel 5

---

## 4. Weg A — Höhe raten: monokulare Tiefenschätzung

### Was ein Tiefenmodell tut

Ein monokulares Tiefenmodell bekommt ein einzelnes Bild und gibt für jeden Pixel
eine Tiefe aus. Es kann das nur, weil es aus Millionen von Trainingsbildern
statistische Zusammenhänge gelernt hat:

- **Bekannte Größen.** Ein Auto ist ungefähr 4,5 m lang. Nimmt es 200 px ein, ist
  die Entfernung ableitbar.
- **Verdeckung.** Was ein anderes Objekt überlappt, ist näher.
- **Texturgradient.** Gras wird mit der Entfernung feiner und kontrastärmer.
- **Perspektivische Linien.** Straßenränder, die konvergieren, geben Tiefe.
- **Dunst.** Ferne Objekte werden blasser und bläulicher.
- **Schattierung.** Wie Licht auf eine Oberfläche fällt, verrät ihre Neigung.

Das sind alles **Prioren** — gelernte Annahmen darüber, wie die Welt normalerweise
aussieht. Nichts davon ist eine Messung. Das Modell rät sehr gut informiert.

### Was in [depth_probe.py](depth_probe.py) benutzt wird

```python
MODELS = {
    "depth_anything": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "depthpro":       "apple/DepthPro-hf",
}
```

Beides sind große Vision Transformer mit Decoder. Interessant: **Depth Anything
ist selbst durch Destillation entstanden** — ein Lehrermodell auf 1,5 M gelabelten
Bildern, dann Pseudolabels für 62 M ungelabelte Bilder, darauf ein Schüler
trainiert. Merk dir das für Kapitel 8; unser Ansatz ist dasselbe Rezept mit einem
anderen Lehrer.

### Warum es hier nicht gut funktioniert

**Erstens: falsche Domäne.** Diese Modelle wurden auf Bildern trainiert, die
Menschen aufnehmen — Straßenszenen, Innenräume, Landschaften vom Boden aus. In
solchen Bildern gibt es einen Horizont, einen Vordergrund, konvergierende Linien,
Objekte bekannter Größe. Ein Kronendach aus 80 m Höhe senkrecht von oben hat
nichts davon. Es ist eine texturierte grüne Fläche ohne jeden der gelernten
Anhaltspunkte. Das Modell ist dort **out of distribution** — es rät weiter, aber
ohne verwertbare Grundlage.

**Zweitens: Halluzination in kontrastarmen Bereichen.** Wo ein Netz keine Anhalts-
punkte hat, erfindet es das Plausibelste — meist eine glatte Fläche. Ausgerechnet
im geschlossenen Kronendach, wo wir die Grenze bräuchten, liefert es also
verlässlich eine sanfte Wölbung ohne Senken.

**Drittens — und das ist der schönste Beleg: Schatten.**

Das Netz hat gelernt, dass dunkle Bereiche oft Vertiefungen, Nischen oder
Verdeckungen sind. Ein Schlagschatten auf einer Wiese ist aber geometrisch
**exakt so hoch wie die Wiese** — nämlich null. Trotzdem taucht er in der
Tiefenkarte als Struktur auf.

In [results_views_tiefe_gegen_parallax/urban/](results_views_tiefe_gegen_parallax/urban/)
lässt sich das direkt ansehen. Die Instanzansicht des Park-Screenshots zeigt bei
vielen freistehenden Bäumen ein Polygon, das die Krone **plus ihren Schlagschatten**
umfasst, konsistent nach rechts-unten versetzt — also in Schattenrichtung. Bei
208 Instanzen und 41 % Abdeckung ist das kein Einzelfall.

Zwei Ursachen wirken hier zusammen:

1. **SAM segmentiert nach Kontrast.** Krone und angehefteter Schatten bilden auf
   heller Wiese *einen* dunklen Fleck. Dessen Außenkante ist der stärkere
   Gradient; die Grenze zwischen Krone und ihrem eigenen Schatten ist viel
   schwächer. SAM nimmt die stärkere Kante. Der Kompaktheitsfilter
   ([segment_sam.py:85](segment_sam.py#L85)) fängt freistehende Schattenbänder ab,
   aber ein direkt anhängender Schatten ergibt eine noch halbwegs runde Form.
2. **Die Tiefenkarte kann es nicht korrigieren.** Der Bodenfilter in
   [segment_hybrid.py:53-55](segment_hybrid.py#L53-L55) schließt Boden über ein
   *Höhenperzentil* aus. Wenn die Tiefenkarte den Schatten für erhaben hält,
   greift dieser Filter nicht.

Merke dir dieses Beispiel — in Kapitel 6 löst es sich von selbst auf.

---

## 5. Weg B — Höhe messen: Parallaxe

### Das Alltagsphänomen

Halte einen Finger vor dein Gesicht und schließe abwechselnd das linke und rechte
Auge. Der Finger springt hin und her. Ein Baum am Horizont springt nicht.

Das ist **Parallaxe**: Bei einem Ortswechsel des Betrachters verschieben sich nahe
Objekte stärker als ferne. Die Stärke der Verschiebung ist ein direktes Maß für
die Entfernung. Dein Gehirn wertet das ständig aus — das ist räumliches Sehen.

Wichtig: das ist keine Schätzung und kein Prior. Es ist eine geometrische
Notwendigkeit.

### Die Rechnung

Zwei Kameras (oder eine Kamera an zwei Orten) im Abstand `B` — die **Basislinie**.
Ein Punkt in Tiefe `Z` erscheint in beiden Bildern an leicht verschiedener
Stelle. Der Unterschied heißt **Disparität** `d`:

```
d = f * B / Z          und umgestellt:          Z = f * B / d
```

Zwei Bilder, zwei Gleichungen, zwei Unbekannte — die Mehrdeutigkeit aus Kapitel 3
ist aufgelöst. **Deshalb ist das eine Messung und kein Raten.**

Ein paar Konsequenzen, die man im Kopf haben sollte:

- **Längere Basislinie = stärkeres Signal.** Doppelter Abstand, doppelte
  Disparität. Deshalb der Parameter `--pair-stride` in
  [stereo_probe.py](stereo_probe.py): größerer Frameabstand = längere Basislinie.
- **Aber:** längere Basislinie = weniger Bildüberlappung und schwierigere
  Zuordnung. Ein Kompromiss, kein freies Mittagessen.
- **Keine Bewegung = kein Signal.** Wenn die Drohne schwebt, ist `B ≈ 0`, also
  `d ≈ 0`. Man misst nur Rauschen. Genau deshalb steht in
  [build_parallax.py:48](build_parallax.py#L48) `--min-displacement 15` — Paare
  darunter werden verworfen.

### Warum wir kein klassisches Stereo machen

In der klassischen Stereo-Vision braucht man:

- zwei **kalibrierte** Kameras (Brennweite, Verzeichnung, Hauptpunkt exakt bekannt)
- die genaue relative Position und Orientierung beider Kameras
- eine **Rektifizierung** — beide Bilder so entzerren, dass korrespondierende
  Punkte auf derselben Bildzeile liegen

Nichts davon haben wir. Es sind Videoframes einer Drohne, ohne Kalibrierprotokoll,
ohne bekannte Pose, ohne Zeitstempel-Synchronisation. Der Bildwinkel im Code ist
ein Schätzwert.

Deshalb nimmt dieses Projekt einen Weg, der **ohne Kalibrierung auskommt** und
dafür auf absolute Meterangaben verzichtet.

---

## 6. Das konkrete Verfahren: Plane + Parallax

Das ist der Kern von [stereo_probe.py](stereo_probe.py). Der Ansatz ist alt und
gut untersucht — er heißt in der Literatur **Plane + Parallax (P+P)**, entwickelt
in den 90ern von Irani, Anandan, Kumar und anderen.

### Die Grundidee in einem Satz

> Rechne heraus, wie sich der **Boden** zwischen zwei Frames verschoben hat, und
> zieh das ab. Was übrig bleibt, kann nur von Objekten stammen, die **nicht** auf
> dem Boden liegen — und ihr Betrag wächst mit der Höhe.

Das Elegante daran: man muss die Kamerabewegung nie explizit bestimmen. Sie steckt
implizit in der Bodenverschiebung und wird mit ihr weggerechnet.

### Schritt 1 — Korrespondenzen finden (SIFT)

Wir brauchen zuerst Punktpaare: "diese Ecke in Frame A ist dieselbe Ecke wie jene
in Frame B".

**SIFT** (Scale-Invariant Feature Transform, Lowe 1999/2004) macht das in zwei
Teilen:

- *Detektor:* findet markante Stellen — Ecken, Flecken, Strukturen, die sich von
  ihrer Umgebung abheben. Eine glatte Wiese liefert nichts, ein Dachfirst oder
  eine Astgabel schon.
- *Deskriptor:* beschreibt die Umgebung jeder Stelle als 128 Zahlen, konstruiert
  so, dass die Beschreibung sich nicht ändert, wenn das Bild gedreht, skaliert
  oder heller wird.

Dann werden Deskriptoren zwischen den Bildern verglichen. Der **Ratio-Test**
([stereo_probe.py:53](stereo_probe.py#L53)) ist dabei wichtiger als er aussieht:

```python
good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
```

Für jeden Punkt aus A werden die zwei ähnlichsten Kandidaten in B gesucht. Nur
wenn der beste **deutlich** besser ist als der zweitbeste (Faktor 0,75), wird die
Zuordnung akzeptiert. In einem Wald sehen sehr viele Stellen sehr ähnlich aus —
ohne diesen Test wäre die Hälfte der Zuordnungen falsch.

### Schritt 2 — Die Bodenebene bestimmen (Homographie + RANSAC)

Eine **Homographie** ist eine 3×3-Matrix, die beschreibt, wie sich eine *ebene
Fläche* zwischen zwei Kameraansichten abbildet. Sie hat eine bemerkenswerte
Eigenschaft: für eine echte Ebene ist die Abbildung **exakt**, egal wie sich die
Kamera bewegt hat — Verschiebung, Drehung, Neigung, Zoom, alles inbegriffen.

Anschauung: fotografiere ein Schachbrett zweimal aus verschiedenen Winkeln. Die
Homographie ist die Transformation, die das eine Foto perfekt auf das andere legt.
Für alles, was aus der Brettebene herausragt, funktioniert sie nicht.

Bei Nadiraufnahmen über Wald ist die dominante Ebene der **Waldboden**. Der ist
nicht perfekt eben, aber gut genug — und er ist die Bezugsfläche, gegen die wir
Höhe messen wollen.

**RANSAC** (Fischler & Bolles 1981) ist der Trick, mit dem das trotz falscher
Zuordnungen funktioniert. Statt alle Punkte gleich zu gewichten:

1. Ziehe zufällig 4 Punktpaare und berechne daraus eine Homographie.
2. Prüfe alle übrigen Paare: wie viele passen zu dieser Hypothese (Fehler unter
   `--ransac-thresh`, hier 3 px)? Das sind die **Inlier**.
3. Wiederhole hunderte Male, behalte die Hypothese mit den meisten Inliern.

Das Ergebnis ist robust gegen einen erheblichen Anteil grober Fehler — und
gleichzeitig gegen die Baumkronen selbst. **Kronen sind für RANSAC Ausreißer**,
weil sie sich nicht wie der Boden verhalten. Genau die wollen wir ja finden.

```python
homography, inliers = cv2.findHomography(points_b, points_a, cv2.RANSAC, args.ransac_thresh)
```

### Schritt 3 — Entzerren

```python
warped = cv2.warpPerspective(gray_b, homography, (gray_a.shape[1], gray_a.shape[0]))
```

Frame B wird so verzogen, dass sein Boden deckungsgleich mit dem Boden von
Frame A liegt. Danach gilt:

- **Boden:** in beiden Bildern an derselben Stelle → Differenz null
- **Wipfel:** noch versetzt, weil sie sich nicht wie die Ebene verhalten
- **Bildrand:** teilweise schwarz, weil B nicht die volle Fläche von A abdeckt

Der letzte Punkt wird in [stereo_probe.py:105](stereo_probe.py#L105) abgefangen:
`valid = warped > 0`.

### Schritt 4 — Restfluss messen (optischer Fluss)

**Optischer Fluss** beantwortet für *jeden einzelnen Pixel*: wohin ist er
gewandert? Ergebnis ist ein Vektorfeld, zwei Zahlen pro Pixel.

Wir wenden ihn auf das bereits entzerrte Paar an. Was hier noch an Bewegung
gefunden wird, ist per Konstruktion nur noch der höhenbedingte Anteil.

```python
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
dis.setFinestScale(args.finest_scale)   # 0 = feinste Stufe
dis.setPatchSize(args.patch_size)       # 8
flow = dis.calc(gray_a, warped, None)
residual = np.linalg.norm(flow, axis=2)  # Betrag des Restvektors
```

**Warum DIS und nicht Farneback:** Farneback passt in einem Fenster (Default hier
41 px) ein Polynom an. Das glättet — und verschmiert genau die kleinräumigen
Kronendetails, um die es geht. DIS arbeitet mit kleinen Patches (8 px) und
räumlicher Propagation und ist subpixelgenau. Das ist entscheidend, wie die
nächste Rechnung zeigt.

### Schritt 5 — Restfluss ist Höhe

Jetzt die Formel. Kamera in Höhe `H` über dem Boden, Punkt in Höhe `h` über dem
Boden, also in Tiefe `Z = H - h`. Basislinie `B`.

```
Verschiebung eines Bodenpunkts:   f * B / H
Verschiebung eines Punkts in h:   f * B / (H - h)

Restfluss = Differenz            =  f * B * h
                                   ------------
                                    H * (H - h)
```

**Rechenbeispiel mit unseren Zahlen** (f = 1280 px, H = 80 m, Basislinie B = 2 m):

| Objekt | Höhe h | Restfluss |
|---|---|---|
| Schatten auf der Wiese | 0 m | **0,00 px** |
| Wiese, Weg, Parkplatz | 0 m | **0,00 px** |
| Auto | 1,5 m | 0,61 px |
| Strauch | 5 m | 2,13 px |
| mittlerer Baum | 15 m | 7,38 px |
| hoher Baum | 25 m | 14,55 px |
| Dachfirst | 12 m | 5,65 px |

Drei Dinge fallen sofort auf:

**(a) Der Effekt ist klein.** Ein 5-m-Strauch bewegt sich um 2 Pixel. Deshalb ist
Subpixelgenauigkeit keine Feinheit, sondern die Voraussetzung — und deshalb DIS.

**(b) Der Zusammenhang ist nichtlinear.** Bei h = 25 m ist der Restfluss nicht
5× der von h = 5 m, sondern fast 7×. Hohe Bäume werden überproportional
hervorgehoben. Für relative Wipfelsuche ist das unschädlich, für metrische
Aussagen wäre es zu korrigieren.

**(c) Der Schatten hat exakt null Parallaxe.** Und das ist der zentrale Punkt.

### Warum das das Schattenproblem löst

Ein Schlagschatten liegt **auf dem Boden**. Er verschiebt sich zwischen zwei
Frames also exakt wie die Bodenebene — und die wird von der Homographie
vollständig weggerechnet. Restfluss null.

Es spielt keine Rolle, wie dunkel er ist. SIFT interessiert sich für Struktur,
nicht für Helligkeit. Optischer Fluss misst Verschiebung, nicht Farbe. Die
Geometrie kennt den Begriff "dunkel" gar nicht.

Noch schärfer formuliert: **der Schatten wandert nicht mit dem Baum.** Er klebt am
Boden, die Krone steht darüber. Zwischen Frame A und Frame B trennen sich die
beiden sichtbar voneinander. Genau diese Trennung ist das Signal — und ein
Einzelbildverfahren kann sie prinzipiell nicht sehen.

Das ist der stärkste inhaltliche Grund für den ganzen Parallaxe-Ansatz.

### Was schiefgehen kann

Ehrlichkeitshalber, das Verfahren hat klare Grenzen:

| Problem | Ursache | Abhilfe im Code |
|---|---|---|
| Drohne schwebt | B ≈ 0, kein Signal | `--min-displacement 15` |
| Zu wenige SIFT-Punkte | homogene Textur, Unschärfe | Rückgabe `None` bei < 20 Punkten |
| Boden nicht sichtbar | im geschlossenen Bestand passt RANSAC evtl. an die Kronenebene | — offen, siehe unten |
| Wind bewegt Äste | echte Bewegung, wird als Höhe fehlgedeutet | — offen |
| Rolling Shutter | Sensor liest zeilenweise, verzerrt bei schneller Bewegung | — offen |
| Zu lange Basislinie | zu wenig Überlappung | `--min-overlap 0.6` |

Der Punkt "Boden nicht sichtbar" verdient Beachtung: RANSAC findet die Ebene mit
den meisten Inliern. In einem wirklich geschlossenen Bestand könnte das die
**Kronenebene** sein statt des Bodens. Dann misst man Höhe relativ zum mittleren
Kronendach — für die Wipfelsuche immer noch brauchbar, aber die Interpretation
ändert sich. Man kann das prüfen: liegen die Inlier im Bild dort, wo Boden sichtbar
ist?

---

## 7. Von einem Paar zur Karte

[stereo_probe.py](stereo_probe.py) rechnet ein Paar. Für die Pipeline braucht es
je Frame **eine** Karte, und möglichst rauscharm. Das macht
[build_parallax.py](build_parallax.py).

### Problem 1: die Skala ist willkürlich und wechselt

Der Restfluss ist proportional zu `B` — der Kamerabewegung zwischen den beiden
Frames. Die kennen wir nicht und sie ist bei jedem Paar anders. Ein Paar mit 4 m
Abstand liefert doppelt so hohe Werte wie eins mit 2 m, **bei identischer Szene**.

Ungewichtet gemittelt würde das längste Paar alle anderen überstimmen.

**Lösung:** jede Karte durch die mediane Verschiebung ihres Paares teilen. Die
mediane Verschiebung ist ein Proxy für die Basislinie. Danach sind alle Karten auf
derselben — weiterhin unbekannten — Skala.

### Problem 2: einzelne Paare sind verrauscht

Optischer Fluss macht Fehler, besonders in homogenen Bereichen. **Lösung:** für
jeden Frame A werden alle anderen Frames desselben Ordners als Partner
durchprobiert und die normierten Ergebnisse gemittelt. Zufällige Fehler mitteln
sich weg, das echte Höhensignal bleibt.

### Was dabei rauskommt

```
/scratch/shared/nik/data/treeclf/parallax_cache/
    100__frame_000537.npy
    80m__frame_000297.npy
    dense__frame_000073.npy
    ...
```

Eine `.npy`-Datei pro Frame, im selben Format wie der Tiefencache — damit die
nachgelagerten Skripte beide Quellen austauschbar verwenden können
([visualize_crowns.py:210](visualize_crowns.py#L210): `--surface depth|parallax`).

### Der aktuelle Umfang — und eine wichtige Einschränkung

```
$ ls parallax_cache/ | sed 's/__.*//' | sort | uniq -c
      4 100      4 80m      4 dense      4 dense1
      4 mixed    4 mixed1   5 pines
```

**29 Frames aus 7 Ordnern.** Der Ordner `urban` fehlt vollständig — er besteht aus
vier voneinander unabhängigen Google-Earth-Screenshots verschiedener Tage, nicht
aus einer Videosequenz. Ohne gemeinsame Szene keine Basislinie, keine Parallaxe.

Das ist bei der Bewertung der bisherigen Ergebnisse **entscheidend**: die
Schattenfehler aus Kapitel 4 stammen aus `urban` und zeigen ausschließlich die
Tiefe-Seite. Der Ordnername `results_views_tiefe_gegen_parallax` ist dort
irreführend.

29 Frames sind außerdem eine sehr schmale Datenbasis für das nächste Kapitel.

---

## 8. Destillation: die Messung in ein Netz überführen

### Das verbleibende Problem

Die Parallaxe ist gemessen statt geraten — aber sie braucht **mehrere Frames
derselben Szene mit Kamerabewegung dazwischen**. Für ein einzelnes Foto ist sie
nicht ausführbar. Für den späteren Betrieb ist das eine harte Einschränkung.

### Was Destillation ist

Klassisch (Hinton et al. 2015): ein großes, langsames **Lehrermodell** erzeugt
Ausgaben, ein kleines **Schülermodell** wird darauf trainiert, diese Ausgaben
nachzuahmen. Der Schüler lernt nicht aus den Originallabels, sondern aus dem, was
der Lehrer produziert. Ziel: fast dieselbe Güte, ein Bruchteil der Kosten.

Bei uns ist der Lehrer **kein Netz, sondern ein Verfahren** — die Parallaxen-
rechnung. Destilliert wird deshalb nicht Rechenaufwand, sondern eine
**Fähigkeit**:

```
vorher:   braucht Videosequenz + SIFT + Homographie + optischer Fluss
nachher:  braucht ein Bild
```

Das Wissen wandert aus dem Algorithmus in Gewichte.

Genau so sind die Modelle aus Kapitel 4 gebaut worden. MegaDepth (2018) ließ
Structure-from-Motion auf Internet-Fotosammlungen laufen und nahm die Ergebnisse
als Trainingsziel für ein Einzelbildnetz. Depth Anything (2024) skalierte das auf
62 Millionen Bilder. Wir machen dasselbe — nur mit einem Lehrer aus **unserer
eigenen Domäne** statt aus generischen Bodenbildern.

### Der Aufbau in [distill_height.py](distill_height.py)

**Backbone: DINOv3 ViT-B/16, eingefroren**

```python
for parameter in backbone.parameters():
    parameter.requires_grad_(False)
backbone.eval()
```

Ein Vision Transformer zerlegt das Bild in Kacheln von 16×16 Pixeln (`PATCH = 16`)
und beschreibt jede Kachel durch einen Vektor mit 768 Zahlen — einen **Patch-Token**.
Bei einem 512×512-Ausschnitt sind das 32×32 = 1024 Tokens.

Man kann sich einen Token als reichhaltige Beschreibung vorstellen: "hier ist
Nadelbaumtextur, mittlerer Kontrast, Kante von links oben nach rechts unten,
Beleuchtung von schräg". Was genau drinsteht, ist gelernt und nicht direkt
interpretierbar — aber es ist wesentlich informativer als die Rohpixel.

Der Backbone stammt aus dem DINOvTree-Checkpoint, ist also bereits auf Baumkronen
feinjustiert. Er wird **nicht weiter trainiert**. Grund: 29 Frames würden ein
86-Millionen-Parameter-Modell in Sekunden überanpassen.

**Kopf: kleiner Faltungsdecoder, ~2 M Parameter**

```python
nn.Conv2d(768, 256, 3), GELU,
nn.Conv2d(256, 256, 3), GELU,
Upsample(×2),
nn.Conv2d(256, 128, 3), GELU,
Upsample(×2),
nn.Conv2d(128,  64, 3), GELU,
nn.Conv2d(64,    1, 1),      # → ein Kanal = Höhe
```

Die Token-Karte (32×32×768) wird schrittweise hochskaliert und auf einen Kanal
reduziert. Nur diese ~2 M Parameter werden trainiert. Das ist wenig genug, dass
29 Frames plus Augmentierung reichen könnten.

**Trainingsdaten: zufällige Ausschnitte**

512×512-Crops aus den Frames, dazu Drehungen um Vielfache von 90° und Spiegelung
([distill_height.py:113](distill_height.py#L113)). Bei Nadiraufnahmen ist
Rotationsaugmentierung physikalisch legitim — es gibt kein "oben" in einem
Senkrechtbild. Aus 29 Frames werden so beliebig viele Trainingsbeispiele, wenn
auch keine beliebig vielen *unabhängigen*.

### Der entscheidende Trick: skaleninvarianter Verlust

Das ist die konzeptionell wichtigste Zeile im ganzen Skript.

```python
loss = F.l1_loss(standardize(prediction), standardize(targets))
```

```python
def standardize(x):
    mean = x.flatten(1).mean(dim=1, keepdim=True)
    std  = x.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-6)
    return ((x.flatten(1) - mean) / std).view_as(x)
```

**Warum das nötig ist:** Die Parallaxe hat keine bekannte Einheit. Selbst nach der
Normierung in [build_parallax.py](build_parallax.py) bleibt ein Restfaktor, der
von der Flugsituation abhängt. Zwei Frames mit identischer Waldstruktur können
Zielwerte haben, die sich um Faktor 3 unterscheiden — nur weil die Drohne
unterschiedlich schnell flog.

Ein gewöhnlicher L1-Verlust würde das Netz zwingen, aus dem Bild die
Fluggeschwindigkeit zu erraten. Das ist unmöglich und das Netz würde sich in
Kompromissen verlieren.

**Was standardize bewirkt** — ein Zahlenbeispiel:

```
Ziel A (langsamer Flug):   [1, 2, 3, 4, 5]     → standardisiert: [-1,41, -0,71, 0, 0,71, 1,41]
Ziel B (schneller Flug):   [10, 20, 30, 40, 50] → standardisiert: [-1,41, -0,71, 0, 0,71, 1,41]
```

Nach der Standardisierung sind beide **identisch**. Skala und Offset sind
entfernt, nur die *Form* des Reliefs bleibt.

Und genau die Form ist alles, was wir brauchen: Wipfelsuche liest lokale Maxima,
Watershed liest Sattelprominenz. Beide sind gegenüber jeder monotonen Streckung
unempfindlich. Absolute Meter würden ohnehin nirgends verwendet.

Diese Idee stammt von Eigen et al. (2014) und ist über MiDaS (Ranftl et al.) zum
Standard in der Tiefenschätzung geworden.

### Split nach Ordnern, nicht nach Frames

```python
parser.add_argument("--val-folders", nargs="*", default=["dense", "mixed"])
```

Das ist methodisch wichtig. Frames desselben Fluges überlappen räumlich — teils
zeigen sie **denselben Baum** aus leicht anderem Winkel. Ein zufälliger
Frame-Split hätte dieselben Bäume in Training und Validierung. Der
Validierungsverlust sähe hervorragend aus und würde nichts über neue Bestände
aussagen.

Deshalb werden ganze Ordner zurückgehalten. Bei 7 Ordnern insgesamt bedeutet das
allerdings: **5 Ordner Training, 2 Ordner Validierung, 29 Frames gesamt.** Das ist
sehr wenig. Die Validierungszahl wird eine hohe Varianz haben.

### Der Gesamtablauf

```
  ┌─────────────────────────────────────────────────────────┐
  │  LEHRER — geometrisch, kein neuronales Netz             │
  │                                                         │
  │  Videoframes  →  SIFT  →  RANSAC-Homographie            │
  │              →  optischer Fluss  →  Restfluss           │
  │              →  normieren, über Partner mitteln         │
  │                                                         │
  │  Ergebnis: gemessene Höhenkarte  (braucht Videofolge)   │
  └───────────────────────┬─────────────────────────────────┘
                          │  dient als Trainingsziel
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  SCHÜLER — neuronales Netz                              │
  │                                                         │
  │  Einzelbild  →  DINOv3 ViT-B/16 (eingefroren)           │
  │              →  Faltungsdecoder (2 M, trainiert)        │
  │              →  skaleninvarianter L1 gegen das Ziel     │
  │                                                         │
  │  Ergebnis: geschätzte Höhenkarte  (braucht ein Bild)    │
  └─────────────────────────────────────────────────────────┘
```

### Die naheliegende Sorge

Der Schüler kann nicht besser werden als sein Lehrer. Sind die Parallaxenkarten
verrauscht, lernt der Kopf das Rauschen mit. MegaDepth berichtet genau dieses
Problem und brauchte explizite Datenbereinigung.

Zum Vergleich: Tolan et al. (2024, Meta) haben **dieselbe Architektur** — DINOv2
eingefroren plus Faltungsdecoder — mit **LiDAR** als Lehrer trainiert und
erreichen 2,8 m mittleren absoluten Fehler. Unser Lehrer ist deutlich schwächer
und unsere Datenmenge um Größenordnungen kleiner. Die Erwartung sollte
entsprechend kalibriert sein.

---

## 9. Was mit der Höhenkarte passiert: CHM, Wipfel, Watershed

Egal welche Quelle — Tiefe, Parallaxe oder destilliertes Netz — die Weiterver-
arbeitung ist dieselbe. Sie steht in [segment_trees.py](segment_trees.py) und
folgt dem forstlichen Standardverfahren.

### Schritt 1: Ersatz-CHM bilden

```python
def build_pseudo_chm(depth, crown_px, detrend_factor):
    surface = -depth.astype(np.float32)                                  # invertieren
    trend   = cv2.GaussianBlur(surface, (0, 0), crown_px * detrend_factor)  # Trend
    return surface - trend                                               # Detrend
```

Drei Operationen:

**Invertieren.** Tiefenmodelle geben Entfernung aus. Näher an der Kamera = höher
über dem Boden. Vorzeichenwechsel.

**Trend berechnen.** Eine sehr starke Weichzeichnung (Sigma = 100 px × 3 = 300 px)
ergibt die großräumige Form ohne jedes Kronendetail.

**Trend abziehen.** Das entfernt:
- die Schräglage der Kamera (ein Bildrand ist weiter weg als der andere)
- Geländeneigung
- den Bodenebenen-Prior des Tiefenmodells

Das ist die Bildverarbeitungs-Entsprechung von **DSM minus DTM** — Oberflächen-
modell minus Geländemodell ergibt Vegetationshöhe. In der Forstfernerkundung das
Standardrezept.

### Schritt 2: Kronenmaske

```python
canopy = smoothed > np.percentile(smoothed, gap_percentile)
```

Die niedrigsten Bereiche sind Lücken, Wege, Boden — die werden ausgeschlossen.
**Hier greift der Schattenfehler durch:** hält die Höhenquelle den Schatten für
erhaben, überlebt er diesen Filter.

### Schritt 3: Wipfel finden

```python
seeds = h_maxima(surface, (high - low) * peak_prominence)
```

Nicht einfach lokale Maxima — **Prominenz**. Der Begriff kommt aus der Topografie:
die Prominenz eines Gipfels ist der Höhenunterschied zur tiefsten Scharte, die man
überqueren muss, um einen höheren Gipfel zu erreichen.

```
              /\  <- prominent: tiefe Scharten auf beiden Seiten
             /  \
        /\  /    \
       /  \/      \      <- die kleine Erhebung links ist
      /  ^         \        ein lokales Maximum, aber nicht prominent
     /   nicht      \
```

Warum das wichtig ist: Watershed ist eine **Partition, kein Detektor**. Es zerlegt
die Fläche in exakt so viele Teile, wie Marker hineingehen. Ein reines lokales
Maximum ist ein viel zu schwaches Kriterium — in flachen Bereichen erzeugt jedes
bisschen Rauschen beliebig viele davon und damit beliebig viele Pseudo-Kronen.
`h_maxima` verlangt eine Mindesterhebung über die Umgebung.

Die Schwelle ist **relativ** zur robusten Spannweite (5. bis 95. Perzentil)
gewählt. Damit hängt sie nicht von der willkürlichen Skala der Höhenquelle ab —
dieselbe Überlegung wie beim skaleninvarianten Verlust.

### Schritt 4: Watershed

Die Metapher: stell dir das umgedrehte CHM als Landschaft vor — Wipfel werden zu
Senken. Lass an jedem Marker Wasser einlaufen. Die Becken wachsen. Wo zwei Becken
sich treffen, wird ein Damm gebaut. Diese Dämme sind die Kronengrenzen.

```python
labels = watershed(-smoothed, markers, mask=canopy)
```

Das Ergebnis ist eine Labelkarte: jedes Pixel trägt die Nummer seiner Krone.

### Schritt 5: Formfilter

Kronen sind halbwegs rund und haben eine plausible Größe. Segmente, die zu klein,
zu groß oder zu langgestreckt sind, fliegen raus
([segment_sam.py:85](segment_sam.py#L85)).

### Der Hybrid

[segment_hybrid.py](segment_hybrid.py) kombiniert zwei Verfahren nach ihren
Stärken:

- **SAM** findet präzise Ränder dort, wo es Kantenkontrast gibt, und lässt
  Unsicheres weg. Hohe Präzision, Abdeckung nur 50–74 %.
- **Watershed** partitioniert vollständig, kachelt aber auch dort zu, wo kein Baum
  steht.

Der Hybrid lässt SAM die sicheren Kronen festlegen und das Watershed danach
**ausschließlich auf der Restfläche** laufen. Jede Krone trägt ihre Herkunft
(`quelle = sam | watershed`) in der Ausgabe, damit sich beide Anteile getrennt
bewerten lassen.

---

## 10. Woher die Modelle kommen: Trainingsdaten und Domänenlücke

Bisher ging es darum, *wie* die Verfahren arbeiten. Dieses Kapitel fragt, *worauf*
sie trainiert wurden — und das erklärt einen großen Teil der Fehlerbilder im
nächsten Kapitel.

In der Pipeline stecken fünf trainierte Modelle. Nur eines davon hat jemals einen
Frame aus diesem Projekt gesehen.

### Übersicht

| Komponente | Trainingsdaten | Region | Maßstab |
|---|---|---|---|
| Höhenkopf ([distill_height.py](distill_height.py)) | 29 eigene Frames, Ziel = Parallaxe | **eigene Flüge** | 6,25 cm/px |
| DINOvTree-B, Artklassifikation | Quebec Trees, 14 Klassen | Québec, Kanada | 1,9 cm/px |
| DeepForest, Detektion | NEON | USA | ~10 cm/px |
| SAM 1/2/3, Segmentierung | SA-1B, generische Alltagsbilder | weltweit, keine Luftbilder | beliebig |
| Depth Anything V2 / DepthPro | gemischte Tiefendatensätze | Bodenperspektiven | beliebig |
| CrownNet ([crownnet.py](crownnet.py)) | BAMFORESTS, 58 228 Kronen | **Deutschland** | 1,70 cm/px |

### Der Höhenkopf — das einzige Modell auf eigenen Daten

- **Ziel:** die gemessenen Parallaxenkarten aus `parallax_cache/`
- **Umfang:** 29 Frames aus 7 Ordnern — `pines` (5 Frames), `100`, `80m`, `dense`,
  `dense1`, `mixed`, `mixed1` (je 4)
- **Split:** `dense` und `mixed` als Validierung → **ca. 21 Frames Training,
  8 Validierung**
- **Trainierte Parameter:** nur die ~2 M des Faltungsdecoders, der Backbone bleibt
  eingefroren

21 Trainingsframes sind auch mit Crop- und Rotationsaugmentierung sehr wenig. Die
Crops eines Frames sind nicht unabhängig voneinander — die effektive Stichprobe
ist deutlich kleiner als die Zahl der gezogenen Ausschnitte suggeriert. Das ist
der Flaschenhals dieser Stufe, und mehr Videoordner würden hier mehr bringen als
jede Änderung an der Architektur.

### DINOvTree — Quebec Trees

Der Checkpoint `dinovtreeb_quebectrees.pth` ist ein DINOv3 ViT-B/16, feinjustiert
auf Daten aus **Québec, Kanada**. Die Kategoriendatei
[quebec_trees_categories.json](third_party/quebec_trees_categories.json) enthält
17 Einträge; nach Ausschluss der drei Supercategories, die im Paper als
Annotator-Unsicherheit gewertet werden (`Pinopsida`, `Magnoliopsida`, `Acer L.`,
siehe `QUEBEC_TREES_EXCLUDE`), bleiben **14 Klassen**:

| Gruppe | Klassen |
|---|---|
| Nadelbäume | *Thuja occidentalis*, *Abies balsamea*, *Larix laricina*, *Tsuga canadensis*, *Pinus strobus*, *Picea* |
| Laubbäume | *Fagus grandifolia*, *Populus*, *Acer pensylvanicum*, *A. saccharum*, *A. rubrum*, *Betula alleghaniensis*, *B. papyrifera* |
| sonstige | `dead` |

Das ist ein borealer bis nordöstlich-nordamerikanischer Artensatz. Für
mitteleuropäischen Wald fehlen unter anderem Fichte (*Picea abies*), Waldkiefer,
Eiche, Esche, Linde und Douglasie. *Fagus grandifolia* ist die amerikanische
Buche, nicht *Fagus sylvatica*.

Ein deutscher Bestand lässt sich mit diesem Kopf also bestenfalls auf die nächste
kanadische Verwandte abbilden. Genau deshalb existiert
[cluster_crowns.py](cluster_crowns.py): es gruppiert Kronen nach ihren
Merkmalsvektoren, statt sie in einen Klassensatz zu zwingen, der die vorhandenen
Arten gar nicht enthält.

Dazu kommt der Maßstabsversatz aus Kapitel 2: Training bei 1,9 cm/px, unsere
Frames bei 6,25 cm/px — Faktor 3,3.

### DeepForest — NEON

`weecology/deepforest-tree`, vortrainiert auf Daten des **National Ecological
Observatory Network** (USA) bei etwa 10 cm/px. Ein reiner Detektor: er liefert
Kronen-Boxen ohne Artangabe. Der Maßstab liegt näher an unseren 6,25 cm/px als
beim Klassifikator, die Vegetation ist aber wieder nordamerikanisch. Die Frage,
wie stark die Detektion vom Maßstab abhängt, untersucht
[detect_scale_test.py](detect_scale_test.py).

### SAM — SA-1B, und kein einziger Wald

[segment_sam.py](segment_sam.py) verwendet `facebook/sam-vit-large`,
[ablate_sam.py](ablate_sam.py) vergleicht sechs Varianten von SAM 1 bis SAM 2.1,
[segment_sam3.py](segment_sam3.py) nutzt das textgepromptete `facebook/sam3`.

Alle wurden auf **SA-1B** trainiert — rund 11 Millionen gewöhnliche Fotos mit
1,1 Milliarden automatisch erzeugten Masken. Luftbilder von Wald sind darin
allenfalls zufällig enthalten.

Das erklärt das Verhalten präzise: **SAM kennt keine Baumkronen.** Es findet
Regionen mit geschlossenem, kontrastreichem Rand. Wo eine Krone einen solchen Rand
hat, funktioniert es hervorragend. Wo der stärkste geschlossene Rand um
*Krone plus Schlagschatten* verläuft, nimmt SAM diesen — nicht aus einem Fehler
heraus, sondern weil es genau das tut, wofür es gebaut wurde. Siehe Kapitel 4.

### Depth Anything / DepthPro — Bodenperspektiven

Beide wurden auf gemischten Tiefendatensätzen trainiert, deren gemeinsamer Nenner
die menschliche Aufnahmeperspektive ist: Straßenszenen, Innenräume, Landschaften.
Depth Anything V2 nutzt 1,5 M gelabelte plus 62 M ungelabelte Bilder — kaum eines
davon ein Senkrechtbild aus 80 m. Kapitel 4 behandelt die Folgen.

### CrownNet — BAMFORESTS, der einzige geografische Treffer

[crownnet.py](crownnet.py) ist die Ausnahme in dieser Aufstellung. Es wird auf
**BAMFORESTS** trainiert: 58 228 annotierte Kronen aus deutschem Wald, nativ bei
1,70 cm/px, im Code per `--scale` auf den Maßstab der eigenen Frames gebracht
([crownnet.py:281](crownnet.py#L281)).

Das ist der einzige Datensatz der Pipeline, der geografisch und in der
Artenzusammensetzung zu den eigenen Aufnahmen passt — und die einzige Stelle mit
einer harten, gegen echte Labels gemessenen Instanzgenauigkeit
([crownnet.py:412](crownnet.py#L412)). Für die Bewertung der gesamten
Segmentierung ist das der wertvollste Bezugspunkt, den das Projekt besitzt.

### Das Muster

```
Deine Drohnenframes   ──> Höhenkopf         (21 Frames)   ← einzige eigene Daten
Québec, Kanada        ──> Artklassifikation
NEON, USA             ──> Detektion
SA-1B, Alltagsbilder  ──> Segmentierung
Bodenperspektiven     ──> Tiefe
Deutscher Wald        ──> CrownNet          (Stufe 2, noch nicht in der Hauptpipeline)
```

Jede Stufe außer dem Höhenkopf arbeitet auf einer Domäne, für die sie nicht
trainiert wurde — anderer Kontinent, anderer Maßstab, andere Perspektive, andere
Artenzusammensetzung.

Die Domänenlücke ist damit nicht *ein* Problem unter mehreren, sondern das
durchgehende Muster. Und sie erklärt, warum der Aufwand um die Parallaxe sich
lohnen kann: sie ist der einzige Baustein der Pipeline, der **überhaupt keine
Trainingsdomäne hat**. SIFT, RANSAC und optischer Fluss funktionieren in Québec
genauso wie in Brandenburg, bei 1,9 cm/px genauso wie bei 6,25 cm/px, bei Nadir
genauso wie schräg. Geometrie kennt keine Domänenlücke.

Das ist der eigentliche Grund, warum sie als Lehrer taugt — und warum ein aus ihr
destillierter Kopf ein Modell in *dieser* Domäne ergibt statt eines geliehenen aus
einer fremden.

---

## 11. Fehlerbilder und wo die Pipeline steht

### Klassisch vs. gelernt — der aktuelle Stand

Von 21 Skripten benutzen 11 direkt `torch`. Netzfrei sind **genau die
Parallaxe-Skripte** ([stereo_probe.py](stereo_probe.py),
[build_parallax.py](build_parallax.py)) sowie die reinen Visualisierungs- und
Auswertungswerkzeuge.

Die Architektur des Projekts lässt sich so zusammenfassen:

> **Klassische Geometrie erzeugt Wahrheit, neuronale Netze machen sie skalierbar
> und semantisch.**

```
SIFT / RANSAC / optischer Fluss  ──> Höhe (gemessen, braucht Video)   [kein Netz]
                                          │  Destillation
                                          ▼
DINOv3 + Faltungskopf            ──> Höhe (geschätzt, ein Bild)       [Netz]
                                          │
SAM / CrownNet / DeepForest      ──> Kroneninstanzen                  [Netz]
                                          │
DINOvTree                        ──> Baumart                          [Netz]
```

Nur die oberste Zeile ist netzfrei — aber sie versorgt alles darunter mit
Trainingssignal. Ihre Qualität ist die Obergrenze für den Rest.

### Bekannte Fehlerbilder

**Schatten werden als Kronenteil erfasst.** Ausführlich in Kapitel 4. Betrifft
nachweislich `urban` (Tiefe-Quelle). Prognose: mit Parallaxe verschwindet der
Fehler, weil ein Schatten null Parallaxe hat. **Noch nicht verifiziert.**

**Falschtrennungen im geschlossenen Kronendach.** Zwei Farben auf einer optisch
durchgehenden Krone. Diagnostizierbar über die `trennungen`-Ansicht in
[visualize_crowns.py](visualize_crowns.py), die jede Grenze nach der Tiefe des
Sattels einfärbt: rot = flacher Sattel, die beiden gehören vermutlich zusammen.

**Kein Vegetationsfilter.** Eine Suche über
[segment_hybrid.py](segment_hybrid.py), [segment_sam.py](segment_sam.py),
[segment_sam3.py](segment_sam3.py) und [refine_crowns.py](refine_crowns.py) findet
nichts zu Grünanteil, HSV oder Excess-Green. Gefiltert wird nur über Form und
Höhe. Ein Helligkeits-/Grünfilter wäre ein billiges Pflaster gegen das
Schattenproblem — aber nur ein Pflaster.

**Maßstabsabhängigkeit.** Der Klassifikations-Checkpoint sah 1,9 cm/px, unsere
Frames haben 6,25 cm/px. Siehe [scale_sweep.py](scale_sweep.py).

### Was als Nächstes zu klären ist

1. **Parallaxe gegen Tiefe an derselben Segmentierung vergleichen.** Es gibt 7
   Ordner mit Parallaxendaten. `visualize_crowns.py --surface parallax` gegen
   `--surface depth` auf `pines` und `dense` wäre die direkte Gegenüberstellung.
   Das Schattenargument aus Kapitel 6 ist damit prüfbar statt nur plausibel.

2. **Die Qualität des Lehrers quantifizieren.** Wie oft schlägt die
   Homographie fehl, wie viele Partnerpaare überleben den
   `--min-displacement`-Filter, wie hoch ist der Restfluss-Median? Diese Werte
   werden in [stereo_probe.py:77](stereo_probe.py#L77) bereits berechnet, aber
   nicht aggregiert ausgewertet.

3. **Prüfen, worauf RANSAC die Ebene legt.** In dichten Beständen könnte es die
   Kronenebene statt des Bodens sein. Inlier-Verteilung visualisieren.

4. **Datenbasis erweitern.** 29 Frames für die Destillation sind sehr wenig. Mehr
   Videoordner würden hier am meisten bringen — mehr als jede Architektur-
   verbesserung.

5. **Gegen Tolan et al. als Baseline vergleichen.** Deren Gewichte sind offen
   verfügbar und laufen auf unseren Frames. Das gäbe eine ehrliche Referenz statt
   nur Depth Anything.

---

## 12. Glossar

| Begriff | Bedeutung |
|---|---|
| **Basislinie** | Abstand zwischen zwei Kamerapositionen. Größer = stärkere Parallaxe. |
| **Backbone** | Der große, vortrainierte Teil eines Netzes, der allgemeine Bildmerkmale liefert. |
| **CHM** | Canopy Height Model — Rasterkarte der Vegetationshöhe über Boden. |
| **Destillation** | Wissen aus einem Lehrer (Modell oder Verfahren) in ein anderes Modell überführen. |
| **Disparität** | Positionsunterschied desselben Punkts zwischen zwei Ansichten, in Pixeln. |
| **DSM / DTM** | Digital Surface / Terrain Model — mit bzw. ohne Vegetation. Differenz = CHM. |
| **Detrend** | Großräumigen Trend abziehen, um lokale Struktur freizulegen. |
| **GSD** | Ground Sample Distance — wie viele Meter ein Pixel am Boden abdeckt. |
| **Homographie** | 3×3-Matrix, die die Abbildung einer *Ebene* zwischen zwei Ansichten beschreibt. |
| **Inlier** | Datenpunkt, der zu einem Modell passt. Gegenteil: Ausreißer. |
| **LiDAR** | Laserscanner, misst Entfernungen direkt. Goldstandard für CHM. |
| **monokular** | Aus einem einzigen Bild. Gegenteil: stereo / multi-view. |
| **Nadir** | Blickrichtung senkrecht nach unten. |
| **Optischer Fluss** | Vektorfeld: wohin ist jeder Pixel zwischen zwei Bildern gewandert? |
| **Parallaxe** | Scheinbare Verschiebung von Objekten bei Ortswechsel des Betrachters. |
| **Patch-Token** | Merkmalsvektor eines Vision Transformers für eine 16×16-Bildkachel. |
| **Prominenz** | Wie weit sich ein Gipfel über die umgebenden Scharten erhebt. |
| **RANSAC** | Robuste Modellschätzung durch zufälliges Ausprobieren und Inlier-Zählen. |
| **Rektifizierung** | Zwei Bilder so entzerren, dass Korrespondenzen auf gleicher Zeile liegen. |
| **SIFT** | Verfahren zum Finden und Beschreiben markanter Bildpunkte. |
| **skaleninvariant** | Unempfindlich gegen Multiplikation mit einem Faktor. |
| **Vision Transformer (ViT)** | Netzarchitektur, die ein Bild als Folge von Kacheln verarbeitet. |
| **Watershed** | Segmentierung nach dem Wasserscheidenprinzip, ausgehend von Markern. |

---

## 13. Literatur

### Zur Geometrie (Kapitel 5–7)

- **Plane + Parallax, ursprünglich:** Irani, Anandan, Kumar u. a., 1990er.
  Formalisierung bei Triggs, *Plane + Parallax, Tensors and Factorization*,
  ECCV 2000 — https://lear.inrialpes.fr/people/triggs/pubs/Triggs-eccv00.pdf
- **Moderne Anwendung, Straßenebene:** *Monocular Road Planar Parallax Estimation*
  — https://arxiv.org/html/2111.11089
  (Homographie auf die Straße, Restfluss = Höhe darüber. Identischer Aufbau, nur
  ist unsere Ebene der Waldboden.)
- **Metrisch genaue Variante:** *DepthP+P* — https://arxiv.org/pdf/2301.02092
- **Satellitenbilder:** *Parallax estimation for push-frame satellite imagery* —
  https://arxiv.org/pdf/2102.02301
  (Enthält explizit die Proportionalität zu Höhe *und* Basislinie — die
  Begründung für unsere Basislinien-Normierung.)
- **SIFT:** Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*,
  IJCV 2004.
- **RANSAC:** Fischler & Bolles, *Random Sample Consensus*, CACM 1981.

### Zur Destillation (Kapitel 8)

- **MegaDepth:** Li & Snavely, *Learning Single-View Depth Prediction from
  Internet Photos*, CVPR 2018 — https://arxiv.org/abs/1804.00607
  *Der strukturell nächste Verwandte:* SfM/MVS als Lehrer, Einzelbildnetz als
  Schüler, skaleninvarianter Verlust wegen unbekannter Skala.
- **MiDaS:** Ranftl et al., *Towards Robust Monocular Depth Estimation*, TPAMI 2022
  — https://arxiv.org/pdf/2307.14460
  (Skalen- und verschiebungsinvarianter Verlust als Standard.)
- **Depth Anything:** Yang et al., CVPR 2024 — https://arxiv.org/pdf/2401.10891
  (Das Modell aus `depth_probe.py` — selbst per Pseudolabel-Destillation gebaut.)
- **Alternative, nicht gewählt:** Zhou et al., *Unsupervised Learning of Depth and
  Ego-Motion from Video*, CVPR 2017 — https://arxiv.org/abs/1704.07813
  (End-to-End aus Video, ohne expliziten Parallaxe-Zwischenschritt. Nachteil für
  uns: keine inspizierbare Zwischenstufe.)
- **Skaleninvarianter Verlust, ursprünglich:** Eigen, Puhrsch, Fergus,
  *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network*,
  NIPS 2014.

### Zur Forstanwendung (Kapitel 1, 9, 10)

- **Tolan et al., 2024:** *Very high resolution canopy height maps from RGB
  imagery using self-supervised vision transformer and convolutional decoder
  trained on aerial lidar*, Remote Sensing of Environment 300
  — https://arxiv.org/abs/2304.07213
  **Der wichtigste Vergleichspunkt.** Fast identische Architektur zu
  `distill_height.py` (DINOv2 eingefroren + Faltungsdecoder), aber LiDAR als
  Lehrer. 2,8 m MAE. Code und Gewichte offen:
  https://github.com/facebookresearch/HighResCanopyHeight
- **UAV-Photogrammetrie als Lehrer:** *Ultrahigh-resolution boreal forest canopy
  mapping*, RSE 2022 —
  https://www.sciencedirect.com/science/article/pii/S0303243422000125
- **Kronenabgrenzung, Grenzen des Verfahrens:** *Individual tree crown delineation
  from high-resolution UAV images in broadleaf forest* —
  https://www.sciencedirect.com/science/article/abs/pii/S1574954120301576
  (Wichtig für die Erwartungshaltung: funktioniert in Nadelbeständen gut, in
  Laub- und Mischbeständen deutlich schlechter. Unsere `mixed`-Ordner werden der
  harte Teil sein, unabhängig von der Höhenquelle.)
- **Benchmark:** *Open-Canopy* — https://arxiv.org/pdf/2407.09392

---

*Stand: 20. August 2026*
