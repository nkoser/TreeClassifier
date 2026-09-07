"""Virtuelle Nadirframes aus den vorbereiteten FORTRESS-Rastern schneiden.

Ein Sample entsteht aus drei Wuerfen: Flughoehe H, Bildwinkel und Position im
Gebiet. Daraus folgt die ganze Geometrie.

    k        = 0.5 / tan(HFOV / 2)      Brennweite geteilt durch Bildbreite
    f_px     = k * Bildbreite_px
    GSD      = H / f_px                 Meter pro Pixel am Boden
    Breite_m = Bildbreite_px * GSD = H / k

`k` ist die eigentliche Groesse, nicht `f_px`: Depth Pro rechnet intern in
**kanonischer inverser Tiefe**, und die haengt nur ueber f/Bildbreite von der
Kamera ab, nicht von der Aufloesung ([image_processing_depth_pro.py:108]).
Deshalb ist es unerheblich, ob ein Ausschnitt mit 768 oder 1920 Pixeln abgelegt
wird -- entscheidend ist der Bildwinkel.

Der Bildwinkel wird breit gewuerfelt (Vorgabe 35 bis 85 Grad) und nicht auf die
73.7 Grad unserer Kamera festgenagelt. Zwei Gruende: bei festem Winkel legt die
Flughoehe die Bodenbreite fest, und aus 80 m sind das 120 m -- fast das ganze
1.7-ha-Gebiet, also genau ein Ausschnitt pro Gebiet. Und breit gestreute Winkel
sind der Weg, auf dem das Modell die kanonische Beziehung lernt statt einer
auswendig gelernten Konstanten.

**Warum die Ausschnitte nicht quadratisch sind.** Depth Pro quetscht jedes Bild
auf 1536 x 1536, ohne Ruecksicht auf das Seitenverhaeltnis. Unsere Frames sind
1920 x 1080, werden in dieser Kette also um Faktor 1.78 in der Hoehe gestaucht.
Wer auf Quadraten trainiert und auf gestauchten Bildern anwendet, hat sich den
Fehler selbst gebaut. Die Ausschnitte kommen deshalb im Seitenverhaeltnis der
Zielframes und laufen anschliessend durch dieselbe Stauchung.

Dazu Augmentierung gegen den Domaenenunterschied: das Ortho ist ein aus vielen
Aufnahmen gerechnetes, gestochen scharfes Produkt, unsere Frames sind einzelne
Videobilder aus 80 m. Weichzeichnung, Rauschen und JPEG-Artefakte schliessen den
Abstand ein Stueck weit.

Die Wahrheit ist bei 5 cm aufgeloest (so liegt das nDSM vor). Unterhalb von etwa
25 m Flughoehe waere die Tiefenkarte glatter als das Bild -- daher die
Untergrenze.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

MODELL_MITTEL = 0.5   # aus preprocessor_config.json von apple/DepthPro-hf
MODELL_STREUUNG = 0.5


def k_von_fov(fov_grad: float) -> float:
    """Brennweite geteilt durch Bildbreite -- die aufloesungsfreie Kamerakonstante."""
    return 0.5 / math.tan(math.radians(fov_grad) / 2.0)


def fov_von_k(k: float) -> float:
    return 2.0 * math.degrees(math.atan(0.5 / k))


class Rasterlager:
    """Haelt die zuletzt gebrauchten Gebietsraster im Speicher.

    Ein Gebiet sind bei 2 cm rund 6500 x 6500 Pixel, also etwa 170 MB als RGB
    plus Hoehe plus Maske. Alle 47 gleichzeitig waeren 8 GB pro Arbeitsprozess.
    Da die Sampleliste nach Gebieten gruppiert ist, reichen wenige.
    """

    def __init__(self, wurzel: Path, groesse: int = 2) -> None:
        self.wurzel, self.groesse = Path(wurzel), groesse
        self.cache: collections.OrderedDict[str, tuple] = collections.OrderedDict()

    def __call__(self, site: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if site in self.cache:
            self.cache.move_to_end(site)
            return self.cache[site]
        rgb = cv2.cvtColor(cv2.imread(str(self.wurzel / f"{site}_rgb.jpg")), cv2.COLOR_BGR2RGB)
        ndsm = cv2.imread(str(self.wurzel / f"{site}_ndsm.png"), cv2.IMREAD_UNCHANGED)
        gueltig = cv2.imread(str(self.wurzel / f"{site}_valid.png"), cv2.IMREAD_GRAYSCALE) > 0
        eintrag = (rgb, ndsm, gueltig)
        self.cache[site] = eintrag
        while len(self.cache) > self.groesse:
            self.cache.popitem(last=False)
        return eintrag


def farbjitter(bild: np.ndarray, rng: np.random.Generator, staerke: float) -> np.ndarray:
    """Helligkeit, Kontrast, Farbstich und Gamma -- Wetter und Weissabgleich."""
    if staerke <= 0:
        return bild
    x = bild.astype(np.float32) / 255.0
    x *= rng.uniform(1 - 0.30 * staerke, 1 + 0.30 * staerke)                  # Helligkeit
    mittel = x.mean()
    x = mittel + (x - mittel) * rng.uniform(1 - 0.30 * staerke, 1 + 0.30 * staerke)  # Kontrast
    x *= rng.uniform(1 - 0.10 * staerke, 1 + 0.10 * staerke, size=(1, 1, 3))  # Farbstich
    x = np.clip(x, 0, 1) ** rng.uniform(1 - 0.25 * staerke, 1 + 0.25 * staerke)
    return (np.clip(x, 0, 1) * 255.0).astype(np.uint8)


def videolook(bild: np.ndarray, rng: np.random.Generator, staerke: float) -> np.ndarray:
    """Weichzeichnung, Sensorrauschen und JPEG-Artefakte.

    Das Orthomosaik ist aus vielen Aufnahmen gerechnet und dadurch schaerfer als
    jedes Einzelbild. Ohne diesen Schritt lernt das Modell auf einer Schaerfe,
    die es im Einsatz nie zu sehen bekommt.
    """
    if staerke <= 0:
        return bild
    if rng.random() < 0.7 * staerke:
        bild = cv2.GaussianBlur(bild, (0, 0), rng.uniform(0.4, 1.6 * staerke))
    if rng.random() < 0.5 * staerke:
        bild = np.clip(bild.astype(np.float32) + rng.normal(0, rng.uniform(1, 6), bild.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.6 * staerke:
        qualitaet = int(rng.integers(45, 92))
        _, puffer = cv2.imencode(".jpg", bild, [cv2.IMWRITE_JPEG_QUALITY, qualitaet])
        bild = cv2.imdecode(puffer, cv2.IMREAD_COLOR)
    return bild


class NadirFrames(torch.utils.data.Dataset):
    """Virtuelle Frames mit metrischer Tiefenwahrheit.

    Die Sampleliste steht vor der Epoche fest und ist nach Gebieten gruppiert --
    so laeuft der Datenlader durch wenige Raster statt bei jedem Zugriff ein
    neues von der Platte zu holen. Innerhalb eines Gebietes wird gemischt.
    """

    def __init__(self, wurzel: Path, split: str, *, crop_px: int = 1536,
                 seitenverhaeltnis: float = 16 / 9,
                 hoehe_min: float = 25.0, hoehe_max: float = 120.0, abstand_min: float = 20.0,
                 fov_min: float = 35.0, fov_max: float = 85.0,
                 pro_gebiet: int = 400, min_gueltig: float = 0.50,
                 augment: bool = True, spiegeln: bool | None = None,
                 domaene: bool | None = None, jitter: float = 1.0, video: float = 1.0,
                 strahl_tiefe: bool = False, cache: int = 2, seed: int = 0,
                 site_block: int = 32) -> None:
        self.wurzel = Path(wurzel)
        index = json.loads((self.wurzel / "index.json").read_text())
        self.sites = sorted(s for s, m in index.items() if m["split"] == split)
        if not self.sites:
            raise SystemExit(f"Keine Gebiete im Split {split!r} unter {self.wurzel}")
        self.meta = index
        self.crop_px, self.pro_gebiet = crop_px, pro_gebiet
        self.seitenverhaeltnis = seitenverhaeltnis
        self.crop_hoch = max(16, int(round(crop_px / seitenverhaeltnis)))
        self.hoehe_min, self.hoehe_max, self.abstand_min = hoehe_min, hoehe_max, abstand_min
        self.fov_min, self.fov_max = fov_min, fov_max
        self.min_gueltig, self.augment = min_gueltig, augment
        # Zwei verschiedene Dinge, die `augment` sonst zusammenwirft. Spiegeln ist
        # Formaugmentierung, Weichzeichnen und JPEG sind Domaenenangleichung.
        # Zum Messen will man oft nur das zweite: feste Ausschnitte, aber in der
        # Schaerfe, die spaeter tatsaechlich anliegt. Ohne Angabe verhaelt sich
        # beides wie `augment` -- das bleibt das alte Verhalten.
        self.spiegeln = augment if spiegeln is None else spiegeln
        self.domaene = augment if domaene is None else domaene
        self.jitter, self.video = jitter, video
        self.strahl_tiefe, self.seed = strahl_tiefe, seed
        self.site_block = max(1, int(site_block))
        self.site_mix = max(1, int(cache))
        self.lager = Rasterlager(self.wurzel / "raster", cache)
        self.set_epoch(0)

    def set_epoch(self, epoche: int) -> None:
        """Neue Wuerfe fuer die Epoche, Gebiete in neuer Reihenfolge."""
        rng = np.random.default_rng(self.seed + 1000 * epoche)
        sites = list(self.sites)
        rng.shuffle(sites)
        pro_site = {
            site: [int(rng.integers(0, 2**31)) for _ in range(self.pro_gebiet)]
            for site in sites
        }
        if not self.augment:
            # Validierung und Test bleiben gebietsweise geordnet: keine Updates,
            # also auch kein Risiko durch lange homogene Folgen, dafuer guter Cache.
            self.plan = [(site, saat) for site in sites for saat in pro_site[site]]
            return

        # Je Cachegruppe kurze Bloecke mehrerer Gebiete abwechseln. Damit sieht
        # der Optimizer nicht ein ganzes Gebiet am Stueck, und jeder Worker kann
        # trotzdem genau die beteiligten Raster im Cache behalten.
        gruppen = [sites[i : i + self.site_mix] for i in range(0, len(sites), self.site_mix)]
        if len(gruppen) > 1 and len(gruppen[-1]) == 1:
            gruppen[-2].append(gruppen[-1].pop())
            gruppen.pop()
        rng.shuffle(gruppen)
        self.plan = []
        for gruppe in gruppen:
            letztes_gebiet = None
            for i in range(0, self.pro_gebiet, self.site_block):
                reihenfolge = list(gruppe)
                rng.shuffle(reihenfolge)
                if len(reihenfolge) > 1 and reihenfolge[0] == letztes_gebiet:
                    reihenfolge = reihenfolge[1:] + reihenfolge[:1]
                for site in reihenfolge:
                    self.plan.extend((site, saat)
                                     for saat in pro_site[site][i : i + self.site_block])
                letztes_gebiet = reihenfolge[-1]

    def __len__(self) -> int:
        return len(self.plan)

    def geometrie(self, rng: np.random.Generator, breite_m: float, hoehe_m: float,
                  wipfel_m: float) -> tuple[float, float]:
        """Flughoehe und Kamerakonstante, passend zu Gebietsgroesse und Bestand.

        Die Kamera muss mit Abstand ueber den hoechsten Wipfeln haengen. Ohne
        diese Schranke entstuenden Ausschnitte, in denen die Baeume fast bis zur
        Linse reichen -- ein Aufnahmefall, den es bei uns nicht gibt und der die
        Tiefenverteilung verzerrt.
        """
        untergrenze = max(self.hoehe_min, wipfel_m + self.abstand_min)
        obergrenze = max(untergrenze * 1.01, self.hoehe_max)
        # Die Bodenbreite H/k muss ins Gebiet passen -- in der Breite direkt, in
        # der Hoehe um das Seitenverhaeltnis entlastet. Unpassende Paare werden
        # neu gezogen. Frueher wurde stattdessen H nachtraeglich verkleinert;
        # dadurch konnte die Kamera unter den zugesicherten Wipfelabstand sinken.
        passt = 0.98 * min(breite_m, hoehe_m * self.seitenverhaeltnis)
        for _ in range(32):
            H = float(np.exp(rng.uniform(math.log(untergrenze), math.log(obergrenze))))
            k = k_von_fov(float(rng.uniform(self.fov_min, self.fov_max)))
            if H / k <= passt:
                return H, k

        # Deterministischer, weiterhin physikalisch gueltiger Notfall. Bei einem
        # zu kleinen Gebiet wird der Bildwinkel enger, nicht die Kamera tiefer.
        H = untergrenze
        k = max(k_von_fov(self.fov_max), H / max(passt, 1e-6))
        return H, k

    def __getitem__(self, i: int):
        site, saat = self.plan[i]
        rng = np.random.default_rng(saat)
        rgb, ndsm_cm, gueltig = self.lager(site)
        meta = self.meta[site]
        g0 = meta["base_gsd_m"]
        H_px, B_px = gueltig.shape

        fenster = None
        for _ in range(24):
            H_flug, k = self.geometrie(rng, meta["breite_m"], meta["hoehe_m"],
                                       meta.get("hoehe_max_m", 40.0))
            nb = int(round((H_flug / k) / g0))                              # Fensterbreite im Bodenraster
            nh = int(round(nb / self.seitenverhaeltnis))
            if nb < 32 or nh < 32 or nb > B_px or nh > H_px:
                continue
            x0 = int(rng.integers(0, B_px - nb + 1))
            y0 = int(rng.integers(0, H_px - nh + 1))
            teil = gueltig[y0 : y0 + nh, x0 : x0 + nb]
            # Grob pruefen reicht und ist bei 6000er Fenstern hundertfach schneller.
            if teil[::8, ::8].mean() >= self.min_gueltig:
                fenster = (x0, y0, nb, nh, H_flug, k)
                break
        if fenster is None:
            # Notfall: das groesste passende Fenster mittig, in der Flughoehe dazu.
            nb = min(B_px, int(H_px * self.seitenverhaeltnis))
            nh = int(round(nb / self.seitenverhaeltnis))
            x0, y0 = (B_px - nb) // 2, (H_px - nh) // 2
            bodenbreite = nb * g0
            untergrenze = max(self.hoehe_min,
                              meta.get("hoehe_max_m", 40.0) + self.abstand_min)
            k = max(k_von_fov(0.5 * (self.fov_min + self.fov_max)),
                    untergrenze / max(bodenbreite, 1e-6))
            fenster = (x0, y0, nb, nh, bodenbreite * k, k)
        x0, y0, nb, nh, H_flug, k = fenster

        sb, sh = self.crop_px, self.crop_hoch
        interp = cv2.INTER_AREA if nb >= sb else cv2.INTER_LINEAR
        bild = cv2.resize(np.ascontiguousarray(rgb[y0 : y0 + nh, x0 : x0 + nb]),
                          (sb, sh), interpolation=interp)

        # Ungueltige nDSM-Pixel liegen auf Platte als 0. Werden Hoehe und Maske
        # unabhaengig skaliert, mischt INTER_AREA diese Nullen an Lochraendern in
        # gueltige Zielpixel. Normalisiertes, maskiertes Resampling verhindert das.
        teil_maske = gueltig[y0 : y0 + nh, x0 : x0 + nb].astype(np.float32)
        teil_hoehe = ndsm_cm[y0 : y0 + nh, x0 : x0 + nb].astype(np.float32) / 100.0
        gewicht = cv2.resize(teil_maske, (sb, sh), interpolation=interp)
        summe = cv2.resize(teil_hoehe * teil_maske, (sb, sh), interpolation=interp)
        hoehe = summe / np.maximum(gewicht, 1e-6)
        # Nur Pixel behalten, deren Resampling-Fussabdruck fast vollstaendig
        # gueltig war. So gelangen keine Fuellkanten in Loss oder Kennzahlen.
        maske = gewicht >= 0.99

        if self.spiegeln:
            # Nur Spiegelungen, keine Vierteldrehung: die wuerde das
            # Seitenverhaeltnis kippen, auf das die ganze Geometrie aufbaut.
            if rng.random() < 0.5:
                bild, hoehe, maske = (a[:, ::-1] for a in (bild, hoehe, maske))
            if rng.random() < 0.5:
                bild, hoehe, maske = (a[::-1] for a in (bild, hoehe, maske))
        bild = np.ascontiguousarray(bild)
        hoehe, maske = np.ascontiguousarray(hoehe), np.ascontiguousarray(maske)
        if self.domaene:
            bild = videolook(farbjitter(bild, rng, self.jitter), rng, self.video)

        tiefe = (H_flug - hoehe).astype(np.float32)
        if self.strahl_tiefe:
            # Abstand entlang des Sehstrahls statt entlang der optischen Achse.
            f_px = k * sb
            gy, gx = np.mgrid[0:sh, 0:sb].astype(np.float32)
            r2 = ((gx - (sb - 1) / 2) ** 2 + (gy - (sh - 1) / 2) ** 2) / (f_px * f_px)
            tiefe *= np.sqrt(1.0 + r2)

        maske &= tiefe > 1.0    # Kamera im Baum waere kein sinnvolles Ziel.
        return {
            "bild": torch.from_numpy(np.ascontiguousarray(bild.transpose(2, 0, 1))),
            "tiefe": torch.from_numpy(tiefe),
            "hoehe": torch.from_numpy(np.ascontiguousarray(hoehe)),
            "maske": torch.from_numpy(maske),
            "k": torch.tensor(k, dtype=torch.float32),
            "flughoehe": torch.tensor(H_flug, dtype=torch.float32),
            "gsd": torch.tensor(H_flug / (k * sb), dtype=torch.float32),
            "site": site,
        }


def auf_modell(bild_uint8: torch.Tensor, modellgroesse: int = 1536) -> torch.Tensor:
    """uint8-Batch auf die Eingangsgroesse von Depth Pro bringen und normieren.

    Genau die Stauchung auf 1536 x 1536, die auch der Bildprozessor von Depth Pro
    vornimmt -- das Seitenverhaeltnis geht dabei verloren, und zwar mit Absicht:
    bei der Anwendung passiert dasselbe.

    Bewusst auf der GPU und nicht im Datenlader: als float32 waere ein einziges
    1536er Bild 28 MB, die sonst durch die Prozessgrenze muessten.
    """
    x = bild_uint8.float().div_(255.0)
    if x.shape[-1] != modellgroesse or x.shape[-2] != modellgroesse:
        x = torch.nn.functional.interpolate(x, size=(modellgroesse, modellgroesse),
                                            mode="bilinear", align_corners=False, antialias=True)
    return (x - MODELL_MITTEL) / MODELL_STREUUNG
