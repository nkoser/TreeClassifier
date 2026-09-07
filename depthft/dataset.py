"""Cut virtual nadir frames out of the prepared FORTRESS rasters.

A sample comes from three draws: flight altitude H, field of view and position
in the site. The whole geometry follows from those.

    k        = 0.5 / tan(HFOV / 2)      focal length divided by image width
    f_px     = k * Bildbreite_px
    GSD      = H / f_px                 metres per pixel on the ground
    width_m  = image_width_px * GSD = H / k

`k` is the quantity that matters, not `f_px`: Depth Pro works internally in
**canonical inverse depth**, which depends on the camera only through
f/image width, not on the resolution ([image_processing_depth_pro.py:108]). So
it does not matter whether a crop is stored at 768 or 1920 pixels -- what counts
is the field of view.

The field of view is drawn from a wide range (default 35 to 85 degrees) rather
than pinned to the 73.7 degrees of our camera. Two reasons: with a fixed angle
the flight altitude fixes the ground width, and from 80 m that is 120 m -- almost
the whole 1.7 ha site, i.e. exactly one crop per site. And widely spread angles
are how the model learns the canonical relation instead of a memorised constant.

**Why the crops are not square.** Depth Pro squeezes every image to 1536 x 1536,
regardless of aspect ratio. Our frames are 1920 x 1080 and are therefore
compressed by a factor of 1.78 in height along this chain. Whoever trains on
squares and applies to squeezed images has built the error themselves. The crops
therefore come in the aspect ratio of the target frames and then go through the
same squeeze.

On top of that, augmentation against the domain difference: the ortho is a
razor-sharp product computed from many captures, while our frames are single
video images from 80 m. Blur, noise and JPEG artefacts close some of that gap.

The truth is resolved at 5 cm (that is how the nDSM comes). Below about 25 m
flight altitude the depth map would be smoother than the image -- hence the lower
bound.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

MODELL_MITTEL = 0.5   # from preprocessor_config.json of apple/DepthPro-hf
MODELL_STREUUNG = 0.5


def k_von_fov(fov_grad: float) -> float:
    """Focal length divided by image width -- the resolution-free camera constant."""
    return 0.5 / math.tan(math.radians(fov_grad) / 2.0)


def fov_von_k(k: float) -> float:
    return 2.0 * math.degrees(math.atan(0.5 / k))


class Rasterlager:
    """Keeps the most recently used site rasters in memory.

    At 2 cm a site is around 6500 x 6500 pixels, i.e. about 170 MB as RGB plus
    height plus mask. All 47 at once would be 8 GB per worker process. Since the
    sample list is grouped by site, a few suffice.
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
    """Brightness, contrast, colour cast and gamma -- weather and white balance."""
    if staerke <= 0:
        return bild
    x = bild.astype(np.float32) / 255.0
    x *= rng.uniform(1 - 0.30 * staerke, 1 + 0.30 * staerke)                  # brightness
    mittel = x.mean()
    x = mittel + (x - mittel) * rng.uniform(1 - 0.30 * staerke, 1 + 0.30 * staerke)  # contrast
    x *= rng.uniform(1 - 0.10 * staerke, 1 + 0.10 * staerke, size=(1, 1, 3))  # colour cast
    x = np.clip(x, 0, 1) ** rng.uniform(1 - 0.25 * staerke, 1 + 0.25 * staerke)
    return (np.clip(x, 0, 1) * 255.0).astype(np.uint8)


def videolook(bild: np.ndarray, rng: np.random.Generator, staerke: float) -> np.ndarray:
    """Blur, sensor noise and JPEG artefacts.

    The orthomosaic is computed from many captures and is therefore sharper than
    any single image. Without this step the model learns on a sharpness it will
    never see in deployment.
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
    """Virtual frames with metric depth truth.

    The sample list is fixed before the epoch and grouped by site -- that way the
    data loader walks through few rasters instead of fetching a new one from disk
    on every access. Within a site the order is shuffled.
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
        # Two different things that `augment` would otherwise conflate. Mirroring
        # is shape augmentation; blur and JPEG are domain matching. For measuring
        # you often want only the second: fixed crops, but at the sharpness that
        # will actually be present later. Left unset, both behave like `augment`
        # -- that stays the old behaviour.
        self.spiegeln = augment if spiegeln is None else spiegeln
        self.domaene = augment if domaene is None else domaene
        self.jitter, self.video = jitter, video
        self.strahl_tiefe, self.seed = strahl_tiefe, seed
        self.site_block = max(1, int(site_block))
        self.site_mix = max(1, int(cache))
        self.lager = Rasterlager(self.wurzel / "raster", cache)
        self.set_epoch(0)

    def set_epoch(self, epoche: int) -> None:
        """New draws for the epoch, sites in a new order."""
        rng = np.random.default_rng(self.seed + 1000 * epoche)
        sites = list(self.sites)
        rng.shuffle(sites)
        pro_site = {
            site: [int(rng.integers(0, 2**31)) for _ in range(self.pro_gebiet)]
            for site in sites
        }
        if not self.augment:
            # Validation and test stay ordered by site: no updates, hence no risk
            # from long homogeneous runs, and a good cache in exchange.
            self.plan = [(site, saat) for site in sites for saat in pro_site[site]]
            return

        # Alternate short blocks of several sites per cache group. That way the
        # optimizer does not see a whole site in one run, and every worker can
        # still keep exactly the rasters involved in its cache.
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
        """Flight altitude and camera constant, matched to site size and stand.

        The camera has to hang well above the tallest treetops. Without that bound
        there would be crops in which the trees almost reach the lens -- a capture
        situation that does not occur for us and that skews the depth
        distribution.
        """
        untergrenze = max(self.hoehe_min, wipfel_m + self.abstand_min)
        obergrenze = max(untergrenze * 1.01, self.hoehe_max)
        # The ground width H/k has to fit into the site -- directly in width, and
        # relieved by the aspect ratio in height. Unsuitable pairs are redrawn.
        # Previously H was shrunk afterwards instead, which could let the camera
        # drop below the guaranteed clearance above the treetops.
        passt = 0.98 * min(breite_m, hoehe_m * self.seitenverhaeltnis)
        for _ in range(32):
            H = float(np.exp(rng.uniform(math.log(untergrenze), math.log(obergrenze))))
            k = k_von_fov(float(rng.uniform(self.fov_min, self.fov_max)))
            if H / k <= passt:
                return H, k

        # A deterministic and still physically valid fallback. For a site that is
        # too small, the field of view narrows, the camera does not descend.
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
            nb = int(round((H_flug / k) / g0))                              # window width in the ground raster
            nh = int(round(nb / self.seitenverhaeltnis))
            if nb < 32 or nh < 32 or nb > B_px or nh > H_px:
                continue
            x0 = int(rng.integers(0, B_px - nb + 1))
            y0 = int(rng.integers(0, H_px - nh + 1))
            teil = gueltig[y0 : y0 + nh, x0 : x0 + nb]
            # A coarse check suffices and is a hundred times faster on 6000 px windows.
            if teil[::8, ::8].mean() >= self.min_gueltig:
                fenster = (x0, y0, nb, nh, H_flug, k)
                break
        if fenster is None:
            # Fallback: the largest fitting window, centred, with the altitude to match.
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

        # Invalid nDSM pixels are stored on disk as 0. If height and mask are
        # scaled independently, INTER_AREA mixes those zeros into valid target
        # pixels at hole edges. Normalised, masked resampling prevents that.
        teil_maske = gueltig[y0 : y0 + nh, x0 : x0 + nb].astype(np.float32)
        teil_hoehe = ndsm_cm[y0 : y0 + nh, x0 : x0 + nb].astype(np.float32) / 100.0
        gewicht = cv2.resize(teil_maske, (sb, sh), interpolation=interp)
        summe = cv2.resize(teil_hoehe * teil_maske, (sb, sh), interpolation=interp)
        hoehe = summe / np.maximum(gewicht, 1e-6)
        # Keep only pixels whose resampling footprint was almost entirely valid.
        # That keeps fill edges out of the loss and out of the metrics.
        maske = gewicht >= 0.99

        if self.spiegeln:
            # Mirroring only, no quarter turns: those would flip the aspect ratio
            # the whole geometry is built on.
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
            # Distance along the viewing ray instead of along the optical axis.
            f_px = k * sb
            gy, gx = np.mgrid[0:sh, 0:sb].astype(np.float32)
            r2 = ((gx - (sb - 1) / 2) ** 2 + (gy - (sh - 1) / 2) ** 2) / (f_px * f_px)
            tiefe *= np.sqrt(1.0 + r2)

        maske &= tiefe > 1.0    # a camera inside a tree would be no sensible target
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
    """Bring a uint8 batch to the input size of Depth Pro and normalise it.

    Exactly the squeeze to 1536 x 1536 that the Depth Pro image processor also
    performs -- the aspect ratio is lost in the process, deliberately: the same
    thing happens at inference.

    Deliberately on the GPU and not in the data loader: as float32 a single 1536
    image would be 28 MB that would otherwise cross the process boundary.
    """
    x = bild_uint8.float().div_(255.0)
    if x.shape[-1] != modellgroesse or x.shape[-2] != modellgroesse:
        x = torch.nn.functional.interpolate(x, size=(modellgroesse, modellgroesse),
                                            mode="bilinear", align_corners=False, antialias=True)
    return (x - MODELL_MITTEL) / MODELL_STREUUNG
