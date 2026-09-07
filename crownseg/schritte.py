"""Every method of phases 2 and 3 in its individual steps, on the same crop.

The result tables say *that* a method reaches 0.25 or 0.62. They do not say
*what* it fails on. These images show the intermediate states of each method --
what it proposes, what it discards and what remains.

All on the same window of the same tile, otherwise you end up comparing image
crops instead of methods. The depth comes from the cache (`depthcache.py`), so
Depth Pro does not have to run.

    python crownseg/schritte.py --nur sam3 split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bamforests as bam  # noqa: E402
import metrics as met  # noqa: E402

K3 = np.ones((3, 3), np.uint8)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def kanten(karte: np.ndarray) -> np.ndarray:
    k = karte.astype(np.uint16)
    return (cv2.dilate(k, K3) != cv2.erode(k, K3)) & (karte > 0)


def panel(bgr: np.ndarray, titel: str, farbe=(255, 255, 255)) -> np.ndarray:
    aus = bgr.copy()
    cv2.rectangle(aus, (0, 0), (aus.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(aus, titel, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, farbe, 1, cv2.LINE_AA)
    return aus


def flaechen(bild: np.ndarray, masken, seed: int = 3) -> np.ndarray:
    """Masks as coloured areas with a white outline."""
    aus = bild.copy()
    tint = aus.copy()
    rng = np.random.default_rng(seed)
    karte = np.zeros(bild.shape[:2], np.uint16)
    for i, m in enumerate(masken, 1):
        m = np.asarray(m, bool)
        tint[m] = rng.integers(60, 255, 3)
        karte[m] = i
    aus = cv2.addWeighted(aus, 0.6, tint, 0.4, 0)
    aus[kanten(karte)] = (255, 255, 255)
    return aus


def umrisse(bild: np.ndarray, masken, farbe) -> np.ndarray:
    aus = bild.copy()
    for m in masken:
        konturen, _ = cv2.findContours(np.asarray(m, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(aus, konturen, -1, farbe, 2)
    return aus


def heatmap(karte: np.ndarray, titel: str) -> np.ndarray:
    norm = (karte - karte.min()) / max(1e-6, karte.max() - karte.min())
    return panel(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO), titel)


def gitter(teile: list[np.ndarray], spalten: int = 2) -> np.ndarray:
    reihen = [np.hstack(teile[i : i + spalten]) for i in range(0, len(teile), spalten)]
    breite = max(r.shape[1] for r in reihen)
    reihen = [np.pad(r, ((0, 0), (0, breite - r.shape[1]), (0, 0))) for r in reihen]
    return np.vstack(reihen)


def ablegen(args, name: str, teile: list[np.ndarray], spalten: int = 2) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    pfad = args.out / f"{name}.jpg"
    cv2.imwrite(str(pfad), gitter(teile, spalten), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  -> {pfad}", flush=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class Fenster:
    """One image crop with depth and truth, shared by every method."""

    def __init__(self, args):
        daten = args.prepared / args.split
        index = json.loads((daten / "annotations.json").read_text())
        stem = args.stem or max(index, key=lambda k: len(index[k]))
        voll = cv2.imread(str(daten / f"{stem}.jpg"))
        tiefe = cv2.imread(str(args.prepared / f"{args.split}_tiefe_depthpro" / f"{stem}.png"),
                           cv2.IMREAD_UNCHANGED).astype(np.float32)
        if tiefe.shape[:2] != voll.shape[:2]:
            tiefe = cv2.resize(tiefe, voll.shape[1::-1], interpolation=cv2.INTER_LINEAR)

        skal = voll.shape[0] / 2048.0
        ringe = [np.asarray(r, np.float32).reshape(-1, 2) * skal for r in index[stem]]
        masken, _ = bam.masks_from_rings(ringe, voll.shape[0], voll.shape[1], int(400 * skal ** 2), 0.0)
        gt = np.zeros(voll.shape[:2], np.uint16)
        for i, m in enumerate(masken, 1):
            gt[m.astype(bool)] = i

        # Put the window where most crowns are annotated.
        S, bestes, y0, x0 = args.fenster, -1, 0, 0
        for yy in range(0, voll.shape[0] - S, S // 4):
            for xx in range(0, voll.shape[1] - S, S // 4):
                n = len(np.unique(gt[yy : yy + S, xx : xx + S])) - 1
                if n > bestes:
                    bestes, y0, x0 = n, yy, xx

        self.stem, self.groesse = stem, S
        self.bgr = np.ascontiguousarray(voll[y0 : y0 + S, x0 : x0 + S])
        self.rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        self.gt = np.ascontiguousarray(gt[y0 : y0 + S, x0 : x0 + S])
        roh = np.ascontiguousarray(tiefe[y0 : y0 + S, x0 : x0 + S])
        self.chm = (roh - roh.min()) / max(1e-6, roh.max() - roh.min())
        self.n_gt = len(np.unique(self.gt)) - 1
        print(f"Kachel {stem}, Fenster {S}x{S} bei ({x0},{y0}) mit {self.n_gt} annotierten Kronen\n", flush=True)

    def wahrheit(self) -> np.ndarray:
        return panel(flaechen(self.bgr, [self.gt == v for v in np.unique(self.gt) if v]),
                     f"Wahrheit: {self.n_gt} annotierte Kronen", (90, 230, 120))


def als_masken(instanzen, form) -> list[np.ndarray]:
    """A list of met.Instance into full-frame masks."""
    heraus = []
    for inst in instanzen:
        voll = np.zeros(form, bool)
        x0, y0, x1, y1 = inst.box
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(form[1], x1), min(form[0], y1)
        if cx0 >= cx1 or cy0 >= cy1:
            continue
        voll[cy0:cy1, cx0:cx1] = inst.mask[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        heraus.append(voll)
    return heraus


def guete(masken, gt: np.ndarray) -> str:
    """F1 against the truth -- so every image carries what it is worth."""
    wahr = [i for i in (met.instance_from_mask(gt == v) for v in np.unique(gt) if v) if i is not None]
    vorher = [i for i in (met.instance_from_mask(np.asarray(m, bool)) for m in masken) if i is not None]
    if not vorher:
        return "F1 0.000"
    z = met.evaluate(vorher, wahr, 0.5)
    return f"F1 {z['f1']:.3f}"


# --------------------------------------------------------------------------- #
# The methods
# --------------------------------------------------------------------------- #

def wipfel(chm, args):
    """Treetops and basins as in segment_prompted.find_peaks."""
    from skimage.measure import label as cc_label
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    glatt = cv2.GaussianBlur(chm, (0, 0), max(0.8, args.crown_px * args.smooth_factor))
    dach = glatt > np.percentile(glatt, args.gap_percentile)
    lo, hi = np.percentile(glatt[dach], [5, 95])
    marker = cc_label(h_maxima(np.where(dach, glatt, glatt.min()), (hi - lo) * args.peak_prominence) > 0)
    becken = watershed(-glatt, marker, mask=dach)
    punkte = np.array([[np.nonzero(marker == r)[1].mean(), np.nonzero(marker == r)[0].mean()]
                       for r in range(1, marker.max() + 1)], np.float32)
    return glatt, dach, marker, becken, punkte


def schritte_crownnet(f: Fenster, args, device) -> None:
    """Three maps -> seeds from the interior -> watershed on the margin map."""
    import crownnet as cn

    stand = torch.load(args.crownnet_ckpt, map_location=device, weights_only=False)
    backbone = cn.build_backbone(args, device)
    kopf = cn.CrownHead().to(device)
    kopf.load_state_dict(stand["head"])
    kopf.eval()
    karten = cn.predict_maps(backbone, kopf, f.rgb, device, args.crownnet_long_side)
    karten = np.stack([cv2.resize(k, (f.groesse, f.groesse)) for k in karten])
    labels = cn.instances_from_maps(karten, args)
    masken = [labels == v for v in np.unique(labels) if v]

    ablegen(args, "01_crownnet", [
        f.wahrheit(),
        heatmap(karten[0], "1. Vorhersage Inneres (Krone ohne Randsaum)"),
        heatmap(karten[1], "2. Vorhersage Rand -- dient als Kostenflaeche"),
        heatmap(karten[2], "3. Vorhersage Zentrum -- im Code ungenutzt"),
        panel(flaechen(f.bgr, [labels == v for v in np.unique(labels) if v], seed=5),
              f"4. Keime aus dem Inneren + Watershed: {len(masken)} Instanzen"),
        panel(umrisse(umrisse(f.bgr, [f.gt == v for v in np.unique(f.gt) if v], (90, 230, 120)),
                      masken, (60, 130, 250)),
              f"5. gegen die Wahrheit (gruen), orange = Vorhersage -- {guete(masken, f.gt)}"),
    ])


def schritte_prompted(f: Fenster, args, device) -> None:
    """Depth says WHERE, SAM says WHERE THE BOUNDARY is."""
    from transformers import SamModel, SamProcessor

    from segment_prompted import pick_candidates, prompt_sam

    glatt, dach, marker, becken, punkte = wipfel(f.chm, args)
    processor = SamProcessor.from_pretrained(args.sam_model)
    modell = SamModel.from_pretrained(args.sam_model).to(device).eval()
    masken3, scores = prompt_sam(modell, processor, f.rgb, punkte, device, args.chunk)

    alle = [masken3[i, o].astype(bool) for i in range(len(masken3)) for o in range(masken3.shape[1])]
    gewaehlt = pick_candidates(masken3, scores, becken, np.arange(1, len(punkte) + 1), args)
    endgueltig = [m for m, _, _ in gewaehlt]

    mit_punkten = heatmap(glatt, f"2. geglaettete Tiefe + {len(punkte)} Wipfel")
    for x, y in punkte:
        cv2.circle(mit_punkten, (int(x), int(y)), 7, (255, 255, 255), -1)
        cv2.circle(mit_punkten, (int(x), int(y)), 7, (0, 0, 0), 2)

    ablegen(args, "02_prompted", [
        f.wahrheit(),
        mit_punkten,
        panel(flaechen(f.bgr, [becken == r for r in range(1, becken.max() + 1)], seed=7),
              "3. Becken je Wipfel -- nur Groessenreferenz, keine Grenze"),
        panel(umrisse(f.bgr, alle, (200, 200, 60)),
              f"4. SAM: {len(alle)} Kandidaten ({masken3.shape[1]} je Wipfel)"),
        panel(flaechen(f.bgr, endgueltig, seed=5),
              f"5. je Wipfel der zum Becken passendste: {len(endgueltig)}"),
        panel(umrisse(umrisse(f.bgr, [f.gt == v for v in np.unique(f.gt) if v], (90, 230, 120)),
                      endgueltig, (60, 130, 250)),
              f"6. gegen die Wahrheit -- {guete(endgueltig, f.gt)}"),
    ])


def schritte_sam3(f: Fenster, args, device) -> None:
    """Tile levels -> discard cut ones -> merge by score -> shape filter."""
    from transformers import Sam3Model, Sam3Processor

    from segment_sam import mask_metrics
    from segment_sam3 import SAM3_MODEL, merge_instances, segment_tile, tile_boxes, touches_inner_edge

    processor = Sam3Processor.from_pretrained(SAM3_MODEL)
    modell = Sam3Model.from_pretrained(SAM3_MODEL).to(device).eval()

    roh, behalten, verworfen = [], [], []
    for stufe in args.sam3_tiles:
        for x0, y0, x1, y1 in tile_boxes(f.groesse, f.groesse, stufe, args.tile_overlap):
            masken, scores = segment_tile(modell, processor, f.rgb[y0:y1, x0:x1],
                                          args.prompt, args.sam3_threshold, device)
            am_rand = (x0 == 0, y0 == 0, x1 == f.groesse, y1 == f.groesse)
            for m, s in zip(masken, scores):
                voll = np.zeros((f.groesse, f.groesse), bool)
                voll[y0:y1, x0:x1] = m
                roh.append(voll)
                (verworfen if touches_inner_edge(m, am_rand) else behalten).append(voll)

    vereint = merge_instances([(m, 1.0) for m in behalten], args.max_overlap)
    erwartet = np.pi * (args.crown_px / 2) ** 2
    endgueltig = []
    for m, _ in vereint:
        w = mask_metrics(m)
        if w is None or not (erwartet * args.min_area_factor <= w["area_px"] <= erwartet * args.max_area_factor):
            continue
        if w["kompaktheit"] < args.min_compactness or w["solidity"] < args.min_solidity:
            continue
        endgueltig.append(m)

    ablegen(args, "03_sam3", [
        f.wahrheit(),
        panel(umrisse(f.bgr, roh, (200, 200, 60)),
              f"1. roh aus {len(args.sam3_tiles)} Kachelstufen {args.sam3_tiles}: {len(roh)} Masken"),
        panel(umrisse(f.bgr, verworfen, (70, 90, 240)),
              f"2. an innerer Kachelkante angeschnitten -> weg: {len(verworfen)}"),
        panel(umrisse(f.bgr, [m for m, _ in vereint], (200, 200, 60)),
              f"3. nach Score zusammengefuehrt: {len(vereint)}"),
        panel(flaechen(f.bgr, endgueltig, seed=5),
              f"4. Form- und Groessenfilter: {len(endgueltig)}"),
        panel(umrisse(umrisse(f.bgr, [f.gt == v for v in np.unique(f.gt) if v], (90, 230, 120)),
                      endgueltig, (60, 130, 250)),
              f"5. gegen die Wahrheit -- {guete(endgueltig, f.gt)}"),
    ])
    return endgueltig


def schritte_split(f: Fenster, args, device, sam3_masken=None) -> None:
    """Break masks spanning several treetops apart at the treetops."""
    from crownseg.sam3_depth import split_by_tops  # noqa: F401  (for completeness only)

    if sam3_masken is None:
        sam3_masken = schritte_sam3(f, args, device)
    glatt, dach, marker, becken, punkte = wipfel(f.chm, args)

    mehrfach, geteilt = [], []
    for m in sam3_masken:
        lokal = np.where(m, marker, 0)
        drin = [v for v in np.unique(lokal) if v > 0]
        if len(drin) <= 1:
            geteilt.append(m)
            continue
        mehrfach.append(m)
        from skimage.segmentation import watershed
        teile = watershed(-glatt, lokal, mask=m)
        geteilt.extend([teile == v for v in drin if (teile == v).any()])

    ablegen(args, "04_teilen", [
        panel(flaechen(f.bgr, sam3_masken, seed=5), f"1. SAM 3: {len(sam3_masken)} Masken"),
        panel(umrisse(f.bgr, mehrfach, (60, 130, 250)),
              f"2. davon mit mehr als einem Wipfel: {len(mehrfach)}"),
        panel(flaechen(f.bgr, geteilt, seed=5),
              f"3. an den Wipfeln aufgebrochen: {len(geteilt)}"),
        panel(umrisse(umrisse(f.bgr, [f.gt == v for v in np.unique(f.gt) if v], (90, 230, 120)),
                      geteilt, (60, 130, 250)),
              f"4. gegen die Wahrheit -- {guete(sam3_masken, f.gt)} -> {guete(geteilt, f.gt)}"),
    ])


def schritte_hybrid(f: Fenster, args, device) -> None:
    """SAM first, watershed only on the remaining area -- the dead end, in a picture."""
    from transformers import Sam3Model, Sam3Processor

    from segment_hybrid import residual_crowns
    from segment_sam3 import SAM3_MODEL, merge_instances, segment_tile, tile_boxes, touches_inner_edge

    processor = Sam3Processor.from_pretrained(SAM3_MODEL)
    modell = Sam3Model.from_pretrained(SAM3_MODEL).to(device).eval()
    kandidaten = []
    for x0, y0, x1, y1 in tile_boxes(f.groesse, f.groesse, 2, args.tile_overlap):
        masken, scores = segment_tile(modell, processor, f.rgb[y0:y1, x0:x1],
                                      args.prompt, args.sam3_threshold, device)
        am_rand = (x0 == 0, y0 == 0, x1 == f.groesse, y1 == f.groesse)
        for m, s in zip(masken, scores):
            if touches_inner_edge(m, am_rand):
                continue
            voll = np.zeros((f.groesse, f.groesse), bool)
            voll[y0:y1, x0:x1] = m
            kandidaten.append((voll, float(s)))
    sam_masken = [m for m, _ in merge_instances(kandidaten, args.max_overlap)]

    belegt = np.zeros((f.groesse, f.groesse), bool)
    for m in sam_masken:
        belegt |= m
    rest = ~belegt
    _, rest_masken = residual_crowns(f.chm, rest, args)

    rest_bild = f.bgr.copy()
    rest_bild[~rest] = (rest_bild[~rest] * 0.25).astype(np.uint8)

    beide = umrisse(umrisse(f.bgr, sam_masken, (90, 230, 120)), rest_masken, (60, 130, 250))
    ablegen(args, "05_hybrid", [
        f.wahrheit(),
        panel(flaechen(f.bgr, sam_masken, seed=5), f"1. SAM 3 zuerst: {len(sam_masken)} Kronen"),
        panel(rest_bild, "2. Restflaeche -- was SAM fuer keinen Baum hielt"),
        panel(flaechen(f.bgr, rest_masken, seed=9),
              f"3. Watershed nur dort: {len(rest_masken)} weitere"),
        panel(beide, "4. gruen = aus SAM, orange = aus dem Watershed"),
        panel(umrisse(f.bgr, [f.gt == v for v in np.unique(f.gt) if v], (90, 230, 120)),
              f"5. Wahrheit -- SAM allein {guete(sam_masken, f.gt)} | "
              f"Rest allein {guete(rest_masken, f.gt)}"),
    ])


def schritte_eomt(f: Fenster, args, device) -> None:
    """Query-based: fixed queries, each with its own mask and its own score."""
    from queryseg import build_model, predict_tiles

    modell = build_model("eomt").to(device)
    modell.load_state_dict(torch.load(args.eomt_ckpt, map_location=device, weights_only=False)["model"])
    modell.eval()

    stufen = []
    for schwelle in args.eomt_schwellen:
        args.score_thresh = schwelle
        instanzen = predict_tiles(modell, f.rgb, device, args)
        stufen.append((schwelle, als_masken(instanzen, (f.groesse, f.groesse))))

    teile = [f.wahrheit()]
    for schwelle, masken in stufen:
        teile.append(panel(flaechen(f.bgr, masken, seed=5),
                           f"Schwelle {schwelle:.2f}: {len(masken)} Kronen -- {guete(masken, f.gt)}"))
    beste = stufen[-1][1]
    teile.append(panel(umrisse(umrisse(f.bgr, [f.gt == v for v in np.unique(f.gt) if v], (90, 230, 120)),
                               beste, (60, 130, 250)),
                       f"gegen die Wahrheit bei Schwelle {stufen[-1][0]:.2f}"))
    ablegen(args, "06_eomt", teile)


def schritte_tiefe(f: Fenster, args, device) -> None:
    """The same model, once on RGB and once on the depth alone.

    `--depth-only` stacks the depth map three times and feeds it to the model as
    an image -- so the network never sees a colour pixel. On Hain it reaches F1
    0.514 that way, against 0.554 with the image.
    """
    from queryseg import build_model, predict_tiles

    tiefe_bild = np.dstack([(f.chm * 255).astype(np.uint8)] * 3)
    laeufe = []
    for name, ckpt, eingabe in (("RGB", args.eomt_ckpt, f.rgb),
                                ("nur Tiefe", args.tiefe_ckpt, tiefe_bild)):
        modell = build_model("eomt").to(device)
        modell.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
        modell.eval()
        instanzen = predict_tiles(modell, eingabe, device, args)
        laeufe.append((name, als_masken(instanzen, (f.groesse, f.groesse))))
        del modell
        torch.cuda.empty_cache()

    beide = f.bgr.copy()
    beide = umrisse(beide, [f.gt == v for v in np.unique(f.gt) if v], (90, 230, 120))
    beide = umrisse(beide, laeufe[0][1], (250, 190, 60))
    beide = umrisse(beide, laeufe[1][1], (60, 130, 250))

    ablegen(args, "07_nur_tiefe", [
        f.wahrheit(),
        panel(cv2.applyColorMap((f.chm * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
              "Das sieht das Tiefe-Modell -- kein Farbpixel"),
        panel(flaechen(f.bgr, laeufe[0][1], seed=5),
              f"aus RGB: {len(laeufe[0][1])} Kronen -- {guete(laeufe[0][1], f.gt)}"),
        panel(flaechen(f.bgr, laeufe[1][1], seed=5),
              f"nur aus der Tiefe: {len(laeufe[1][1])} Kronen -- {guete(laeufe[1][1], f.gt)}"),
        panel(beide, "gruen = Wahrheit, hellblau = aus RGB, orange = nur Tiefe"),
        panel(flaechen(tiefe_bild, laeufe[1][1], seed=5), "dieselben Kronen auf der Tiefenkarte"),
    ])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nur", nargs="*", default=["crownnet", "prompted", "sam3", "teilen", "hybrid", "eomt", "tiefe"])
    p.add_argument("--prepared", type=Path, default=Path("/scratch/shared/nik/data/bamforests/crownseg"))
    p.add_argument("--split", default="test1")
    p.add_argument("--stem", default=None)
    p.add_argument("--fenster", type=int, default=768)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "results_views_schritte")
    # Depth / treetops
    p.add_argument("--crown-px", type=float, default=275.0)
    p.add_argument("--smooth-factor", type=float, default=0.045)
    p.add_argument("--gap-percentile", type=float, default=25.0)
    p.add_argument("--peak-prominence", type=float, default=0.10)
    # Shape
    p.add_argument("--min-area-factor", type=float, default=0.12)
    p.add_argument("--max-area-factor", type=float, default=5.0)
    p.add_argument("--min-compactness", type=float, default=0.25)
    p.add_argument("--min-solidity", type=float, default=0.65)
    # SAM / SAM 3
    p.add_argument("--sam-model", default="facebook/sam-vit-huge")
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--select", default="basin")
    p.add_argument("--prompt", default="tree")
    p.add_argument("--sam3-tiles", type=int, nargs="*", default=[2, 3])
    p.add_argument("--sam3-threshold", type=float, default=0.15)
    p.add_argument("--tile-overlap", type=float, default=0.15)
    p.add_argument("--max-overlap", type=float, default=0.30)
    # crownnet
    p.add_argument("--crownnet-ckpt", type=Path,
                   default=Path("/scratch/shared/nik/data/treeclf/checkpoints/crownnet_fuse.pth"))
    p.add_argument("--crownnet-long-side", type=int, default=768)
    p.add_argument("--ckpt", type=Path,
                   default=Path("/scratch/shared/nik/data/treeclf/checkpoints/dinovtreeb_quebectrees.pth"))
    p.add_argument("--categories", type=Path,
                   default=Path(__file__).resolve().parent.parent / "third_party" / "quebec_trees_categories.json")
    p.add_argument("--interior-thresh", type=float, default=0.5)
    p.add_argument("--crown-thresh", type=float, default=0.5)
    p.add_argument("--min-seed-px", type=int, default=200)
    # EoMT
    p.add_argument("--eomt-ckpt", type=Path,
                   default=Path("/scratch/shared/nik/data/treeclf/checkpoints/crownseg_eomt.pth"))
    p.add_argument("--tiefe-ckpt", type=Path,
                   default=Path("/scratch/shared/nik/data/treeclf/checkpoints/crownseg_eomt_nurtiefe_depthpro.pth"))
    p.add_argument("--eomt-schwellen", type=float, nargs="*", default=[0.50, 0.25])
    p.add_argument("--eval-tile", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=768)
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--min-area", type=int, default=400)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    f = Fenster(args)
    sam3_cache = None
    for name in args.nur:
        print(f"[{name}]", flush=True)
        try:
            if name == "crownnet":
                schritte_crownnet(f, args, device)
            elif name == "prompted":
                schritte_prompted(f, args, device)
            elif name == "sam3":
                sam3_cache = schritte_sam3(f, args, device)
            elif name == "teilen":
                schritte_split(f, args, device, sam3_cache)
            elif name == "hybrid":
                schritte_hybrid(f, args, device)
            elif name == "eomt":
                schritte_eomt(f, args, device)
            elif name == "tiefe":
                schritte_tiefe(f, args, device)
        except Exception as fehler:            # one method must not take the others down
            print(f"  FEHLER: {type(fehler).__name__}: {fehler}", flush=True)


if __name__ == "__main__":
    main()
