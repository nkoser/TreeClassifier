"""FORTRESS: Kronen segmentieren und aus den Artpolygonen beschriften.

FORTRESS (Schiefer, Frey & Kattenborn 2022, CC BY 4.0) liefert 47 UAV-Gebiete im
Suedschwarzwald zu je 1.7 ha bei 0.77 bis 1.57 cm Bodenaufloesung, dazu 9553
Artpolygone und ein normalisiertes Hoehenmodell. Die Arten sind die, die dem
Quebec-Checkpoint fehlen:

    Picea abies 3560 | Fagus sylvatica 1824 | Abies alba 1191
    Pinus sylvestris 685 | Acer pseudoplatanus 244 | Pseudotsuga menziesii 221
    Fraxinus excelsior 175 | Larix decidua 161 | Quercus 72 | Betula pendula 53
    dazu forest floor 776 und deadwood 389 als Nicht-Baum-Klassen

Die Polygone sind aber **semantisch**: sie sagen, welche Art an einer Stelle
steht, nicht welcher Baum. Einzelne Baeume kommen aus unserer Segmentierung, die
Art aus der Verschneidung -- gemessen auf Quebec liefert das bei vorhergesagten
Kronen 71 % brauchbare Ausschnitte mit nahezu fehlerfreier Artzuordnung.

Der Massstab wird je Gebiet angeglichen: das Segmentierungsmodell hat auf
BAMFORESTS bei 1.70 cm/px gelernt, FORTRESS liegt darunter. Ohne Angleichung
saehe es Kronen in falscher Groesse -- der Fehler, der sich durch dieses Projekt
zieht.

    python crownseg/fortress.py --sites CFB014 CFB019 --out .../fortress_kronen
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as met  # noqa: E402
from label_from_semantic import assign  # noqa: E402
from queryseg import build_model, predict_tiles  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segment_sam import mask_metrics  # noqa: E402
from segment_sam3 import (  # noqa: E402
    SAM3_MODEL, merge_instances, segment_tile, tile_boxes, touches_inner_edge,
)

BAM_GSD_M = 0.0170
NICHT_BAUM = {"forest floor", "deadwood", "other"}


@torch.no_grad()
def sam3_instanzen(model, processor, patch: np.ndarray, args, device) -> list[met.Instance]:
    """Kronen einer Kachel mit SAM 3 ueber mehrere Kachelstufen.

    Dieselbe Kette wie in `segment_sam3.py`: jede Stufe bestimmt, wie gross eine
    Krone dem Modell erscheint, angeschnittene Instanzen fallen raus (die
    Nachbarkachel enthaelt dasselbe Objekt vollstaendig), danach gierig nach
    Score zusammenfuehren.
    """
    hoehe, breite = patch.shape[:2]
    kandidaten = []
    for stufe in args.sam3_tiles:
        for x0, y0, x1, y1 in tile_boxes(breite, hoehe, stufe, args.tile_overlap):
            masken, scores = segment_tile(model, processor, patch[y0:y1, x0:x1],
                                          args.prompt, args.sam3_threshold, device)
            am_rand = (x0 == 0, y0 == 0, x1 == breite, y1 == hoehe)
            for maske, score in zip(masken, scores):
                if touches_inner_edge(maske, am_rand):
                    continue
                voll = np.zeros((hoehe, breite), dtype=bool)
                voll[y0:y1, x0:x1] = maske
                kandidaten.append((voll, float(score)))

    erwartet = np.pi * (args.crown_px / 2) ** 2
    heraus = []
    for maske, score in merge_instances(kandidaten, args.max_overlap):
        werte = mask_metrics(maske)
        if werte is None:
            continue
        if not (erwartet * args.min_area_factor <= werte["area_px"] <= erwartet * args.max_area_factor):
            continue
        if werte["kompaktheit"] < args.min_compactness or werte["solidity"] < args.min_solidity:
            continue
        instanz = met.instance_from_mask(maske, score)
        if instanz is not None:
            heraus.append(instanz)
    return heraus


def read_shp(path: Path):
    """Polygonringe eines Shapefiles in Weltkoordinaten."""
    b = path.read_bytes()
    offset = 100
    while offset < len(b):
        _, laenge = struct.unpack(">II", b[offset : offset + 8])
        typ, = struct.unpack("<I", b[offset + 8 : offset + 12])
        if typ == 5:
            n_teile, n_punkte = struct.unpack("<II", b[offset + 44 : offset + 52])
            teile = struct.unpack("<" + "I" * n_teile, b[offset + 52 : offset + 52 + 4 * n_teile])
            start = offset + 52 + 4 * n_teile
            punkte = np.frombuffer(b, "<f8", 2 * n_punkte, start).reshape(-1, 2)
            grenzen = list(teile) + [n_punkte]
            yield [punkte[grenzen[i] : grenzen[i + 1]] for i in range(n_teile)]
        offset += 8 + 2 * laenge


def read_dbf(path: Path, feld: str = "species"):
    b = path.read_bytes()
    n_rec, hdr, rec = struct.unpack("<IHH", b[4:12])
    felder, offset = [], 32
    while b[offset] != 0x0D:
        felder.append((b[offset : offset + 11].split(b"\x00")[0].decode("latin-1"), b[offset + 16]))
        offset += 32
    for i in range(n_rec):
        zeile = b[hdr + i * rec : hdr + (i + 1) * rec]
        p, werte = 1, {}
        for name, laenge in felder:
            werte[name] = zeile[p : p + laenge].decode("latin-1").strip()
            p += laenge
        yield werte[feld]


def main() -> None:
    import rasterio
    from rasterio.windows import Window

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/scratch/shared/nik/data/fortress"))
    parser.add_argument("--out", type=Path, default=Path("/scratch/shared/nik/data/fortress/kronen"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/scratch/shared/nik/data/treeclf/checkpoints/crownseg_eomt.pth"))
    parser.add_argument("--sites", nargs="*", default=None, help="Vorgabe: alle.")
    parser.add_argument("--tile", type=int, default=2048, help="Kachel im Massstab von BAMFORESTS.")
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--footprint-factor", type=float, default=2.4)
    parser.add_argument("--crop-px", type=int, default=224, help="Kantenlaenge der abgelegten Ausschnitte.")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--min-purity", type=float, default=0.7)
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--score-thresh", type=float, default=0.25)
    parser.add_argument("--eval-tile", type=int, default=1024)
    parser.add_argument("--eval-overlap", type=int, default=768)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--segmenter", default="eomt", choices=("eomt", "sam3"),
                        help="Woher die Kroneninstanzen kommen, in die beschriftet wird.")
    parser.add_argument("--prompt", default="tree")
    parser.add_argument("--sam3-tiles", type=int, nargs="*", default=[2, 3, 4],
                        help="Kachelstufen je Kachel -- wie beim Lauf auf den eigenen Frames.")
    parser.add_argument("--sam3-threshold", type=float, default=0.15)
    parser.add_argument("--tile-overlap", type=float, default=0.15)
    parser.add_argument("--max-overlap", type=float, default=0.30)
    parser.add_argument("--crown-px", type=float, default=275.0,
                        help="Erwarteter Kronendurchmesser bei 1.70 cm/px; steuert den Groessenfilter.")
    parser.add_argument("--min-area-factor", type=float, default=0.12)
    parser.add_argument("--max-area-factor", type=float, default=5.0)
    parser.add_argument("--min-compactness", type=float, default=0.25)
    parser.add_argument("--min-solidity", type=float, default=0.65)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    processor = None
    if args.segmenter == "sam3":
        from transformers import Sam3Model, Sam3Processor
        processor = Sam3Processor.from_pretrained(SAM3_MODEL)
        model = Sam3Model.from_pretrained(SAM3_MODEL).to(device).eval()
        print(f"SAM 3, Prompt {args.prompt!r}, Kachelstufen {args.sam3_tiles}, "
              f"Schwelle {args.sam3_threshold}", flush=True)
    else:
        model = build_model("eomt").to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"])
        model.eval()

    shapes = args.root / "10.35097-538/data/dataset/shapefiles/shapefile"
    orthos = args.root / "orthomosaic/orthomosaic"
    sites = args.sites or sorted(p.stem.replace("_ortho", "") for p in orthos.glob("*_ortho.tif"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "crops").mkdir(exist_ok=True)

    # Einheitliche Klassennummern ueber alle Gebiete.
    alle_arten = sorted({a for s in sites if (shapes / f"poly_{s}.dbf").exists()
                         for a in read_dbf(shapes / f"poly_{s}.dbf")})
    zu_id = {a: i + 1 for i, a in enumerate(alle_arten)}
    (args.out / "klassen.json").write_text(json.dumps(zu_id, indent=2, ensure_ascii=False))
    print(f"{len(alle_arten)} Klassen: {', '.join(alle_arten)}\n", flush=True)

    zeilen, gezaehlt = [], collections.Counter()
    for site in sites:
        ortho = orthos / f"{site}_ortho.tif"
        shp = shapes / f"poly_{site}.shp"
        if not ortho.exists() or not shp.exists():
            continue

        ringe = list(read_shp(shp))
        arten = list(read_dbf(shp.with_suffix(".dbf")))

        with rasterio.open(ortho) as src:
            gsd = abs(src.transform.a)
            inverse = ~src.transform
            # Auf den Massstab bringen, in dem das Modell gelernt hat.
            faktor = gsd / BAM_GSD_M
            in_pixeln = []
            for teile, art in zip(ringe, arten):
                for ring in teile:
                    cols, rows = inverse * (ring[:, 0], ring[:, 1])
                    in_pixeln.append((np.stack([cols, rows], 1).astype(np.float32), zu_id[art]))

            quelle = int(round(args.tile / faktor))   # Fenster im Originalbild
            schritt = int(round((args.tile - args.overlap) / faktor))
            behalten = 0
            for y0 in range(0, max(1, src.height - quelle), schritt):
                for x0 in range(0, max(1, src.width - quelle), schritt):
                    patch = np.transpose(src.read((1, 2, 3), window=Window(x0, y0, quelle, quelle),
                                                  out_shape=(3, args.tile, args.tile)), (1, 2, 0))
                    patch = np.ascontiguousarray(patch)
                    if (patch.max(axis=2) == 0).mean() > 0.10:
                        continue

                    semantic = np.zeros((args.tile, args.tile), np.uint8)
                    skal = args.tile / quelle
                    for ring, klasse in in_pixeln:
                        p = np.round((ring - (x0, y0)) * skal).astype(np.int32)
                        if (p[:, 0].max() < 0 or p[:, 1].max() < 0
                                or p[:, 0].min() >= args.tile or p[:, 1].min() >= args.tile):
                            continue
                        cv2.fillPoly(semantic, [p], int(klasse))
                    if not semantic.any():
                        continue

                    instanzen = (sam3_instanzen(model, processor, patch, args, device)
                                 if args.segmenter == "sam3"
                                 else predict_tiles(model, patch, device, args))
                    if not instanzen:
                        continue
                    labels = np.zeros((args.tile, args.tile), np.uint16)
                    for i, inst in enumerate(sorted(instanzen, key=lambda x: x.score), start=1):
                        bx0, by0, bx1, by1 = inst.box
                        cx0, cy0 = max(0, bx0), max(0, by0)
                        cx1, cy1 = min(args.tile, bx1), min(args.tile, by1)
                        if cx0 >= cx1 or cy0 >= cy1:
                            continue
                        labels[cy0:cy1, cx0:cx1][inst.mask[cy0 - by0 : cy1 - by0, cx0 - bx0 : cx1 - bx0]] = i
                    if labels.max() == 0:
                        continue

                    frame = assign(labels, semantic, {0}, args.min_coverage, args.min_purity, args.min_area)
                    if frame.empty:
                        continue

                    umkehr = {v: k for k, v in zu_id.items()}
                    for row in frame[frame["brauchbar"]].itertuples():
                        art = umkehr[row.klasse]
                        durchmesser = max(row.x1 - row.x0, row.y1 - row.y0)
                        size = max(32, int(round(durchmesser * args.footprint_factor)))
                        half = size // 2
                        sx, sy = int(row.cx) - half, int(row.cy) - half
                        if sx < 0 or sy < 0 or sx + size > args.tile or sy + size > args.tile:
                            continue
                        crop = cv2.resize(patch[sy : sy + size, sx : sx + size],
                                          (args.crop_px, args.crop_px), interpolation=cv2.INTER_AREA)
                        name = f"{site}_{x0}_{y0}_{row.instanz}.jpg"
                        cv2.imwrite(str(args.out / "crops" / name),
                                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                        zeilen.append({"datei": name, "gebiet": site, "art": art, "klasse": row.klasse,
                                       "durchmesser_m": durchmesser * BAM_GSD_M,
                                       "abdeckung": row.abdeckung, "reinheit": row.reinheit})
                        gezaehlt[art] += 1
                        behalten += 1

            print(f"{site}: GSD {gsd*100:.2f} cm | {behalten} beschriftete Kronen", flush=True)

    tabelle = pd.DataFrame(zeilen)
    tabelle.to_csv(args.out / "kronen.csv", index=False)
    print(f"\n{len(tabelle)} beschriftete Kronenausschnitte -> {args.out}")
    for art, anzahl in gezaehlt.most_common():
        print(f"  {art:24s} {anzahl}")


if __name__ == "__main__":
    main()
