"""Turn the depth maps into point clouds -- anchored to the ground, not the camera.

A depth map is already 2.5D: one distance per pixel. The route to 3D is the
inverse of the projection equation,

    X = (u - cx) * d / f_px        Y = -(v - cy) * d / f_px        Z = ground - d

where `f_px = k * image width` and `k = 0.5 / tan(HFOV/2)`.

**Why the field of view distorts the cloud, but not the way you would think.** In
X and Y it cancels out: `d` is proportional to `k`, and so is `f_px`. If the
assumed field of view is wrong, crown diameters therefore stay correct -- only
the height is stretched or compressed. Trees become too pointed or too flat, not
too wide.

**Why points at crown edges are discarded.** There the depth jumps from the crown
to the ground. The depth map is continuous, though, and puts intermediate values
in between -- in 3D those become points hanging free in space, and trees taper
downwards into cones instead of touching down. `--max-neigung` throws them out.

**Why the ground does not come from the flight altitude.** One could compute
`Z = H - d`. But then everything hangs on a number that is often estimated, and
sloping terrain tilts the whole cloud. Instead the ground is taken from the data,
as in forestry practice with LiDAR: divide the depth map into tiles, take the
deepest point per tile as a ground candidate, smooth, and normalise the cloud
onto it. That carries terrain slope along.

One residual remains: in a closed canopy the deepest **visible** point is not the
ground. Measured from the FORTRESS height models it lies at 0.917 of the flight
altitude at the median. `--boden-faktor` factors that out.

    python depthft/punktwolke.py --frames 80m/frame_000297.jpg
    python depthft/punktwolke.py --alle --format ply las --schritt 2
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inferenz  # noqa: E402


def schreibe_ply(pfad: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Binary PLY with colour -- read by CloudCompare, MeshLab, Blender, QGIS."""
    kopf = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment erzeugt von depthft/punktwolke.py\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    satz = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("r", "u1"), ("g", "u1"), ("b", "u1")])
    satz["x"], satz["y"], satz["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    satz["r"], satz["g"], satz["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(pfad, "wb") as f:
        f.write(kopf)
        f.write(satz.tobytes())


def schreibe_las(pfad: Path, xyz: np.ndarray, rgb: np.ndarray, klasse: np.ndarray | None = None,
                 skala: float = 0.001) -> None:
    """LAS 1.2, point format 2 (with colour) -- for lidR, LAStools, CloudCompare.

    Written by hand because there is no LAS library in the container. The
    coordinates are stored as int32 in multiples of `skala`; at 1 mm the value
    range covers a good 2000 km, i.e. more than enough.
    """
    mins = xyz.min(axis=0) if len(xyz) else np.zeros(3)
    maxs = xyz.max(axis=0) if len(xyz) else np.zeros(3)
    ganz = np.round(xyz / skala).astype(np.int32)

    punkte = np.zeros(len(xyz), dtype=[
        ("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("intensity", "<u2"),
        ("flags", "u1"), ("klasse", "u1"), ("winkel", "i1"), ("nutzer", "u1"),
        ("quelle", "<u2"), ("r", "<u2"), ("g", "<u2"), ("b", "<u2")])
    punkte["x"], punkte["y"], punkte["z"] = ganz[:, 0], ganz[:, 1], ganz[:, 2]
    punkte["flags"] = 1                                    # return 1 of 1
    punkte["klasse"] = 1 if klasse is None else klasse     # 1 = unclassified
    # LAS expects 16 bits per colour channel; 8-bit values are scaled up.
    punkte["r"], punkte["g"], punkte["b"] = (rgb[:, 0].astype(np.uint16) * 257,
                                             rgb[:, 1].astype(np.uint16) * 257,
                                             rgb[:, 2].astype(np.uint16) * 257)

    kopf = struct.pack(
        "<4sHH16sBB32s32sHHHLLBHL",
        b"LASF", 0, 0, b"\x00" * 16, 1, 2,
        b"TreeClassifier".ljust(32, b"\x00"), b"depthft/punktwolke.py".ljust(32, b"\x00"),
        1, 2026, 227, 227, 0, 2, 26, len(xyz),
    )
    kopf += struct.pack("<5L", len(xyz), 0, 0, 0, 0)
    kopf += struct.pack("<3d", skala, skala, skala)
    kopf += struct.pack("<3d", 0.0, 0.0, 0.0)
    kopf += struct.pack("<6d", maxs[0], mins[0], maxs[1], mins[1], maxs[2], mins[2])
    assert len(kopf) == 227, len(kopf)

    with open(pfad, "wb") as f:
        f.write(kopf)
        f.write(punkte.tobytes())


def kamera_pruefen(bild, pfad, erwartete_breite: int = 1920, erwartetes_verhaeltnis: float = 16 / 9,
                   toleranz: float = 0.02) -> str | None:
    """Warn when an image cannot come from the same camera.

    The supplied field of view holds for the drone frames, 1920x1080. A screenshot
    or a cropped image has a different framing and therefore a different field of
    view -- the depths would be wrong by an unknown factor without it showing in
    the result. The folder `urban` contained exactly such files.
    """
    h, w = bild.shape[:2]
    verhaeltnis = w / max(h, 1)
    if w == erwartete_breite and abs(verhaeltnis - erwartetes_verhaeltnis) < toleranz:
        return None
    return (f"{pfad.name}: {w}x{h} (Verhaeltnis {verhaeltnis:.2f}) statt "
            f"{erwartete_breite}x{int(erwartete_breite / erwartetes_verhaeltnis)} -- "
            f"der vorgegebene Bildwinkel gilt hier vermutlich nicht.")


def kantenmaske(tiefe: np.ndarray, gsd_m: float, max_neigung: float) -> np.ndarray:
    """Discard points at depth jumps -- in reality they do not exist.

    At a crown edge the depth jumps from the crown to the ground. The depth map is
    continuous, though, so it places intermediate values in between, and when
    spanned into 3D those become points hanging free in space -- trees taper
    downwards into cones instead of touching down. In the literature they are
    called *flying pixels*.

    They are detected by their slope: `max_neigung` is the ratio of depth change
    to ground sampling. 8 means a slope steeper than 8:1 (about 83 degrees) no
    longer passes as a surface.
    """
    dy, dx = np.gradient(tiefe.astype(np.float32))
    gefaelle = np.hypot(dx, dy) / max(gsd_m, 1e-6)
    steil = gefaelle > max_neigung
    # Dilate by one pixel: the jump itself is sharp, the veil sits beside it.
    return cv2.dilate(steil.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0


def bodenmodell(tiefe: np.ndarray, gsd_m: float, kachel_m: float, perzentil: float,
                boden_faktor: float) -> np.ndarray:
    """Terrain model from the depth map -- the deepest point per tile.

    The same method by which a terrain model is made from a LiDAR cloud: ground
    candidates per tile, then smoothing. With two conditions, without which it
    goes wrong:

    **The model must nowhere lie above the observed surface.** Otherwise points
    end up below the ground -- measured up to 29 m deep in a stand with a strong
    slope, where the smoothed surface did not follow the real one. The final
    `maximum` step enforces it: where the surface lies deeper than the estimate,
    it is itself the ground.

    **The smoothing must not iron out the slope.** It therefore runs over just
    under one tile, not over several.

    One residual remains: in a closed canopy the deepest **visible** point is not
    the ground. Measured from the FORTRESS height models it lies at 0.917 of the
    flight altitude at the median, which `boden_faktor` compensates. The same
    value comes out if it is instead optimised against the truth
    (`hoehe_pruefen.py`) -- two independent routes to the same factor.

    Measured against the nDSM of the test sites, this route to height gives
    MAE 5.64 m and is therefore **better than the calculation from a known flight
    altitude** (7.46 m): `Z = H - d` cannot represent slope, a terrain model can.
    """
    # Individual outlying pixels would pull the ground down locally.
    tiefe_r = cv2.medianBlur(tiefe.astype(np.float32), 5)

    kachel_px = max(8, int(round(kachel_m / max(gsd_m, 1e-6))))
    h, w = tiefe_r.shape
    nz, ns = max(2, h // kachel_px), max(2, w // kachel_px)
    grob = np.zeros((nz, ns), np.float32)
    for i in range(nz):
        z0, z1 = i * h // nz, (i + 1) * h // nz
        for j in range(ns):
            s0, s1 = j * w // ns, (j + 1) * w // ns
            teil = tiefe_r[z0:z1, s0:s1]
            grob[i, j] = np.percentile(teil, perzentil) if teil.size else 0.0

    fein = cv2.resize(grob, (w, h), interpolation=cv2.INTER_LINEAR)
    fein = cv2.GaussianBlur(fein, (0, 0), max(2.0, kachel_px * 0.4))

    # The decisive step: never above the surface.
    boden = np.maximum(fein, tiefe_r)
    return boden / max(boden_faktor, 1e-6)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--karten", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft_karten"),
                        help="Precomputed depth maps; if missing, the model is loaded.")
    parser.add_argument("--ft", type=Path,
                        default=Path("/scratch/shared/nik/runs/depthft/bestes"))
    parser.add_argument("--modellart", default="tiefe", choices=("tiefe", "hoehe"),
                        help="tiefe: via the depth, with a terrain model. hoehe: outputs metres "
                             "directly, but does not distinguish between stands.")
    parser.add_argument("--out", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft_wolken"))
    parser.add_argument("--frames", nargs="*", default=None, metavar="ORDNER/DATEI",
                        help="Individual frames; the default is one per folder.")
    parser.add_argument("--alle", action="store_true", help="Every frame instead of one per folder.")
    parser.add_argument("--format", nargs="*", default=["ply"], choices=("ply", "las"))
    parser.add_argument("--hfov-deg", type=float, default=48.0)
    parser.add_argument("--schritt", type=int, default=2,
                        help="Every n-th pixel. 1 gives a good 2 million points per frame.")
    parser.add_argument("--boden", default="modell", choices=("modell", "flughoehe", "roh"),
                        help="modell: terrain model from the data. flughoehe: Z = H - d. "
                             "roh: Z = -d, with the origin at the camera.")
    parser.add_argument("--kachel-m", type=float, default=35.0,
                        help="Tile size for the terrain model. Smaller follows the slope "
                             "better, larger is calmer. Measured on the FORTRESS test sites, "
                             "35 m is the best compromise: MAE 5.64 m at practically no "
                             "offset.")
    parser.add_argument("--boden-perzentil", type=float, default=97.0)
    parser.add_argument("--boden-faktor", type=float, default=0.917,
                        help="Deepest visible point relative to the real ground.")
    parser.add_argument("--max-neigung", type=float, default=8.0,
                        help="Discard points at depth jumps. Ratio of depth change to "
                             "ground sampling; 0 turns it off.")
    parser.add_argument("--min-hoehe", type=float, default=None,
                        help="Discard points below this, e.g. 2 keeps only vegetation.")
    parser.add_argument("--altitudes", nargs="*", metavar="ORDNER=HOEHE",
                        default=["dense=51", "dense1=69", "mixed=92", "mixed1=103",
                                 "pines=60", "urban=120"])
    parser.add_argument("--altitude", type=float, default=100.0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    vorgaben = {e.split("=")[0]: float(e.split("=")[1]) for e in args.altitudes}
    k = inferenz.k_von_fov(args.hfov_deg)

    ordner = sorted(p for p in args.input.iterdir() if p.is_dir())
    if args.frames:
        auswahl = [args.input / f for f in args.frames]
    elif args.alle:
        auswahl = [f for o in ordner for f in sorted(o.iterdir())
                   if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    else:
        auswahl = []
        for o in ordner:
            treffer = sorted(f for f in o.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if treffer:
                auswahl.append(treffer[0])

    model = None
    zeilen = []
    warnungen: list[str] = []
    for pfad in auswahl:
        ordnername = pfad.parent.name
        name = f"{ordnername}_{pfad.stem}"
        bild = cv2.cvtColor(cv2.imread(str(pfad)), cv2.COLOR_BGR2RGB)
        hinweis = kamera_pruefen(bild, pfad)
        if hinweis:
            warnungen.append(hinweis)
            print(f"  ACHTUNG {hinweis}", flush=True)
        H = vorgaben.get(ordnername)
        if H is None:
            import re
            m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*m?\s*", ordnername, re.IGNORECASE)
            H = float(m.group(1)) if m else args.altitude

        if model is None:
            device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
            model = inferenz.lade(str(args.ft), device, fov_head=False)
            print(f"Modell geladen auf {device} ({args.modellart})", flush=True)
        tiefe, hoehe_direkt = inferenz.karten_von(model, bild, art=args.modellart, k=k,
                                                  flughoehe_m=H, device=device)

        h, w = tiefe.shape
        f_px = k * w
        gsd = H / f_px

        # If the model predicts the height directly, no terrain model is needed --
        # the ground reference is already inside the prediction.
        if hoehe_direkt is not None:
            boden = tiefe + hoehe_direkt
        elif args.boden == "modell":
            boden = bodenmodell(tiefe, gsd, args.kachel_m, args.boden_perzentil, args.boden_faktor)
        elif args.boden == "flughoehe":
            boden = np.full_like(tiefe, H)
        else:
            boden = np.zeros_like(tiefe)

        s = max(1, args.schritt)
        v, u = np.mgrid[0:h:s, 0:w:s].astype(np.float32)
        d = tiefe[::s, ::s]
        # X and Y are independent of the assumed field of view: d is proportional
        # to k and so is f_px, so both cancel out here.
        X = (u - (w - 1) / 2.0) * d / f_px
        Y = -(v - (h - 1) / 2.0) * d / f_px
        Z = boden[::s, ::s] - d

        xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
        rgb = bild[::s, ::s].reshape(-1, 3).astype(np.uint8)
        gueltig = np.isfinite(xyz).all(axis=1)
        if args.max_neigung > 0:
            gueltig &= kantenmaske(tiefe, gsd, args.max_neigung)[::s, ::s].ravel()
        if args.min_hoehe is not None:
            gueltig &= xyz[:, 2] >= args.min_hoehe
        xyz, rgb = xyz[gueltig], rgb[gueltig]

        for endung in args.format:
            ziel = args.out / f"{name}.{endung}"
            (schreibe_ply if endung == "ply" else schreibe_las)(ziel, xyz, rgb)

        zeilen.append({
            "datei": name, "ordner": ordnername, "punkte": len(xyz),
            "flughoehe_m": H, "hfov_grad": args.hfov_deg, "gsd_cm": gsd * 100,
            "boden": args.boden, "schritt": s,
            "breite_m": float(xyz[:, 0].max() - xyz[:, 0].min()) if len(xyz) else 0.0,
            "laenge_m": float(xyz[:, 1].max() - xyz[:, 1].min()) if len(xyz) else 0.0,
            "unter_boden": float(np.mean(xyz[:, 2] < -1.0)) if len(xyz) else 0.0,
            "z_p02_m": float(np.percentile(xyz[:, 2], 2)) if len(xyz) else 0.0,
            "z_min_m": float(xyz[:, 2].min()) if len(xyz) else 0.0,
            "z_p95_m": float(np.percentile(xyz[:, 2], 95)) if len(xyz) else 0.0,
            "z_max_m": float(xyz[:, 2].max()) if len(xyz) else 0.0,
        })
        print(f"  {name:32s} {len(xyz):8d} Punkte | {zeilen[-1]['breite_m']:5.1f} x "
              f"{zeilen[-1]['laenge_m']:5.1f} m | Z {zeilen[-1]['z_p02_m']:6.1f} bis "
              f"{zeilen[-1]['z_max_m']:5.1f} m (p95 {zeilen[-1]['z_p95_m']:5.1f}) | "
              f"unter Boden {zeilen[-1]['unter_boden']*100:4.1f} %", flush=True)

    tabelle = pd.DataFrame(zeilen)
    tabelle.to_csv(args.out / "wolken.csv", index=False)
    groesse = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"\n{len(tabelle)} Wolken -> {args.out}  ({groesse:.0f} MB)")
    if warnungen:
        print(f"\n{len(warnungen)} Datei(en) passen nicht zur angenommenen Kamera:")
        for h in warnungen:
            print(f"  {h}")
        print("  Deren Hoehen sind um einen unbekannten Faktor falsch.")
    print("\nZ ist die Hoehe ueber Boden: 0 ist der Waldboden, positiv nach oben.")
    if args.modellart == "hoehe":
        print("Die Hoehe kommt unmittelbar aus dem Modell -- ohne Gelaendemodell und ohne "
              "Bildwinkel. Der Bildwinkel geht nur noch in X und Y ein, also in die "
              "Kronendurchmesser, nicht mehr in die Hoehe.")
    elif args.boden == "modell":
        print(f"Boden aus den Daten geschaetzt ({args.kachel_m:.0f}-m-Kacheln, "
              f"{args.boden_perzentil:.0f}. Perzentil, Faktor {args.boden_faktor:.3f}) -- "
              "unabhaengig von der angenommenen Flughoehe.")


if __name__ == "__main__":
    main()
