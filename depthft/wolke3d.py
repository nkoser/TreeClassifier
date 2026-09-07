"""Render the point cloud above the original frame -- as stills and as an orbit.

The frame lies in the scene as a ground plane, with the point cloud floating
above it. That shows at a glance which tree in the image corresponds to which
rise in the cloud -- and whether the heights are plausible at all.

The rendering is done by hand, because there is no 3D library in the container.
That is less work than it sounds:

  ground     Project the four corners of the ground plane, derive a homography
             from them, and place the frame in with `warpPerspective`.
  points     Project perspectively, sort by depth, and draw from back to front.
             The front ones overwrite the ones behind, which replaces a depth
             buffer.

    python depthft/wolke3d.py --frames 80m/frame_000297.jpg
    python depthft/wolke3d.py --frames 80m/frame_000297.jpg --umlauf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inferenz  # noqa: E402
from punktwolke import bodenmodell, kamera_pruefen, kantenmaske  # noqa: E402

BILDENDUNGEN = {".jpg", ".jpeg", ".png"}


def kamera(mitte: np.ndarray, abstand: float, azimut_grad: float, hoehe_grad: float):
    """Viewpoint and axis frame of a camera looking at `mitte`."""
    az, el = np.radians(azimut_grad), np.radians(hoehe_grad)
    richtung = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    auge = mitte + abstand * richtung
    vorne = mitte - auge
    vorne /= np.linalg.norm(vorne)
    rechts = np.cross(vorne, np.array([0.0, 0.0, 1.0]))
    rechts /= max(np.linalg.norm(rechts), 1e-9)
    oben = np.cross(rechts, vorne)
    return auge, np.stack([rechts, oben, vorne])


def projizieren(punkte: np.ndarray, auge: np.ndarray, achsen: np.ndarray,
                breite: int, hoehe: int, fov_grad: float) -> tuple[np.ndarray, np.ndarray]:
    """World points onto the image plane. Returns image coordinates and camera depth."""
    lokal = (punkte - auge) @ achsen.T
    tiefe = lokal[:, 2]
    f = (breite / 2.0) / np.tan(np.radians(fov_grad) / 2.0)
    sicher = np.maximum(tiefe, 1e-6)
    u = breite / 2.0 + f * lokal[:, 0] / sicher
    v = hoehe / 2.0 - f * lokal[:, 1] / sicher
    return np.stack([u, v], axis=1), tiefe


def boden_legen(leinwand: np.ndarray, frame_bgr: np.ndarray, ecken_welt: np.ndarray,
                auge, achsen, fov_grad: float, abdunkeln: float) -> None:
    """Place the frame into the scene as a ground plane."""
    hoehe, breite = leinwand.shape[:2]
    ziel, tiefe = projizieren(ecken_welt, auge, achsen, breite, hoehe, fov_grad)
    if (tiefe <= 0.1).any():
        return                      # a corner lies behind the camera
    fh, fw = frame_bgr.shape[:2]
    quelle = np.float32([[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]])
    matrix = cv2.getPerspectiveTransform(quelle, ziel.astype(np.float32))
    gelegt = cv2.warpPerspective(frame_bgr, matrix, (breite, hoehe), flags=cv2.INTER_LINEAR)
    maske = cv2.warpPerspective(np.full((fh, fw), 255, np.uint8), matrix, (breite, hoehe),
                                flags=cv2.INTER_NEAREST) > 0
    leinwand[maske] = (gelegt[maske].astype(np.float32) * abdunkeln).astype(np.uint8)


def punkte_setzen(leinwand: np.ndarray, bild: np.ndarray, tiefe: np.ndarray,
                  farben: np.ndarray, groesse: int) -> int:
    """Draw from back to front -- the front ones overwrite those behind."""
    hoehe, breite = leinwand.shape[:2]
    sichtbar = (tiefe > 0.1) & np.isfinite(bild).all(axis=1)
    if not sichtbar.any():
        return 0
    bild, tiefe, farben = bild[sichtbar], tiefe[sichtbar], farben[sichtbar]
    reihe = np.argsort(-tiefe)                      # farthest first
    u = np.round(bild[reihe, 0]).astype(np.int32)
    v = np.round(bild[reihe, 1]).astype(np.int32)
    f = farben[reihe]
    for dy in range(groesse):
        for dx in range(groesse):
            uu, vv = u + dx, v + dy
            drin = (uu >= 0) & (uu < breite) & (vv >= 0) & (vv < hoehe)
            leinwand[vv[drin], uu[drin]] = f[drin]
    return int(sichtbar.sum())


def hoehenfarbe(z: np.ndarray, unten: float, oben: float) -> np.ndarray:
    x = np.clip((z - unten) / max(oben - unten, 1e-6), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8).reshape(-1, 1),
                             cv2.COLORMAP_VIRIDIS).reshape(-1, 3)


def beschriftung(leinwand: np.ndarray, zeilen: list[str]) -> None:
    for i, text in enumerate(zeilen):
        cv2.putText(leinwand, text, (18, 34 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62 if i == 0 else 0.5, (240, 240, 240), 2 if i == 0 else 1, cv2.LINE_AA)


def szene_rendern(frame_bgr, xyz, farben, azimut, hoehe_grad, breite, hoehe, fov_grad,
                  abstand_faktor, punkt_groesse, abdunkeln, zeilen,
                  boden_ebene: float | None = None) -> np.ndarray:
    """`boden_ebene=None` places the frame below the deepest visible points.

    At 0 -- the estimated ground -- a gap of about 9 % of the flight altitude would
    otherwise open up, because the ground is not visible in the stand at all.
    Physically right, but misleading as an image: the texture shows crowns from above.
    """
    leinwand = np.full((hoehe, breite, 3), 18, np.uint8)
    mitte = np.array([xyz[:, 0].mean(), xyz[:, 1].mean(),
                      float(np.percentile(xyz[:, 2], 50))])
    spanne = max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]))
    auge, achsen = kamera(mitte, spanne * abstand_faktor, azimut, hoehe_grad)

    x0, x1 = xyz[:, 0].min(), xyz[:, 0].max()
    y0, y1 = xyz[:, 1].min(), xyz[:, 1].max()
    z_ebene = float(np.percentile(xyz[:, 2], 1)) if boden_ebene is None else boden_ebene
    ecken = np.array([[x0, y1, z_ebene], [x1, y1, z_ebene],
                      [x1, y0, z_ebene], [x0, y0, z_ebene]])
    boden_legen(leinwand, frame_bgr, ecken, auge, achsen, fov_grad, abdunkeln)

    bild, tiefe = projizieren(xyz, auge, achsen, breite, hoehe, fov_grad)
    punkte_setzen(leinwand, bild, tiefe, farben, punkt_groesse)
    beschriftung(leinwand, zeilen)
    return leinwand


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--karten", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft_karten"))
    parser.add_argument("--ft", type=Path,
                        default=Path("/scratch/shared/nik/runs/depthft/bestes"))
    parser.add_argument("--modellart", default="tiefe", choices=("tiefe", "hoehe"),
                        help="tiefe: via the depth, with a terrain model. hoehe: outputs metres "
                             "directly, but does not distinguish between stands.")
    parser.add_argument("--out", type=Path,
                        default=Path("/home/nik/workspace/TreeClassifier/results_depthft_3d"))
    parser.add_argument("--frames", nargs="*", default=None, metavar="ORDNER/DATEI")
    parser.add_argument("--hfov-deg", type=float, default=48.0)
    parser.add_argument("--schritt", type=int, default=2)
    parser.add_argument("--breite", type=int, default=1600)
    parser.add_argument("--hoehe", type=int, default=1000)
    parser.add_argument("--blickwinkel", type=float, nargs="*", default=[35.0, 20.0, 60.0],
                        help="Elevation angle of the stills in degrees above the horizon.")
    parser.add_argument("--azimut", type=float, default=225.0)
    parser.add_argument("--kamera-fov", type=float, default=45.0, help="Field of view of the viewing camera.")
    parser.add_argument("--abstand", type=float, default=1.5, help="Multiple of the scene width.")
    parser.add_argument("--punkt", type=int, default=2, help="Edge length of a point in pixels.")
    parser.add_argument("--boden-ebene", type=float, default=None,
                        help="Height of the ground plane in metres. Default: below the "
                             "deepest visible points. 0 places it at the estimated "
                             "ground, which is not visible inside the stand.")
    parser.add_argument("--boden-dunkel", type=float, default=0.45,
                        help="How much the ground image is darkened so that points stand out.")
    parser.add_argument("--faerben", default="hoehe", choices=("hoehe", "bild"),
                        help="hoehe: coloured by height. bild: original colours.")
    parser.add_argument("--max-neigung", type=float, default=8.0,
                        help="Discard points at depth jumps; 0 turns it off.")
    parser.add_argument("--min-hoehe", type=float, default=1.0,
                        help="Omit points below this -- otherwise the ground carpet hides everything.")
    parser.add_argument("--umlauf", action="store_true", help="Also render an orbiting video.")
    parser.add_argument("--umlauf-bilder", type=int, default=72)
    parser.add_argument("--altitudes", nargs="*", metavar="ORDNER=HOEHE",
                        default=["dense=51", "dense1=69", "mixed=92", "mixed1=103",
                                 "pines=60", "urban=120"])
    parser.add_argument("--altitude", type=float, default=100.0)
    parser.add_argument("--kachel-m", type=float, default=35.0)
    parser.add_argument("--boden-faktor", type=float, default=0.917)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    vorgaben = {e.split("=")[0]: float(e.split("=")[1]) for e in args.altitudes}
    k = inferenz.k_von_fov(args.hfov_deg)

    ordner = sorted(p for p in args.input.iterdir() if p.is_dir())
    if args.frames:
        auswahl = [args.input / f for f in args.frames]
    else:
        auswahl = []
        for o in ordner:
            treffer = sorted(f for f in o.iterdir() if f.suffix.lower() in BILDENDUNGEN)
            if treffer:
                auswahl.append(treffer[0])

    model = None
    for pfad in auswahl:
        ordnername = pfad.parent.name
        name = f"{ordnername}_{pfad.stem}"
        frame_bgr = cv2.imread(str(pfad))
        bild_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hinweis = kamera_pruefen(bild_rgb, pfad)
        if hinweis:
            print(f"  ACHTUNG {hinweis}", flush=True)

        H = vorgaben.get(ordnername)
        if H is None:
            m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*m?\s*", ordnername, re.IGNORECASE)
            H = float(m.group(1)) if m else args.altitude

        import torch
        if model is None:
            device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
            model = inferenz.lade(str(args.ft), device, fov_head=False)
        tiefe, hoehe_direkt = inferenz.karten_von(model, bild_rgb, art=args.modellart, k=k,
                                                  flughoehe_m=H, device=device)

        h, w = tiefe.shape
        f_px = k * w
        gsd = H / f_px
        # With the height model the ground reference is already in the prediction.
        boden = (tiefe + hoehe_direkt if hoehe_direkt is not None
                 else bodenmodell(tiefe, gsd, args.kachel_m, 97.0, args.boden_faktor))

        s = max(1, args.schritt)
        v, u = np.mgrid[0:h:s, 0:w:s].astype(np.float32)
        d = tiefe[::s, ::s]
        X = (u - (w - 1) / 2.0) * d / f_px
        Y = -(v - (h - 1) / 2.0) * d / f_px
        Z = boden[::s, ::s] - d
        xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
        rgb = bild_rgb[::s, ::s].reshape(-1, 3)

        behalten = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] >= args.min_hoehe)
        if args.max_neigung > 0:
            behalten &= kantenmaske(tiefe, gsd, args.max_neigung)[::s, ::s].ravel()
        xyz, rgb = xyz[behalten], rgb[behalten]
        if len(xyz) < 100:
            print(f"  {name}: zu wenige Punkte ueber {args.min_hoehe} m", flush=True)
            continue

        unten, oben = float(np.percentile(xyz[:, 2], 2)), float(np.percentile(xyz[:, 2], 98))
        farben = (hoehenfarbe(xyz[:, 2], unten, oben) if args.faerben == "hoehe"
                  else rgb[:, ::-1].astype(np.uint8))

        zeilen = [f"{ordnername}/{pfad.name}",
                  f"{len(xyz)} Punkte | Hoehe ueber Boden {unten:.1f} - {oben:.1f} m",
                  f"Flughoehe {H:.0f} m, Bildwinkel {args.hfov_deg:.0f} Grad, "
                  f"Flaeche {np.ptp(xyz[:, 0]):.0f} x {np.ptp(xyz[:, 1]):.0f} m"]

        for winkel in args.blickwinkel:
            leinwand = szene_rendern(frame_bgr, xyz, farben, args.azimut, winkel,
                                     args.breite, args.hoehe, args.kamera_fov,
                                     args.abstand, args.punkt, args.boden_dunkel,
                                     zeilen + [f"Blickwinkel {winkel:.0f} Grad ueber dem Horizont"],
                                     args.boden_ebene)
            cv2.imwrite(str(args.out / f"{name}_blick{int(winkel):02d}.jpg"), leinwand,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {name:32s} {len(xyz):7d} Punkte | Z {unten:5.1f} - {oben:5.1f} m | "
              f"{len(args.blickwinkel)} Ansichten", flush=True)

        if args.umlauf:
            ziel = args.out / f"{name}_umlauf.mp4"
            schreiber = cv2.VideoWriter(str(ziel), cv2.VideoWriter_fourcc(*"mp4v"), 24,
                                        (args.breite, args.hoehe))
            for i in range(args.umlauf_bilder):
                az = args.azimut + 360.0 * i / args.umlauf_bilder
                schreiber.write(szene_rendern(frame_bgr, xyz, farben, az, args.blickwinkel[0],
                                              args.breite, args.hoehe, args.kamera_fov,
                                              args.abstand, args.punkt, args.boden_dunkel, zeilen,
                                              args.boden_ebene))
            schreiber.release()
            print(f"    Umlauf -> {ziel.name}", flush=True)

    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
