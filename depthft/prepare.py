"""Bring FORTRESS sites into a ground raster that virtual frames are cut from.

FORTRESS supplies one orthomosaic per site (0.77 to 1.57 cm ground sampling) and
a normalised height model (nDSM, metres above ground). Together they are the
truth Depth Pro lacks: the actual height for every pixel.

You cannot train on it directly, though. An orthomosaic is not a photograph -- it
has no camera, no field of view, no depth. Depth only arises from an assumption:
hang a nadir camera at height H above the stand, and the depth at every point is

    d = H - nDSM

and the ground sampling of the resulting image is H / f_px. A site thus yields
arbitrarily many frames at any flight altitude -- with a depth map.

This script does the expensive part once: read the 40 GB of orthos, bring them
onto a common raster at a fixed resolution (default 2 cm, fine enough for any
flight altitude wanted later) and fit the nDSM onto it. `dataset.py` then cuts
the frames out of it in seconds.

The split is by **site**, not by crop: crops of the same stand are too similar,
and a random split would flatter the result. The same principle as in
`distill_height.py`.

    python depthft/prepare.py --sites CFB014 CFB019
    python depthft/prepare.py                          # all 47 sites
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Every 9th site goes into evaluation, not into training. The rest trains.
VAL_JEDES = 9      # Index % VAL_JEDES == 0  -> val
TEST_VERSATZ = 4   # Index % VAL_JEDES == TEST_VERSATZ -> test

MAX_HOEHE_M = 60.0   # cap for the nDSM; no tree in the Black Forest is taller


def render_site(ortho: Path, ndsm_pfad: Path, base_gsd: float, *,
                null_schwelle: float = 0.01, nullen_verwerfen: bool = True
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Ortho and nDSM onto a common raster at `base_gsd` metres per pixel."""
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(ortho) as src:
        gsd = abs(src.transform.a)
        breite = max(1, int(round(src.width * gsd / base_gsd)))
        hoehe = max(1, int(round(src.height * gsd / base_gsd)))
        # Resampling.average rather than nearest: when downscaling by a factor of
        # 2 to 8, nearest would throw away every second to eighth pixel and create
        # aliasing -- crown structure that never existed.
        rgb = np.transpose(src.read((1, 2, 3), out_shape=(3, hoehe, breite),
                                    resampling=Resampling.average), (1, 2, 0))
        rgb = np.ascontiguousarray(rgb.astype(np.uint8))
        if src.count >= 4:
            alpha = src.read(4, out_shape=(hoehe, breite), resampling=Resampling.nearest)
            gueltig = alpha > 0
        else:
            gueltig = rgb.max(axis=2) > 0
        ziel_transform = src.transform * Affine.scale(src.width / breite, src.height / hoehe)
        ziel_crs = src.crs

    with rasterio.open(ndsm_pfad) as nsrc:
        if ziel_crs is None or nsrc.crs is None:
            raise SystemExit(f"Ohne Koordinatensystem laesst sich {ndsm_pfad.name} nicht "
                             f"auf {ortho.name} einpassen (Ortho: {ziel_crs}, nDSM: {nsrc.crs}).")
        ndsm = np.full((hoehe, breite), np.nan, np.float32)
        reproject(
            source=rasterio.band(nsrc, 1), destination=ndsm,
            src_transform=nsrc.transform, src_crs=nsrc.crs, src_nodata=nsrc.nodata,
            dst_transform=ziel_transform, dst_crs=ziel_crs, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        ndsm_gsd = abs(nsrc.transform.a)

    # Invalid is anything without an image, without a height value, or with a
    # nonsensical height. Slightly negative values are normal roughness in the
    # terrain model and are pulled to 0; strongly negative ones indicate errors.
    gueltig &= np.isfinite(ndsm) & (ndsm > -3.0) & (ndsm < MAX_HOEHE_M * 1.5)

    # Exact zeros are fill, not terrain. In the FORTRESS height models 0.00 is by
    # far the most frequent single value -- 9 to 27 % of the area, in large
    # contiguous blocks under which the orthomosaic shows closed forest. Real
    # ground scatters around 0, it does not hit it exactly tens of thousands of
    # times. Learning those areas as ground would be exactly the wrong truth:
    # crowns at height zero.
    # The price is that real ground pixels get discarded too -- but they
    # contribute little, while mislabelled forest does immediate harm.
    anteil_null = 0.0
    if nullen_verwerfen:
        fuellung = np.abs(np.nan_to_num(ndsm, nan=1e3)) < null_schwelle
        anteil_null = float(fuellung.mean())
        # Dilate by a few pixels: at the edge the resampling mixes fill with real
        # values and creates a seam just above the threshold.
        fuellung = cv2.dilate(fuellung.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        gueltig &= ~fuellung

    ndsm = np.clip(np.nan_to_num(ndsm, nan=0.0), 0.0, MAX_HOEHE_M)
    werte = ndsm[gueltig]
    meta = {
        "ortho_gsd_m": float(gsd),
        "ndsm_gsd_m": float(ndsm_gsd),
        "base_gsd_m": float(base_gsd),
        "breite_px": int(breite),
        "hoehe_px": int(hoehe),
        "breite_m": float(breite * base_gsd),
        "hoehe_m": float(hoehe * base_gsd),
        "anteil_gueltig": float(gueltig.mean()),
        "anteil_null": anteil_null,
        "hoehe_p50_m": float(np.percentile(werte, 50)) if werte.size else 0.0,
        "hoehe_p95_m": float(np.percentile(werte, 95)) if werte.size else 0.0,
        "hoehe_max_m": float(werte.max()) if werte.size else 0.0,
    }
    return rgb, ndsm, gueltig, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/scratch/shared/nik/data/fortress"))
    parser.add_argument("--out", type=Path, default=Path("/scratch/shared/nik/data/fortress/depthft"))
    parser.add_argument("--sites", nargs="*", default=None, help="Default: all of them.")
    parser.add_argument("--base-gsd", type=float, default=0.02,
                        help="Resolution of the ground raster in m/px. Finer than any "
                             "flight altitude needed later demands, but not so fine "
                             "that the storage explodes.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--null-schwelle", type=float, default=0.01,
                        help="Heights below this count as fill, not as ground.")
    parser.add_argument("--nullen-behalten", dest="nullen_verwerfen", action="store_false",
                        help="Keep the fill areas -- for comparison only.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    orthos = args.root / "orthomosaic/orthomosaic"
    ndsms = args.root / "nDSM/nDSM"
    alle = sorted(p.stem.replace("_ortho", "") for p in orthos.glob("*_ortho.tif"))
    if not alle:
        raise SystemExit(f"Keine Orthos unter {orthos}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raster").mkdir(exist_ok=True)

    # The split depends on the complete site list, not on --sites. Otherwise a
    # partial run would have a different split from the full one.
    split_von = {}
    for i, site in enumerate(alle):
        rest = i % VAL_JEDES
        split_von[site] = "val" if rest == 0 else "test" if rest == TEST_VERSATZ else "train"

    sites = args.sites or alle
    index_pfad = args.out / "index.json"
    index = json.loads(index_pfad.read_text()) if index_pfad.exists() else {}

    for site in sites:
        ortho, ndsm_pfad = orthos / f"{site}_ortho.tif", ndsms / f"nDSM_{site}.tif"
        if not ortho.exists() or not ndsm_pfad.exists():
            print(f"{site}: Ortho oder nDSM fehlt -- uebersprungen", flush=True)
            continue
        ziel = args.out / "raster" / f"{site}_rgb.jpg"
        if ziel.exists() and not args.overwrite and site in index:
            print(f"{site}: vorhanden", flush=True)
            continue

        rgb, ndsm, gueltig, meta = render_site(ortho, ndsm_pfad, args.base_gsd,
                                               null_schwelle=args.null_schwelle,
                                               nullen_verwerfen=args.nullen_verwerfen)
        cv2.imwrite(str(ziel), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        # Height in centimetres as uint16: 1 cm resolution is ample for tree
        # heights and costs half of float32.
        cv2.imwrite(str(args.out / "raster" / f"{site}_ndsm.png"),
                    np.round(ndsm * 100.0).astype(np.uint16))
        cv2.imwrite(str(args.out / "raster" / f"{site}_valid.png"),
                    gueltig.astype(np.uint8) * 255)

        meta["split"] = split_von[site]
        index[site] = meta
        index_pfad.write_text(json.dumps(dict(sorted(index.items())), indent=2))
        print(f"{site:8s} {meta['split']:5s} | Ortho {meta['ortho_gsd_m']*100:.2f} cm, "
              f"nDSM {meta['ndsm_gsd_m']*100:.1f} cm | {meta['breite_m']:.0f}x{meta['hoehe_m']:.0f} m "
              f"| gueltig {meta['anteil_gueltig']*100:.0f} % (Fuellung {meta['anteil_null']*100:.0f} %) "
              f"| Hoehe p95 {meta['hoehe_p95_m']:.1f} m",
              flush=True)

    zaehler = {"train": 0, "val": 0, "test": 0}
    for site, meta in index.items():
        zaehler[meta["split"]] += 1
    print(f"\n{len(index)} Gebiete -> {args.out}")
    print(f"  train {zaehler['train']} | val {zaehler['val']} | test {zaehler['test']}")


if __name__ == "__main__":
    main()
