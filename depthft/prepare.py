"""FORTRESS-Gebiete in ein Bodenraster bringen, aus dem virtuelle Frames entstehen.

FORTRESS liefert je Gebiet ein Orthomosaik (0.77 bis 1.57 cm Bodenaufloesung) und
ein normalisiertes Hoehenmodell (nDSM, Meter ueber Boden). Beides zusammen ist
die Wahrheit, die Depth Pro fehlt: zu jedem Bildpunkt die tatsaechliche Hoehe.

Nur direkt trainieren laesst sich damit nicht. Ein Orthomosaik ist kein Foto --
es hat keine Kamera, keinen Bildwinkel, keine Tiefe. Die entsteht erst durch eine
Annahme: haenge eine Nadirkamera in Hoehe H ueber den Bestand, dann ist die Tiefe
an jedem Punkt

    d = H - nDSM

und die Bodenaufloesung des so entstehenden Bildes ist H / f_px. Aus einem Gebiet
werden damit beliebig viele Frames in beliebiger Flughoehe -- mit Tiefenkarte.

Dieses Skript macht den teuren Teil einmal: die 40 GB Orthos einlesen, auf ein
gemeinsames Raster mit fester Aufloesung bringen (Vorgabe 2 cm, fein genug fuer
jede spaeter gewuenschte Flughoehe) und das nDSM darauf einpassen. Die Frames
schneidet dann `dataset.py` im Sekundenbereich daraus.

Aufgeteilt wird nach **Gebieten**, nicht nach Ausschnitten: Ausschnitte desselben
Bestandes sind sich zu aehnlich, eine zufaellige Aufteilung wuerde die Guete
schoenrechnen. Dasselbe Prinzip wie in `distill_height.py`.

    python depthft/prepare.py --sites CFB014 CFB019
    python depthft/prepare.py                          # alle 47 Gebiete
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Alle 9 Gebiete gehen in die Auswertung, nicht ins Training. Der Rest trainiert.
VAL_JEDES = 9      # Index % VAL_JEDES == 0  -> val
TEST_VERSATZ = 4   # Index % VAL_JEDES == TEST_VERSATZ -> test

MAX_HOEHE_M = 60.0   # Deckel fuer das nDSM; hoeher wird im Schwarzwald kein Baum.


def render_site(ortho: Path, ndsm_pfad: Path, base_gsd: float, *,
                null_schwelle: float = 0.01, nullen_verwerfen: bool = True
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Ortho und nDSM auf ein gemeinsames Raster mit `base_gsd` Metern pro Pixel."""
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(ortho) as src:
        gsd = abs(src.transform.a)
        breite = max(1, int(round(src.width * gsd / base_gsd)))
        hoehe = max(1, int(round(src.height * gsd / base_gsd)))
        # Resampling.average statt nearest: beim Verkleinern um Faktor 2 bis 8
        # wuerde nearest jedes zweite bis achte Pixel wegwerfen und Aliasing
        # erzeugen -- Kronenstruktur, die es so nie gab.
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

    # Ungueltig ist alles ohne Bild, ohne Hoehenwert oder mit unsinniger Hoehe.
    # Leicht negative Werte sind normale Rauheit im Bodenmodell und werden auf 0
    # gezogen; stark negative deuten auf Fehler in der Photogrammetrie.
    gueltig &= np.isfinite(ndsm) & (ndsm > -3.0) & (ndsm < MAX_HOEHE_M * 1.5)

    # Exakte Nullen sind Fuellung, nicht Gelaende. In den FORTRESS-Hoehenmodellen
    # ist 0.00 der mit Abstand haeufigste Einzelwert -- 9 bis 27 % der Flaeche,
    # in grossen zusammenhaengenden Bloecken, unter denen im Orthomosaik
    # geschlossener Wald steht. Echter Boden streut um 0 herum, er trifft ihn
    # nicht Zehntausende Male exakt. Diese Flaechen als Boden zu lernen waere
    # genau die falsche Wahrheit: Kronen auf Hoehe null.
    # Der Preis ist, dass echte Bodenpixel mit verworfen werden -- die tragen
    # aber wenig, waehrend falsch beschrifteter Wald unmittelbar schadet.
    anteil_null = 0.0
    if nullen_verwerfen:
        fuellung = np.abs(np.nan_to_num(ndsm, nan=1e3)) < null_schwelle
        anteil_null = float(fuellung.mean())
        # Ein paar Pixel weiten: an der Kante mischt die Neuabtastung Fuellung
        # mit echten Werten und erzeugt einen Saum knapp ueber der Schwelle.
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
    parser.add_argument("--sites", nargs="*", default=None, help="Vorgabe: alle.")
    parser.add_argument("--base-gsd", type=float, default=0.02,
                        help="Aufloesung des Bodenrasters in m/px. Feiner als jede "
                             "spaeter gebrauchte Flughoehe verlangt, aber nicht so fein, "
                             "dass die Ablage explodiert.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--null-schwelle", type=float, default=0.01,
                        help="Hoehen darunter gelten als Fuellung, nicht als Boden.")
    parser.add_argument("--nullen-behalten", dest="nullen_verwerfen", action="store_false",
                        help="Fuellflaechen nicht verwerfen -- nur zum Vergleichen.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    orthos = args.root / "orthomosaic/orthomosaic"
    ndsms = args.root / "nDSM/nDSM"
    alle = sorted(p.stem.replace("_ortho", "") for p in orthos.glob("*_ortho.tif"))
    if not alle:
        raise SystemExit(f"Keine Orthos unter {orthos}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raster").mkdir(exist_ok=True)

    # Der Split haengt an der vollstaendigen Gebietsliste, nicht an --sites.
    # Sonst haette ein Teillauf eine andere Aufteilung als der volle.
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
        # Hoehe in Zentimetern als uint16: 1 cm Aufloesung reicht fuer Baumhoehen
        # bei Weitem und kostet die Haelfte von float32.
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
