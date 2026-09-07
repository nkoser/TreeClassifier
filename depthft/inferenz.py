"""Depth Pro anwenden -- mit vorgegebener Kamera statt geschaetztem Bildwinkel.

Depth Pro liefert kanonische inverse Tiefe. Meter werden daraus ueber

    d = k / D_roh        mit  k = f_px / Bildbreite = 0.5 / tan(HFOV / 2)

Der Bildwinkelkopf des Modells schaetzt `k` mit, wenn man ihn laesst. Bei einer
Drohne mit bekanntem Objektiv ist das die schlechtere Wahl: der Kopf ist auf
Bodenperspektiven trainiert und liegt bei Nadiraufnahmen aus 80 m regelmaessig
daneben -- und ein Fehler in `k` geht linear in jede Tiefe ein.

Deshalb ist `k` hier vorgabefaehig. Genau so wurde auch feinabgestimmt.

Dieses Modul ist bewusst schlank und ohne Projektabhaengigkeiten, damit es dem
feinabgestimmten Checkpoint beigelegt werden kann.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

MODELL_MITTEL = 0.5
MODELL_STREUUNG = 0.5
MODELLGROESSE = 1536
MIN_D = 1e-6


def k_von_fov(fov_grad: float) -> float:
    """Brennweite geteilt durch Bildbreite."""
    return 0.5 / math.tan(math.radians(fov_grad) / 2.0)


def fov_von_k(k: float) -> float:
    return 2.0 * math.degrees(math.atan(0.5 / k))


def gsd_von_flughoehe(flughoehe_m: float, fov_grad: float, breite_px: int) -> float:
    """Bodenaufloesung in m/px bei Nadirblick."""
    return 2.0 * flughoehe_m * math.tan(math.radians(fov_grad) / 2.0) / breite_px


def lade(quelle: str, device, *, fov_head: bool = False):
    """Modell laden. `fov_head=False` spart einen zweiten Encoderdurchlauf."""
    from transformers import DepthProForDepthEstimation

    model = DepthProForDepthEstimation.from_pretrained(quelle, dtype=torch.float32).to(device).eval()
    model.use_fov_model = bool(fov_head) and model.fov_model is not None
    return model


@torch.no_grad()
def roh(model, bild_rgb: np.ndarray, *, device=None, modellgroesse: int = MODELLGROESSE,
        bf16: bool = True) -> tuple[np.ndarray, float | None]:
    """Kanonische inverse Tiefe in Originalgroesse, dazu der geschaetzte Bildwinkel.

    Getrennt von `tiefe`, damit dieselbe Vorhersage gegen mehrere Annahmen ueber
    die Kamera gerechnet werden kann, ohne das Modell erneut laufen zu lassen.
    """
    device = device or next(model.parameters()).device
    h, w = bild_rgb.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(bild_rgb.transpose(2, 0, 1)))[None].to(device)
    x = x.float().div_(255.0)
    x = F.interpolate(x, size=(modellgroesse, modellgroesse), mode="bilinear",
                      align_corners=False, antialias=True)
    x = (x - MODELL_MITTEL) / MODELL_STREUUNG

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
        ausgabe = model(pixel_values=x)

    D = ausgabe.predicted_depth.float()
    if D.dim() == 2:
        D = D[None]
    D = F.interpolate(D[:, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    fov = None if ausgabe.field_of_view is None else float(ausgabe.field_of_view[0])
    return D.clamp_min(MIN_D).cpu().numpy(), fov


def tiefe(model, bild_rgb: np.ndarray, *, k: float | None = None, device=None,
          modellgroesse: int = MODELLGROESSE, bf16: bool = True) -> tuple[np.ndarray, float]:
    """Metrische Tiefenkarte in Originalgroesse plus das benutzte `k`.

    `k=None` heisst: den Bildwinkelkopf fragen. Dafuer muss das Modell mit
    `fov_head=True` geladen sein.
    """
    D, fov = roh(model, bild_rgb, device=device, modellgroesse=modellgroesse, bf16=bf16)
    if k is None:
        if fov is None:
            raise ValueError("Kein Bildwinkel vorhergesagt -- Modell mit fov_head=True laden oder k vorgeben.")
        k = k_von_fov(fov)
    return k / D, float(k)


def hoehe_ueber_boden(tiefe_m: np.ndarray, flughoehe_m: float) -> np.ndarray:
    """Aus Tiefe wird Hoehe, sobald die Flughoehe bekannt ist."""
    return flughoehe_m - tiefe_m


def karten_von(model, bild_rgb: np.ndarray, *, art: str, k: float | None = None,
               flughoehe_m: float | None = None, device=None,
               modellgroesse: int = MODELLGROESSE) -> tuple[np.ndarray, np.ndarray | None]:
    """Tiefe und Hoehe ueber Boden -- aus dem Tiefen- oder dem Hoehenmodell.

    Es gibt zwei Checkpoints, die dieselbe Architektur, aber verschiedene
    Ausgaben haben:

    `art="tiefe"`   Die Ausgabe ist kanonische inverse Tiefe; Meter entstehen
                    ueber `d = k / D`. Die Hoehe bleibt offen -- sie braucht
                    entweder die Flughoehe oder ein Gelaendemodell.
    `art="hoehe"`   Die Ausgabe ist unmittelbar die Hoehe ueber Boden in Metern.
                    Weder Bildwinkel noch Bodenbezug noetig. Die Tiefe folgt aus
                    `d = H - h` und wird nur fuer die Rueckprojektion nach 3D
                    gebraucht.

    Gibt `(tiefe_m, hoehe_m_oder_None)` zurueck.
    """
    karte, _ = roh(model, bild_rgb, device=device, modellgroesse=modellgroesse)
    if art == "hoehe":
        if flughoehe_m is None:
            raise ValueError("Fuer die Tiefe aus dem Hoehenmodell wird die Flughoehe gebraucht.")
        return (flughoehe_m - karte).astype(np.float32), karte.astype(np.float32)
    if k is None:
        raise ValueError("Fuer das Tiefenmodell wird k gebraucht.")
    return (k / karte).astype(np.float32), None
