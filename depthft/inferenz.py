"""Apply Depth Pro -- with a supplied camera instead of an estimated field of view.

Depth Pro returns canonical inverse depth. Metres come from it via

    d = k / D_roh        mit  k = f_px / Bildbreite = 0.5 / tan(HFOV / 2)

The field-of-view head of the model will estimate `k` if you let it. With a drone
whose lens is known that is the worse choice: the head is trained on ground
perspectives and is regularly wrong on nadir captures from 80 m -- and an error
in `k` enters every depth linearly.

So `k` can be supplied here. The fine-tuning was done exactly that way too.

This module is deliberately lean and free of project dependencies, so that it can
ship alongside the fine-tuned checkpoint.
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
    """Focal length divided by image width."""
    return 0.5 / math.tan(math.radians(fov_grad) / 2.0)


def fov_von_k(k: float) -> float:
    return 2.0 * math.degrees(math.atan(0.5 / k))


def gsd_von_flughoehe(flughoehe_m: float, fov_grad: float, breite_px: int) -> float:
    """Ground sampling in m/px for a nadir view."""
    return 2.0 * flughoehe_m * math.tan(math.radians(fov_grad) / 2.0) / breite_px


def lade(quelle: str, device, *, fov_head: bool = False):
    """Load the model. `fov_head=False` saves a second encoder pass."""
    from transformers import DepthProForDepthEstimation

    model = DepthProForDepthEstimation.from_pretrained(quelle, dtype=torch.float32).to(device).eval()
    model.use_fov_model = bool(fov_head) and model.fov_model is not None
    return model


@torch.no_grad()
def roh(model, bild_rgb: np.ndarray, *, device=None, modellgroesse: int = MODELLGROESSE,
        bf16: bool = True) -> tuple[np.ndarray, float | None]:
    """Canonical inverse depth at original size, plus the estimated field of view.

    Separate from `tiefe`, so that the same prediction can be evaluated against
    several assumptions about the camera without running the model again.
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
    """Metric depth map at original size, plus the `k` that was used.

    `k=None` means: ask the field-of-view head. The model then has to be loaded
    with `fov_head=True`.
    """
    D, fov = roh(model, bild_rgb, device=device, modellgroesse=modellgroesse, bf16=bf16)
    if k is None:
        if fov is None:
            raise ValueError("Kein Bildwinkel vorhergesagt -- Modell mit fov_head=True laden oder k vorgeben.")
        k = k_von_fov(fov)
    return k / D, float(k)


def hoehe_ueber_boden(tiefe_m: np.ndarray, flughoehe_m: float) -> np.ndarray:
    """Depth becomes height as soon as the flight altitude is known."""
    return flughoehe_m - tiefe_m


def karten_von(model, bild_rgb: np.ndarray, *, art: str, k: float | None = None,
               flughoehe_m: float | None = None, device=None,
               modellgroesse: int = MODELLGROESSE) -> tuple[np.ndarray, np.ndarray | None]:
    """Depth and height above ground -- from the depth or from the height model.

    There are two checkpoints with the same architecture but different outputs:

    `art="tiefe"`   The output is canonical inverse depth; metres come from
                    `d = k / D`. The height stays open -- it needs either the
                    flight altitude or a terrain model.
    `art="hoehe"`   The output is directly the height above ground in metres.
                    Neither field of view nor ground reference is needed. The
                    depth follows from `d = H - h` and is only needed for the
                    back-projection into 3D.

    Returns `(tiefe_m, hoehe_m_oder_None)`.
    """
    karte, _ = roh(model, bild_rgb, device=device, modellgroesse=modellgroesse)
    if art == "hoehe":
        if flughoehe_m is None:
            raise ValueError("Fuer die Tiefe aus dem Hoehenmodell wird die Flughoehe gebraucht.")
        return (flughoehe_m - karte).astype(np.float32), karte.astype(np.float32)
    if k is None:
        raise ValueError("Fuer das Tiefenmodell wird k gebraucht.")
    return (k / karte).astype(np.float32), None
