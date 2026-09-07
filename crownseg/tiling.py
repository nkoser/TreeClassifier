"""Gleitendes Fenster und Ausgabeformate -- gemeinsam fuer alle Verfahren.

Herausgezogen, damit Mask R-CNN, EoMT und Mask2Former nachweislich dieselbe
Fensterlogik benutzen. Ein Vergleich, bei dem die Verfahren unterschiedlich
kacheln, misst die Kachelung mit.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

import metrics as met


def suppress(instances: list[met.Instance], threshold: float) -> list[met.Instance]:
    """Maskenbasierte Unterdrueckung -- raeumt Dopplungen aus dem Fensterrand."""
    kept: list[met.Instance] = []
    for candidate in sorted(instances, key=lambda i: -i.score):
        if all(met.iou(candidate, other) < threshold for other in kept):
            kept.append(candidate)
    return kept


def slide(image_rgb: np.ndarray, tile: int, overlap: int,
          predict_window: Callable[[np.ndarray], list[met.Instance]],
          nms: float = 0.5) -> list[met.Instance]:
    """Gleitendes Fenster, Zuordnung ueber den Kronenmittelpunkt.

    Jedes Fenster ist nur fuer seinen Kern zustaendig -- das Fenster ohne den
    halben Ueberlapp an den Seiten, an denen ein Nachbarfenster anschliesst. Die
    Kerne kacheln das Bild lueckenlos, jede Krone wird also genau einmal
    vergeben, naemlich dort, wo ihr Mittelpunkt liegt.

    Der Ueberlapp muss groesser sein als die groesste erwartete Krone
    (BAMFORESTS: p95 bei 842 px in Hain). Sonst ragen Kronen, deren Mittelpunkt
    im Kern liegt, ueber den Fensterrand hinaus und werden dort abgeschnitten.
    Eine Regel "verwirf alles, was den Rand beruehrt" waere falsch: eine Krone
    breiter als der Ueberlapp beruehrt in *jedem* Fenster einen Rand und
    verschwindet komplett.
    """
    height, width = image_rgb.shape[:2]
    step = max(1, tile - overlap)
    xs = list(range(0, max(1, width - overlap), step)) or [0]
    ys = list(range(0, max(1, height - overlap), step)) or [0]

    collected: list[met.Instance] = []
    for y in ys:
        for x in xs:
            y0, x0 = min(y, max(0, height - tile)), min(x, max(0, width - tile))
            window = image_rgb[y0 : y0 + tile, x0 : x0 + tile]
            if window.shape[0] < 32 or window.shape[1] < 32:
                continue
            margin = overlap // 2
            core = (
                x0 + (margin if x0 > 0 else 0),
                y0 + (margin if y0 > 0 else 0),
                x0 + window.shape[1] - (margin if x0 + window.shape[1] < width else 0),
                y0 + window.shape[0] - (margin if y0 + window.shape[0] < height else 0),
            )
            for instance in predict_window(window):
                bx0, by0, bx1, by1 = instance.box
                instance.box = (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0)
                center_x = (instance.box[0] + instance.box[2]) / 2
                center_y = (instance.box[1] + instance.box[3]) / 2
                if core[0] <= center_x < core[2] and core[1] <= center_y < core[3]:
                    collected.append(instance)

    return suppress(collected, nms) if len(xs) * len(ys) > 1 else collected


def to_label_map(instances: list[met.Instance], height: int, width: int) -> np.ndarray:
    """Fuer Weiterverarbeitung und Betrachter; Ueberlappungen gewinnt der Sicherere."""
    labels = np.zeros((height, width), dtype=np.uint16)
    for index, instance in enumerate(sorted(instances, key=lambda i: i.score), start=1):
        x0, y0, x1, y1 = instance.box
        # Instanzen aus hochskalierten Vorhersagen ragen nach dem Zurueckrechnen
        # gelegentlich um ein bis zwei Pixel ueber den Bildrand. Fenster und
        # Maske muessen deshalb gemeinsam beschnitten werden -- sonst passen die
        # Formen nicht mehr zueinander, und zwar abhaengig von der Rundung.
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(width, x1), min(height, y1)
        if cx0 >= cx1 or cy0 >= cy1:
            continue
        clipped = instance.mask[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        labels[cy0:cy1, cx0:cx1][clipped] = index
    return labels


def draw_overlay(image_bgr: np.ndarray, labels: np.ndarray, caption: str) -> np.ndarray:
    overlay = image_bgr.copy()
    kernel = np.ones((3, 3), np.uint8)
    borders = (cv2.dilate(labels, kernel) != cv2.erode(labels, kernel)) & (labels > 0)
    overlay[borders] = (80, 230, 120)
    cv2.rectangle(overlay, (0, 0), (360, 34), (0, 0, 0), -1)
    cv2.putText(overlay, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay
