"""Sliding window and output formats -- shared by every method.

Pulled out so that Mask R-CNN, EoMT and Mask2Former demonstrably use the same
window logic. A comparison in which the methods tile differently measures the
tiling as well.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

import metrics as met


def suppress(instances: list[met.Instance], threshold: float) -> list[met.Instance]:
    """Mask-based suppression -- clears duplicates coming from the window margin."""
    kept: list[met.Instance] = []
    for candidate in sorted(instances, key=lambda i: -i.score):
        if all(met.iou(candidate, other) < threshold for other in kept):
            kept.append(candidate)
    return kept


def slide(image_rgb: np.ndarray, tile: int, overlap: int,
          predict_window: Callable[[np.ndarray], list[met.Instance]],
          nms: float = 0.5) -> list[met.Instance]:
    """Sliding window, assignment via the crown centroid.

    Every window is responsible only for its core -- the window minus half the
    overlap on the sides where a neighbouring window adjoins. The cores tile the
    image without gaps, so every crown is assigned exactly once, namely where its
    centroid lies.

    The overlap has to be larger than the largest expected crown (BAMFORESTS: p95
    at 842 px in Hain). Otherwise crowns whose centroid lies in the core extend
    beyond the window edge and get cut off there. A rule "discard everything that
    touches the edge" would be wrong: a crown wider than the overlap touches an
    edge in *every* window and disappears completely.
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
    """For further processing and viewing; on overlap the more confident one wins."""
    labels = np.zeros((height, width), dtype=np.uint16)
    for index, instance in enumerate(sorted(instances, key=lambda i: i.score), start=1):
        x0, y0, x1, y1 = instance.box
        # Instances from upscaled predictions occasionally stick out one or two
        # pixels beyond the image border after being converted back. Window and
        # mask therefore have to be clipped together -- otherwise the shapes no
        # longer match, in a way that depends on the rounding.
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
