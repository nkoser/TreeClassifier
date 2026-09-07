"""Instance metrics for crown segmentation -- without pycocotools.

Two numbers that answer different questions:

  F1 @ IoU 0.5   How many crowns are right at the threshold actually used? That
                 is the number that counts when a crop per crown later goes to
                 the species classifier.
  AP @ IoU 0.5   How good is the ranking across all thresholds? Independent of
                 the choice of confidence threshold, and therefore comparable
                 with the numbers papers report on this dataset.

Instances are carried as `(box, mask within the box crop, score)`. Full
2048x2048 masks per crown would be 1.2 GB at 300 predictions per tile -- the
crop costs a fiftieth of that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Instance:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1
    mask: np.ndarray                # bool, shape (y1-y0, x1-x0)
    score: float = 1.0

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def instance_from_mask(mask: np.ndarray, score: float = 1.0) -> Instance | None:
    """Crop a full-frame mask down to its own outline."""
    mask = mask.astype(bool)
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return Instance((x0, y0, x1, y1), mask[y0:y1, x0:x1], score)


def iou(a: Instance, b: Instance) -> float:
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    overlap = int(np.logical_and(
        a.mask[y0 - ay0 : y1 - ay0, x0 - ax0 : x1 - ax0],
        b.mask[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0],
    ).sum())
    if not overlap:
        return 0.0
    return overlap / (a.area + b.area - overlap)


def match(predictions: list[Instance], truth: list[Instance], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy assignment in order of descending confidence.

    Returns `(treffer, iou_je_vorhersage)`, both in the order of the predictions
    (sorted by score). A ground-truth crown is claimed at most once, so multiple
    hits count as false alarms -- exactly as in COCO.
    """
    order = np.argsort([-p.score for p in predictions])
    hits = np.zeros(len(predictions), dtype=bool)
    scores = np.zeros(len(predictions), dtype=np.float32)
    taken = np.zeros(len(truth), dtype=bool)

    for rank, index in enumerate(order):
        best, best_iou = -1, threshold
        for j, gt in enumerate(truth):
            if taken[j]:
                continue
            value = iou(predictions[index], gt)
            if value >= best_iou:
                best, best_iou = j, value
        if best >= 0:
            taken[best] = True
            hits[rank] = True
            scores[rank] = best_iou
    return hits, scores


def average_precision(hits: np.ndarray, n_truth: int) -> float:
    """101-point interpolation over the precision-recall curve (COCO)."""
    if not n_truth:
        return float("nan")
    if not len(hits):
        return 0.0
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    recall = tp / n_truth
    precision = tp / np.maximum(1, tp + fp)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    return float(np.interp(grid, recall, precision, left=precision[0], right=0.0).mean())


def evaluate(predictions: list[Instance], truth: list[Instance],
             threshold: float = 0.5) -> dict[str, float]:
    hits, ious = match(predictions, truth, threshold)
    order = np.argsort([-p.score for p in predictions])
    n_hit = int(hits.sum())
    precision = n_hit / max(1, len(predictions))
    recall = n_hit / max(1, len(truth))
    return {
        "n_pred": len(predictions),
        "n_true": len(truth),
        "treffer": n_hit,
        "praezision": precision,
        "trefferquote": recall,
        "f1": 2 * precision * recall / max(1e-9, precision + recall),
        "mittlere_iou": float(ious[hits].mean()) if n_hit else 0.0,
        "ap": average_precision(hits, len(truth)),
        # For the AP pooled across tiles: pass hits and their scores through raw.
        # Without them only a per-tile AP could be computed and averaged -- that
        # is not the same thing and not comparable with COCO.
        "_hits": hits,
        "_scores": np.array([predictions[i].score for i in order], dtype=np.float32),
    }


def accumulate(rows: list[dict[str, float]]) -> dict[str, float]:
    """Collapse per-tile counts into one number per area.

    The AP is computed over all tiles jointly, not per tile and then averaged.
    The difference is not small: at around 20 crowns per tile a single curve is
    short and jumpy, and the mean of such curves lies systematically below the
    joint one. Only the joint variant is what COCO and the literature report as
    AP50 -- an averaging computed here earlier was not comparable with published
    numbers.
    """
    n_pred = sum(r["n_pred"] for r in rows)
    n_true = sum(r["n_true"] for r in rows)
    treffer = sum(r["treffer"] for r in rows)
    precision = treffer / max(1, n_pred)
    recall = treffer / max(1, n_true)
    weights = np.array([r["treffer"] for r in rows], dtype=np.float64)
    return {
        "kacheln": len(rows),
        "kronen_gt": n_true,
        "kronen_pred": n_pred,
        "praezision": precision,
        "trefferquote": recall,
        "f1": 2 * precision * recall / max(1e-9, precision + recall),
        "mittlere_iou": float(np.average([r["mittlere_iou"] for r in rows], weights=weights))
        if weights.sum() else 0.0,
        "ap50": pooled_ap(rows),
    }


def pooled_ap(rows: list[dict]) -> float:
    """AP over all tiles jointly, sorted by confidence."""
    if not rows or "_hits" not in rows[0]:
        return float(np.mean([r["ap"] for r in rows])) if rows else float("nan")
    hits = np.concatenate([r["_hits"] for r in rows if len(r["_hits"])]) if any(
        len(r["_hits"]) for r in rows) else np.zeros(0, bool)
    scores = np.concatenate([r["_scores"] for r in rows if len(r["_scores"])]) if any(
        len(r["_scores"]) for r in rows) else np.zeros(0, np.float32)
    n_truth = sum(r["n_true"] for r in rows)
    if not len(hits):
        return 0.0
    order = np.argsort(-scores)
    return average_precision(hits[order], n_truth)
