"""Instanzmasse fuer Kronensegmentierung -- ohne pycocotools.

Zwei Zahlen, die verschiedene Fragen beantworten:

  F1 @ IoU 0.5   Wie viele Kronen sitzen bei der Schwelle, die man tatsaechlich
                 faehrt? Das ist die Zahl, die zaehlt, wenn hinterher je Krone
                 ein Ausschnitt an den Artklassifikator geht.
  AP @ IoU 0.5   Wie gut ist die Rangfolge ueber alle Schwellen? Unabhaengig von
                 der Wahl der Konfidenzschwelle und damit vergleichbar mit den
                 Zahlen, die Paper zu diesem Datensatz berichten.

Instanzen werden als `(box, maske im box-Ausschnitt, score)` gefuehrt. Ganze
2048x2048-Masken je Krone waeren bei 300 Vorhersagen pro Kachel 1.2 GB -- der
Ausschnitt kostet ein Fuenfzigstel davon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Instance:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1
    mask: np.ndarray                # bool, Form (y1-y0, x1-x0)
    score: float = 1.0

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def instance_from_mask(mask: np.ndarray, score: float = 1.0) -> Instance | None:
    """Vollbildmaske auf ihren Umriss zuschneiden."""
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
    """Gierige Zuordnung in der Reihenfolge fallender Konfidenz.

    Gibt `(treffer, iou_je_vorhersage)` zurueck, beides in der Reihenfolge der
    (nach Score sortierten) Vorhersagen. Ein GT wird hoechstens einmal belegt,
    Mehrfachtreffer zaehlen also als Fehlalarm -- genau wie in COCO.
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
    """101-Punkt-Interpolation ueber die Praezisions-Trefferquoten-Kurve (COCO)."""
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
        # Fuer die kachieluebergreifende AP: Treffer und zugehoerige Scores
        # roh mitgeben. Ohne sie liesse sich nur je Kachel eine AP rechnen und
        # mitteln -- das ist nicht dasselbe und nicht mit COCO vergleichbar.
        "_hits": hits,
        "_scores": np.array([predictions[i].score for i in order], dtype=np.float32),
    }


def accumulate(rows: list[dict[str, float]]) -> dict[str, float]:
    """Kachelweise Zaehlungen zu einer Zahl je Gebiet zusammenziehen.

    Die AP wird ueber alle Kacheln gemeinsam gerechnet, nicht je Kachel und dann
    gemittelt. Der Unterschied ist nicht klein: bei rund 20 Kronen je Kachel ist
    eine einzelne Kurve kurz und sprunghaft, und der Mittelwert solcher Kurven
    liegt systematisch unter der gemeinsamen. Nur die gemeinsame Variante ist
    das, was COCO und die Literatur unter AP50 berichten -- eine frueher hier
    gerechnete Mittelung war mit veroeffentlichten Zahlen nicht vergleichbar.
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
    """AP ueber alle Kacheln gemeinsam, nach Konfidenz sortiert."""
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
