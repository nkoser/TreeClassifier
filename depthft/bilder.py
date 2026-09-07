"""Hoehenkarten sichtbar machen -- Farbskala, Reliefschattierung, Beschriftung."""

from __future__ import annotations

import cv2
import numpy as np


def hoehenbild(hoehe: np.ndarray, obergrenze: float, untergrenze: float = 0.0) -> np.ndarray:
    """Hoehe ueber Boden farbig, mit vorgegebener Skala fuer die Vergleichbarkeit.

    NaN wird schwarz -- so ist im Bild zu sehen, wo gar keine Wahrheit vorliegt,
    statt dass die Luecke als Boden durchgeht.
    """
    spanne = max(obergrenze - untergrenze, 1e-6)
    fehlt = ~np.isfinite(hoehe)
    x = np.clip((np.nan_to_num(hoehe) - untergrenze) / spanne, 0, 1)
    bild = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    bild[fehlt] = 0
    return bild


def hillshade(flaeche: np.ndarray, azimut_grad: float = 315.0, hoehe_grad: float = 45.0,
              ueberhoehung: float = 40.0) -> np.ndarray:
    """Reliefschattierung -- macht feine Hoehenunterschiede fuer das Auge sichtbar."""
    dy, dx = np.gradient(cv2.GaussianBlur(flaeche.astype(np.float32), (0, 0), 2.0))
    neigung = np.arctan(np.hypot(dx, dy) * ueberhoehung)
    richtung = np.arctan2(-dx, dy)
    az, alt = np.radians(360.0 - azimut_grad + 90.0), np.radians(hoehe_grad)
    schatten = np.sin(alt) * np.cos(neigung) + np.cos(alt) * np.sin(neigung) * np.cos(az - richtung)
    return np.clip(schatten, 0, 1)


def grauwert(x: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def beschriften(bild: np.ndarray, text: str, zweite: str | None = None) -> np.ndarray:
    """Kopfzeile ins Bild, damit ein Vergleichsstreifen ohne Legende lesbar ist."""
    bild = np.ascontiguousarray(bild)
    hoehe = 34 if zweite is None else 58
    cv2.rectangle(bild, (0, 0), (bild.shape[1], hoehe), (0, 0, 0), -1)
    cv2.putText(bild, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if zweite:
        cv2.putText(bild, zweite, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 180), 1, cv2.LINE_AA)
    return bild


def balkendiagramm(werte: list[tuple[str, float, tuple[int, int, int]]], breite: int,
                   hoehe: int, titel: str) -> np.ndarray:
    """Waagerechte Balken mit Zahl dran -- die Aussage ohne Farbskalen-Umweg."""
    bild = np.full((hoehe, breite, 3), 22, np.uint8)
    cv2.putText(bild, titel, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
    hoechster = max(max((w for _, w, _ in werte), default=1.0), 1e-6)
    links, oben = 300, 58
    balken_h = max(28, (hoehe - 80) // max(len(werte), 1) - 16)
    for i, (name, wert, farbe) in enumerate(werte):
        y = oben + i * (balken_h + 16)
        laenge = max(3, int((wert / hoechster) * (breite - links - 220)))
        cv2.rectangle(bild, (links, y), (links + laenge, y + balken_h), farbe, -1)
        cv2.putText(bild, name, (24, y + balken_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(bild, f"{wert:.1f} m", (links + laenge + 16, y + balken_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, farbe, 2, cv2.LINE_AA)
    return bild


def einfache_abbildung(rgb_bgr: np.ndarray, karten: list, kopf: str, unterzeile: str = "Originalaufnahme",
                       ziel_h: int = 500, balken_titel: str =
                       "Wie hoch sind die Baeume? (Wipfel minus Kronenluecke, in Metern)") -> np.ndarray:
    """Jede Karte mit eigener Skala -- und der Massstab als Balken darunter.

    Eine gemeinsame Farbskala ist sachlich richtig, macht die Kachel des puren
    Modells aber zu einer einfarbigen Flaeche: es liegt um Faktor 50 daneben,
    also faellt alles jenseits des Skalenendes zusammen. Eine Abbildung, in der
    man nichts erkennt, erklaert nichts -- deshalb hier jede Karte gespreizt auf
    ihren eigenen Wertebereich, der als Text dabeisteht, und der eigentliche
    Unterschied als Balken.
    """
    h, w = rgb_bgr.shape[:2]
    faktor = ziel_h / h
    klein = lambda a: cv2.resize(a, (int(w * faktor), ziel_h), interpolation=cv2.INTER_AREA)  # noqa: E731

    kacheln = [beschriften(klein(rgb_bgr), kopf, unterzeile)]
    balken = []
    for name, karte, farbe in karten:
        unten, oben = float(np.nanpercentile(karte, 2)), float(np.nanpercentile(karte, 98))
        if oben - unten < 1e-3:
            oben = unten + 1.0
        kacheln.append(beschriften(hoehenbild(klein(karte), oben, unten), name,
                                   f"gespreizt auf {unten:.1f} - {oben:.1f} m"))
        balken.append((name, oben - unten, farbe))

    streifen = np.hstack(kacheln)
    return np.vstack([streifen, balkendiagramm(balken, streifen.shape[1], 230, balken_titel)])


def farbskala(breite: int, hoehe: int, obergrenze: float, schritte: int = 5) -> np.ndarray:
    """Senkrechter Farbkeil mit Beschriftung in Metern."""
    keil = hoehenbild(np.linspace(obergrenze, 0, hoehe)[:, None].repeat(breite, 1), obergrenze)
    for i in range(schritte + 1):
        y = int(i * (hoehe - 1) / schritte)
        wert = obergrenze * (1 - i / schritte)
        cv2.putText(keil, f"{wert:.0f}", (4, max(12, min(hoehe - 4, y + 5))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return keil
