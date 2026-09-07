"""Depth Pro auf FORTRESS feinabstimmen -- metrische Tiefe aus Nadiraufnahmen.

Warum ueberhaupt. Depth Pro ist auf Bodenperspektiven trainiert: Strassen,
Innenraeume, Portraits. Eine Nadiraufnahme aus 80 m kommt darin nicht vor, und
das sieht man den Ergebnissen an -- die Tiefe stimmt in der Struktur, aber die
Skala ist verschoben, und Kronen sitzen zu hoch. Genau das ist reparierbar,
wenn Wahrheit vorliegt: FORTRESS liefert zu jedem Bildpunkt die Hoehe ueber
Boden.

**In welchem Raum trainiert wird.** Depth Pro gibt nicht Meter aus, sondern
kanonische inverse Tiefe. Metrisch wird daraus erst im Nachlauf des Prozessors:

    d = (f_px / Bildbreite) / D_roh  =  k / D_roh

`k` haengt allein am Bildwinkel. Da wir bei unseren Frames die Kamera kennen,
geben wir `k` vor, statt es vom Bildwinkelkopf schaetzen zu lassen -- der bleibt
eingefroren und unveraendert im Checkpoint, damit er weiterhin zur Verfuegung
steht. Die Ausgabe bleibt kanonische inverse Tiefe, der Verlust wird aber nach
`d = k/D` und `h = H-d` in metrischer Hoehe berechnet. Der Checkpoint bleibt
mit den normalen Hugging-Face-Klassen ladbar. Fuer korrekte metrische Tiefe muss
der bekannte Bildwinkel jedoch weiterhin bei der Nachrechnung vorgegeben
werden; der eingefrorene Bildwinkelkopf ist fuer Nadirbilder unzuverlaessig.

**Der Verlust** hat zwei Teile. Ein Huber-Verlust misst den metrischen Fehler
der Hoehe `h = H - k/D` und optimiert damit unmittelbar die Zielgroesse. Eine
Gradientenanpassung ueber vier Skalen auf derselben Hoehendifferenz schaerft
Kronengrenzen. Ein frueherer Stand benutzte Log-Tiefe; das waere ein relativer
Fehler der Kameradistanz `H-h`, nicht der Baumhoehe, und war fuer dieses Ziel
deshalb falsch gewichtet.

**Warum der Kopf vorher vorgespannt wird.** Pures Depth Pro liegt bei diesem
Aufnahmefall um Faktor 50 daneben. Wegen `d = k/D` ist die Abbildung nahe null
sehr steil; ein Optimizer soll diesen grossen Skalenwechsel nicht erst durch
viele instabile Schritte lernen. Die Vorspannung setzt die Ausgabe vor dem
ersten Update in den physikalisch relevanten Bereich.

Der Ausweg ist, den Sprung gar nicht erst zu verlangen. `--vorspannen auto`
misst den Skalenfehler auf ein paar Stapeln und skaliert damit die letzte
Faltung des Kopfes. Weil sie eine 1x1-Faltung vor der abschliessenden ReLU ist
und der Faktor positiv, ist das exakt aequivalent zu `D -> faktor * D` -- aber
als echte Gewichtsaenderung. Der ausgelieferte Checkpoint bleibt damit ohne
Sonderweg brauchbar. Die Lernrate dieser einen Schicht wird mit demselben
Faktor skaliert, sonst rissen Adam-Schritte in gewohnter Groesse die nun um
Groessenordnungen kleineren Gewichte sofort auseinander.

**Was trainiert wird.** Vorgabe ist `decoder`: Nacken, Fusionsstufe und Kopf,
rund 60 M Parameter. Der Encoder laeuft eingefroren unter `no_grad` -- das
spart den Grossteil des Speichers und reicht, denn die Skala sitzt im Kopf,
nicht in den Merkmalen. `all` stimmt alles mit ab und braucht deutlich mehr GPU.

    python depthft/finetune.py --epochs 8
    python depthft/finetune.py --trainable all --batch 1 --accum 16
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import NadirFrames, auf_modell  # noqa: E402

MODELL = "apple/DepthPro-hf"
MIN_D = 1e-6      # kanonische inverse Tiefe; der Kopf endet auf ReLU, kann also 0 liefern
TIEFE_MIN_M = 2.0     # Sicherung: darunter haengt die Kamera im Baum,


def gruppen(model, was: str) -> list[torch.nn.Parameter]:
    """Welche Teile mitlernen. Der Bildwinkelkopf nie -- wir geben k vor."""
    for p in model.parameters():
        p.requires_grad_(False)
    teile = {
        "head": [model.head],
        "decoder": [model.depth_pro.neck, model.fusion_stage, model.head],
        "all": [model.depth_pro, model.fusion_stage, model.head],
    }[was]
    params = []
    for teil in teile:
        for p in teil.parameters():
            p.requires_grad_(True)
            params.append(p)
    if model.fov_model is not None:
        for p in model.fov_model.parameters():
            p.requires_grad_(False)
    return params


def vorhersage(model, pixel_values: torch.Tensor, encoder_frozen: bool) -> torch.Tensor:
    """Kanonische inverse Tiefe. Bei eingefrorenem Encoder ohne Aktivierungsspeicher."""
    if not encoder_frozen:
        return model(pixel_values=pixel_values).predicted_depth
    with torch.no_grad():
        encodings = model.depth_pro.encoder(pixel_values, return_dict=True)
    features = [f.detach() for f in encodings[1]]
    features = model.depth_pro.neck(features)
    return model.head(model.fusion_stage(features)[-1])


@torch.no_grad()
def vorspannen(model, lader, args, device, encoder_frozen: bool, stapel: int = 8) -> float:
    """Den Kopf auf den Aufnahmefall einstellen, bevor ueberhaupt trainiert wird.

    Gemessen wird der Skalenfehler `d_vorhergesagt / d_wahr` als Median ueber
    einige Stapel. Genau dieser Faktor geht in die letzte Faltung des Kopfes:
    ist die Vorhersage um Faktor 50 zu klein, wird die kanonische inverse Tiefe
    mit 0.02 skaliert, und die Tiefe stimmt in der Groessenordnung.
    """
    trainingsmodus(model, encoder_frozen, False)
    faktoren = []
    for i, batch in enumerate(lader):
        if i >= stapel:
            break
        d_gt = batch["tiefe"].to(device)
        maske = batch["maske"].to(device)
        k = batch["k"].to(device).view(-1, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            D = vorhersage(model, auf_modell(batch["bild"].to(device), args.model_size), encoder_frozen)
        D = D.float()
        if D.shape[-2:] != d_gt.shape[-2:]:
            D = F.interpolate(D.unsqueeze(1), size=d_gt.shape[-2:], mode="bilinear",
                              align_corners=False).squeeze(1)
        d_pred = k / D.clamp_min(MIN_D)
        if maske.sum() > 0:
            faktoren.append(float(torch.median(d_pred[maske] / d_gt[maske])))
    trainingsmodus(model, encoder_frozen, True)
    if not faktoren:
        return 1.0
    return float(torch.median(torch.tensor(faktoren)))


def kopf_skalieren(model, faktor: float) -> torch.nn.Conv2d:
    """Letzte 1x1-Faltung des Kopfes skalieren -- aequivalent zu `D -> faktor * D`.

    Sie sitzt vor der abschliessenden ReLU; ein positiver Faktor kommutiert mit
    dieser, die Skalierung ist also exakt und nicht bloss ungefaehr.
    """
    letzte = [schicht for schicht in model.head.layers if isinstance(schicht, torch.nn.Conv2d)][-1]
    with torch.no_grad():
        letzte.weight.mul_(faktor)
        if letzte.bias is not None:
            letzte.bias.mul_(faktor)
    return letzte


def gradientenanpassung(rest: torch.Tensor, maske: torch.Tensor, stufen: int = 4) -> torch.Tensor:
    """Mehrskalige Gradientenanpassung auf einer normierten Fehlerkarte.

    Auf jeder Stufe wird der Rest halb so fein betrachtet. Feine Stufen schaerfen
    Kronengrenzen, grobe verhindern ein langsames Wegdriften ueber das Bild.
    """
    verlust = rest.new_zeros(())
    for stufe in range(stufen):
        schritt = 2 ** stufe
        r, m = rest[:, ::schritt, ::schritt], maske[:, ::schritt, ::schritt]
        if r.shape[-1] < 4:
            break
        dx, mx = (r[:, :, 1:] - r[:, :, :-1]).abs(), m[:, :, 1:] & m[:, :, :-1]
        dy, my = (r[:, 1:] - r[:, :-1]).abs(), m[:, 1:] & m[:, :-1]
        n = mx.sum() + my.sum()
        if n > 0:
            verlust = verlust + ((dx * mx).sum() + (dy * my).sum()) / n
    return verlust


def kennzahlen(d_pred: torch.Tensor, d_gt: torch.Tensor, maske: torch.Tensor) -> dict[str, float]:
    """Metrische Guete in Metern.

    `mae_m` ist zugleich der Fehler der Hoehe ueber Boden: die ist Flughoehe
    minus Tiefe, und die Flughoehe kuerzt sich in der Differenz heraus. Also
    genau die Zahl, um die es bei der Baumhoehe geht.
    """
    if maske.sum() == 0:
        return {}
    p, g = d_pred[maske], d_gt[maske]
    verhaeltnis = torch.maximum(p / g, g / p)
    return {
        "absrel": float(((p - g).abs() / g).mean()),
        "rmse_m": float(torch.sqrt(((p - g) ** 2).mean())),
        "mae_m": float((p - g).abs().mean()),
        "bias_m": float((p - g).mean()),
        "delta125": float((verhaeltnis < 1.25).float().mean()),
        "log_rmse": float(torch.sqrt(((p.log() - g.log()) ** 2).mean())),
    }


def durchlauf(model, batch, args, device, encoder_frozen: bool):
    """Ein Vorwaertsschritt: Verlust und Kennzahlen."""
    bild = batch["bild"].to(device, non_blocking=True)
    d_gt = batch["tiefe"].to(device, non_blocking=True)
    maske = batch["maske"].to(device, non_blocking=True)
    k = batch["k"].to(device, non_blocking=True).view(-1, 1, 1)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pixel_values = auf_modell(bild, args.model_size)
        D = vorhersage(model, pixel_values, encoder_frozen)

    D = D.float()
    if D.shape[-2:] != d_gt.shape[-2:]:
        D = F.interpolate(D.unsqueeze(1), size=d_gt.shape[-2:], mode="bilinear",
                          align_corners=False).squeeze(1)

    # Vorwaerts den exakten metrischen Hoehenfehler benutzen. Rueckwaerts ist
    # die Ableitung von k/D nahe D=0 singulaer; deshalb bekommt der Fehler den
    # am Ziel D_gt linearisierten Jacobian d_gt**2/k. Der Forward-Wert und damit
    # der optimierte Huber-Loss bleiben exakt, der Gradient ist jedoch endlich
    # und zeigt auch bei groben Fehlern in die richtige Richtung.
    D_roh = D.clamp_min(MIN_D)
    d_pred = k / D_roh
    D_gt = k / d_gt.clamp_min(TIEFE_MIN_M)
    rest_exakt = d_gt - d_pred
    rest_linear = (D - D_gt) * (d_gt.square() / k.clamp_min(MIN_D))
    rest_m = rest_linear + (rest_exakt - rest_linear).detach()

    # h_pred - h_gt = (H-d_pred) - (H-d_gt) = d_gt-d_pred. H kuerzt sich
    # algebraisch, der Fehler ist aber in Metern und nicht relativ zur grossen
    # Kameradistanz gewichtet.
    n = maske.sum().clamp_min(1)
    huber = (F.smooth_l1_loss(rest_m, torch.zeros_like(rest_m),
                             reduction="none", beta=args.huber_beta) * maske).sum()
    huber = huber / n / args.bezugshoehe
    grad = gradientenanpassung(rest_m / args.bezugshoehe, maske, args.grad_stufen)
    verlust = huber + args.lambda_grad * grad
    return (verlust, huber.detach(), grad.detach(), d_pred.detach(), d_gt, maske)


def lade_modell(quelle: str, device, args):
    from transformers import DepthProForDepthEstimation
    model = DepthProForDepthEstimation.from_pretrained(quelle, dtype=torch.float32).to(device)
    if args.grad_checkpointing and args.trainable == "all":
        # Nur dann sinnvoll: bei eingefrorenem Encoder faellt der grosse
        # Aktivierungsspeicher ohnehin nicht an.
        try:
            model.gradient_checkpointing_enable()
        except Exception as fehler:      # nicht jede Version kann das
            print(f"Gradient-Checkpointing nicht verfuegbar: {fehler}", flush=True)
    return model


def trainingsmodus(model, encoder_frozen: bool, an: bool) -> None:
    """Ein eingefrorener Encoder bleibt auch im Training in `eval`."""
    model.train(an)
    if encoder_frozen:
        model.depth_pro.encoder.eval()


def bauen_loader(args, split: str, augment: bool, pro_gebiet: int, seed: int):
    """Zum Validieren feste Ausschnitte, aber in der Schaerfe der Zielframes.

    Die Ausschnitte stehen fest (kein Spiegeln, fester Wurf), damit Epochen
    vergleichbar bleiben. Weichgezeichnet wird trotzdem: sonst misst die
    Validierung gestochen scharfe Orthoausschnitte, waehrend Training und
    Anwendung mit Videobildern zu tun haben.
    """
    daten = NadirFrames(
        args.data, split, crop_px=args.crop_px, seitenverhaeltnis=args.seitenverhaeltnis,
        hoehe_min=args.hoehe_min, hoehe_max=args.hoehe_max, abstand_min=args.abstand_min,
        fov_min=args.fov_min, fov_max=args.fov_max,
        pro_gebiet=pro_gebiet, min_gueltig=args.min_gueltig,
        augment=augment, domaene=augment or args.val_domaene,
        jitter=args.jitter, video=args.video,
        strahl_tiefe=args.strahl_tiefe, cache=args.cache_sites, seed=seed,
        site_block=args.site_block,
    )
    # shuffle=False mit Absicht: NadirFrames mischt reproduzierbar in kurzen
    # Gebietsblocks. Ein zweites Mischen hier wuerde den Rastercache zerstoeren.
    lader = torch.utils.data.DataLoader(
        daten, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=augment, persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    return daten, lader


@torch.no_grad()
def bewerten(model, lader, args, device, encoder_frozen: bool) -> dict[str, float]:
    trainingsmodus(model, encoder_frozen, False)
    summe, anzahl = {}, 0
    for batch in lader:
        verlust, huber, grad, d_pred, d_gt, maske = durchlauf(model, batch, args, device, encoder_frozen)
        werte = kennzahlen(d_pred, d_gt, maske)
        werte["verlust"] = float(verlust.detach())
        werte["huber_hoehe"] = float(huber)
        for name, wert in werte.items():
            summe[name] = summe.get(name, 0.0) + wert
        anzahl += 1
    trainingsmodus(model, encoder_frozen, True)
    return {name: wert / max(1, anzahl) for name, wert in summe.items()}


def start_epoche_bekannt(args) -> bool:
    """Beim Fortsetzen ist der Kopf schon vorgespannt -- nicht ein zweites Mal."""
    pfad = args.out / "verlauf.json"
    return pfad.exists() and bool(json.loads(pfad.read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/scratch/shared/nik/data/fortress/depthft"))
    parser.add_argument("--out", type=Path, default=Path("/scratch/shared/nik/runs/depthft_huber_v2"))
    parser.add_argument("--start", default=MODELL, help="Ausgangsgewichte oder ein eigener Checkpoint.")
    parser.add_argument("--trainable", default="decoder", choices=("head", "decoder", "all"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8, help="Schritte bis zur Gewichtsaktualisierung.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-encoder", type=float, default=1e-5, help="Nur bei --trainable all.")
    parser.add_argument("--vorspannen", default="auto",
                        help="'auto' misst den Skalenfehler und stellt den Kopf darauf ein, "
                             "eine Zahl setzt den Faktor fest, 'aus' laesst den Kopf wie er ist.")
    parser.add_argument("--vorspann-stapel", type=int, default=12)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.05, help="Anteil der Schritte zum Aufwaermen.")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lambda-grad", type=float, default=0.5)
    parser.add_argument("--grad-stufen", type=int, default=4)
    parser.add_argument("--bezugshoehe", type=float, default=20.0,
                        help="Normierung des metrischen Hoehen-Losses in Metern.")
    parser.add_argument("--huber-beta", type=float, default=2.0,
                        help="Breite des quadratischen Huber-Bereichs in Metern.")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--crop-px", type=int, default=1536,
                        help="Bildbreite der Ausschnitte -- zugleich die Aufloesung, in der "
                             "Verlust und Wahrheit leben. 1536 heisst: kein Umskalieren "
                             "zwischen Ausschnitt und Modelleingang.")
    parser.add_argument("--seitenverhaeltnis", type=float, default=16 / 9,
                        help="Breite durch Hoehe der Ausschnitte. Vorgabe ist das unserer Frames.")
    parser.add_argument("--model-size", type=int, default=1536, help="Eingangsgroesse von Depth Pro.")
    parser.add_argument("--pro-gebiet", type=int, default=200, help="Ausschnitte je Gebiet und Epoche.")
    parser.add_argument("--pro-gebiet-val", type=int, default=40)
    parser.add_argument("--val-scharf", dest="val_domaene", action="store_false", default=True,
                        help="Validierung auf ungefilterten Orthoausschnitten statt in der "
                             "Schaerfe der Zielframes. Misst dann eine Verteilung, die im "
                             "Einsatz nicht vorkommt.")
    parser.add_argument("--hoehe-min", type=float, default=25.0)
    parser.add_argument("--hoehe-max", type=float, default=120.0)
    parser.add_argument("--abstand-min", type=float, default=20.0,
                        help="Mindestabstand der Kamera ueber dem hoechsten Wipfel des Gebietes.")
    parser.add_argument("--fov-min", type=float, default=35.0)
    parser.add_argument("--fov-max", type=float, default=85.0)
    parser.add_argument("--min-gueltig", type=float, default=0.80)
    parser.add_argument("--jitter", type=float, default=1.0)
    parser.add_argument("--video", type=float, default=1.0)
    parser.add_argument("--strahl-tiefe", action="store_true",
                        help="Tiefe entlang des Sehstrahls statt entlang der optischen Achse.")
    parser.add_argument("--melden", type=int, default=50, help="Zwischenstand alle N Batches.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-sites", type=int, default=4,
                        help="Rastercache je Worker und Zahl gemeinsam gemischter Gebiete.")
    parser.add_argument("--site-block", type=int, default=32,
                        help="Maximal aufeinanderfolgende Samples desselben Gebiets.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    if args.resume and (args.out / "verlauf.json").exists():
        meta_pfad = args.out / "bestes" / "depthft.json"
        meta = json.loads(meta_pfad.read_text()) if meta_pfad.exists() else {}
        if meta.get("trainingsformat_version") != 3:
            raise SystemExit(
                "Dieser Lauf stammt von einer inkompatiblen Loss-Version. "
                "Bitte einen neuen --out-Ordner verwenden."
            )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    letztes, bestes = args.out / "letztes", args.out / "bestes"

    quelle = str(letztes) if (args.resume and letztes.exists()) else args.start
    model = lade_modell(quelle, device, args)
    params = gruppen(model, args.trainable)
    encoder_frozen = args.trainable != "all"
    # Der Bildwinkelkopf wird im Training nicht gerechnet -- er ist eingefroren
    # und kostet einen zweiten Encoderdurchlauf. Die Gewichte bleiben erhalten
    # und werden beim Ablegen wieder mitgeschrieben.
    model.use_fov_model = False
    trainingsmodus(model, encoder_frozen, True)

    n_trainierbar = sum(p.numel() for p in params)
    print(f"Device: {device} | Start: {quelle}", flush=True)
    print(f"Trainierbar ({args.trainable}): {n_trainierbar/1e6:.1f} M von "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f} M", flush=True)

    train_daten, train_lader = bauen_loader(args, "train", True, args.pro_gebiet, args.seed)
    val_daten, val_lader = bauen_loader(args, "val", False, args.pro_gebiet_val, 12345)

    # Den Kopf auf den Aufnahmefall einstellen, bevor der Optimierer gebaut wird
    # -- die skalierte Schicht braucht eine eigene, mitskalierte Lernrate.
    vorgespannt, kopf_schicht = 1.0, None
    fortsetzung = args.resume and start_epoche_bekannt(args)
    if fortsetzung:
        beschreibung = bestes / "depthft.json"
        if beschreibung.exists():
            vorgespannt = float(json.loads(beschreibung.read_text()).get("vorgespannt", 1.0))
            kopf_schicht = kopf_skalieren(model, 1.0)   # nur greifen, nicht skalieren
            print(f"Fortgesetzt; Kopf war mit {vorgespannt:.4f} vorgespannt.", flush=True)
    elif args.vorspannen != "aus":
        if args.vorspannen == "auto":
            vorgespannt = vorspannen(model, train_lader, args, device, encoder_frozen,
                                     args.vorspann_stapel)
            print(f"Skalenfehler gemessen: {vorgespannt:.4f} "
                  f"(1.00 waere richtig, gemessen ueber {args.vorspann_stapel} Stapel)", flush=True)
        else:
            vorgespannt = float(args.vorspannen)
        if not (0 < vorgespannt < 1e6):
            print(f"Faktor {vorgespannt} unbrauchbar -- Kopf bleibt unveraendert.", flush=True)
            vorgespannt = 1.0
        if vorgespannt != 1.0:
            kopf_schicht = kopf_skalieren(model, vorgespannt)
            print(f"Kopf mit {vorgespannt:.4f} vorgespannt; Lernrate dieser Schicht "
                  f"{args.lr * vorgespannt:.2e} statt {args.lr:.2e}", flush=True)

    kopf_ids = set() if kopf_schicht is None else {id(p) for p in kopf_schicht.parameters()}
    gruppe_kopf = [p for p in params if id(p) in kopf_ids]
    uebrige = [p for p in params if id(p) not in kopf_ids]
    if args.trainable == "all":
        encoder_ids = {id(p) for p in model.depth_pro.encoder.parameters()}
        gruppen_liste = [
            {"params": [p for p in uebrige if id(p) in encoder_ids], "lr": args.lr_encoder},
            {"params": [p for p in uebrige if id(p) not in encoder_ids], "lr": args.lr},
        ]
    else:
        gruppen_liste = [{"params": uebrige, "lr": args.lr}]
    if gruppe_kopf:
        gruppen_liste.append({"params": gruppe_kopf, "lr": args.lr * vorgespannt})
    optimizer = torch.optim.AdamW([g for g in gruppen_liste if g["params"]],
                                  weight_decay=args.weight_decay)
    schritte_je_epoche = max(1, len(train_lader) // args.accum)
    gesamt = schritte_je_epoche * args.epochs
    aufwaermen = max(1, int(args.warmup * gesamt))
    print(f"Train {len(train_daten)} Ausschnitte aus {len(train_daten.sites)} Gebieten "
          f"({', '.join(train_daten.sites[:6])}...)", flush=True)
    print(f"Val   {len(val_daten)} Ausschnitte aus {len(val_daten.sites)} Gebieten "
          f"({', '.join(val_daten.sites)})", flush=True)
    print(f"{gesamt} Aktualisierungen, {aufwaermen} davon zum Aufwaermen\n", flush=True)

    def lr_faktor(schritt: int) -> float:
        if schritt < aufwaermen:
            return (schritt + 1) / aufwaermen
        anteil = (schritt - aufwaermen) / max(1, gesamt - aufwaermen)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, anteil)))

    plan = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_faktor)
    verlauf_pfad = args.out / "verlauf.json"
    verlauf = json.loads(verlauf_pfad.read_text()) if (args.resume and verlauf_pfad.exists()) else []
    bester_wert = min((e["val"]["mae_m"] for e in verlauf if e.get("val")), default=float("inf"))
    start_epoche = len(verlauf)

    # Beim Fortsetzen muessen Momente und Lernrate mit: sonst faengt der Plan
    # wieder beim Aufwaermen an und AdamW ohne Momente reisst die Gewichte an.
    zustand = args.out / "optimierer.pt"
    if start_epoche and zustand.exists():
        optimizer.load_state_dict(torch.load(zustand, map_location=device, weights_only=False))
        for _ in range(start_epoche * schritte_je_epoche):
            plan.step()
        print(f"Fortgesetzt ab Epoche {start_epoche}, lr {plan.get_last_lr()[0]:.2e}", flush=True)
    elif start_epoche:
        print(f"Fortgesetzt ab Epoche {start_epoche}, aber ohne {zustand.name} -- "
              "Momente und Lernrate beginnen neu.", flush=True)

    for epoche in range(start_epoche, args.epochs):
        train_daten.set_epoch(epoche)
        t0, laufend, gesehen = time.time(), {}, 0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_lader):
            verlust, huber, grad, d_pred, d_gt, maske = durchlauf(model, batch, args, device, encoder_frozen)
            (verlust / args.accum).backward()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                plan.step()

            werte = kennzahlen(d_pred, d_gt, maske)
            werte["verlust"] = float(verlust.detach())
            werte["huber_hoehe"], werte["grad"] = float(huber), float(grad)
            for name, wert in werte.items():
                laufend[name] = laufend.get(name, 0.0) + wert
            gesehen += 1
            if gesehen % max(1, args.melden) == 0:
                mittel = {n: w / gesehen for n, w in laufend.items()}
                print(f"  Epoche {epoche} {i+1}/{len(train_lader)} | "
                      f"Verlust {mittel['verlust']:.4f} | AbsRel {mittel.get('absrel', 0):.3f} | "
                      f"MAE {mittel.get('mae_m', 0):.2f} m | lr {plan.get_last_lr()[0]:.2e}", flush=True)

        train_mittel = {n: w / max(1, gesehen) for n, w in laufend.items()}
        val_daten.set_epoch(0)      # feste Ausschnitte, damit Epochen vergleichbar sind
        val_mittel = bewerten(model, val_lader, args, device, encoder_frozen)
        print(f"Epoche {epoche} in {(time.time()-t0)/60:.1f} min | "
              f"train AbsRel {train_mittel.get('absrel', 0):.3f} | "
              f"val AbsRel {val_mittel.get('absrel', 0):.3f}, MAE {val_mittel.get('mae_m', 0):.2f} m, "
              f"Bias {val_mittel.get('bias_m', 0):+.2f} m, d1 {val_mittel.get('delta125', 0):.3f}", flush=True)

        verlauf.append({"epoche": epoche, "train": train_mittel, "val": val_mittel})
        verlauf_pfad.write_text(json.dumps(verlauf, indent=2))

        model.use_fov_model = model.fov_model is not None   # vollstaendig ablegen
        model.save_pretrained(letztes)
        torch.save(optimizer.state_dict(), args.out / "optimierer.pt")
        if val_mittel.get("mae_m", float("inf")) < bester_wert:
            bester_wert = val_mittel["mae_m"]
            model.save_pretrained(bestes)
            (bestes / "depthft.json").write_text(json.dumps({
                "epoche": epoche, "val": val_mittel, "trainable": args.trainable,
                "crop_px": args.crop_px, "seitenverhaeltnis": args.seitenverhaeltnis,
                "hoehe_min": args.hoehe_min, "hoehe_max": args.hoehe_max,
                "abstand_min": args.abstand_min,
                "fov_min": args.fov_min, "fov_max": args.fov_max,
                "strahl_tiefe": args.strahl_tiefe, "vorgespannt": vorgespannt,
                "trainingsformat_version": 3,
                "auswahlmetrik": "mae_m",
                "loss": "huber_hoehe", "huber_beta": args.huber_beta,
                "bezugshoehe": args.bezugshoehe, "lambda_grad": args.lambda_grad,
                "grad_stufen": args.grad_stufen, "lr": args.lr,
                "lr_encoder": args.lr_encoder, "weight_decay": args.weight_decay,
                "batch": args.batch, "accum": args.accum, "seed": args.seed,
                "jitter": args.jitter, "video": args.video,
                "site_block": args.site_block,
                "train_gebiete": train_daten.sites,
            }, indent=2))
            print(f"  neuer Bestwert -> {bestes}", flush=True)
        model.use_fov_model = False

    print(f"\nFertig. Bester Hoehen-MAE {bester_wert:.3f} m -> {bestes}")


if __name__ == "__main__":
    main()
