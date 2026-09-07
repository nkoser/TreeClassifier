"""Tune Depth Pro on the **height above ground** instead of on the depth.

So far the model predicts depth and we compute the height from it. That detour
costs twice.

**It optimises the wrong thing.** The loss minimises the depth error, while what
is of interest is the height error. The two are related, but not favourably: the
height is a difference of two large numbers (80 m altitude minus 60 m depth gives
a 20 m tree). A depth error of 5 m carries undiminished into the height, where it
weighs four times as much in relative terms. Measured: a 12 % relative error on
the depth becomes about 19 % on the height.

**It needs two numbers we do not know.** Depth becomes height only with a field
of view and a ground reference. Our field of view is back-calculated and accurate
to 10 %, the ground reference is estimated. Both enter every height linearly.

If the model predicts the height directly, both drop away. That is precisely a
**canopy height model**, and FORTRESS supplies the truth for it as an nDSM.

**The output space.** The head of Depth Pro ends on a ReLU, so it outputs nothing
negative -- exactly right for heights above ground. Its output is read directly
as metres, without conversion. Pre-scaling works as in the depth training, only
against the height.

**The loss** is L1 on metres, divided by a reference height so that the magnitude
is around one and `--lambda-grad` keeps the same meaning as in the depth
training. L1 on metres and not on the logarithm, because the height goes to zero
at the ground and because the error at issue is measured in metres.

    python depthft/finetune_hoehe.py --epochs 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import NadirFrames, auf_modell  # noqa: E402
from finetune import (  # noqa: E402
    MODELL, gradientenanpassung, gruppen, kopf_skalieren, lade_modell,
    start_epoche_bekannt, trainingsmodus, vorhersage,
)

MIN_HOEHE_FUERS_MESSEN = 2.0   # below knee height the ratio is not meaningful


def kennzahlen(h_pred: torch.Tensor, h_gt: torch.Tensor, maske: torch.Tensor) -> dict[str, float]:
    """Quality in metres. `mae_m` is the number that matters."""
    if maske.sum() == 0:
        return {}
    p, g = h_pred[maske], h_gt[maske]
    fehler = p - g
    # Correlation over the deviations from the mean -- measures whether the relief
    # is right, independently of a uniform offset.
    pz, gz = p - p.mean(), g - g.mean()
    nenner = (pz.norm() * gz.norm()).clamp_min(1e-6)
    hoch = g > MIN_HOEHE_FUERS_MESSEN
    return {
        "mae_m": float(fehler.abs().mean()),
        "rmse_m": float(torch.sqrt((fehler ** 2).mean())),
        "bias_m": float(fehler.mean()),
        "korrelation": float((pz * gz).sum() / nenner),
        "relativ": float((fehler[hoch].abs() / g[hoch]).mean()) if hoch.any() else float("nan"),
    }


def durchlauf(model, batch, args, device, encoder_frozen: bool):
    bild = batch["bild"].to(device, non_blocking=True)
    h_gt = batch["hoehe"].to(device, non_blocking=True)
    maske = batch["maske"].to(device, non_blocking=True)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        D = vorhersage(model, auf_modell(bild, args.model_size), encoder_frozen)

    h_pred = D.float()
    if h_pred.shape[-2:] != h_gt.shape[-2:]:
        h_pred = F.interpolate(h_pred.unsqueeze(1), size=h_gt.shape[-2:], mode="bilinear",
                               align_corners=False).squeeze(1)
    h_pred = h_pred.clamp(0.0, args.hoehe_deckel)

    rest = h_pred - h_gt
    n = maske.sum().clamp_min(1)
    l1 = (rest.abs() * maske).sum() / n / args.bezugshoehe
    grad = gradientenanpassung(rest / args.bezugshoehe, maske, args.grad_stufen)
    return l1 + args.lambda_grad * grad, l1.detach(), grad.detach(), h_pred.detach(), h_gt, maske


@torch.no_grad()
def vorspannen(model, lader, args, device, encoder_frozen: bool, stapel: int) -> float:
    """Set the head to metres before training starts.

    Measurement runs over pixels with an appreciable height: at ground level the
    ratio of prediction to truth is not meaningful.
    """
    trainingsmodus(model, encoder_frozen, False)
    faktoren = []
    for i, batch in enumerate(lader):
        if i >= stapel:
            break
        h_gt = batch["hoehe"].to(device)
        maske = batch["maske"].to(device) & (h_gt > MIN_HOEHE_FUERS_MESSEN)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            D = vorhersage(model, auf_modell(batch["bild"].to(device), args.model_size), encoder_frozen)
        h_pred = D.float()
        if h_pred.shape[-2:] != h_gt.shape[-2:]:
            h_pred = F.interpolate(h_pred.unsqueeze(1), size=h_gt.shape[-2:], mode="bilinear",
                                   align_corners=False).squeeze(1)
        if maske.sum() > 0:
            faktoren.append(float(torch.median(h_pred[maske].clamp_min(1e-4) / h_gt[maske])))
    trainingsmodus(model, encoder_frozen, True)
    return float(torch.median(torch.tensor(faktoren))) if faktoren else 1.0


def bauen_loader(args, split: str, augment: bool, pro_gebiet: int, seed: int):
    daten = NadirFrames(
        args.data, split, crop_px=args.crop_px, seitenverhaeltnis=args.seitenverhaeltnis,
        hoehe_min=args.hoehe_min, hoehe_max=args.hoehe_max, abstand_min=args.abstand_min,
        fov_min=args.fov_min, fov_max=args.fov_max,
        pro_gebiet=pro_gebiet, min_gueltig=args.min_gueltig,
        augment=augment, domaene=augment or args.val_domaene,
        jitter=args.jitter, video=args.video, cache=args.cache_sites, seed=seed,
    )
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
        verlust, l1, grad, h_pred, h_gt, maske = durchlauf(model, batch, args, device, encoder_frozen)
        werte = kennzahlen(h_pred, h_gt, maske)
        werte["verlust"] = float(verlust.detach())
        for name, wert in werte.items():
            summe[name] = summe.get(name, 0.0) + wert
        anzahl += 1
    trainingsmodus(model, encoder_frozen, True)
    return {name: wert / max(1, anzahl) for name, wert in summe.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/scratch/shared/nik/data/fortress/depthft"))
    parser.add_argument("--out", type=Path, default=Path("/scratch/shared/nik/runs/depthft_hoehe"))
    parser.add_argument("--start", default=MODELL)
    parser.add_argument("--trainable", default="decoder", choices=("head", "decoder", "all"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lambda-grad", type=float, default=0.5)
    parser.add_argument("--grad-stufen", type=int, default=4)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--bezugshoehe", type=float, default=20.0,
                        help="The loss is divided by this, so that it sits around one.")
    parser.add_argument("--hoehe-deckel", type=float, default=70.0,
                        help="Cap on the prediction; no tree in the Black Forest is taller.")
    parser.add_argument("--vorspannen", default="auto")
    parser.add_argument("--vorspann-stapel", type=int, default=12)
    parser.add_argument("--crop-px", type=int, default=1536)
    parser.add_argument("--seitenverhaeltnis", type=float, default=16 / 9)
    parser.add_argument("--model-size", type=int, default=1536)
    parser.add_argument("--pro-gebiet", type=int, default=200)
    parser.add_argument("--pro-gebiet-val", type=int, default=40)
    parser.add_argument("--val-scharf", dest="val_domaene", action="store_false", default=True)
    parser.add_argument("--hoehe-min", type=float, default=25.0)
    parser.add_argument("--hoehe-max", type=float, default=120.0)
    parser.add_argument("--abstand-min", type=float, default=20.0)
    parser.add_argument("--fov-min", type=float, default=35.0)
    parser.add_argument("--fov-max", type=float, default=85.0)
    parser.add_argument("--min-gueltig", type=float, default=0.50)
    parser.add_argument("--jitter", type=float, default=1.0)
    parser.add_argument("--video", type=float, default=1.0)
    parser.add_argument("--melden", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-sites", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    letztes, bestes = args.out / "letztes", args.out / "bestes"

    quelle = str(letztes) if (args.resume and letztes.exists()) else args.start
    model = lade_modell(quelle, device, args)
    params = gruppen(model, args.trainable)
    encoder_frozen = args.trainable != "all"
    model.use_fov_model = False
    trainingsmodus(model, encoder_frozen, True)

    print(f"Device: {device} | Start: {quelle} | Ziel: Hoehe ueber Boden", flush=True)
    print(f"Trainierbar ({args.trainable}): {sum(p.numel() for p in params)/1e6:.1f} M von "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f} M", flush=True)

    train_daten, train_lader = bauen_loader(args, "train", True, args.pro_gebiet, args.seed)
    val_daten, val_lader = bauen_loader(args, "val", False, args.pro_gebiet_val, 12345)

    vorgespannt, kopf_schicht = 1.0, None
    fortsetzung = args.resume and start_epoche_bekannt(args)
    if fortsetzung:
        beschreibung = bestes / "depthft.json"
        if beschreibung.exists():
            vorgespannt = float(json.loads(beschreibung.read_text()).get("vorgespannt", 1.0))
            kopf_schicht = kopf_skalieren(model, 1.0)
            print(f"Fortgesetzt; Kopf war mit {vorgespannt:.4f} vorgespannt.", flush=True)
    elif args.vorspannen != "aus":
        vorgespannt = (vorspannen(model, train_lader, args, device, encoder_frozen, args.vorspann_stapel)
                       if args.vorspannen == "auto" else float(args.vorspannen))
        if not (0 < vorgespannt < 1e6):
            vorgespannt = 1.0
        if vorgespannt != 1.0:
            # The head should output metres; what is measured is the factor it is
            # off by, and its last convolution is scaled by exactly that.
            kopf_schicht = kopf_skalieren(model, 1.0 / vorgespannt)
            print(f"Kopf um {1/vorgespannt:.2f} vorgespannt (Ausgabe lag um Faktor "
                  f"{vorgespannt:.4f} neben der Hoehe)", flush=True)

    kopf_ids = set() if kopf_schicht is None else {id(p) for p in kopf_schicht.parameters()}
    gruppe_kopf = [p for p in params if id(p) in kopf_ids]
    uebrige = [p for p in params if id(p) not in kopf_ids]
    lr_kopf = args.lr * min(1.0, 1.0 / max(vorgespannt, 1e-6)) if kopf_schicht is not None else args.lr
    gruppen_liste = [{"params": uebrige, "lr": args.lr}]
    if args.trainable == "all":
        encoder_ids = {id(p) for p in model.depth_pro.encoder.parameters()}
        gruppen_liste = [
            {"params": [p for p in uebrige if id(p) in encoder_ids], "lr": args.lr_encoder},
            {"params": [p for p in uebrige if id(p) not in encoder_ids], "lr": args.lr},
        ]
    if gruppe_kopf:
        gruppen_liste.append({"params": gruppe_kopf, "lr": lr_kopf})
    optimizer = torch.optim.AdamW([g for g in gruppen_liste if g["params"]],
                                  weight_decay=args.weight_decay)

    schritte_je_epoche = max(1, len(train_lader) // args.accum)
    gesamt = schritte_je_epoche * args.epochs
    aufwaermen = max(1, int(args.warmup * gesamt))
    print(f"Train {len(train_daten)} Ausschnitte aus {len(train_daten.sites)} Gebieten", flush=True)
    print(f"Val   {len(val_daten)} aus {len(val_daten.sites)} ({', '.join(val_daten.sites)})", flush=True)
    print(f"{gesamt} Aktualisierungen, {aufwaermen} zum Aufwaermen\n", flush=True)

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

    zustand = args.out / "optimierer.pt"
    if start_epoche and zustand.exists():
        optimizer.load_state_dict(torch.load(zustand, map_location=device, weights_only=False))
        for _ in range(start_epoche * schritte_je_epoche):
            plan.step()

    for epoche in range(start_epoche, args.epochs):
        train_daten.set_epoch(epoche)
        t0, laufend, gesehen = time.time(), {}, 0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_lader):
            verlust, l1, grad, h_pred, h_gt, maske = durchlauf(model, batch, args, device, encoder_frozen)
            (verlust / args.accum).backward()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                plan.step()

            werte = kennzahlen(h_pred, h_gt, maske)
            werte["verlust"] = float(verlust.detach())
            for name, wert in werte.items():
                laufend[name] = laufend.get(name, 0.0) + wert
            gesehen += 1
            if gesehen % max(1, args.melden) == 0:
                m = {n: w / gesehen for n, w in laufend.items()}
                print(f"  Epoche {epoche} {i+1}/{len(train_lader)} | Verlust {m['verlust']:.4f} | "
                      f"MAE {m.get('mae_m', 0):.2f} m | r {m.get('korrelation', 0):.3f} | "
                      f"lr {plan.get_last_lr()[0]:.2e}", flush=True)

        train_mittel = {n: w / max(1, gesehen) for n, w in laufend.items()}
        val_daten.set_epoch(0)
        val_mittel = bewerten(model, val_lader, args, device, encoder_frozen)
        print(f"Epoche {epoche} in {(time.time()-t0)/60:.1f} min | train MAE "
              f"{train_mittel.get('mae_m', 0):.2f} m | val MAE {val_mittel.get('mae_m', 0):.2f} m, "
              f"Bias {val_mittel.get('bias_m', 0):+.2f} m, r {val_mittel.get('korrelation', 0):.3f}, "
              f"relativ {val_mittel.get('relativ', 0)*100:.0f} %", flush=True)

        verlauf.append({"epoche": epoche, "train": train_mittel, "val": val_mittel})
        verlauf_pfad.write_text(json.dumps(verlauf, indent=2))

        model.use_fov_model = model.fov_model is not None
        model.save_pretrained(letztes)
        torch.save(optimizer.state_dict(), args.out / "optimierer.pt")
        if val_mittel.get("mae_m", float("inf")) < bester_wert:
            bester_wert = val_mittel["mae_m"]
            model.save_pretrained(bestes)
            (bestes / "depthft.json").write_text(json.dumps({
                "ziel": "hoehe_ueber_boden_m", "epoche": epoche, "val": val_mittel,
                "trainable": args.trainable, "crop_px": args.crop_px,
                "seitenverhaeltnis": args.seitenverhaeltnis, "vorgespannt": vorgespannt,
                "hoehe_deckel": args.hoehe_deckel, "train_gebiete": train_daten.sites,
            }, indent=2))
            print(f"  neuer Bestwert -> {bestes}", flush=True)
        model.use_fov_model = False

    print(f"\nFertig. Bestes MAE {bester_wert:.2f} m -> {bestes}")


if __name__ == "__main__":
    main()
