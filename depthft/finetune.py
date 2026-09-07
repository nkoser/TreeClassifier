"""Fine-tune Depth Pro on FORTRESS -- metric depth from nadir captures.

Why at all. Depth Pro is trained on ground perspectives: streets, interiors,
portraits. A nadir capture from 80 m does not occur in that, and it shows in the
results -- the depth is right in structure, but the scale is shifted and crowns
sit too high. That is exactly what can be repaired once truth is available:
FORTRESS supplies the height above ground for every pixel.

**Which space training happens in.** Depth Pro does not output metres but
canonical inverse depth. It only becomes metric in the processor post-processing:

    d = (f_px / image width) / D_raw  =  k / D_raw

`k` depends solely on the field of view. Since we know the camera of our frames,
we supply `k` instead of letting the field-of-view head estimate it -- that head
stays frozen and unchanged in the checkpoint so that it remains available. The
output stays canonical inverse depth, but the loss is computed in metric height
via `d = k/D` and `h = H-d`. The checkpoint remains loadable with the normal
Hugging Face classes. For correct metric depth, however, the known field of view
still has to be supplied during the conversion; the frozen field-of-view head is
unreliable on nadir images.

**The loss** has two parts. A Huber loss measures the metric error of the height
`h = H - k/D` and thereby optimises the target quantity directly. A gradient
matching over four scales on the same height difference sharpens crown
boundaries. An earlier state used log depth; that would be a relative error of
the camera distance `H-h`, not of the tree height, and was therefore weighted
wrongly for this goal.

**Why the head is pre-scaled first.** Pure Depth Pro is off by a factor of 50 in
this capture situation. Because of `d = k/D` the mapping is very steep near zero;
an optimizer should not have to learn that large change of scale through many
unstable steps. The pre-scaling puts the output into the physically relevant
range before the first update.

The way out is not to demand the jump in the first place. `--vorspannen auto`
measures the scale error over a few batches and scales the last convolution of
the head with it. Because that is a 1x1 convolution before the final ReLU and the
factor is positive, this is exactly equivalent to `D -> factor * D` -- but as a
real weight change. The shipped checkpoint therefore stays usable without a
special case. The learning rate of that one layer is scaled by the same factor,
otherwise Adam steps of the usual size would immediately tear apart weights that
are now orders of magnitude smaller.

**What is trained.** The default is `decoder`: neck, fusion stage and head,
around 60 M parameters. The encoder runs frozen under `no_grad` -- that saves the
bulk of the memory and is enough, because the scale sits in the head, not in the
features. `all` tunes everything and needs considerably more GPU.

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
MIN_D = 1e-6      # canonical inverse depth; the head ends on a ReLU, so it can return 0
TIEFE_MIN_M = 2.0     # safeguard: below this the camera hangs inside the tree,


def gruppen(model, was: str) -> list[torch.nn.Parameter]:
    """Which parts take part in learning. Never the FOV head -- we supply k."""
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
    """Canonical inverse depth. With a frozen encoder, without activation memory."""
    if not encoder_frozen:
        return model(pixel_values=pixel_values).predicted_depth
    with torch.no_grad():
        encodings = model.depth_pro.encoder(pixel_values, return_dict=True)
    features = [f.detach() for f in encodings[1]]
    features = model.depth_pro.neck(features)
    return model.head(model.fusion_stage(features)[-1])


@torch.no_grad()
def vorspannen(model, lader, args, device, encoder_frozen: bool, stapel: int = 8) -> float:
    """Set the head to the capture situation before any training happens.

    The scale error `d_predicted / d_true` is measured as the median over a few
    batches. Exactly that factor goes into the last convolution of the head: if
    the prediction is a factor of 50 too small, the canonical inverse depth is
    scaled by 0.02, and the depth is right in order of magnitude.
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
    """Scale the last 1x1 convolution of the head -- equivalent to `D -> factor * D`.

    It sits before the final ReLU; a positive factor commutes with it, so the
    scaling is exact and not merely approximate.
    """
    letzte = [schicht for schicht in model.head.layers if isinstance(schicht, torch.nn.Conv2d)][-1]
    with torch.no_grad():
        letzte.weight.mul_(faktor)
        if letzte.bias is not None:
            letzte.bias.mul_(faktor)
    return letzte


def gradientenanpassung(rest: torch.Tensor, maske: torch.Tensor, stufen: int = 4) -> torch.Tensor:
    """Multi-scale gradient matching on a normalised error map.

    At each level the residual is viewed half as finely. Fine levels sharpen crown
    boundaries, coarse ones prevent a slow drift across the image.
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
    """Metric quality, in metres.

    `mae_m` is at the same time the error of the height above ground: that is
    flight altitude minus depth, and the altitude cancels out in the difference.
    So exactly the number that matters for tree height.
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
    """One forward step: loss and metrics."""
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

    # Use the exact metric height error in the forward pass. Backwards, the
    # derivative of k/D is singular near D=0; so the error gets the Jacobian
    # d_gt**2/k linearised at the target D_gt. The forward value, and hence the
    # optimised Huber loss, stays exact, while the gradient is finite and points
    # in the right direction even for large errors.
    D_roh = D.clamp_min(MIN_D)
    d_pred = k / D_roh
    D_gt = k / d_gt.clamp_min(TIEFE_MIN_M)
    rest_exakt = d_gt - d_pred
    rest_linear = (D - D_gt) * (d_gt.square() / k.clamp_min(MIN_D))
    rest_m = rest_linear + (rest_exakt - rest_linear).detach()

    # h_pred - h_gt = (H-d_pred) - (H-d_gt) = d_gt-d_pred. H cancels
    # algebraically, but the error is in metres and not weighted relative to the
    # large camera distance.
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
        # Only sensible then: with a frozen encoder the large activation memory
        # does not arise anyway.
        try:
            model.gradient_checkpointing_enable()
        except Exception as fehler:      # not every version supports this
            print(f"Gradient-Checkpointing nicht verfuegbar: {fehler}", flush=True)
    return model


def trainingsmodus(model, encoder_frozen: bool, an: bool) -> None:
    """A frozen encoder stays in `eval` even during training."""
    model.train(an)
    if encoder_frozen:
        model.depth_pro.encoder.eval()


def bauen_loader(args, split: str, augment: bool, pro_gebiet: int, seed: int):
    """Fixed crops for validation, but at the sharpness of the target frames.

    The crops are fixed (no mirroring, a fixed draw) so that epochs stay
    comparable. Blur is applied nonetheless: otherwise the validation measures
    razor-sharp ortho crops while training and inference deal with video images.
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
    # shuffle=False on purpose: NadirFrames shuffles reproducibly in short site
    # blocks. A second shuffle here would destroy the raster cache.
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
    """On resume the head is already pre-scaled -- do not do it a second time."""
    pfad = args.out / "verlauf.json"
    return pfad.exists() and bool(json.loads(pfad.read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/scratch/shared/nik/data/fortress/depthft"))
    parser.add_argument("--out", type=Path, default=Path("/scratch/shared/nik/runs/depthft_huber_v2"))
    parser.add_argument("--start", default=MODELL, help="Starting weights, or a checkpoint of your own.")
    parser.add_argument("--trainable", default="decoder", choices=("head", "decoder", "all"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8, help="Steps until the weights are updated.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-encoder", type=float, default=1e-5, help="Only with --trainable all.")
    parser.add_argument("--vorspannen", default="auto",
                        help="'auto' measures the scale error and sets the head to it, "
                             "a number fixes the factor, 'aus' leaves the head as it is.")
    parser.add_argument("--vorspann-stapel", type=int, default=12)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.05, help="Share of the steps used for warm-up.")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lambda-grad", type=float, default=0.5)
    parser.add_argument("--grad-stufen", type=int, default=4)
    parser.add_argument("--bezugshoehe", type=float, default=20.0,
                        help="Normalisation of the metric height loss, in metres.")
    parser.add_argument("--huber-beta", type=float, default=2.0,
                        help="Width of the quadratic Huber region, in metres.")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--crop-px", type=int, default=1536,
                        help="Image width of the crops -- also the resolution in which loss "
                             "and truth live. 1536 means: no rescaling between crop and "
                             "model input.")
    parser.add_argument("--seitenverhaeltnis", type=float, default=16 / 9,
                        help="Width over height of the crops. Defaults to that of our frames.")
    parser.add_argument("--model-size", type=int, default=1536, help="Input size of Depth Pro.")
    parser.add_argument("--pro-gebiet", type=int, default=200, help="Crops per site and epoch.")
    parser.add_argument("--pro-gebiet-val", type=int, default=40)
    parser.add_argument("--val-scharf", dest="val_domaene", action="store_false", default=True,
                        help="Validate on unfiltered ortho crops instead of at the sharpness "
                             "of the target frames. That measures a distribution which does "
                             "not occur in deployment.")
    parser.add_argument("--hoehe-min", type=float, default=25.0)
    parser.add_argument("--hoehe-max", type=float, default=120.0)
    parser.add_argument("--abstand-min", type=float, default=20.0,
                        help="Minimum clearance of the camera above the tallest treetop of the site.")
    parser.add_argument("--fov-min", type=float, default=35.0)
    parser.add_argument("--fov-max", type=float, default=85.0)
    parser.add_argument("--min-gueltig", type=float, default=0.80)
    parser.add_argument("--jitter", type=float, default=1.0)
    parser.add_argument("--video", type=float, default=1.0)
    parser.add_argument("--strahl-tiefe", action="store_true",
                        help="Depth along the viewing ray instead of along the optical axis.")
    parser.add_argument("--melden", type=int, default=50, help="Progress report every N batches.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-sites", type=int, default=4,
                        help="Raster cache per worker, and number of sites shuffled together.")
    parser.add_argument("--site-block", type=int, default=32,
                        help="Maximum consecutive samples from the same site.")
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
    # The field-of-view head is not computed during training -- it is frozen and
    # costs a second encoder pass. Its weights are preserved and written back out
    # when saving.
    model.use_fov_model = False
    trainingsmodus(model, encoder_frozen, True)

    n_trainierbar = sum(p.numel() for p in params)
    print(f"Device: {device} | Start: {quelle}", flush=True)
    print(f"Trainierbar ({args.trainable}): {n_trainierbar/1e6:.1f} M von "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f} M", flush=True)

    train_daten, train_lader = bauen_loader(args, "train", True, args.pro_gebiet, args.seed)
    val_daten, val_lader = bauen_loader(args, "val", False, args.pro_gebiet_val, 12345)

    # Set the head to the capture situation before the optimizer is built -- the
    # scaled layer needs its own, co-scaled learning rate.
    vorgespannt, kopf_schicht = 1.0, None
    fortsetzung = args.resume and start_epoche_bekannt(args)
    if fortsetzung:
        beschreibung = bestes / "depthft.json"
        if beschreibung.exists():
            vorgespannt = float(json.loads(beschreibung.read_text()).get("vorgespannt", 1.0))
            kopf_schicht = kopf_skalieren(model, 1.0)   # only fetch it, do not scale
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

    # On resume the moments and the learning rate have to come along: otherwise
    # the schedule restarts at warm-up and AdamW without moments jolts the weights.
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
        val_daten.set_epoch(0)      # fixed crops, so that epochs stay comparable
        val_mittel = bewerten(model, val_lader, args, device, encoder_frozen)
        print(f"Epoche {epoche} in {(time.time()-t0)/60:.1f} min | "
              f"train AbsRel {train_mittel.get('absrel', 0):.3f} | "
              f"val AbsRel {val_mittel.get('absrel', 0):.3f}, MAE {val_mittel.get('mae_m', 0):.2f} m, "
              f"Bias {val_mittel.get('bias_m', 0):+.2f} m, d1 {val_mittel.get('delta125', 0):.3f}", flush=True)

        verlauf.append({"epoche": epoche, "train": train_mittel, "val": val_mittel})
        verlauf_pfad.write_text(json.dumps(verlauf, indent=2))

        model.use_fov_model = model.fov_model is not None   # save it complete
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
