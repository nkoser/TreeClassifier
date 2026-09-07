"""Query-basierte Kroneninstanzen: EoMT und Mask2Former auf BAMFORESTS.

Beide Architekturen sagen feste Anfragen ("queries") vorher, jede mit einer
eigenen Maske und einem eigenen Score -- keine Anker, kein NMS, keine
Watershed-Nachbearbeitung. Das raeumt strukturell den Fehler aus, an dem Mask
R-CNN in Hain gescheitert ist: dort deckten die Anker 32 bis 512 px ab, die
Kronen reichen bis 842 px, und alles darueber konnte die RPN gar nicht erst
vorschlagen. Eine Query hat keine Groessenannahme.

  eomt          Encoder-only Mask Transformer (CVPR 2025) mit DINOv3-Backbone.
                Kein Pixeldecoder, kein Transformerdecoder -- der ViT selbst
                traegt die Queries. Passt zum Projekt, weil DINOv3 ueber den
                DINOvTree-Checkpoint ohnehin schon da ist.
  mask2former   Masked-Attention-Decoder auf Swin. Reifer und breiter erprobt,
                besonders bei dicht gedraengten Instanzen.

Beide laufen ueber denselben Datenpfad (`bamforests.CrownCrops`), dieselbe
Fensterlogik (`tiling.slide`) und dieselbe Metrik (`metrics`). Was verglichen
wird, ist damit die Architektur und nicht die Umgebung drumherum.

Beide sehen denselben Bodenausschnitt: ein 1024-px-Ausschnitt der Kachel, auf
die Eingabegroesse des Modells skaliert. Gleiche Flaeche, gleiche Kronengroesse
in Metern -- nur die Pixelzahl unterscheidet sich, und die gehoert zur
Architektur.

    python crownseg/queryseg.py --arch eomt --mode train
    python crownseg/queryseg.py --arch mask2former --mode eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bamforests as bam  # noqa: E402
import metrics as met  # noqa: E402
from tiling import draw_overlay, slide, suppress, to_label_map  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = Path(f"/scratch/shared/{os.environ.get('USER', 'nik')}/data/treeclf/checkpoints")

# Die `tue-mps/<task>_eomt_<...>`-Repos sind das Originalformat der Autoren
# und haben keine config.json. Die nach transformers konvertierten liegen
# unter der Bindestrich-Form `eomt-dinov3-coco-instance-large-640`.
ARCHS = {
    "eomt": "tue-mps/eomt-dinov3-coco-instance-large-640",
    "mask2former": "facebook/mask2former-swin-base-coco-instance",
}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# Die Tiefe ist je Kachel auf 0..255 gespreizt, also ungefaehr gleichverteilt.
# Mittelwert 0.5 und Streuung 0.29 machen daraus etwa denselben Wertebereich wie
# bei den ImageNet-normierten Farbkanaelen.
DEPTH_MEAN, DEPTH_STD = 0.5, 0.29


def stats(channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    if channels == 3:
        return IMAGENET_MEAN, IMAGENET_STD
    return (torch.cat([IMAGENET_MEAN, torch.tensor([[[DEPTH_MEAN]]])]),
            torch.cat([IMAGENET_STD, torch.tensor([[[DEPTH_STD]]])]))


# --------------------------------------------------------------------------- #
# Modell
# --------------------------------------------------------------------------- #


def add_depth_channel(model) -> bool:
    """Die erste Faltung von 3 auf 4 Eingangskanaele erweitern.

    Der neue Kanal startet mit Nullgewichten. Das Modell verhaelt sich damit im
    ersten Schritt exakt wie das RGB-Modell und muss sich den Nutzen der Tiefe
    erst erarbeiten -- initialisiert man ihn stattdessen mit dem Mittel der
    Farbgewichte, sieht das Netz die Kachelstruktur sofort doppelt und der
    Vergleich gegen den RGB-Lauf misst diesen Sprung mit.
    """
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            weight = module.weight.data
            expanded = torch.zeros(weight.shape[0], 4, *weight.shape[2:], dtype=weight.dtype)
            expanded[:, :3] = weight
            module.in_channels = 4
            module.weight = nn.Parameter(expanded)
            return True
    return False


class GatedDepthEmbed(torch.nn.Module):
    """Eigene Eingangsfaltung fuer die Tiefe, dazuaddiert ueber ein gelerntes Gewicht.

    Der vierte Eingabekanal mit Nullgewichten ist gemessen wirkungslos geblieben
    (+0.005, unter der Streuung), obwohl die Tiefe allein 0.514 erreicht -- das
    Netz *darf* ihn ignorieren, und solange RGB allein traegt, entsteht kein
    Grund, ihn zu benutzen.

    Hier bekommt die Tiefe deshalb eine eigene, aus den RGB-Gewichten kopierte
    Faltung. Die vortrainierten Filter sind Kanten- und Texturdetektoren, die auf
    einer Hoehenkarte genauso sinnvoll ansetzen wie auf einem Bild. Das Gewicht
    `gate` startet bei 0.5: die Tiefe ist von der ersten Iteration an im
    Tokenbild, und das Netz muss sie aktiv herausdrehen, statt sie nie
    hereinzulassen.

    Der gelernte Wert ist zugleich das Messergebnis -- bleibt er nahe null, hat
    das Netz die Tiefe auch dann verworfen, als sie ihm aufgedraengt wurde.
    """

    def __init__(self, base: torch.nn.Conv2d, start: float) -> None:
        super().__init__()
        import copy

        self.rgb = base
        self.depth = copy.deepcopy(base)
        self.gate = torch.nn.Parameter(torch.tensor(float(start)))

    @property
    def weight(self) -> torch.Tensor:
        """Der umgebende Code liest hierueber den Datentyp der Eingangsfaltung."""
        return self.rgb.weight

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        colour = self.rgb(pixels[:, :3])
        if pixels.shape[1] < 4:
            return colour
        height = self.depth(pixels[:, 3:4].expand(-1, 3, -1, -1))
        return colour + self.gate * height


def add_gated_depth(model, start: float) -> bool:
    """Die erste 3-Kanal-Faltung durch die gegatete Variante ersetzen."""
    import torch.nn as nn

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                setattr(module, child_name, GatedDepthEmbed(child, start))
                print(f"Fusion an {name}.{child_name}, Startgewicht {start}", flush=True)
                return True
    return False


def build_model(arch: str, checkpoint: str | None = None, depth: bool = False,
                fusion: str = "kanal", gate_start: float = 0.5):
    """Vortrainierten Kopf auf eine Klasse umbauen.

    Der COCO-Kopf sagt 80 Klassen vorher, hier gibt es nur `tree`. Die
    Klassenschicht passt damit nicht und wird neu initialisiert
    (`ignore_mismatched_sizes`); Backbone, Pixeldecoder und Maskenkopf bleiben
    vortrainiert -- dort steckt das, was uebertragbar ist.
    """
    from transformers import AutoModelForUniversalSegmentation

    model = AutoModelForUniversalSegmentation.from_pretrained(
        checkpoint or ARCHS[arch],
        id2label={0: "tree"},
        label2id={"tree": 0},
        ignore_mismatched_sizes=True,
    )
    if depth:
        added = add_gated_depth(model, gate_start) if fusion == "gate" else add_depth_channel(model)
        if not added:
            raise RuntimeError("Keine 3-Kanal-Faltung gefunden -- Tiefe nicht eingebaut.")
    return model


def normalize(image_uint8: np.ndarray, size: int) -> torch.Tensor:
    resized = cv2.resize(image_uint8, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().div_(255.0)
    mean, std = stats(tensor.shape[0])
    return (tensor - mean) / std


# --------------------------------------------------------------------------- #
# Daten
# --------------------------------------------------------------------------- #


class QueryCrops(torch.utils.data.Dataset):
    """Ausschnitte von `CrownCrops` im Format der Universal-Segmentation-Modelle."""

    def __init__(self, base: bam.CrownCrops, size: int) -> None:
        self.base, self.size = base, size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target = self.base[index]
        pixels = F.interpolate(image[None], size=(self.size, self.size),
                               mode="bilinear", align_corners=False)[0]
        mean, std = stats(pixels.shape[0])
        pixels = (pixels - mean) / std

        masks = target["masks"]
        if len(masks):
            masks = F.interpolate(masks[None].float(), size=(self.size, self.size),
                                  mode="nearest")[0]
            keep = masks.flatten(1).sum(1) > 0  # beim Verkleinern verschwundene Kronen
            masks = masks[keep]
        else:
            masks = torch.zeros((0, self.size, self.size))
        return pixels, masks, torch.zeros(len(masks), dtype=torch.int64)


def collate(batch):
    pixels, masks, classes = zip(*batch)
    return torch.stack(pixels), list(masks), list(classes)


# --------------------------------------------------------------------------- #
# Vorhersage
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict_window(model, window_rgb: np.ndarray, device, args) -> list[met.Instance]:
    """Queries mit Score ueber der Schwelle als Instanzen im Fenstermassstab."""
    height, width = window_rgb.shape[:2]
    pixels = normalize(window_rgb, args.input_size)[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(pixel_values=pixels)

    # Letzte Klasse ist "kein Objekt"; Klasse 0 ist die Krone.
    scores = output.class_queries_logits[0].float().softmax(-1)[:, 0]
    keep = scores >= args.score_thresh
    if not keep.any():
        return []

    logits = output.masks_queries_logits[0][keep].float()
    logits = F.interpolate(logits[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    masks = (logits.sigmoid() > 0.5).cpu().numpy()

    instances = [met.instance_from_mask(m, float(s))
                 for m, s in zip(masks, scores[keep].cpu().numpy())]
    return [i for i in instances if i is not None and i.area >= args.min_area]


def predict_tiles(model, image_rgb: np.ndarray, device, args) -> list[met.Instance]:
    return slide(image_rgb, args.eval_tile, args.overlap,
                 lambda window: predict_window(model, window, device, args))


def rescale(instance: met.Instance, factor: float) -> met.Instance | None:
    """Eine Instanz aus einem skalierten Bild in die Originalkoordinaten holen."""
    x0, y0, x1, y1 = (int(round(v * factor)) for v in instance.box)
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    mask = cv2.resize(instance.mask.astype(np.uint8), (width, height),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    if not mask.any():
        return None
    return met.Instance((x0, y0, x0 + width, y0 + height), mask, instance.score)


def predict_multiscale(model, image_bgr: np.ndarray, device, args,
                       scales: list[float]) -> list[met.Instance]:
    """Dieselbe Aufnahme in mehreren Aufloesungen, Ergebnisse zusammengefuehrt.

    Der Bildmassstab der eigenen Frames ist nur geschaetzt, und ein trainiertes
    Modell sucht Kronen in der Groesse, die es gelernt hat. Mehrere Massstaebe
    nebeneinander machen die Vorhersage von dieser Schaetzung unabhaengig -- es
    ist derselbe Grund, aus dem `segment_sam3.py` ueber mehrere Kachelstufen
    laeuft. Zusammengefuehrt wird nach Konfidenz, stark ueberlappende Masken aus
    benachbarten Massstaeben fallen dabei weg.
    """
    height, width = image_bgr.shape[:2]
    collected: list[met.Instance] = []
    for scale in scales:
        work = cv2.resize(image_bgr, (int(round(width * scale)), int(round(height * scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        found = predict_tiles(model, cv2.cvtColor(work, cv2.COLOR_BGR2RGB), device, args)
        collected.extend(i for i in (rescale(f, 1.0 / scale) for f in found) if i is not None)
    return suppress(collected, args.merge_iou) if len(scales) > 1 else collected


# --------------------------------------------------------------------------- #
# Modi
# --------------------------------------------------------------------------- #


def roots(args) -> list[Path]:
    return args.prepared if isinstance(args.prepared, list) else [args.prepared]


def depth_dir_for(args, split: str, root: Path | None = None) -> Path | None:
    if not (args.depth or args.depth_only):
        return None
    directory = (root or roots(args)[0]) / f"{split}_tiefe_{args.depth_model}"
    if not directory.exists():
        raise FileNotFoundError(f"Tiefencache fehlt: {directory} -- erst depthcache.py laufen lassen.")
    return directory


def load_image(directory: Path, stem: str, args) -> np.ndarray:
    """Kachel als RGB oder RGB+Tiefe, je nach Betriebsart."""
    image = cv2.cvtColor(cv2.imread(str(directory / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
    depth_dir = depth_dir_for(args, directory.name, directory.parent)
    if depth_dir is None:
        return image
    depth = cv2.imread(str(depth_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    return np.dstack([depth] * 3) if args.depth_only else np.dstack([image, depth])


def validate_instances(model, args, device) -> float:
    """Instanz-F1 auf ganzen Validierungskacheln -- die Groesse, nach der ausgewaehlt wird."""
    model.eval()
    rows = []
    for directory in [root / "val" for root in roots(args)]:
        index = json.loads((directory / "annotations.json").read_text())
        stems = sorted(index)[:: max(1, len(index) // max(1, args.val_f1_tiles))][: args.val_f1_tiles]
        rows.extend(_validate_tiles(model, args, device, directory, index, stems))
    model.train()
    return met.accumulate(rows)["f1"]


def _validate_tiles(model, args, device, directory, index, stems) -> list[dict]:
    rows = []
    for stem in stems:
        _, rings = bam.load_tile(directory, stem, index)
        image = load_image(directory, stem, args)
        masks, _ = bam.masks_from_rings(rings, *image.shape[:2], args.min_area, 0.0)
        truth = [i for i in (met.instance_from_mask(m) for m in masks) if i is not None]
        rows.append(met.evaluate(predict_tiles(model, image, device, args), truth, args.iou_thresh))
    return rows


def run_training(args, device) -> None:
    model = build_model(args.arch, args.checkpoint_from, args.depth,
                        args.fusion, args.gate_start).to(device)

    sources = roots(args)
    loaders = {}
    for split, steps, augment in (("train", args.steps_per_epoch, True), ("val", args.val_steps, False)):
        # Gleiche Gewichtung je Quelle, nicht nach Kachelzahl. BAMFORESTS hat
        # 1438 Trainingskacheln, Quebec 543 -- nach Groesse gewichtet kaeme der
        # Datensatz mit den kleinen Kronen kaum vor, und genau der fehlt.
        # Die Orthomosaik-Quelle zaehlt mit, sonst liefert der Ladevorgang mehr
        # Schritte als der Lernratenplan vorsieht.
        n_parts = len(sources) + (1 if args.cog_root and augment else 0)
        per_source = max(1, (steps * args.batch_size) // n_parts)
        parts = [QueryCrops(bam.CrownCrops(
            root, split, args.crop, per_source, augment=augment,
            scale_jitter=tuple(args.scale_jitter) if augment else (1.0, 1.0),
            depth_dir=depth_dir_for(args, split, root), depth_only=args.depth_only), args.input_size)
            for root in sources]
        # Ausschnitte mit frei gewaehltem Bildfeld direkt aus dem Orthomosaik.
        # Die vorgeschnittenen Kacheln koennen den Massstab nur bis rund
        # 5.4 cm/px aufweiten; hier sind bis 8.7 cm/px moeglich, begrenzt durch
        # die 200 Anfragen des Modells und nicht durch die Daten.
        if args.cog_root and augment:
            from quebec_cog import CogCrops

            parts.append(QueryCrops(CogCrops(
                args.cog_root, args.cog_zones, args.cog_date, args.input_size,
                per_source, gsd_range=tuple(args.cog_gsd), augment=True,
                max_instances=args.cog_max_instances), args.input_size))

        dataset = parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)
        loaders[split] = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=len(parts) > 1,
            num_workers=args.workers, drop_last=augment, persistent_workers=args.workers > 0)

    quelle = "nur Tiefe" if args.depth_only else "RGB + Tiefe" if args.depth else "RGB"
    print(f"{args.arch}: {ARCHS[args.arch]} | Eingabe: {quelle} | "
          f"Quellen: {', '.join(r.parent.name + '/' + r.name for r in sources)}"
          f"{' + Orthomosaik ' + str(tuple(args.cog_gsd)) + ' cm/px' if args.cog_root else ''}\n"
          f"Ausschnitt {args.crop} px Boden -> {args.input_size} px Eingabe | "
          f"Massstab {args.scale_jitter[0]:.2f}-{args.scale_jitter[1]:.2f}\n", flush=True)

    # Der vortrainierte Backbone braucht eine kleinere Schrittweite als der
    # neu initialisierte Klassenkopf, sonst wird sein Wissen im ersten Epoch
    # ueberschrieben.
    backbone, rest = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if ("encoder" in name or "backbone" in name) else rest).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr * args.backbone_lr_factor}, {"params": rest, "lr": args.lr}],
        weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr * args.backbone_lr_factor, args.lr],
        total_steps=args.epochs * args.steps_per_epoch, pct_start=0.1)

    best = -1.0
    last_checkpoint = args.checkpoint.with_name(args.checkpoint.stem + "_letzte.pth")
    for epoch in range(1, args.epochs + 1):
        losses = {}
        for phase, loader in loaders.items():
            model.train()
            total, count = 0.0, 0
            for pixels, masks, classes in loader:
                pixels = pixels.to(device)
                masks = [m.to(device) for m in masks]
                classes = [c.to(device) for c in classes]
                with torch.set_grad_enabled(phase == "train"), \
                        torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = model(pixel_values=pixels, mask_labels=masks, class_labels=classes).loss
                if phase == "train":
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(backbone + rest, 1.0)
                    optimizer.step()
                    scheduler.step()
                total += float(loss.detach()) * len(pixels)
                count += len(pixels)
            losses[phase] = total / max(1, count)

        f1 = validate_instances(model, args, device)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        state = {"model": model.state_dict(), "val_loss": losses["val"], "val_f1": f1,
                 "epoch": epoch, "arch": args.arch, "args": vars(args)}
        torch.save(state, last_checkpoint)
        marker = ""
        if f1 > best:
            best, marker = f1, "  <- gespeichert"
            torch.save(state, args.checkpoint)
        gates = [f"{p.item():+.3f}" for n, p in model.named_parameters() if n.endswith(".gate")]
        gate_text = f"  Tiefengewicht {' '.join(gates)}" if gates else ""
        print(f"Epoche {epoch:3d}  train {losses['train']:.4f}  val {losses['val']:.4f}  "
              f"F1 {f1:.3f}{gate_text}{marker}", flush=True)

    print(f"\nBeste Instanz-F1: {best:.3f} -> {args.checkpoint}")


def load_trained(args, device):
    model = build_model(args.arch, depth=args.depth,
                        fusion=args.fusion, gate_start=args.gate_start).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"{args.arch} geladen (Epoche {state.get('epoch', '?')}, val F1 {state.get('val_f1', float('nan')):.3f})")
    return model


def run_fusion(args, device) -> None:
    """Zwei Modelle nebeneinander laufen lassen und die Instanzen vereinigen.

    RGB und Tiefe haben gemessen komplementaere Profile: auf Hain ist RGB
    praeziser (0.578 gegen 0.474), die Tiefe findet mehr (Trefferquote 0.562
    gegen 0.532). Der Versuch, beides frueh zu verbinden -- die Tiefe als vierter
    Eingabekanal -- brachte nichts, weil der Kanal mit Nullgewichten startet und
    das Modell ihn schlicht ignorieren kann, solange RGB allein traegt.

    Hier laufen beide vollstaendig getrennt, und erst die fertigen Instanzen
    werden nach Konfidenz zusammengefuehrt. Dieselbe Mechanik wie bei Multiskala:
    stark ueberlappende Masken gelten als Dopplung, der Rest bleibt.
    """
    import collections

    import pandas as pd

    rgb_args = argparse.Namespace(**{**vars(args), "depth_only": False, "depth": False})
    depth_args = argparse.Namespace(**{**vars(args), "depth_only": True, "depth": False})
    rgb_args.checkpoint = args.checkpoint
    depth_args.checkpoint = args.fuse_with

    rgb_model = load_trained(rgb_args, device)
    depth_model = load_trained(depth_args, device)

    rows = []
    for split in args.splits:
        directory = roots(args)[0] / split
        index = json.loads((directory / "annotations.json").read_text())
        stems = sorted(index)
        if args.eval_tiles:
            stems = stems[:: max(1, len(stems) // args.eval_tiles)][: args.eval_tiles]

        per_area = collections.defaultdict(list)
        for stem in stems:
            _, rings = bam.load_tile(directory, stem, index)
            image_rgb = load_image(directory, stem, rgb_args)
            image_depth = load_image(directory, stem, depth_args)
            masks, _ = bam.masks_from_rings(rings, *image_rgb.shape[:2], args.min_area, 0.0)
            truth = [i for i in (met.instance_from_mask(m) for m in masks) if i is not None]

            found = (predict_tiles(rgb_model, image_rgb, device, rgb_args)
                     + predict_tiles(depth_model, image_depth, device, depth_args))
            per_area[stem.split("_")[0]].append(
                met.evaluate(suppress(found, args.merge_iou), truth, args.iou_thresh))

        for area, tiles in sorted(per_area.items()):
            rows.append({"arch": f"{args.arch}+tiefe", "split": split, "gebiet": area,
                         **met.accumulate(tiles)})
            print(f"  {split}/{area}: {rows[-1]['f1']:.3f} F1", flush=True)

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "eval_fusion.csv", index=False)
    print(f"\n=== RGB + Tiefe spaet vereinigt, IoU >= {args.iou_thresh} ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def run_evaluation(args, device) -> None:
    import collections

    import pandas as pd

    model = load_trained(args, device)
    rows = []
    for split in args.splits:
        directory = roots(args)[0] / split
        index = json.loads((directory / "annotations.json").read_text())
        stems = sorted(index)
        if args.eval_tiles:
            stems = stems[:: max(1, len(stems) // args.eval_tiles)][: args.eval_tiles]

        per_area = collections.defaultdict(list)
        for stem in stems:
            _, rings = bam.load_tile(directory, stem, index)
            image = load_image(directory, stem, args)
            masks, _ = bam.masks_from_rings(rings, *image.shape[:2], args.min_area, 0.0)
            truth = [i for i in (met.instance_from_mask(m) for m in masks) if i is not None]
            predicted = predict_tiles(model, image, device, args)
            per_area[stem.split("_")[0]].append(met.evaluate(predicted, truth, args.iou_thresh))

        for area, tiles in sorted(per_area.items()):
            rows.append({"arch": args.arch, "split": split, "gebiet": area, **met.accumulate(tiles)})
            print(f"  {split}/{area}: {rows[-1]['f1']:.3f} F1", flush=True)

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / f"eval_{args.arch}.csv", index=False)
    print(f"\n=== {args.arch}, IoU >= {args.iou_thresh}, Score >= {args.score_thresh} ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# Bodenaufloesung von BAMFORESTS. Fremde Aufnahmen muessen darauf gebracht
# werden, damit die Kronen in der gelernten Groesse ankommen.
BAM_GSD_CM = 1.70


def frame_scale(args, folder: str) -> float:
    """Massstabsfaktor aus Flughoehe und Bildwinkel.

    Ein 1920-px-Frame aus 100 m bei 73.7 Grad hat rund 7.8 cm/px, BAMFORESTS
    1.70 cm/px -- Faktor 4.6. Ohne diese Korrektur sieht das Modell Kronen von
    56 px, wo es 258 px gelernt hat, und findet nichts.
    """
    altitude = args.altitudes.get(folder, args.altitude)
    gsd_cm = 100 * altitude * 2 * np.tan(np.radians(args.hfov_deg) / 2) / args.frame_width
    return float(gsd_cm / BAM_GSD_CM)


def run_prediction(args, device) -> None:
    model = load_trained(args, device)
    args.out.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in args.frames_dir.iterdir() if p.is_dir()) or [args.frames_dir]

    for folder in folders:
        frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        out_folder = args.out / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)
        for frame_path in frames:
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            height, width = image_bgr.shape[:2]
            base = (args.scales.get(folder.name)
                    or (args.predict_scale if args.predict_scale > 0 else frame_scale(args, folder.name)))
            scales = [base * step for step in args.scale_steps]
            instances = predict_multiscale(model, image_bgr, device, args, scales)
            labels = to_label_map(instances, height, width)
            caption = (f"{int(labels.max())} Kronen ({args.arch}, x{base:.2f}"
                       f"{'' if len(scales) == 1 else ' x' + '/'.join(f'{s:.2f}' for s in args.scale_steps)})")
            cv2.imwrite(str(out_folder / f"{frame_path.stem}_labels.png"), labels)
            cv2.imwrite(str(out_folder / f"{frame_path.stem}_{args.arch}.jpg"),
                        draw_overlay(image_bgr, labels, caption), [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  {folder.name}/{frame_path.name}: {caption}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=list(ARCHS), default="eomt")
    parser.add_argument("--mode", choices=("train", "eval", "predict", "fuse"), default="train")
    parser.add_argument("--fuse-with", type=Path, default=None,
                        help="Zweiter Checkpoint (Tiefe) fuer --mode fuse.")
    parser.add_argument("--prepared", type=Path, nargs="+", default=[bam.BAMFORESTS / "crownseg"],
                        help="Ein oder mehrere aufbereitete Datensaetze; mehrere werden gemischt.")
    parser.add_argument("--splits-per-source", nargs="*", default=[],
                        help="Abweichende Splitnamen je Quelle, z.B. test fuer Quebec statt test1.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results_queryseg")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-from", default=None, help="Abweichender Startcheckpoint.")

    parser.add_argument("--crop", type=int, default=1024, help="Bodenausschnitt in Kachelpixeln.")
    parser.add_argument("--input-size", type=int, default=640, help="Eingabegroesse des Modells.")
    parser.add_argument("--scale-jitter", type=float, nargs=2, default=(0.6, 1.8))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--val-steps", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--depth", action="store_true",
                        help="Tiefe als vierten Eingabekanal verwenden.")
    parser.add_argument("--depth-only", action="store_true",
                        help="Nur die Hoehenkarte, dreifach kopiert -- misst ihren "
                             "Informationsgehalt ohne Farbe.")
    parser.add_argument("--depth-model", default="depthpro", choices=("depthpro", "dav2"))
    parser.add_argument("--fusion", choices=("kanal", "gate"), default="kanal",
                        help="kanal: vierter Eingabekanal (gemessen wirkungslos). "
                             "gate: eigene Faltung fuer die Tiefe mit gelerntem Gewicht.")
    parser.add_argument("--gate-start", type=float, default=0.5)

    parser.add_argument("--splits", nargs="*", default=["test1", "test2"])
    parser.add_argument("--eval-tile", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=768)
    parser.add_argument("--eval-tiles", type=int, default=40, help="0 = alle Kacheln.")
    parser.add_argument("--val-f1-tiles", type=int, default=10)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5)

    parser.add_argument("--frames-dir", type=Path, default=Path("/cold/Mahfuz/chosen_frames"))
    parser.add_argument("--predict-scale", type=float, default=0.0,
                        help="Fester Faktor; 0 = aus Flughoehe und Bildwinkel bestimmen.")
    parser.add_argument("--altitude", type=float, default=100.0)
    parser.add_argument("--altitudes", nargs="*", default=[], help="ORDNER=HOEHE, z.B. pines=35")
    parser.add_argument("--scales", nargs="*", default=[],
                        help="ORDNER=FAKTOR, gemessen mit scale_probe.py. Schlaegt --altitudes.")
    parser.add_argument("--cog-root", type=Path, default=None,
                        help="Orthomosaik-Wurzel fuer Ausschnitte mit variablem Bildfeld.")
    parser.add_argument("--cog-zones", nargs="*", default=["zone1"])
    parser.add_argument("--cog-date", default="2021-09-02")
    parser.add_argument("--cog-gsd", type=float, nargs=2, default=(2.7, 8.7),
                        help="Massstabsspanne in cm je Eingabepixel.")
    parser.add_argument("--cog-max-instances", type=int, default=150,
                        help="Ausschnitte mit mehr Kronen verwerfen -- das Modell hat 200 Anfragen.")
    parser.add_argument("--scale-steps", type=float, nargs="*", default=[0.7, 1.0, 1.4],
                        help="Vielfache des Grundmassstabs, die zusammengefuehrt werden. "
                             "Ein einzelner Wert schaltet Multiskala ab.")
    parser.add_argument("--merge-iou", type=float, default=0.4,
                        help="Ab dieser Ueberlappung gilt eine Maske als Dopplung.")
    parser.add_argument("--hfov-deg", type=float, default=73.7)
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    args.altitudes = {p.split("=")[0]: float(p.split("=")[1]) for p in args.altitudes}
    args.scales = {p.split("=")[0]: float(p.split("=")[1]) for p in args.scales}
    if args.checkpoint is None:
        suffix = (f"_nurtiefe_{args.depth_model}" if args.depth_only
                  else f"_tiefe{'gate' if args.fusion == 'gate' else ''}_{args.depth_model}"
                  if args.depth else "")
        args.checkpoint = CHECKPOINTS / f"crownseg_{args.arch}{suffix}.pth"
    return args


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device} | {args.arch} | Modus: {args.mode}\n", flush=True)
    {"train": run_training, "eval": run_evaluation,
     "predict": run_prediction, "fuse": run_fusion}[args.mode](args, device)


if __name__ == "__main__":
    main()
