# crownseg — individual tree crowns from real annotations

A folder of its own, so that nothing in the existing pipeline has to be touched.
The scripts here import only from each other, never from the repository root.

## Yes, the dataset has ground-truth segmentations

**BAMFORESTS** (Troles et al. 2024, *Remote Sensing* 16(11), 1935;
CC BY-NC-SA 4.0) lives under `/scratch/shared/$USER/data/bamforests`. These are
**instance** polygons, not just boxes and not just a foreground mask —
hand-digitised crown outlines in COCO format, one class `tree`, 2048x2048 tiles
at **1.70 cm/px**:

| Split | Tiles | Crowns | Areas |
|---|---|---|---|
| train | 1439 | 58,228 | Stadtwald, Tretzendorf |
| val | 382 | 15,177 | Stadtwald, Tretzendorf |
| **test1** | 313 | 6,720 | **Hain** — contained in neither training nor val |
| test2 | 322 | 12,320 | Stadtwald, Tretzendorf |

On average 40 crowns per tile, median crown width 258 px (≈ 4.4 m), p95 530 px.
No multipolygon in the set; every crown is exactly one ring.

**test1 is the only honest number.** Hain appears nowhere in training; test2
only measures how well the already-seen areas fit. Always report both
separately.

The fourth TIFF band is the alpha channel of the orthomosaic and is discarded.

## Why not simply carry on with `crownnet.py`

The existing approach predicts three maps (interior, edge, centre) and cuts the
instances out afterwards by watershed. Measured on BAMFORESTS:

| Area | Precision | Recall | F1 @ IoU 0.5 |
|---|---|---|---|
| Hain | 0.074 | 0.056 | **0.063** |
| Stadtwald | 0.176 | 0.090 | **0.119** |
| Tretzendorf | 0.076 | 0.030 | **0.043** |

Two structural reasons why this will not get better in that form:

1. **Crowns overlap.** A single label map cannot represent that — when
   rasterising, the last crown overwrites the previous one. With interlocking
   broadleaves that is the rule, not the exception, and so the target itself is
   wrong.
2. **The network never predicts an instance**, only where an interior ends. The
   actual separation decision is made by the watershed, on a map that was not
   optimised for it. There is also no per-crown confidence, hence no dial
   between precision and recall.

On top of that came the frozen backbone: what was trained were ~2 M parameters
of a head, on images downscaled by a factor of 0.34.

## The plan

**Stage 1 — preparation.** TIFF to JPEG at original resolution, polygons per
tile as `annotations.json`. No merging into label maps, no downscaling.
`bamforests.py`

**Stage 2 — Mask R-CNN (`maskrcnn.py`).** Every crown its own mask with its own
confidence, overlap allowed. ResNet50-FPN-v2, COCO-pretrained, fully fine-tuned.
The mask head runs at 56x56 instead of the usual 28x28 (`--mask-pool 28`),
because the crowns are enormous by COCO standards. Training happens on 1024
crops with scale jitter 0.5–1.0; inference runs in a sliding window, where
crowns cut at the window edge are discarded and the overlap catches them
completely in the neighbouring window.

**Stage 3 — measuring (`metrics.py`).** F1 at IoU 0.5 (the number that counts
when a crop per crown later goes to the species classifier) and AP@0.5
(threshold-independent, comparable with published numbers). Separately per area.
Crowns below 400 px area drop out on both sides — the GT contains polygons down
to 3 px, which no method can hit.

**Stage 4 — transfer to our own frames.** This is where the actual risk sits,
not in the model: BAMFORESTS has 1.70 cm/px, a 1920 px frame from 100 m altitude
at 73.7° field of view has 7.8 cm/px — a factor of 4.6. A crown that was 258 px
wide in training is 56 px wide in the frame. `--mode predict` therefore scales
the frames up to the BAMFORESTS scale, derived from altitude and field of view,
before the model sees them. For the folders with unknown altitude that remains
an estimate — the same open point as with species identification.

**Stage 5, open.** Once Mask R-CNN stands and the number on test1 is solid: the
56x56 mask is still coarse for 300 px crowns (≈ 5 px per mask pixel). Sharpening
the outlines — SAM3 with the predicted box as prompt, or an edge head — is the
next lever, but measure first, then build.

## Usage

```bash
sbatch crownseg/sbatch/run_prepare.sbatch              # once, ~2456 tiles
sbatch crownseg/sbatch/run_maskrcnn.sbatch             # training
MODE=eval    sbatch crownseg/sbatch/run_maskrcnn.sbatch
MODE=inspect sbatch crownseg/sbatch/run_maskrcnn.sbatch
MODE=predict sbatch crownseg/sbatch/run_maskrcnn.sbatch
EXTRA="--epochs 40 --scale-jitter 0.3 1.0" sbatch crownseg/sbatch/run_maskrcnn.sbatch
```

| Where | What |
|---|---|
| `/scratch/shared/$USER/data/bamforests/crownseg/` | prepared tiles |
| `/scratch/shared/$USER/data/treeclf/checkpoints/crownseg_maskrcnn.pth` | weights |
| `results_crownseg/` | evaluation, overlays (gitignored) |

## Measured

First run (jobs 666/667, selection by validation loss, i.e. epoch 2; 40 tiles per
split, score >= 0.5, IoU >= 0.5):

| Split | Area | GT | Predictions | Precision | Recall | F1 | mean IoU | AP50 |
|---|---|---|---|---|---|---|---|---|
| test2 | Stadtwald | 2126 | 3180 | 0.523 | 0.783 | **0.627** | 0.761 | 0.661 |
| test1 | Hain | 780 | 2628 | 0.094 | 0.315 | **0.144** | 0.675 | 0.128 |

For comparison, `crownnet.py`: 0.119 (Stadtwald) and 0.063 (Hain).

So it works on known ground and not on the unfamiliar one — and for a measurable
reason, not a vague one:

| | Crowns/tile | annotated area | median Ø | p95 Ø |
|---|---|---|---|---|
| Stadtwald (train) | 47.5 | 0.57 | 281 px | 503 px |
| Tretzendorf (train) | 32.3 | 0.56 | 324 px | 664 px |
| **Hain (test1)** | 21.5 | 0.57 | **392 px** | **842 px** |

Same area coverage, half as many crowns: the trees in Hain are larger. The model
breaks them into pieces (2628 predictions for 780 crowns), and the ones it does
hit sit properly (mean IoU 0.675). Three causes, all fixed:

1. **Anchors.** Mask R-CNN proposes 32–512 px by default. An 842 px crown cannot
   be proposed by the RPN at all, no matter how long you train.
   `--anchor-scale 2.0` shifts the ladder to 64–1024 px.
2. **Scale jitter only went downwards** (0.5–1.0), so the network never saw
   crowns larger than those in training. Now 0.6–1.8.
3. **Model selection by validation loss.** With Mask R-CNN that routinely keeps
   rising while accuracy is still improving — so epoch 2 of 25 was selected. Now
   instance F1 on whole validation tiles decides.

Two further bugs found and fixed on the way: the sliding window discarded
everything touching a window edge — a crown wider than the overlap touches an
edge in *every* window and disappeared completely (now assigned via the crown
centroid); and the evaluation's tile sample took the first N instead of sampling
evenly, which dropped Tretzendorf out of test2 entirely.

## Method comparison on a common basis

All methods on the same 40 tiles from test1 (Hain, GT 780 crowns), IoU >= 0.5,
measured with `eval_labels.py` over label maps. This also makes the methods from
the repository root comparable without touching them.

| Method | Predictions | Precision | Recall | F1 | mean IoU |
|---|---|---|---|---|---|
| **EoMT (DINOv3), trained** | 814 | 0.478 | 0.499 | **0.488** | 0.755 |
| Mask R-CNN, trained | 424 | 0.583 | 0.317 | 0.410 | 0.714 |
| SAM 3 + Depth Pro, split + seed | 932 | 0.320 | 0.382 | 0.348 | 0.725 |
| SAM 3 + Depth Pro, split only | 570 | 0.405 | 0.296 | 0.342 | 0.764 |
| SAM 3 `tree`, corrected crown size | 509 | 0.395 | 0.258 | 0.312 | 0.780 |
| Hybrid: SAM 1 + depth watershed | 812 | 0.241 | 0.251 | 0.246 | 0.776 |
| Depth + SAM 1, promptable | 304 | 0.405 | 0.158 | 0.227 | 0.759 |
| `crownnet.py` (starting point) | – | 0.074 | 0.056 | 0.063 | 0.682 |

### Architecture comparison, full splits

40 evenly sampled tiles per split, direct instances instead of label maps:

| Architecture | test2 Stadtwald | test2 Tretzendorf | **test1 Hain** | IoU Hain | AP50 Hain |
|---|---|---|---|---|---|
| **EoMT (DINOv3)** | 0.738 | 0.723 | **0.568** | 0.769 | 0.491 |
| Mask2Former (Swin) | 0.565 | 0.562 | 0.454 | 0.708 | 0.372 |
| Mask R-CNN | 0.629 | 0.544 | 0.367 | 0.706 | 0.241 |

Identical data path, identical ground footprint (1024 px tile), identical window
logic, identical metric — what is being compared is the architecture, not the
environment. Both query-based models start from COCO instance weights.

On the unfamiliar area EoMT holds a mean IoU of **0.769**, i.e. SAM level
(0.76–0.79), which no other trained model reaches. The obvious plan of combining
good selection with SAM edges is thereby moot — EoMT delivers both.

### What the numbers do not show

The visual comparison (`show_gt.py --labels`) exposes two kinds of error:

1. **Merging in dense stands.** Where the GT separates three or four trees,
   there is often a single EoMT mask. This matches the picture given by recall
   0.499 at good IoU: what is hit is hit well.
2. **Non-trees.** House roofs get segmented as crowns. BAMFORESTS knows only the
   class `tree`; everything else is unlabelled background, from which the model
   never learned that it is not a tree.

### What depth contributes

Broken down by role, measured rather than assumed:

| Role | Effect |
|---|---|
| **split** — break up a SAM mask spanning several peaks | 0.312 → 0.342, better with Depth Pro than with DA-V2 (0.342 vs 0.315) |
| **seed** — free peaks as point prompts | 170 free peaks → 113 candidates → 101 overlapping too much → **12 crowns**. Hardly any effect |
| **complete** — watershed on the remaining area | **0 hits out of 291 crowns** at IoU 0.5, 2.7 % at IoU 0.1 |

The peak *positions* are good (64–67 % lie inside a real crown), but they mostly
point at trees SAM 3 has already found. For the missed crowns, depth does not
supply a peak of its own either. The ceiling of F1 0.606 computed in advance was
therefore too optimistic: it assumed every free peak would become an independent
new crown.

### A deadlock that cost compute time

Both training runs stalled after 17 and 16 epochs — main and worker processes in
`do_poll`, GPU idle, SLURM still reporting RUNNING. The cause is the OpenCV
thread pool in forked DataLoader workers; fixed with `cv2.setNumThreads(0)` in
`bamforests.py`. The reported values come from the best epochs (13 and 12), both
of which lay before the hang and had been declining for four epochs.

## Status

- [x] Data layer, model, metric, job scripts
- [x] Preparation: 2456 tiles, 5.2 GB, GSD 1.6998 cm/px (GeoTIFF tag)
- [x] First training run + evaluation + error diagnosis
- [x] Second run with anchors, jitter and F1 selection: test1 0.367, test2 0.629 / 0.544
- [x] Method comparison against SAM 3, depth prompt, hybrid, Depth Pro
- [x] EoMT and Mask2Former trained and compared: EoMT wins clearly (test1 0.568)
- [ ] Depth as a fourth input channel, measured against the EoMT RGB run
- [ ] Train EoMT through cleanly (the runs were aborted after the deadlock)
- [ ] Application to our own frames
