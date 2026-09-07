# Individual crowns on BAMFORESTS — experiment report

Status: 2026-08-20 · Dataset: BAMFORESTS, 92,445 crown polygons · Test area: Hain (test1), never seen in training

## Summary

The goal is a segmentation that captures every tree individually — the
prerequisite for later handing one crop per crown to species identification. The
starting point (`crownnet.py`) hit 6 % of the crowns. Today's state hits 55 %.

The jump did **not** come from combining the existing tools more cleverly. SAM,
SAM 3, Depth Pro and depth watershed got stuck between 23 % and 35 % in every
wiring that was tested. It came from training on real crown annotations with a
query-based architecture.

| | |
|---|---|
| F1 at the start (crownnet) | 0.063 |
| best tool combination | 0.348 |
| **EoMT, trained** | **0.554** |
| mean IoU of the hits | 0.773 |

## Why an external dataset at all

Our own drone frames have no annotations. Without truth, any segmentation can
only be looked at, not judged — and that is exactly where the earlier work got
stuck: it was rated by crown count, area coverage and the share of "suspicious
separations". The last one is defined via saddle prominence in the *estimated*
depth, i.e. via the same depth that produced the separation. A quantity that
grades itself.

**BAMFORESTS** (Troles et al. 2024) solves that. 2456 tiles of 2048 × 2048 px
from four forests around Bamberg, 1.70 cm ground sampling distance, 92,445
hand-digitised crown polygons in COCO format — real instance boundaries, not just
boxes.

| Split | Tiles | Crowns | Areas | Role |
|---|---|---|---|---|
| train | 1439 | 58,228 | Stadtwald, Tretzendorf | training |
| val | 382 | 15,177 | Stadtwald, Tretzendorf | model selection |
| **test1** | 313 | 6,720 | **Hain** | the only real transfer test |
| test2 | 322 | 12,320 | Stadtwald, Tretzendorf | known areas |

**Only test1 counts.** Hain occurs neither in training nor in validation. test2
merely measures how well the already-seen areas fit — anyone throwing both
numbers together reports a better result than they have.

> **A peculiarity of the annotation.** Only around 57 % of the tile area belongs
> to a crown at all. The rest is shadow, undergrowth, gap — unlabelled. Anything
> a method finds there counts as a false alarm, even if a tree actually stands
> there. And there is only the one class `tree`: no model can learn from this
> data that a house roof is not a tree.

## The measuring rig

Before the experiments came a method-independent measurement, so that everything
afterwards stays comparable and not every variant brings its own definition of
success.

| Quantity | What it answers |
|---|---|
| F1 @ IoU 0.5 | Share of crowns hit at the confidence threshold actually used. Greedy assignment by descending confidence, each true crown claimed at most once |
| mean IoU | How well the **hit** crowns sit — separates "finds little but cleanly" from "finds a lot but imprecisely" |
| AP50 | Threshold-independent quality of the ranking. Only meaningful for methods with a per-instance confidence |
| per area | Reported separately, never averaged. Hain behaves differently from Stadtwald, and that difference is exactly the interesting quantity |

All methods write label maps in the same format, regardless of which code
produced them (`eval_labels.py`). That made it possible to rank the older runs
retrospectively without touching them. Likewise the window logic was pulled into
a shared module (`tiling.py`) — a comparison in which the methods tile
differently measures the tiling as well.

## The experiments

### 01 · The starting point: three maps plus watershed — *rejected*

`crownnet.py` predicts interior, edge and centre and cuts the instances out
afterwards by watershed. Measured on Hain: **F1 0.063**.

Two structural reasons that no amount of extra training fixes. First, crowns
overlap — a single label map cannot represent that; when rasterising, the last
crown overwrites the previous one. The training target itself is therefore
wrong. Second, the network never predicts an instance, only where an interior
ends; the separation decision is made by the watershed on a map that was not
optimised for it. There is also no per-crown confidence.

### 02 · Mask R-CNN: every crown its own mask — *partially*

First training run: **0.627** on known ground, **0.144** on Hain — with 2628
predictions for 780 crowns. Massive over-production.

The cause was measurable, not vague: at the same annotated area coverage, Hain
has only 21.5 crowns per tile against 47.5 in the Stadtwald. The trees there are
larger — median diameter 392 px against 281 px, p95 842 px against 503 px. The
model broke them apart.

Three corrections: anchor ladder stretched from 32–512 px to 64–1024 px (an
842 px crown could not even be proposed by the RPN before), scale jitter upwards
as well as downwards, and model selection by instance F1 instead of validation
loss. Result: Hain **0.367**.

### 03 · SAM 3 with the text prompt `tree` — *partially*

Without any training, **0.281** — more than the first Mask R-CNN state. The mean
IoU of the hits was 0.772, the best boundaries in the whole field.

The run was held back by a wrong size prior: `--crown-px 100` sets the expected
crown area to 7854 px², and with the factor of 5 everything above 39,270 px² was
thrown out. A real Hain crown has around 59,000 px² — correctly found large
crowns were discarded as "too big". On top of that, the 2×2 tiling inevitably
cuts in when 274 px crowns meet 717 px windows. Corrected: **0.312**.

### 04 · Depth for splitting — *works*

A SAM mask lying across several treetops is broken up at the tops — watershed
inside the mask, prominence measured relative to the respective instance.

With Depth Pro: **0.312 → 0.342**. With Depth-Anything-V2 only 0.315. For
separating two neighbouring tops what counts is the edge sharpness of the depth,
not its metric correctness — the conjecture was confirmed.

### 05 · Depth as a detector on the remaining area — *rejected*

Treetops in the area not captured by SAM get a watershed basin. Of **291 crowns
added this way, not a single one hits** a real crown at IoU 0.5; at IoU 0.1 it is
2.7 %. Not a threshold artefact.

The reason is structural: by construction, the remaining area is what SAM 3
considered not to be a tree. Searching for crowns there means working against the
decision of a model that reaches a mean IoU of 0.78 on this task. The same step
sits unchanged in `segment_hybrid.py` and explains its 0.246.

### 06 · Treetops as additional prompts — *hardly any effect*

An obvious objection: in the completion step the *shape* came from the watershed
— a point prompt takes only the *where* from the depth and leaves the boundary to
the image model. The preliminary measurement supported it: 64–67 % of the free
tops lie inside a crown SAM 3 missed, ceiling F1 0.606.

Built and measured: 170 free tops → 113 candidates → 101 overlapping too much →
**12 crowns**. Overall result 0.348 instead of 0.342, and that gain comes almost
entirely from more aggressive splitting, not from the 12 seeded crowns.

The ceiling was too optimistic because it assumed every free top would become an
independent *new* crown. In fact the tops mostly point at trees SAM 3 already
has. For the missed crowns, depth does not supply a top of its own either.

### 07 · Query-based architectures — *works clearly*

EoMT with a DINOv3 backbone and Mask2Former with Swin, both over the same data
path, the same ground footprint, the same window logic and the same metric. Both
predict fixed queries, each with its own mask and its own score — no anchors, no
NMS, no post-processing. The anchor problem from experiment 02 cannot occur
there.

On Hain: **EoMT 0.554**, Mask2Former 0.454, Mask R-CNN 0.367. EoMT holds a mean
IoU of 0.773 in doing so — SAM level, which no other trained model reaches.

That makes a plan considered in the meantime unnecessary, namely combining good
selection with SAM boundaries: EoMT delivers both at once.

## Overall result

All methods on the same 40 tiles from test1, 780 true crowns, IoU ≥ 0.5, measured
over label maps.

| Method | Predictions | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|
| **EoMT (DINOv3), trained** | 814 | 0.478 | 0.499 | **0.488** | 0.755 |
| Mask R-CNN, trained | 424 | 0.583 | 0.317 | 0.410 | 0.714 |
| SAM 3 + Depth Pro, split + seed | 932 | 0.320 | 0.382 | 0.348 | 0.725 |
| SAM 3 + Depth Pro, split only | 570 | 0.405 | 0.296 | 0.342 | 0.764 |
| SAM 3 `tree`, corrected | 509 | 0.395 | 0.258 | 0.312 | 0.780 |
| Hybrid: SAM 1 + watershed | 812 | 0.241 | 0.251 | 0.246 | 0.776 |
| Depth + SAM 1, promptable | 304 | 0.405 | 0.158 | 0.227 | 0.759 |
| crownnet (starting point) | — | 0.074 | 0.056 | 0.063 | 0.682 |

Architecture comparison over the full splits, 40 evenly sampled tiles per split,
direct instances instead of label maps:

| Architecture | test2 Stadtwald | test2 Tretzendorf | **test1 Hain** | IoU Hain | AP50 Hain |
|---|---|---|---|---|---|
| **EoMT (DINOv3)** | 0.688 | 0.698 | **0.624** | 0.773 | 0.507 |
| EoMT, short schedule | 0.721 | 0.687 | 0.554 | 0.773 | 0.406 |
| Mask2Former (Swin) | 0.565 | 0.562 | 0.454 | 0.708 | 0.372 |
| Mask R-CNN | 0.629 | 0.544 | 0.367 | 0.706 | 0.241 |

What is remarkable is not only the ranking but the smaller drop from the known to
the unfamiliar area: EoMT falls from 0.72 to 0.55, Mask R-CNN from 0.59 to 0.37.
It transfers better, not merely scores better in absolute terms.

The EoMT numbers come from the complete 30-epoch run. The earlier, aborted run
was at 0.568 / 0.738 / 0.723 — so the spread between two runs of the same
configuration is around 0.015 to 0.035. Differences of that magnitude are not
interpretable; the gap to Mask2Former (0.100) and Mask R-CNN (0.187) is clearly
above it.

## What the numbers do not show

The visual comparison against the truth (`show_gt.py --labels`) exposes two kinds
of error that appear in no metric.

**Merging in dense stands.** Where the annotation separates three or four trees,
there is often a single EoMT mask. That fits the picture given by recall 0.499
together with good IoU of the hits: what is hit sits well, but in a crowd things
get lumped together.

**Non-trees.** House roofs get segmented as crowns. The dataset knows only the
class `tree`, everything else is unlabelled background — that a roof is not a
tree is stated nowhere in this data.

## Errors along the way

Four of them distorted results before they were noticed. They are recorded here
because they explain why individual intermediate states looked the way they did.

- **The sliding window discarded large crowns entirely.** The rule "discard
  everything touching a window edge" hits a crown wider than the overlap in
  *every* window — it disappeared completely. Replaced by assignment via the
  crown centroid, with an overlap larger than the biggest expected crown.
- **The tile sample took the first N alphabetically.** Tretzendorf thereby fell
  out of test2 entirely; the early test2 numbers were pure Stadtwald. Replaced by
  even sampling across the split.
- **Model selection by validation loss.** With detection models the loss
  routinely keeps rising while accuracy is still improving — so epoch 2 of 25 was
  selected. Replaced by instance F1 on whole validation tiles.
- **The ceiling was computed against the wrong run.** The 0.606 for the seeding
  step referred to the masks of a different configuration from the one it was
  supposed to judge.

> **Operational incident.** Both training runs stalled after 17 and 16 epochs —
> main and worker processes in `do_poll`, GPU idle, while SLURM kept listing them
> as running. The cause is the OpenCV thread pool in forked DataLoader workers;
> fixed with `cv2.setNumThreads(0)`. The repeat run over the full 30 epochs
> confirms the state (best validation F1 0.707 against 0.718 in the aborted run,
> on test1 0.554 against 0.568 — spread, not a difference). The numbers reported
> here are throughout those of the complete run.

## What is currently running

- **Depth channel: done, with no demonstrable benefit.** Depth Pro precomputed
  over all 2456 tiles, EoMT trained with depth as a fourth input channel (zero
  weights in the new channel, so that the comparison does not also measure an
  initialisation jump).

  | | test1 Hain | test2 Stadtwald | test2 Tretzendorf |
  |---|---|---|---|
  | RGB | 0.554 | 0.721 | 0.687 |
  | RGB + depth | 0.559 | 0.733 | 0.698 |

  All three differences are positive, but at +0.005 to +0.012 they lie **below
  the spread between two identical runs** (0.015–0.035). No benefit can be
  derived from that; establishing one would need several runs per variant.

  **Addendum: the depth-only run.** The same setup without any colour pixel
  (`--depth-only`), for a long time measured only on the validation set and
  therefore never reported here. Caught up on test1:

  | | test1 Hain | test2 Stadtwald | test2 Tretzendorf | IoU Hain | AP50 |
  |---|---|---|---|---|---|
  | RGB | 0.554 | 0.721 | 0.687 | 0.773 | 0.406 |
  | **depth only** | **0.514** | 0.646 | 0.666 | 0.748 | 0.379 |
  | RGB + depth | 0.559 | 0.733 | 0.698 | — | — |

  **93 % of the RGB quality without the image**, at almost the same boundary
  quality. That re-frames the depth strand: the reason fusion brings nothing is
  not *little information about crowns* but **the same** information. Depth Pro
  estimates monocularly from exactly this image — the depth map is not a second
  measurement but a learned transformation of the first. That it almost carries
  the task alone shows that the crown structure survives the transformation; that
  it adds nothing when combined shows that it brings nothing new.

  That also explains why Ruschhaupt et al. arrive at the same result with a
  *real* photogrammetric CHM: in a closed canopy the separating information is
  missing from the measured height as well.

  **On the fine-tuned maps from `depthft/`:** there Depth Pro was retrained on
  FORTRESS and the scale error fixed (AbsRel 0.98 → 0.120). For segmentation that
  is irrelevant, because `depthcache.py` stores the depth **normalised per tile**
  — the absolute scale that fine-tuning repairs is discarded before the model
  ever sees the map. In the relative structure we actually use, the pure model is
  even ahead when given the scale factor (AbsRel 0.074 against 0.120). What
  remains open is the edge sharpness at crown boundaries, which AbsRel does not
  measure and which made the difference once before (Depth Pro 0.342 against
  Depth-Anything 0.315).
- **Evaluation.** Done: test1 0.554, test2 0.721 / 0.687. The numbers above are
  already those of the complete run.
- **Open: retraining on our own annotations.** There the scale is in the way:
  BAMFORESTS has 1.70 cm/px, a 1920 px frame from 100 m altitude around
  7.8 cm/px — a factor of 4.6. The frames are upscaled accordingly, but **it
  cannot be checked** as long as no annotations exist on our own imagery. Ten
  labelled frames would be enough to turn the impression into a number.

## Application to our own frames

All models run via `crownseg/sbatch/run_frames.sbatch` on
`/cold/Mahfuz/chosen_frames` and write label maps to `results_frames_<model>/`
plus the four diagnostic views to `results_views_<model>/`.

### The image scale had been assumed wrongly

The calculation "100 m flight altitude → 7.8 cm/px → factor 4.6" is formally
correct, but the 100 m were never measured, they were a default value. With
factor 4.6 Mask R-CNN found **exactly zero** crowns on all 33 frames, even at
confidence threshold 0.05.

`scale_probe.py` instead measures the scale from the model's response: the same
ground footprint at various resolutions, looking for the maximum of instance
count and confidence. Mask R-CNN is suitable for this precisely because of its
fixed anchor sizes — at the wrong scale its confidence collapses, whereas a
query-based model delivers something at any scale.

| Folder | Maximum at | Crowns | Confidence | → GSD | → altitude |
|---|---|---|---|---|---|
| pines | ×1.0–1.4 | 39 | 0.95 | ~1.2–1.7 cm/px | 16–22 m |
| dense | ×1.0–1.4 | 32 | 0.95 | ~1.2–1.7 cm/px | 16–22 m |
| dense1 | ×1.0–1.4 | 32 | 0.93 | ~1.2–1.7 cm/px | 16–22 m |
| mixed | ×1.0 | 20 | 0.94 | 1.70 cm/px | 22 m |
| 100, 80m, mixed1, urban | no interior maximum | ≤ 5 | — | undetermined | undetermined |

The probe has a range that has to be respected: it uses Mask R-CNN as the
measuring instrument, and that responds hardly at all to urban material (at most
1.2 instances at confidence 0.54). For those four folders it is blind. The
default of ×1.0 set as a result is **not a neutral assumption** — it claims that
the image scale happens to match that of BAMFORESTS exactly.

For `urban` a visual comparison over ×0.4 to ×3.0 gave a clear optimum at
**×1.5** (11 → 26 → 40 crowns at ×0.8 / ×1.0 / ×1.5, worse again beyond). At ×2.0
and above — run as a single scale — the model fills whole search windows with
masks where there is lawn: masks with straight edges that cannot be a crown.

The obvious conclusion that the upper multi-scale level is responsible for the
mask over the park meadow has, however, been **checked and is false**: with
levels 0.7/1.0/1.4 (up to ×2.1) and with 0.7/1.0 (up to ×1.5) it arises equally,
41 against 39 crowns with an otherwise nearly identical result. Together with the
failed texture filter this doubly confirms what the dataset already suggested: it
is not a parameter problem.

Two restrictions: from factor ×3 upwards no model finds anything any more,
because bicubically upscaled JPEG frames lose their texture — there "wrong scale"
cannot be distinguished from "image destroyed". And the folder names `100` and
`80m` contradict the measurement; that can only be resolved with the real flight
data.

### Multi-scale and threshold

The visual comparison against the earlier SAM 3 runs showed markedly lower canopy
coverage. Two causes, both fixed:

| Variant (pines/frame_000006) | Crowns | Coverage |
|---|---|---|
| SAM 3 multi-scale | 125 | 77 % |
| EoMT, one scale, threshold 0.5 | 131 | 63 % |
| EoMT multi-scale, threshold 0.5 | 158 | 68 % |
| **EoMT multi-scale, threshold 0.25** | **181** | **69 %** |

**Multi-scale** (`--scale-steps 0.7 1.0 1.4`) is insurance against the unknown
scale, exactly like the tile levels in `segment_sam3.py`. On BAMFORESTS, where
the scale is known, it changes nothing (0.469 against 0.488, within the spread);
on the frames it brings coverage from 63 % to 68 %.

**The threshold of 0.5 was an unchecked default.** Measured on test1:

| Threshold | Predictions | Precision | Recall | F1 | mean IoU |
|---|---|---|---|---|---|
| 0.50 | 818 | 0.578 | 0.532 | 0.554 | 0.773 |
| 0.35 | 896 | 0.557 | 0.561 | **0.559** | 0.770 |
| 0.25 | 957 | 0.533 | **0.574** | 0.553 | 0.769 |
| 0.15 | 1091 | 0.481 | 0.591 | 0.530 | 0.767 |

From 0.50 to 0.25 recall rises by 8 % relative while F1 stays unchanged.
Crucially, the mean IoU stays at 0.77 — the additional crowns are delineated just
as cleanly, no fragment junk comes with them.

### Non-trees: not repairable downstream

On the urban frames EoMT segments park meadows, path edges and roof edges. The
cause is the dataset: BAMFORESTS is pure forest with the single class `tree`,
everything else is unlabelled background and **not a counter-example**. The model
never learned that lawn is not a tree, because it never saw one.

The attempt to fix that with a filter on finished masks (`reject.py`, texture via
the Laplacian magnitude and relief against a ring in the surrogate CHM) failed
when measured:

| Texture threshold | Predictions | Precision | Recall | F1 |
|---|---|---|---|---|
| none | 789 | 0.466 | 0.472 | 0.469 |
| 0.8 | 767 | 0.473 | 0.465 | 0.469 |
| 1.2 | 740 | 0.482 | 0.458 | 0.470 |
| 1.4 | 716 | 0.486 | 0.446 | 0.465 |

Precision rises, recall falls by the same amount, F1 stays flat across the whole
range. The filter removes no non-crowns; it acts like a stricter confidence
threshold and cuts away borderline cases indiscriminately.

The distributions explain why: the urban frames have *higher* texture than the
forest frames (median 1.9–2.9 against 1.7–1.9) — they are high-contrast satellite
images with sharp building edges. Texture and relief measure properties that
dense crowns and dense meadows share.

The obvious way out does not work either: using unannotated BAMFORESTS areas as
counter-examples goes wrong, because 43 % of the area there is unlabelled and a
large part of it does contain trees.

What helps are real counter-examples — annotated tiles in which meadow, path and
roof are marked as non-tree.

### What remains

Coverage saturates at around 69 % and does not reach SAM 3's 77 %. But the
difference is not what it looks like: EoMT finds **more** crowns on **less** area
(181 against 125), it divides the canopy more finely. Which view is right depends
on the actual tree size in the stand.

The remaining gap is attributable to BAMFORESTS' annotation policy — 57 %
annotated area, shadows and gaps unlabelled — and cannot be closed further via
parameters. On your frames every tile counts as forest; in the dataset it does
not.

Coverage per folder in the final state: 58 % (`urban`) to 76 % (`dense1`), around
68 % on average.

## Positioning against the literature

Checked only late, and the comparison corrects several of our own assumptions.

**The scale had been measured correctly.** The BAMFORESTS paper gives a GSD of
1.61–1.82 cm (Stadtwald 1.70 cm) — exactly the value read from the GeoTIFF tag.

**Test set 1 is deliberately the hard case**, but for a different reason than
assumed. It is not only crown size that differs: the species distribution is
radically different (Pinus 36.2 % in training, 1.1 % in Hain; "Other" 6.8 %
against 52.9 %), and Hain was captured with **a different drone and a different
sensor** (DJI Phantom 4, 85 m, 84° against Trinity F90+ / Sony RX1 RII, 120 m,
63°).

**The depth findings are independently confirmed — with measured height.**
Ruschhaupt, Troles & Schmid (2025) investigated the same question, with a
photogrammetric canopy height model (DSM minus the official terrain model) in the
alpha channel, i.e. the same fourth channel. Result: RGB beats RGBA by 0.87 %
(Mask R-CNN) and 1.18 % (Mask2Former); height information has "a negative
influence". That also answers the question that was considered unanswerable here:
*measured* height helps just as little as estimated height.

**Fine-tuning SAM 3 is not worth it.** Measured ceiling on test1: of 428 GT
crowns, only 271 have any of the 3743 raw masks reaching IoU ≥ 0.5 — **63.3 %**.
A perfect selector on those proposals would therefore hardly get beyond EoMT's
current recall of 0.651. On top of that, `Sam3Model` in transformers has no
training path; the Hungarian matching and four loss terms would have to be
written from scratch.

**An error in our own metric.** AP was computed per tile and averaged. At around
20 crowns per tile a short curve can easily run to 1.0, which flatters the mean —
the pooled, COCO-standard computation is lower (test1 0.406 instead of 0.458).
Only the pooled one is comparable with published numbers.

**Where we stand.** Ruschhaupt et al. report 69.05 % AP50 (Mask R-CNN) and
68.89 % (Mask2Former) on Stadtwald+Tretzendorf; we reach 59.9 % and 58.8 % there.
So we are **below** the published baselines — the earlier phrasing "EoMT is the
clear winner" held only against our own, weaker Mask R-CNN implementation. The
test sets are however not identical (their split of 1621 tiles at 1024 px against
our 40 tiles at 2048 px).

### What the learning-rate schedule does

The attempt to close the gap through longer training showed more than expected.
At 80 epochs of 400 steps the best state fell on **epoch 5** — i.e. on 8000
crops, fewer than the old run saw in total. The result on test1 is nevertheless
markedly better: F1 0.624 instead of 0.554, AP50 0.507 instead of 0.406.

The cause is not the number of steps but the schedule: OneCycleLR spreads warm-up
and decay over the *planned* total length. At 32,000 planned steps, epoch 5 is
still in the warm-up at a low rate; in the old run with 6000 steps it was long
past the peak by then. The learning rate of 1e-4 was simply too high.

**And the validation misled us in the process.** It runs on 8 tiles from
Stadtwald and Tretzendorf — areas where little changes. While it stagnated, test1
improved by 0.07. Model selection by that validation does not optimise what
matters.

## A second dataset and a variable field of view

On the pine frames the model lumps several trees into one mask. The obvious
explanation: BAMFORESTS annotates crowns with a median of 4.78 m (Stadtwald) to
6.66 m (Hain), and Nik's pines are about half that size.

**Quebec Trees** (Cloutier et al. 2023, CC-BY-4.0) covers the missing range.
Measured on the polygons themselves, not taken from the publication:

| | median ⌀ | p5 | p95 | GSD |
|---|---|---|---|---|
| Quebec Trees, 22,933 crowns | 4.09 m | **1.82 m** | 8.54 m | 1.64 cm/px |
| BAMFORESTS Stadtwald | 4.78 m | – | – | 1.70 cm/px |
| BAMFORESTS Hain | 6.66 m | – | – | 1.82 cm/px |

By species: Abies balsamea 2.78 m (n=2895), Thuja occidentalis 2.97 m (n=1510),
Picea 3.02 m (n=599) — the range of the pines in `pines`.

Prepared with `quebec.py`: polygons from UTM into pixel coordinates, tiles of
2048 px as with BAMFORESTS, zone 3 held back completely as the test area.
543 / 459 / 214 tiles, 112 crowns per tile against 40 in BAMFORESTS.

**Variable field of view** (`quebec_cog.py`): crops taken directly from the
orthomosaic instead of from pre-cut tiles. From a 2048 tile the scale can only be
widened to 5.4 cm/px; from the 40,000 × 42,000 px mosaic, arbitrarily far. The
COGs have overview levels down to 1/128, and a 7500 px window read down to 640
costs 7 ms — less than a JPEG tile.

The limit is set not by the data but by the model: **EoMT has 200 queries.** At
360–440 crowns per hectare, around 150 crowns fit into a field of view of 0.31 ha,
which corresponds to 8.7 cm/px. Crops that are too full are discarded instead of
giving the network an unsolvable target (measured: 0.5 % empty crops, median 31
crowns).

### Result

| | test1 Hain | test2 Stadtwald | test2 Tretzendorf | Quebec zone 3 | crown ⌀ `pines` |
|---|---|---|---|---|---|
| BAMFORESTS only | **0.624** | 0.688 | 0.698 | – | 2.45 m |
| + Quebec | 0.583 | **0.730** | **0.717** | 0.631 | 2.32 m |
| + variable field of view | 0.558 | 0.721 | 0.714 | **0.638** | 2.37 m |

**Quebec brings** +0.04 on the BAMFORESTS core areas and delivers, for the first
time, a value for fine-crowned stands (zone 3: F1 0.631, recall 0.777 — the
highest measured). It costs 0.04 on the hard transfer test, Hain.

**The variable field of view brings nothing:** +0.007 on Quebec, −0.009 to −0.025
on BAMFORESTS. The sampler demonstrably works correctly, so the result is not a
measurement error.

**And the actual question stays open.** The predicted crown diameter on `pines`
lies between 2.32 and 2.45 m across all three models. Five explanations for the
coarse segmentation were tested and rejected: scale at inference, multi-scale
levels, confidence threshold, a finer dataset, variable field of view. Whether
2.4 m is too large cannot be decided — the number comes from the model's own
output.

### The scale of the urban imagery

Determined from the errors themselves: the objects along the row of parking
spaces wrongly segmented as crowns are 18–28 px long with an elongation of
1.5–2.0, i.e. cars from above. At a vehicle length of 4.5 m that gives
**17–25 cm/px**.

| | GSD | factor to BAMFORESTS |
|---|---|---|
| Training data | 1.6–1.8 cm/px | 1 |
| Drone frames | ~2.0 cm/px | 1.2 |
| **urban screenshots** | **~20 cm/px** | **12** |

For `urban` the models are mis-scaled by a factor of 12. That explains missing
trees and cars-as-crowns at the same time. It would be reachable with around 800
queries instead of 200, or with a dataset at that resolution —
[OAM-TCD](https://huggingface.co/datasets/restor/tcd) offers 10 cm/px, 280,000
individual trees, 5072 tiles, CC-BY-4.0, 3.55 GB.

## Stage 2: species identification on the instances

The instances from `crownseg` go to the DINOvTree head (`classify.py`), using the
centroid of the mask instead of the box centre and the scale from the measurement
instead of from an assumed flight altitude.

### The classifier works — on its own domain

On Nik's frames it delivers nonsense: in the pine stand, 384 of 158 crowns are
listed as yellow birch. Two explanations were possible — transfer, or our own
wiring — and Quebec zone 3 with species labels separates them.

**90.1 % accuracy** (1303 of 1446 crowns), against 27.2 % for the most frequent
class and 7.1 % for chance. Per species between 48 % (Acer saccharum) and 100 %
(Pinus strobus, Tsuga canadensis). So the wiring is correct; the nonsensical
answers on our own frames are a pure transfer problem. The label set knows
*Pinus strobus*, not *Pinus sylvestris* — the model has to answer and picks the
most similar option.

Worrying in this: mean confidence **0.99 on correct and 0.90 on wrong**
predictions. Outside the domain, confidence is useless as a warning signal.

### Clustering solves the label problem

Instead of forcing things into Canadian classes: group the feature vectors,
without labels. Tested on Quebec zone 3, clustered without the labels, compared
afterwards:

| Clusters | ARI | NMI | Purity |
|---|---|---|---|
| 8 | **0.764** | 0.728 | 82.9 % |
| 12 | 0.686 | 0.725 | 86.8 % |
| 14 | 0.595 | 0.720 | **89.0 %** |
| 20 | 0.410 | 0.666 | 89.8 % |
| *labelled class head* | *0.762* | *0.773* | *90.1 %* |
| *random groups* | *−0.001* | *0.022* | – |

**Clustering without labels reaches the same ARI as the labelled classifier.**
The features carry the species information completely; what was missing were only
the names. In practice that means: name twelve to twenty clusters instead of
thousands of trees, with around nine out of ten crowns assigned correctly.

### The crop size was an inherited assumption

DINOvTree was trained on 9.73 m crops, and that number had been adopted
unquestioned. Measured on Quebec, with the crop as a multiple of the *respective*
crown diameter:

| Factor | Crown share | Accuracy | ARI | Purity |
|---|---|---|---|---|
| 1.5 | 44 % | 80.6 % | 0.496 | 83.2 % |
| **2.4** | **17 %** | **86.2 %** | **0.509** | **86.1 %** |
| 3.5 | 8 % | 84.9 % | 0.489 | 82.8 % |
| 5.0 | 4 % | 79.8 % | 0.430 | 80.3 % |
| 8.0 | 1.6 % | 69.2 % | 0.344 | 71.2 % |

The optimum is at factor 2.4. Too tight is worse as well — a bit of surroundings
contributes. With Quebec's 3.41 m crowns, the 9.73 m correspond to factor 2.9 and
are thus close to the optimum; with Nik's 2.4 m pines they mean factor 4 and
around four points of loss. `classify.py` now determines the crop per crown.

That fixes a parameter error, not the transfer problem: with an adapted crop the
pine remains a birch.

## Stage 2 with European species: FORTRESS

The previous two sections end at the same point: the features carry the species
information, but the head points at 14 Canadian classes. The only route that
makes the pine a pine is a label set with European species.

**FORTRESS** (Schiefer, Frey & Kattenborn 2022, DOI 10.35097/538, CC BY 4.0): 47
drone surveys in the southern Black Forest, 79 ha, GSD 0.65–1.87 cm, 9389 species
polygons in 16 classes, plus one nDSM per site. The scale is in the same range as
BAMFORESTS (1.70 cm) and as Nik's imagery; the species are the ones that actually
stand in his stands.

### Labelling by intersection

FORTRESS supplies species polygons, not individual crowns — the polygons often
cover several trees of the same species. `fortress.py` re-tiles the orthomosaics
to 1.70 cm/px, segments with our EoMT, rasterises the species polygons and
intersects the two: every predicted crown gets the species that covers the
majority of its mask, plus `abdeckung` (what share of the mask carries a class at
all) and `reinheit` (share of the majority species).

The first test of this was worthless: I built the semantic map from the same
polygons that also served as truth — 100 % is then not a measurement. Repeated on
predicted crowns: **98.8 % assigned correctly** over 607 crowns, 100 % for the
71 % that pass the coverage and purity thresholds.

Result across all 47 sites: **9373 labelled crown crops**.

| Species | n | | Species | n |
|---|---:|---|---|---:|
| Picea abies | 4669 | | Pseudotsuga menziesii | 157 |
| Fagus sylvatica | 1710 | | Larix decidua | 74 |
| Abies alba | 1018 | | Quercus spec. | 32 |
| *forest floor* | 843 | | Betula pendula | 29 |
| **Pinus sylvestris** | **458** | | *other* | 12 |
| *deadwood* | 189 | | Fraxinus excelsior | 8 |
| Acer pseudoplatanus | 173 | | | |

Forest floor and deadwood are not by-catch here but the first time a dataset
supplies counter-examples at all — until now every predicted crown had to be a
tree.

Notable: ash falls from 175 polygons to 8 crops. The explanation is probably the
intersection — narrow polygons at the crown margin lose the majority to their
neighbours.

### The new head

`train_head.py`: the same frozen DINOv3 backbone, a new head onto the eight
classes with at least 100 examples. Oak, birch, ash and larch drop out — with 8
to 74 examples they can neither be learned nor measured.

Two decisions determine whether the number is worth anything:

**Split by site, not by crop.** Crops from the same site share illumination,
capture date and sometimes the same tree. Eleven of the 47 sites are held back;
the number is therefore a transfer number, just like Hain for BAMFORESTS and zone
3 for Quebec.

**Balanced accuracy instead of raw.** Spruce makes up 59 % of the test set. A raw
accuracy would above all be a measure of how often spruce is recognised
correctly.

**72.8 % raw, 65.8 % balanced** over eight classes — against 59.2 % for
always-spruce and 12.5 % for guessing.

| Class | n | Recall | mostly predicted |
|---|---:|---:|---|
| **Pinus sylvestris** | 98 | **77.6 %** | Pinus sylvestris |
| Picea abies | 1431 | 72.4 % | Picea abies |
| Abies alba | 287 | ~70 % | Abies alba |
| Fagus sylvatica | 277 | ~67 % | Fagus sylvatica |
| Pseudotsuga menziesii | 20 | ~45 % | Pseudotsuga menziesii |
| Acer pseudoplatanus | 48 | ~35 % | **Fagus sylvatica** |
| *forest floor* | 177 | ~86 % | forest floor |
| *deadwood* | 78 | ~86 % | deadwood |

Three observations the numbers themselves yield:

*Sycamore becomes beech.* The confusion goes almost entirely in one direction.
Both are broadleaves with similar crown texture, and beech has ten times as many
training examples — in case of doubt the more frequent one wins.

*Long training buys almost nothing but spruce.* Over 60 epochs the raw accuracy
rises from 78 % to 80 %, the balanced one stalls at around 64 %. Selection by
balanced accuracy therefore keeps an early state.

*Two points of spread between runs.* The same setup with a different random seed
gave 78.2 % / 68.2 % instead of 72.8 % / 65.8 %. The stored head is the weaker of
the two; swapping it for the better one would mean selecting on the test set. The
seed has been fixed since.

### Positioning

The Quebec head reaches 90.1 % — but on hand-drawn individual crowns. Our
labelling comes from intersecting our own segmentation with the species polygons;
the errors of the segmentation are baked into the label. So the gap measures not
only the model but also the labelling source.

What matters is not the comparison but that the head **can** answer *Pinus
sylvestris*. That was the entire reason for taking FORTRESS.

### Does the instance source matter?

The head above learned on crowns found by our EoMT. The obvious control: the same
chain with SAM 3 as the instance source. `fortress.py` got a switch
`--segmenter sam3` for that — prompt "tree", tile levels 2/3/4, threshold 0.15,
cut instances discarded, merged by score, i.e. exactly the settings of the run on
our own frames. Everything after that stays the same: intersection, crop factor
2.4, the same frozen backbone, the same 47 sites, the same split, the same seed.

SAM 3 delivers 8840 crops against 9373, but better distributed: pine 555 instead
of 458, fir 1336 instead of 1018, Douglas fir 219 instead of 157, spruce 3881
instead of 4669. The crowns are smaller (median 4.05 m against 4.56 m).

| Class | from SAM 3 instances | from EoMT instances |
|---|---:|---:|
| **balanced overall** | **66.8 %** | **65.8 %** |
| Abies alba | 73.4 % | ~70 % |
| Pinus sylvestris | 72.2 % | 77.6 % |
| Acer pseudoplatanus | 65.0 % | ~35 % |
| Picea abies | 59.7 % | 72.4 % |
| Fagus sylvatica | 56.5 % | ~67 % |
| Pseudotsuga menziesii | 35.7 % | ~45 % |
| *deadwood* / *forest floor* | 91.2 / 80.7 % | ~86 / ~86 % |

**No difference.** One point lies within the two points that already fluctuate
between two runs of the same setup. The raw accuracy falls from 72.8 % to 64.8 %,
but only because the test set is composed differently — spruce makes up 50.7 %
here instead of 59.2 %. The jump for sycamore rests on 20 test crowns and does
not carry weight.

The application to our own frames shows the same: the EoMT head and the SAM 3
head arrive at the same species distribution per folder (pine share in `dense`
56 % against 50 %, in `pines` 3 % against 4 %). For species identification it
therefore does not matter who cut the crown out — as long as the crop contains
the tree, the texture carries the information. That closes one of the open
questions and saves the comparison in future.

A side finding: the SAM 3 head is less convinced of itself (mean confidence 0.65
against 0.79 in `pines`). Given what the Quebec head showed — confidence 0.90 on
*wrong* answers outside the domain — the lower number is a good sign rather than
a bad one.

### Dense DINOv3 features: do the patch features separate on their own?

So far everything ran over crops — a crown is cut out, the backbone delivers *one*
vector. DINOv3, however, delivers its own vector per 16×16 patch, and these patch
features are known to carry an emergent segmentation (LOST, TokenCut, STEGO get
object boundaries out of them without any mask). `dinocluster.py` measures how
much of that is usable for us. The backbone is the DINOv3 fine-tuned on Quebec,
k-means on L2-normalised vectors.

**Area → species.** Against the species polygons of FORTRESS, 40,752 labelled
patches from 16 tiles in four sites, without segmentation and without labels:

| Clusters | DINOv3 NMI / purity | colour only | position only |
|---|---|---|---|
| 4 | **0.485** / 66.7 % | 0.116 / 42.6 % | 0.032 / 33.5 % |
| 8 | 0.445 / 66.1 % | 0.143 / 44.7 % | 0.037 / 33.9 % |
| 12 | 0.430 / 70.8 % | 0.142 / 45.4 % | 0.051 / 36.6 % |
| 20 | 0.476 / **79.0 %** | 0.147 / 48.0 % | 0.063 / 38.9 % |

Most frequent species alone: 33.5 %.

Both controls are necessary and both exonerate the result. **Colour** explains
only a third of the mutual information — the suspicion that autumn colours rather
than species are being sorted here now has a counter-number. **Position** was the
more serious objection: species polygons are large contiguous areas, and the
cluster maps look blocky, so a good NMI could simply come from both being
spatially smooth. Clustering position alone gives 0.03 to 0.06 and purity at
guessing level. The objection was wrong.

**Area → individual crown.** Against the crown polygons of BAMFORESTS test1, with
connected components of the clusters as instances:

| Clusters | Instances | true | Hits | F1 | mean IoU |
|---|---:|---:|---:|---:|---:|
| 4 | 266 | 273 | 12 | 0.045 | 0.652 |
| 8 | 416 | 273 | 25 | 0.073 | 0.648 |
| 12 | 484 | 273 | 43 | 0.114 | 0.656 |
| 20 | 711 | 273 | 70 | **0.142** | 0.660 |

Against 0.624 for the trained EoMT. The emergent segmentation separates species
and stands, not neighbouring trees of the same species — two spruces standing
next to each other are the same thing in the features, and separating exactly
those is the hard part of crown delineation. The mean IoU of 0.66 says: what is
hit is hit cleanly; it is just that almost nothing is hit.

Images in `results_views_dinov3/` — per frame the original next to the cluster map
at 6 and at 20 clusters.

Practical conclusion: the patch features do not replace the segmentation, but they
could carry the species stage — as a species map over the area that one intersects
with our instances. That is the same route `fortress.py` takes, only with learned
clusters instead of labelled polygons.

### Fine-tune Depth Pro on FORTRESS?

Tempting, because FORTRESS ships a measured nDSM per site: RGB in, height in
metres out, no scale ambiguity, 79 ha at around 1 cm. For the segmentation it is
still not worth it — four fusion routes in this report and Ruschhaupt et al. with
a real photogrammetric CHM arrive independently at the same result, that height
does not improve crown segmentation. Only if tree height is itself a goal is the
effort justified.

---

All numbers measured on BAMFORESTS · Troles, Schmid, Fan & Tian (2024),
*Remote Sensing* 16(11), 1935 · CC BY-NC-SA 4.0
