# Depth Pro fine-tuned on FORTRESS — metric tree heights from nadir images

> **Note on the training state:** the weights evaluated in this report, under
> `/scratch/shared/$USER/runs/depthft`, were produced with the log-depth loss used
> at the time. The training code meanwhile uses a corrected Huber loss on the
> metric height error and writes new runs to
> `/scratch/shared/$USER/runs/depthft_huber_v2`. The old measurements documented
> here do not change as a result; the new loss requires retraining.

Everything for this report lives in [`depthft/`](depthft/); the technical
documentation of the individual scripts is in
[`depthft/README.md`](depthft/README.md).

---

## Summary

Depth Pro produces depth maps for our drone frames whose height is wrong.
Measured on FORTRESS nadir images it predicts **0.7 to 2.6 m of depth** where
there are in fact 24 to 85 m — a scale factor of **0.02**, independently of the
flight altitude. Too little depth means: everything sits too close to the camera,
the trees appear too tall. Exactly the symptom observed.

Fine-tuned on FORTRESS, the relative error on test sites that never occurred in
training drops from **AbsRel 0.98 to 0.120**; the share of pixels within 25 % of
the true depth rises from **0.000 to 0.843**.

On our own frames, pure Depth Pro turns every stand into **bushes 0.3 to 2.0 m
tall**. The fine-tuned model delivers 13 to 47 m — plausible tree heights — and
hits the ground level to within **0.6 m**.

Two results were not expected and matter more than the numbers themselves:

1. **The field of view of our camera is not 73.7° but around 48°.** The value in
   the code was a default, not a camera specification. It enters every depth
   linearly — all previous heights from these frames are too small by a factor of
   1.7.
2. **Depth Pro's relative structure was never the problem.** Given the scale
   factor for free, the pure model reaches AbsRel 0.074 and thereby beats the
   fine-tuned one. What was missing was the scale, and only the scale.

---

## 1. Why Depth Pro fails here

Depth Pro is trained on ground-level perspectives — streets, interiors,
portraits. A nadir shot from 80 m does not occur in that. From the check run over
ten crops at 73.7° field of view:

| | Truth | Pure Depth Pro |
|---|---|---|
| Depth at 27 m altitude | 3.4–27.5 m | 1.0–1.9 m |
| Depth at 64 m altitude | 22.7–64.0 m | 1.1–1.6 m |
| Scale factor | 1.00 | **0.06** |
| Field of view (its own head) | 73.7° | 18–41° |

Two things stand out. The scale error is **independent of flight altitude** — the
model always outputs roughly the same thing, it does not read the altitude out of
the image at all. And its built-in field-of-view head, which co-determines the
metric scale, is off by a factor of two.

---

## 2. Where the truth comes from

**FORTRESS** (Schiefer, Frey & Kattenborn 2022, CC BY 4.0): 47 UAV sites in the
southern Black Forest, 1.7 ha each, orthomosaic at 0.77–1.57 cm/px, plus one
normalised height model (nDSM) per site at 5 cm — metres above ground, for every
pixel.

You cannot train on that directly: an orthomosaic has no camera, no field of
view, no depth. Depth only arises from an assumption — hang a nadir camera at
altitude `H` above the stand:

```
depth        d    = H − nDSM
ground samp. GSD  = H / f_px
ground width      = 2 · H · tan(HFOV / 2)
```

A site thus yields arbitrarily many virtual frames **with an exact metric depth
map**, at any flight altitude. Altitude, field of view and position are drawn at
random per crop.

### Training happens in the model's home space

Depth Pro does not output metres but canonical inverse depth. It only becomes
metric in the post-processing:

```
d = (f_px / image width) / D_raw  =  k / D_raw
```

`k` depends solely on the field of view. The old run evaluated here was trained
directly against `D_gt = k / d_gt`. The checkpoint is loadable with the normal
Hugging Face classes; for metric depth, `k` still has to be supplied as in
`inferenz.py`, because the unchanged field-of-view head is unreliable on nadir
images.

The loss is L1 on the log depth (penalises relative rather than absolute error)
plus multi-scale gradient matching (keeps crown boundaries sharp).

---

## 3. Three traps that would have silently spoiled the training

### Exact zeros in the nDSM are fill, not terrain

In the height models `0.00` is by far the most frequent single value — 6 % of the
area at the median, **43 %** in the worst site, in large contiguous blocks under
which the orthomosaic shows closed forest. Real ground scatters around zero; it
does not hit it exactly ten thousand times. These are the places where
photogrammetry could not reconstruct a height.

Learning those areas as ground would mean: crowns at height zero. They are
discarded. The valid share afterwards lies between 51 % and 97 %, at the median
82 %.

### The gradient explodes at the change of scale

The loss lives on `log d = log k − log D`, so its gradient with respect to the
model output is `1/D`. While the model drives `D` down by a factor of 50, that
gradient itself grows by a factor of 50 — a self-reinforcing descent that
overshoots past zero. Behind the head's ReLU it is then dead, and the depth runs
to a hundred thousand times its value. Observed exactly like that: clean descent
up to step 250, then AbsRel 7000 and MAE 400,000 m.

**Solution: do not demand the jump in the first place.** The scale error is
measured over a few batches (0.0315) and used to scale the last 1×1 convolution
of the head. Because it sits before the final ReLU and the factor is positive,
this is exactly equivalent to `D → factor · D` — but as a real weight change, not
as an instruction leaflet. The learning rate of that layer is scaled along with
it, otherwise Adam steps of the usual size would tear the now tiny weights apart.

Effect, even before the first learning step: loss 3.67 → 0.43, AbsRel
0.97 → 0.37.

### The camera has to hang above the treetops

Without a bound there would be crops in which 30 m trees almost reach the lens at
27 m flight altitude. The flight altitude is set per site to at least *highest
treetop + 20 m*.

---

## 4. Results on the test sites

Five sites, 200 crops, seen neither in training nor in validation.

| Variant | AbsRel | MAE | δ<1.25 | Scale error |
|---|---|---|---|---|
| `pur_skalenangleich` *(oracle)* | 0.074 | 4.41 m | 0.941 | 1.000 |
| **`feinabgestimmt_kamera`** | **0.120** | **7.36 m** | **0.843** | 0.945 |
| `feinabgestimmt_hoehenanker` | 0.167 | 8.96 m | 0.797 | 1.170 |
| `pur_hoehenanker` | 0.226 | 11.98 m | 0.632 | 1.231 |
| `pur_fovkopf` | 0.939 | 56.11 m | 0.000 | 0.061 |
| `pur_kamera` | 0.976 | 58.14 m | 0.000 | 0.024 |

`mae_m` is at the same time the error of the **height above ground** — that is
flight altitude minus depth, and the altitude cancels out in the difference.

### The fine-tuning works

AbsRel from 0.98 to 0.120, δ<1.25 from 0.000 to 0.843, scale error from 0.024 to
0.945 — still 5.5 % off at the median.

### But the structure was never the problem

`pur_skalenangleich` is pure Depth Pro, scaled globally so that the median is
exactly right. It reaches **0.074 and thereby beats the fine-tuned model.**

That is not an applicable method — the factor comes from the truth, which you do
not have in deployment. The row is an upper bound, and read correctly it says two
things: Depth Pro's *relative* depth structure on nadir images is excellent, and
what it lacks is exclusively the scale. The fine-tuning comes close to that bound
without any help (0.120), but does not overtake it.

> I had assumed the opposite for a while — because the predicted depth range
> *within* an image was a good metre where in reality there are 40 m, the contrast
> seemed broken too. That was wrong: the small range is a consequence of the
> scale error, not a defect of its own.

### The obvious anchor does not work

A drone knows its flight altitude from barometer and GPS. It is tempting to scale
with that instead of training a model: scale until the deepest point in the image
matches the flight altitude. Measured, that is **worse** — 0.226 instead of 0.976
for pure, but also worse than the 0.120 of the fine-tuning. Even applied to the
fine-tuned model the anchor makes things worse (0.167).

The reason is in the scale error of 1.23: **in a closed canopy the deepest
visible point is not the ground.** Measured from the nDSM data it lies at **0.917
of the flight altitude** at the median (5th–95th percentile: 0.82–0.99). The
learned scale is more reliable than that geometric assumption.

### Image sharpness hardly matters

Brought down to video image quality with blur, noise and JPEG artefacts: AbsRel
0.135 instead of 0.120. The model transfers.

---

## 5. The field of view — the biggest single error in the project

The 73.7° come from a default value in the code, not from a camera
specification. The value enters every depth **linearly**, and you cannot see it
in the depth map: it looks right and is not.

Without EXIF it can be bounded if the flight altitude is known:
`k = ground factor · H / p95(1/D)`. That has a built-in check — **different
flight altitudes have to give the same field of view.**

| Folder | Known altitude | Back-calculated field of view |
|---|---|---|
| `80m` | 80 m | 46.9° (spread 1.0°) |
| `100` | 100 m | 50.4° (spread 1.1°) |

Range 3.4° — the check passes. **Recommended value: 48.0°**, uncertainty band
44.8°–52.9° (dominated by the question of how well the ground is visible). The
previous depths are therefore to be corrected by a **factor of 1.68**.

For contrast: pure Depth Pro back-calculates to 1.4–1.6° and is *more consistent*
in doing so (spread 0.3°). That is not a quality indicator but the opposite — it
always outputs the same thing regardless of flight altitude.

**This back-calculation is not a substitute for a real calibration.** It presumes
that the depth model is correct and is thus partly circular. The solid part is
the consistency: that *one* field of view fits both known altitudes is something
the model could not achieve if it were not actually reading the flight altitude
out of the image. **AnyCam** on the original videos would be the independent
route — it estimates the intrinsics from image motion, i.e. via a completely
different information path. That needs the videos; the single frames are not
enough.

---

## 6. Application to our frames

With a calibrated field of view (48°), over all 33 frames:

| | Pure | Fine-tuned |
|---|---|---|
| Estimated flight altitude | 2.98 m | **83.0 m** |
| **Ground error** | −80.7 m | **−0.61 m** |
| Crown height | 0.89 m | 28.2 m |
| Points below the ground | 0 % | 6 % |

Per folder, crown height (assumption-free — a difference needs no reference
point):

| Folder | Altitude | Source | Pure | Fine-tuned |
|---|---|---|---|---|
| `100` | 100 m | folder name | 1.0 m | 35.4 m |
| `80m` | 80 m | folder name | 0.6 m | 18.7 m |
| `dense` | 51 m | estimated | 0.6 m | 13.1 m |
| `dense1` | 69 m | estimated | 0.5 m | 16.7 m |
| `mixed` | 92 m | estimated | 1.1 m | 45.9 m |
| `mixed1` | 103 m | estimated | 2.0 m | 47.1 m |
| `pines` | 60 m | estimated | 0.6 m | 14.5 m |
| `urban` | 120 m | estimated | 0.8 m | 37.4 m |

The flight altitudes of the folders without a number in the name are estimated by
the model, not measured.

**The ground error of 0.6 m is partly built in**, because the field of view was
back-calculated from exactly these frames. What is not built in is the
consistency: a single field of view brings all eight folders to plausible values
simultaneously, and only 6 % of the pixels end up below the ground.

**Caveat on crown height:** that is the depth range in the image, not necessarily
the tree height. On sloping terrain the relief is included — the 46–47 m for
`mixed`/`mixed1` are unrealistic for Central European trees and are likely to
come from that.

> **`urban` is not a drone capture at all.** The folder contains four
> **screenshots** at varying resolutions (1463×705, 1378×709, …), not video
> frames. The field of view of 48° does not hold for those: a screenshot shows a
> crop, i.e. a narrower field of view, and by how much is unknown. All `urban`
> values in this report — 37.4 m crown height, 120 m estimated altitude — are
> therefore wrong by an unknown factor and do not belong in the evaluation.
> `karten_export.py` and `punktwolke.py` have warned since then when an image is
> not 1920×1080 in a 16:9 ratio.

---

## 7. Limits

- **Flight altitude 25–120 m, nadir view, forest.** Oblique captures did not
  occur in training.
- **100 m is a slight extrapolation.** The sites are 130 m wide; from 80 m a shot
  at 73.7° covers exactly 120 m — just inside; from 100 m it would be 150 m. The
  effective resolution in training ends at 8.3 cm, our 100 m frames would need
  9.8 cm. A factor of 1.18 beyond it.
- **The truth comes from orthomosaics**, not from real single captures. An ortho
  shows every tree from exactly above, a photograph shows crown flanks towards the
  image edge. Hardly relevant for the height of a tree, somewhat more so for the
  exact position of its edge. `--strahl-tiefe` switches to depth along the viewing
  ray.
- **Low heights are slightly under-represented.** The discarded fill areas sit
  preferentially in canopy gaps and shadows — where ground would be visible.
- **Only the decoder was trained** (40 M of 952 M parameters). Whether a full pass
  brings more is untested.

---

## 7b. Point clouds

`depthft/punktwolke.py` turns the depth maps into 3D clouds, as `.ply`
(CloudCompare, MeshLab, Blender) and `.las` 1.2 (lidR, LAStools). Both formats are
written directly; there is no point-cloud library in the container.

**The ground comes from the data, not from the flight altitude — and that is not
the makeshift solution but the better one.** Measured against the nDSM of the test
sites (`hoehe_pruefen.py`, 100 crops):

| Route to height above ground | MAE | Bias | Requires |
|---|---|---|---|
| **Terrain model** (35 m, factor 0.917) | **5.64 m** | **+0.30 m** | nothing |
| From known flight altitude, `Z = H − d` | 7.46 m | +4.67 m | the altitude |

`Z = H − d` cannot represent slope; a terrain model can. The ground factor of
0.917 was estimated from the nDSM data beforehand and comes out independently a
second time here when optimised against the truth.

**Two conditions without which it goes wrong.** The model must **never lie above
the observed surface** — otherwise points end up below the ground, measured up to
29 m deep in a stand with 62 m of terrain drop, of which over-strong smoothing
reproduced only 25 m. And the smoothing must not iron out the slope. After both
corrections: 0.0 % points below the ground instead of 6.7 %.

**The height is less accurate than the depth, and that is arithmetic.** On the
depth the relative error is 12 %; but the height is a difference of two large
numbers (80 m altitude − 60 m depth = 20 m tree). A depth error of 5 m carries
undiminished into the height, where it weighs four times as much in relative terms
— around 19 % at tree heights of about 30 m. The correlation between estimated and
true height is 0.69. For "which tree is taller than its neighbour" that is enough,
for "this tree is 24.3 m tall" it is not.

**The field of view distorts differently than expected.** In the back-projection
`X = (u−cx)·Z/f` it cancels out in X and Y, because `Z` and `f_px` are both
proportional to it. A wrong field of view therefore leaves crown diameters correct
and stretches only the height — trees become too pointed or too flat, not too
wide.

The geometry is independently confirmed: the frame from 80 m gives a cloud 70.3 m
wide, where 2·80·tan(24°) = 71.2 m is expected; from 100 m it is 88.9 m against
89.1 m expected.

| Frame | Ground (Z p02) | Treetop (Z p95) | Points below ground |
|---|---|---|---|
| `100` | 6.2 m | 41.1 m | 0.1 % |
| `80m` | 3.0 m | 23.4 m | 0.2 % |
| `dense` | 3.2 m | 17.5 m | 0.6 % |
| `dense1` | 1.4 m | 23.7 m | 0.8 % |
| `mixed` | 0.4 m | 43.0 m | 1.2 % |
| `mixed1` | 0.4 m | 24.8 m | 0.0 % |
| `pines` | 1.7 m | 21.9 m | 1.4 % |

That Z does not start at zero is correct: in a closed canopy the ground is not
visible, and the terrain model estimates it around 9 % of the flight altitude
below the deepest visible point. So the cloud does not float; it begins where the
line of sight ends.

## 7c. An experiment that did not work out: training on height directly

The detour via depth has an obvious flaw — it optimises the depth error while what
interests us is the height error, and the height is a difference of two large
numbers, which quadruples the relative error. A second model
(`finetune_hoehe.py`) therefore predicts the height above ground directly, with
the nDSM as the target.

At first glance a success: MAE 5.34 m against 5.64 m, correlation over all pixels
0.77 against 0.69, and it needs neither field of view nor ground reference.

**At second glance, not.** On our frames it delivered 22–29 m for every stand —
strikingly uniform. The check confirms the suspicion:

| Route | r **between** stands | predicted spread | true spread |
|---|---|---|---|
| direct height model | **−0.16** | ± 1.1 m | ± 6.0 m |
| terrain model from the depth | 0.10 | ± 5.4 m | ± 6.0 m |
| from known flight altitude | **0.64** | ± 8.6 m | ± 6.0 m |

The model guesses the training mean. It differentiates well *within* an image —
treetop against gap — and is blind *between* stands.

The reason is fundamental and hits the terrain model too: **without a scale
reference, a 30 m tree from 90 m altitude cannot be distinguished from a 15 m tree
from 45 m.** Same apparent size, same level of detail. The information has to come
from outside — and the only source for it is the flight altitude. My reasoning
that the height model makes itself independent of the uncertain field of view is
correct; it just also makes itself independent of the only scale reference there
is.

Why the MAE did not show this: it averages over all pixels, where the variation
within an image dominates. And it *rewards* guessing the mean — by definition the
mean is the best estimator when you know nothing. A metric that punishes the
application.

**Practical consequence:**

| What is needed | Route |
|---|---|
| Segmenting crowns, relative structure | terrain model — locally accurate, without altitude |
| Comparing stands, absolute height | `Z = H − d` with a known flight altitude |
| Point clouds for looking at | terrain model |

The checkpoint sits under `/scratch/shared/$USER/runs/depthft_hoehe/bestes` and
can be selected via `--modellart hoehe`, but it is not the default.

## 8. What is worth doing next

1. **Confirm the field of view independently** — EXIF of the original files, or
   AnyCam on the videos. That is the biggest lever: a factor of 1.68 on every
   height.
2. **`pure + a learned scale estimator`.** The oracle row shows that a tiny
   network estimating only *one* scale factor per image could theoretically reach
   0.074 — better than the fully fine-tuned model and orders of magnitude cheaper
   to train. After the finding in 7c, however, one would expect that this
   estimator too guesses the mean as long as it does not receive the flight
   altitude as an input. **Giving the flight altitude to the model as an
   additional input** would be the genuinely promising route: on a drone it is
   known, and the model would then no longer have to guess the scale.
3. **Ground detection instead of a percentile.** The terrain model estimates the
   ground from the deepest visible places. A real ground mask — recognising pixels
   where ground is actually visible — would do without the blanket factor of 0.917
   and would be markedly more accurate in open stands.
4. **`--trainable all`** with gradient checkpointing, to see whether the encoder
   still contributes anything.

---

## 9. The shipping package

```
/scratch/shared/$USER/runs/depthft/versand/
  depthpro-fortress-nadir/          3.81 GB
  depthpro-fortress-nadir.tar.gz    2.30 GB
```

Contains the weights in Hugging Face format, the matching image processor, the
lean `inferenz.py`, a `beispiel.py` and a model card. The model can be loaded with
`DepthProForDepthEstimation.from_pretrained(path)`. The metric conversion then
runs via `inferenz.py`, so that the field of view is supplied and not estimated by
the unchanged head.

Anyone publishing results should cite FORTRESS (Schiefer, Frey & Kattenborn 2022,
CC BY 4.0) — that applies to colleagues who only receive the weights, too. It says
so in the model card.
