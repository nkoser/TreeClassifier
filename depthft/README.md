# depthft — fine-tuning Depth Pro on FORTRESS

> **State of the weights:** the existing run under
> `/scratch/shared/$USER/runs/depthft` was still trained with the old log-depth
> loss. The corrected code writes new runs to
> `/scratch/shared/$USER/runs/depthft_huber_v2` by default; the old weights are
> not overwritten.

A folder of its own, so that nothing in the existing pipeline has to be touched.
The scripts here import only from each other, never from the repository root.

## The problem

Depth Pro produces depth maps for our frames, but the height is wrong: crowns
sit too high, tree heights come out too small. That is not a coincidence. Depth
Pro is trained on ground-level perspectives — streets, interiors, portraits. A
nadir shot from 80 m does not occur in that. What the model has learned is the
**structure** of a scene; what it has not learned is the **scale** of a capture
situation it has never seen.

That is exactly what can be repaired, once truth is available.

## Where the truth comes from

**FORTRESS** (Schiefer, Frey & Kattenborn 2022, CC BY 4.0) lives under
`/scratch/shared/$USER/data/fortress`: 47 UAV sites in the southern Black
Forest, 1.7 ha each, orthomosaic at 0.77–1.57 cm/px, plus one **normalised
height model** (nDSM) per site — metres above ground, for every pixel.

You cannot train on that directly. An orthomosaic is not a photograph: it has no
camera, no field of view, no depth. Depth only arises from an assumption — hang
a nadir camera at altitude `H` above the stand:

```
depth        d    = H - nDSM
ground samp. GSD  = H / f_px
ground width      = 2 * H * tan(HFOV / 2)
```

A site thereby yields arbitrarily many **virtual frames with an exact metric
depth map**, at any flight altitude. Altitude, field of view and position are
drawn at random per crop.

The price of that assumption: an ortho shows every tree from exactly above, a
real photograph shows crown flanks towards the image edge. For the question *how
tall is this tree* that hardly matters, for the question *where exactly is its
edge* somewhat more. If you want to be strict, `--strahl-tiefe` switches to
depth along the viewing ray instead of along the optical axis.

## The space in which training happens

Depth Pro does not output metres but **canonical inverse depth**. It only
becomes metric in the processor's post-processing
([`image_processing_depth_pro.py:108`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/depth_pro/image_processing_depth_pro.py)):

```
d = (f_px / image width) / D_raw  =  k / D_raw
```

`k` depends solely on the field of view and not on resolution. That has three
consequences, and they determine the whole setup:

1. **`k` is supplied, not estimated.** Depth Pro has a field-of-view head that
   estimates `k` as well. With a known drone camera that is the worse choice —
   the head is trained on ground perspectives, and an error in `k` enters every
   depth *linearly*. The head stays frozen and is preserved in the checkpoint,
   but it is not used.
2. **The output stays canonical inverse depth.** Training runs over the metric
   height computed from it, `h = H - k/D`. The checkpoint remains loadable with
   the normal Hugging Face classes. Because the frozen field-of-view head is
   unreliable on nadir images, `k` still has to be supplied when converting to
   metres.
3. **The resolution of the crops is free.** They are set to 1536 px width, so
   that no rescaling happens between crop and model input at all.

### The crop is 16:9, not square

Depth Pro squeezes every image to 1536×1536, regardless of aspect ratio. Our
frames are 1920×1080 and are therefore compressed by a factor of 1.78 in height
along this chain. Whoever trains on squares and applies to squeezed images has
built the error themselves. The crops therefore come in the aspect ratio of the
target frames (`--seitenverhaeltnis`, default 16/9) and then go through the same
squeeze.

For the same reason the only augmentation is mirroring and not quarter-turns —
those would flip the aspect ratio.

The loss has two parts:

| Part | Effect |
|---|---|
| Huber on the metric height | optimises the height error in metres directly and is robust against individual nDSM outliers. |
| Gradient matching over 4 scales | keeps crown boundaries sharp. A pure per-pixel loss rewards softer transitions. |

### The head is pre-scaled before training

Pure Depth Pro is off by roughly a factor of 50 in this capture situation.
Because of `d = k/D` the mapping is very steep near zero. The pre-scaling puts
the output into the physically relevant range before the first optimizer step
and thereby avoids an unnecessarily unstable change of scale.

The way out is not to demand the jump in the first place. `--vorspannen auto`
measures the scale error over a few batches and scales the last convolution of
the head with it. Because that is a 1×1 convolution before the final ReLU and
the factor is positive, this is **exactly** equivalent to `D → factor · D` — but
as a real weight change, not as a special case at inference time. The shipped
checkpoint therefore stays usable without an instruction leaflet.

The learning rate of that one layer is scaled by the same factor. Otherwise Adam
steps of the usual size would immediately tear apart weights that are now orders
of magnitude smaller — Adam normalises the step size away, so it depends solely
on the learning rate.

The metric error is evaluated exactly in the forward pass. For the backward
pass, the Jacobian of `k/D` is linearised at the respective target value. The
optimisation direction thus stays exact at the target, but can no longer become
singular near `D=0`. In addition, training samples run in short, mixed site
blocks instead of 200 crops of the same stand back to back.

The default is `--trainable decoder`: neck, fusion stage and head, around 60 M
parameters. The encoder runs frozen under `no_grad` — that saves the bulk of the
memory and is sufficient, because the scale sits in the head, not in the
features. `--trainable all` tunes everything and needs considerably more GPU.

## Order of operations

```bash
sbatch depthft/sbatch/run_prepare.sbatch        # 47 orthos -> ground raster, ~1 h
sbatch depthft/sbatch/run_check.sbatch          # MANDATORY, see below
sbatch depthft/sbatch/run_finetune.sbatch       # fine-tuning
sbatch depthft/sbatch/run_evaluate.sbatch       # metric, on the test sites
sbatch depthft/sbatch/run_apply_frames.sbatch   # on our own frames
sbatch depthft/sbatch/run_export.sbatch         # shippable package for colleagues
```

All scripts take their settings from environment variables, e.g.

```bash
EPOCHS=12 TRAINABLE=all BATCH=1 ACCUM=16 sbatch depthft/sbatch/run_finetune.sbatch
SITES="CFB014 CFB019" sbatch depthft/sbatch/run_prepare.sbatch
ALTITUDES="pines=35 dense=60 urban=50" sbatch depthft/sbatch/run_apply_frames.sbatch
```

### Two peculiarities of the FORTRESS height models

Both only surfaced in the check run, and both would have silently spoiled the
training.

**Exact zeros are fill, not ground.** In the nDSM files `0.00` is by far the most
frequent single value — 9 to 27 % of the area, in large contiguous blocks under
which the orthomosaic shows closed forest. Real ground scatters around zero, it
does not hit it exactly ten thousand times. Learning those areas as ground would
mean: crowns at height zero. `prepare.py` therefore discards them
(`--nullen-behalten` turns that off for comparison). Across all 47 sites: fill
6 % at the median, 43 % in the worst site; the valid share afterwards lies
between 51 % and 97 %, at the median 82 % (recorded per site in `index.json`).

The discarded areas sit preferentially in canopy gaps and shadows — where
photogrammetry could not reconstruct a height. That means **low heights are
slightly under-represented in training**, particularly the ground reference. For
sites with 0 % fill — the majority — the distribution stays complete, which is
why it is acceptable. When evaluating the ground level it is still worth a look.

**The camera has to hang above the treetops.** Without a bound there would be
crops in which 30 m trees almost reach the lens at 27 m flight altitude — a
capture situation that does not occur for us. `--abstand-min` (default 20 m)
sets the flight altitude per site to at least *highest treetop + 20 m*.

### What pure Depth Pro achieves here — the bar

From the check run over ten crops at 73.7° field of view:

| | Truth | Pure Depth Pro |
|---|---|---|
| Depth at 27 m altitude | 3.4–27.5 m | 1.0–1.9 m |
| Depth at 64 m altitude | 22.7–64.0 m | 1.1–1.6 m |
| Scale factor | 1.00 | **0.06** |
| AbsRel | — | **0.94** |
| Field of view (head) | 73.7° | 18–41° |

On nadir forest images Depth Pro collapses to roughly one metre of depth,
independently of the flight altitude. Too little depth means: everything sits
too close to the camera, the trees appear too tall — exactly the symptom
observed.

The second row is striking: the predicted **range** within one image is a good
half metre where in reality there are 40 m. One might conclude from that that
not only the scale is wrong but also the contrast — and that a global scale
factor therefore cannot help.

**That conclusion is wrong, and the evaluation proves it.** Scaled globally to
the correct median, pure Depth Pro reaches AbsRel 0.074 and thereby beats the
fine-tuned model. The relative structure is excellent; the small absolute range
is merely a consequence of the scale error, not a defect of its own. What is
missing is the scale, and only the scale.

Except: in deployment you do not know the right factor. See below.

### `run_check.sbatch` is not optional

The most expensive error in this chain would be an offset between orthomosaic
and height model: the georeferencing is off, the crowns in the nDSM sit two
metres beside those in the image, and the training patiently learns nonsense —
for 48 hours, with a falling loss. `check.py` puts both side by side and checks
the geometry against the formulas above. **The comparison strips under
`results_depthft/check/` have to be looked at** before the training starts: the
relief of the nDSM must lie on the crowns in the image.

Along the way `check.py` supplies the starting position (table above) and
reports per crop how much of the area carries truth at all. In the comparison
strip, missing truth is black — it must not pass as ground.

## What is measured

`evaluate.py` compares four variants on sites that never occurred in training.
The third is the most revealing:

| Variant | What it answers |
|---|---|
| `pur_fovkopf` | Depth Pro exactly as you take it off the shelf. |
| `pur_kamera` | The same run with `k` supplied. Separates field-of-view errors from depth errors. |
| `pur_skalenangleich` | Scaled globally so that the median is exactly right. **Not an applicable method but an oracle** — the factor comes from the truth. Measures how good the *relative* structure is. |
| `*_hoehenanker` | Scaled until the deepest point in the image matches the known flight altitude. Applicable, because a drone knows its altitude. |
| `feinabgestimmt_kamera` | The result: the scale comes from the image itself. |

### Measured, 200 crops from five test sites

| Variant | AbsRel | MAE | δ<1.25 | Scale error |
|---|---|---|---|---|
| `pur_skalenangleich` *(oracle)* | 0.074 | 4.41 m | 0.941 | 1.000 |
| **`feinabgestimmt_kamera`** | **0.120** | **7.36 m** | **0.843** | 0.945 |
| `feinabgestimmt_hoehenanker` | 0.167 | 8.96 m | 0.797 | 1.170 |
| `pur_hoehenanker` | 0.226 | 11.98 m | 0.632 | 1.231 |
| `pur_fovkopf` | 0.939 | 56.11 m | 0.000 | 0.061 |
| `pur_kamera` | 0.976 | 58.14 m | 0.000 | 0.024 |

Three things are in there.

**The fine-tuning works.** From AbsRel 0.98 to 0.120, from δ<1.25 = 0.000 to
0.843. The scale error goes from 0.024 to 0.945 — still 5.5 % off at the median.

**The structure was never the problem.** Given the scale factor for free, the
pure model reaches 0.074. The fine-tuning reaches 0.120 without any help and
thereby comes close to that bound, but it does not overtake it.

**The obvious anchor does not work.** A drone knows its flight altitude — it is
tempting to scale with that instead of training a model. Measured, that is
*worse* (0.226 against 0.976 for pure, but also worse than 0.120), and the
reason is in the scale error of 1.23: **in a closed canopy the deepest visible
point is not the ground.** Even for the fine-tuned model the anchor makes the
result worse (0.167 instead of 0.120) — the learned scale is more reliable than
the geometric assumption.

Image sharpness hardly matters: with `--videolook` (blur, noise, JPEG) the
result is 0.135 instead of 0.120. So the model transfers to video image quality.

`mae_m` is at the same time the error of the **height above ground**: that is
flight altitude minus depth, and the altitude cancels out in the difference.

## Measuring without truth — on our own frames

For `/cold/Mahfuz/chosen_frames` there is no nDSM. It can still be measured, and
on something that needs no height map at all: **the depth to the ground is the
flight altitude.**

```
estimated altitude = 95th percentile of the depth
crown height       = 95th percentile - 2nd percentile of the depth
```

The second number is the more important one, because it needs no assumption at
all: the span between ground and treetop is the tree height, no matter how high
the drone actually hung. A model that compresses the stand to 6 m is caught
immediately here — even when its depth map looks pretty.

If the flight altitude is known (a number in the folder name such as `80m`, or
via `--altitudes`), two more checks are added: the ground level has to be at
0 m, and no pixel may sit below the ground.

> **Open point: 100 m is a small extrapolation.** The sites are 130 m wide. At
> 73.7° field of view a shot from 80 m covers exactly 120 m — just about inside.
> From 100 m it would be 150 m, more than the site provides. In training the
> ground width therefore ends at around 127 m, i.e. at an effective resolution of
> 8.3 cm on the 1536 input, while our 100 m frames need 9.8 cm. A factor of 1.18
> beyond it — defensible, but worth keeping in mind when evaluating the folder
> `100`. The 80 m case is fully covered.

> **`urban` contains screenshots, not drone frames** — four screen captures at
> varying resolutions. The prescribed field of view does not hold there, and all
> values are wrong by an unknown factor. `karten_export.py` and `punktwolke.py`
> warn on images that are not 1920×1080.

> **Open point.** The field of view of 73.7° is an estimate in the code, not a
> measured camera specification (see `REPORT_height_from_images.md`). It enters
> every depth linearly. If EXIF data is available, the value should come from
> there — `--hfov-deg` accepts it. Likewise the flight altitudes of the folders
> without a number in the name (`dense`, `mixed`, `pines`, `urban`) are unknown
> and fall back to 100 m; that distorts the ground level there, but **not** the
> crown height.

## The shipping package

`export.py` builds a folder that someone without this repository can use:
weights in Hugging Face format, the matching image processor, the lean
`inferenz.py`, an `beispiel.py` and a model card stating the one thing that
everything else depends on — that the field of view belongs supplied, not
estimated. With `--tar` a `tar.gz` is placed next to it.

```bash
sbatch depthft/sbatch/run_export.sbatch
# -> /scratch/shared/$USER/runs/depthft/versand/depthpro-fortress-nadir[.tar.gz]
```

## Files

| File | Task |
|---|---|
| `prepare.py` | 47 orthos + nDSM → common ground raster at 2 cm/px, ~1.2 GB. The expensive part, once. |
| `dataset.py` | cuts virtual nadir frames from it; geometry and augmentation. |
| `check.py` | truth against image, geometry against formula, starting position of the pure model. |
| `finetune.py` | the fine-tuning. |
| `inferenz.py` | inference with a supplied camera. No project dependencies, ships with the package. |
| `evaluate.py` | pure vs. fine-tuned, metric, on the test sites. |
| `apply_frames.py` | pure vs. fine-tuned on our own frames. |
| `export.py` | shipping package. |
| `karten_export.py` | depth and height maps as npy/png/jpg for further processing. |
| `punktwolke.py` | 3D clouds as `.ply` and `.las`, anchored to the ground rather than the camera. |
| `kalibrieren.py` | field of view back-calculated from frames with known flight altitude. |
| `vergleichsbild.py` | figures pure vs. fine-tuned, readable instead of saturated. |
| `bilder.py` | colour scale, hillshading, labelling, bar chart. |

## Provenance of the data

FORTRESS: Schiefer, F., Frey, J. & Kattenborn, T. (2022), CC BY 4.0. Anyone
publishing results from this model should cite the dataset — including the
colleague who receives the weights. It says so in the model card.
