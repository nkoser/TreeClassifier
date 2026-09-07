# Species inference on our own drone frames

Two-stage pipeline applying the published **DINOvTree-B** checkpoint (ECCV 2026,
[RolnickLab/DINOvTree](https://github.com/RolnickLab/DINOvTree)) to our own,
non-georeferenced drone frames. Operating instructions are in
[`README.md`](README.md).

```
Frame (1920x1080 JPG)
  └─ instances                                 → one tree per segment
      ├─ segment_hybrid.py (best result)        SAM + watershed for the remaining area
      ├─ segment_sam.py                        SAM → crown polygons on real image edges
      ├─ segment_trees.py                      depth + watershed → crown polygons
      └─ infer_species.py --detector deepforest   DeepForest → boxes (weakest variant)
          └─ tree-centred crop, at 512x512      → one crop per tree
              └─ DINOvTree-B (Quebec Trees)     → species probability + height
```

> **Status:** `segment_trees.py` delivers markedly better instances than
> DeepForest, but is not yet wired into `infer_species.py` — the results in
> `results/` are still based on DeepForest boxes.

DINOvTree itself detects **nothing** — it always classifies the tree at the
centre of a 512×512 crop. The instances therefore have to come from outside; in
the original paper they are manually annotated crown polygons, here DeepForest
replaces them.

## Setup

Everything runs through SLURM in the existing `vesselgpt_7.sif`, nothing
directly on the login node:

```bash
sbatch sbatch/setup_env.sbatch      # env on /scratch + checkpoint, once
squeue -u $USER                     # status
```

The container brings Python 3.11.8 and **torch 2.10.0+cu128** — exactly what the
node's RTX PRO 6000 Blackwell needs (`sm_120`, driver 580.x / CUDA 13.0; older
CUDA builds import cleanly and then die on the first kernel). Instead of a
second torch installation, `setup_env.sbatch` creates a venv **with
`--system-site-packages`** under `/scratch/shared/$USER/envs/treeclf`: it
inherits torch from the container and adds only what is missing (deepforest,
opencv, omegaconf). That saves 7 GB in the home directory and keeps the torch
version consistent with your other jobs.

The venv is only usable **inside** the container — its interpreter points at
`/miniconda3`. All runs therefore go through `apptainer exec --nv`.

Locations:

| What | Where |
|---|---|
| Env | `/scratch/shared/$USER/envs/treeclf` |
| Checkpoint | `/scratch/shared/$USER/data/treeclf/checkpoints/` |
| HF / torch cache | `/scratch/shared/$USER/hf_cache` |
| Job logs | `/scratch/shared/$USER/runs/logs/%x_%j.{out,err}` |
| Results | `results/` in the repo (gitignored) |

`third_party/DINOvTree/` is a copy of the original repository (only the model
code is imported). **Meta's DINOv3 weights are not needed** — the checkpoint
contains the complete fine-tuned backbone. The original repository nevertheless
insists on fetching them from a URL in `dinov3_urls.json`, which is why
`build_dinovtree()` briefly redirects `torch.hub.load` and builds the
architecture uninitialised before the checkpoint is laid over it with
`strict=True`.

## Usage

```bash
sbatch sbatch/run_infer_species.sbatch                       # defaults
ALTITUDES="pines=35 dense=60" sbatch sbatch/run_infer_species.sbatch
OUT=results_v2 MAX_TREES=200 sbatch sbatch/run_infer_species.sbatch
FRAMES="/cold/Mahfuz/chosen_frames/dense/frame_000073.jpg" sbatch sbatch/run_scale_sweep.sbatch
```

A full pass over all 33 frames with 150 trees per frame takes **50 seconds** (for
comparison: 27 minutes on CPU for only 40 trees per frame). Job parameters are
given as environment variables before `sbatch`: `INPUT`, `OUT`, `MAX_TREES`,
`BATCH`, `ALTITUDES`, `CKPT`, `SIF`, `ENV_DIR`, `EXTRA` (any further flags).

The script's own important parameters:

| Flag | Default | Meaning |
|---|---|---|
| `--altitudes FOLDER=ALTITUDE` | – | flight altitude per folder, e.g. `--altitudes pines=35 dense=60` |
| `--altitude` | 100 | fallback flight altitude in m |
| `--hfov-deg` | 73.7 | horizontal field of view (DJI 24 mm equivalent, 16:9) |
| `--crop-mode` | `gsd` | `gsd`: the crop covers 9.73 m as in training. `relative`: a multiple of the crown box, without camera knowledge |
| `--footprint-m` | 9.73 | edge length of the crop on the ground |
| `--min-score` | 0.35 | DeepForest confidence threshold |
| `--max-trees-per-frame` | 40 | only the N most confident detections (GPU ≈ 0.015 s/tree, CPU ≈ 0.85 s/tree) |
| `--device` | `auto` | `cuda` if available, otherwise `cpu` |

Output per frame: `results/<folder>/<frame>_trees.csv` (boxes, top-3 species with
probabilities, predicted height, entropy) and `<frame>_overlay.jpg`. Plus
`results/all_trees.csv` and `results/summary_by_folder.csv`.

## Scale: the critical parameter

The checkpoint has seen **exclusively** crops of 9.73 m edge length (512 px ×
1.9 cm/px, without resampling). A frame without georeferencing, however, has no
known scale. The GSD is therefore estimated from flight altitude and field of
view:

```
GSD = 2 · altitude · tan(HFOV/2) / image width
crop edge length in source pixels = 9.73 m / GSD
```

`scale_sweep.py` shows how sensitive this is — the same detections at different
crop sizes:

```bash
FRAMES=/cold/Mahfuz/chosen_frames/pines/frame_000006.jpg \
  CROP_SIZES="100 150 250 400 600" N_TREES=6 sbatch sbatch/run_scale_sweep.sbatch
```

On the test frames the top-1 species flips between broadleaf- and
conifer-dominated depending on whether 150 px or 400 px are interpreted as
9.73 m. Without a solid flight altitude per folder the species predictions are
correspondingly shaky.

## Instances: three approaches compared

| Method | Crowns (33 frames) | mean diameter | area coverage | character |
|---|---|---|---|---|
| DeepForest | 3617 (out of 20563 raw) | 27 px | – | crown fragments, not trees |
| Depth + watershed | 4697 | 94–117 px | 71–99 % | tiles the image seamlessly, even where there is no tree |
| SAM (vit-large) | 4848 | 65–103 px | 50–74 % | real crown edges, leaves out the uncertain parts |
| **Hybrid (SAM + watershed)** | **5897** | 63–97 px | 65–76 % | SAM precision, watershed fills the gaps |

The decisive difference between the last two: watershed is a *partition* — it
divides the mask into exactly as many parts as markers go into it, regardless of
whether trees stand there. The high area coverage is therefore not a quality
indicator but an artefact. SAM segments along actual image edges and leaves out
areas for which it has no evidence.

### Ablation of the SAM variants

`ablate_sam.py` compares the checkpoints under identical filter rules
(4 representative frames, averages):

| Model | Raw masks | Crowns | Coverage | s/frame |
|---|---|---|---|---|
| **sam_vit_large** | 283 | 140 | **0.61** | 3.4 |
| sam_vit_base | 275 | 140 | 0.60 | 2.9 |
| sam_vit_huge | 263 | 133 | 0.56 | 3.4 |
| sam2.1_large | 140 | 81 | 0.37 | 2.0 |
| sam2_large | 130 | 82 | 0.37 | 2.1 |
| sam2.1_base | 132 | 79 | 0.34 | 2.1 |

Two unexpected findings: **SAM 1 clearly beats SAM 2**, and within SAM 1
`vit-huge` is worse than `vit-large` *and* `vit-base` — bigger is not better
here.

The obvious objection, that SAM 2 is disadvantaged by shared thresholds (its
IoU/stability scores are calibrated differently), was checked: even with
completely open thresholds (`--pred-iou-thresh 0.0 --stability-score-thresh 0.5`)
SAM 2.1 only reaches 41–49 % coverage. The difference is real, not an artefact.

### SAM 3 — result

It runs (given access) and is the best variant on the difficult stands, but it is
no free lunch. Three findings:

1. **Only the prompt `"tree"` works.** `"tree crown"` yields 0–8 instances per
   frame, `"treetop"` nothing at all. SAM 3 knows the term, not the paraphrase.
2. **The shape filter has to be off** (`--no-shape-filter`) and the threshold
   lowered to 0.15. With the defaults inherited from SAM 1 you discard 44 % of
   the instances — SAM 3 already delivers instances instead of a scale stack, so
   the filter is superfluous.
3. **Tile edges have to be handled.** Without that you get dead-straight cuts
   across crowns: instances running over a tile boundary get split and both
   halves are kept. `--drop-cut` (on by default) discards cut instances — thanks
   to the overlap the same object is contained completely in the neighbouring
   tile.

| Folder | Hybrid | SAM 3 | Δ |
|---|---|---|---|
| `mixed` | 0.65 | **0.76** | +0.11 |
| `dense` | 0.75 | **0.83** | +0.08 |
| `mixed1` | 0.66 | **0.70** | +0.04 |
| `pines` | 0.68 | **0.72** | +0.04 |
| `80m` | 0.76 | 0.77 | +0.01 |
| `dense1` | 0.74 | 0.74 | 0.00 |
| `100` | **0.76** | 0.72 | −0.04 |
| `urban` | **0.74** | 0.47 | −0.27 |

SAM 3 wins on five of eight stands, with **fewer and larger** instances (4587
instead of 5897, median 91 instead of 81 px) — that is, less fragmentation. It
loses clearly on `urban`; those are the screenshots with deviating image sizes,
where the fixed 2×2 tiling does not fit.

The share of suspicious separations is higher for SAM 3 at 51 % than for the
hybrid (40 %) — partly real, partly an artefact of the metric, which measures
against the same weak monocular depth and punishes larger crowns harder.

### Access

`facebook/sam3` is access-restricted. Request access on the model page, then
create a read token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and
deposit it — `huggingface_hub` reads that path under `HF_HOME` by itself, and all
scripts work afterwards without further change:

```bash
printf 'hf_YOUR_TOKEN' > /scratch/shared/$USER/hf_cache/token
chmod 600 /scratch/shared/$USER/hf_cache/token
sbatch sbatch/run_segment_sam3.sbatch
```

SAM 3 works fundamentally differently from SAM 1/2 and therefore needs its own
code path (`segment_sam3.py`): it receives the term as **text**
(`--prompt tree`) and returns instances with a score, instead of sampling a point
grid and returning a scale stack of leaf/branch/crown. The shape filter is thus
optional (`--no-shape-filter`).

Because a 100 px crown shrinks to a good 50 px during the internal resize, the
script processes the frame in 2×2 overlapping tiles by default and afterwards
merges the instances through an overlap resolution (`--tiles`, `--tile-overlap`).

## Depth as a prompt source for SAM

`segment_prompted.py` combines both sources of information where they complement
each other best:

```
depth  →  WHERE a tree is        (CHM treetop as prompt point, with prominence check)
SAM    →  WHERE its border is    (promptable segmentation, one point per crown)
```

Previously SAM had to work out for itself where a tree begins — via a blind point
grid (SAM 1/2) or via a text term (SAM 3). Depth knows that better: a treetop is
a local maximum in the surrogate CHM. Conversely SAM draws the border from real
image edges instead of from the smoothed depth surface, as `segment_trees.py`
does.

SAM returns three candidates of different extent per point (part, object,
context). What is selected is **not the one with the highest score** — that often
aims at the whole canopy — but the one whose area best matches the expected crown
size.

```bash
sbatch sbatch/run_segment_prompted.sbatch
PROMINENCE=0.08 sbatch sbatch/run_segment_prompted.sbatch
```

| Method | Crowns | Compactness | Coverage | suspicious separations |
|---|---|---|---|---|
| SAM 3 multi-scale | 5957 | 0.53–0.65 | 0.51–0.84 | 57 % |
| Hybrid | 5897 | 0.55–0.62 | 0.65–0.76 | 40 % |
| **Depth prompt** | 3502 | **0.65–0.74** | 0.37–0.67 | **44 %** |

The depth prompt delivers markedly **fewer but cleaner** instances: compactness
is consistently 0.1 above the other methods, and the shapes are recognisably
crown-round instead of frayed. The price is coverage — of 247 treetops found,
only 154 survive the area and shape check.

## Connecting image and depth: splitting and merging

`refine_crowns.py` is the union of both information sources. SAM 3 delivers the
instances from the **image**, and the monocular **depth** corrects them in both
directions:

| Correction | Trigger |
|---|---|
| **Split** | One instance contains two prominent treetops with a notch between them → two trees were merged. Separated at the saddle, by watershed inside the instance. |
| **Merge** | Two neighbouring instances have no saddle between them *and* the same colour → one tree was cut apart. |

Split first, then merge: wrongly merged blobs get broken up, and afterwards the
fragments are grouped correctly. What matters when splitting is that prominence
is measured **within the respective instance** — a low crown has a smaller height
range than a tall one, and a globally set threshold would never trigger on it.

```bash
sbatch sbatch/run_refine_crowns.sbatch
SPLITPROM=0.25 sbatch sbatch/run_refine_crowns.sbatch
EXTRA="--no-merge" sbatch sbatch/run_refine_crowns.sbatch
```

| Variant | Instances | Splits | Merges | suspicious separations |
|---|---|---|---|---|
| SAM 3 multi-scale (raw) | 5957 | – | – | 2361 (57 %) |
| merge only | 5223 | – | 734 | 1691 (50 %) |
| `SPLITPROM=0.50` | 5676 | 103 | 368 | – |
| **`SPLITPROM=0.35`** | **5807** | 211 | 375 | **1954 (48 %)** |
| `SPLITPROM=0.25` | 5974 | 350 | 402 | – |
| split only | 6167 | 196 | – | – |

The **area coverage stays at 80 %** — the correction only changes *which*
instances exist, not how much canopy is captured. That was precisely the goal:
keep SAM 3's coverage and only repair the borders it set wrongly.

### Runtime

The first version needed 10 minutes for 4 frames, because a mask over the whole
image was built for every instance (250 instances × 2 megapixels per round). Via
`regionprops` bounding boxes the same thing runs in **2.5 minutes for all 33
frames**.

## Merging wrongly separated crowns

`merge_crowns.py` merges neighbouring instances when **two independent criteria**
indicate they belong to the same tree:

- **Saddle prominence** — between the tops of two genuinely neighbouring trees
  there is a notch. If the border runs across a continuous dome, the saddle is
  flat.
- **Colour distance** in Lab space — two parts of the same crown are almost
  identical in colour, two different trees usually differ measurably.

Both have to agree. That matters, because saddle prominence measures against the
estimated monocular depth — the weakest point of the pipeline. The colour
distance is entirely independent of it and partly catches its errors. An area
ceiling prevents runaway growth into giant blobs, and because every merge changes
tops and saddles, the whole thing runs in several rounds.

```bash
sbatch sbatch/run_merge_crowns.sbatch
SPLIT=0.10 COLOR=16 sbatch sbatch/run_merge_crowns.sbatch
```

| Threshold | Crowns | suspicious separations |
|---|---|---|
| none | 5957 | 2361 (57 %) |
| `SPLIT=0.06 COLOR=12` | 5553 | 1926 (52 %) |
| `SPLIT=0.10 COLOR=16` | 5223 | 1691 (50 %) |
| `SPLIT=0.15 COLOR=20` | 4815 | 1451 (50 %) |

## Diagnostic views

`visualize_crowns.py` renders four views from the stored label map — without
model inference, in seconds, as often as you like:

| View | Answers |
|---|---|
| `instanzen` | Every crown in its own colour, semi-transparent. Two colours on a visually continuous crown = false separation. The image stays at full brightness everywhere. |
| `luecken` | Uncaptured area is **hatched instead of darkened** — the texture stays visible, so you can judge how much structure is still there. |
| `relief` | Crown borders on the hillshading. Does the line follow a ridge or does it cut a dome? |
| `trennungen` | Every border between two crowns coloured by **saddle prominence**: red = flat saddle, the two probably belong together; green = deep notch, the separation is backed by the relief. |

```bash
OUT=results_views_<name> SEGMENTS=<segment folder> sbatch sbatch/run_visualize.sbatch
FRAMES="dense/frame_000073.jpg" VIEWS="trennungen" sbatch sbatch/run_visualize.sbatch
```

**Convention:** one folder `results_views_<name>` per change, so that the
variants keep existing side by side. Rendering takes about two minutes for all 33
frames and needs no model — the label maps are enough.

| Folder | Segmentation |
|---|---|
| `results_views_verfeinert` | SAM 3 + depth, split & merge (current state) |
| `results_views_verfeinert_p025` / `_p050` | the same, split more aggressively / more cautiously |
| `results_views_sam3_multiskala` | SAM 3 raw, tile levels 2+3+4 |
| `results_views_verschmolzen` | SAM 3 + merge only |
| `results_views_hybrid` | SAM 1 + watershed |
| `results_views_tiefenprompt` | CHM treetops → SAM (3 test frames only) |

Saddle prominence is the objective version of the question "were two trees cut
apart here?": how far does the surface fall from the lower of the two tops to the
highest point of their shared border, relative to the range of the image. Across
all 33 frames **1222 of 3037 separations (40 %) are suspicious** — with clear
differences between stands (`dense` 13 %, `100` 49 %).

## Hybrid: SAM + watershed for the remaining area

`segment_hybrid.py` combines both methods where each is strong: SAM first fixes
the crowns with a clear edge, then the depth watershed runs **exclusively on the
area SAM did not capture**. It thereby loses its main weakness — it can no longer
tile the image over its full area, only close gaps.

```bash
sbatch sbatch/run_segment_hybrid.sbatch
```

One detail that was necessary: the prominence threshold of the treetop search is
computed on the *remaining area*, not on the whole image. Otherwise the relief of
the already-found SAM crowns dominates the statistics and the rest fails
wholesale.

The gain is concentrated exactly where SAM was weak:

| Folder | SAM alone | Hybrid | watershed share |
|---|---|---|---|
| `dense` | 53 % | **75 %** | 188 of 554 |
| `pines` | 50 % | **68 %** | 281 of 852 |
| `mixed1` | 57 % | 66 % | 103 of 638 |
| `80m` | 74 % | 76 % | 96 of 868 |

The origin is recorded in the CSV column `quelle`. That matters for
interpretation, because the shape quality of the two parts differs markedly: SAM
crowns have a median compactness of 0.71 and solidity 0.94, the watershed
additions only 0.47 and 0.78. The additions are thus visibly more irregular —
which is to be expected, since by construction they are the leftovers between
already placed crowns. Anyone who needs only clean instances filters on
`quelle == "sam"`.

## Crown delineation with SAM

```bash
sbatch sbatch/run_segment_sam.sbatch
FRAMES="100/frame_000537.jpg" EXTRA="--pred-iou-thresh 0.6" sbatch sbatch/run_segment_sam.sbatch
```

SAM produces masks at all scales simultaneously (leaf, branch, crown, stand), and
the work is in the selection: an area window around the expected crown size,
compactness against shadow bands, and a greedy overlap resolution by score that
reduces the scale stack to one mask per crown.

The most effective parameters are **not** our own filters but SAM's internal
quality thresholds `--pred-iou-thresh` (0.70 here instead of 0.88) and
`--stability-score-thresh` (0.85 instead of 0.95). With the library defaults SAM
discards the bulk of the crowns before they ever come out — in a canopy the
borders are objectively fuzzy, and the defaults are meant for everyday objects.
Lowering them raised area coverage from 53 % to 67 %.

## Crown delineation via monocular depth

`segment_trees.py` replaces the DeepForest boxes with real crown polygons. The
trick: individual tree delineation needs height information — two neighbouring
green crowns often have no visible border in RGB. Forestry practice solves that
with a CHM, which is missing here. A monocular depth model
(`Depth-Anything-V2-Metric-Outdoor`, already present in the `hf_cache`) supplies
a surrogate surface with the same structure:

1. Estimate depth, invert it (closer to the camera = higher).
2. **Detrend**: subtract the large-scale component — removes camera tilt and the
   model's ground-plane prior. Analogous to DSM minus DTM.
3. Smooth, so that foliage texture does not create phantom treetops.
4. Local maxima as treetop markers, minimum distance 0.3 × crown diameter.
5. Marker-based watershed on the inverted surface.
6. Filter segments by area and aspect ratio.

```bash
sbatch sbatch/run_segment_trees.sbatch
CROWN_PX=120 EXTRA="--save-chm" sbatch sbatch/run_segment_trees.sbatch
```

The depth maps are cached under
`/scratch/shared/$USER/data/treeclf/depth_cache`, so parameter tuning afterwards
runs without model inference. The three effective knobs are `--crown-px` (sets
all scales), `--smooth-factor` and `--min-distance-factor`; the latter two
determine how many treetops survive, and thereby the area coverage.

Result across all 33 frames: **4697 crowns**, 69–190 per frame, median diameter
94–117 px, canopy area coverage 71–99 %. For comparison, DeepForest: 20563 raw
detections with a median of 27 px, hitting crown fragments instead of trees.

## Limits — please read before interpreting results

1. **The class space is Canadian.** The checkpoint knows exactly 14 classes from
   the Quebec Trees dataset and cannot output anything else:
   `dead`, *Thuja occidentalis*, *Abies balsamea*, *Larix laricina*, *Tsuga
   canadensis*, *Fagus grandifolia*, *Populus* (genus), *Acer pensylvanicum*,
   *Acer saccharum*, *Acer rubrum*, *Pinus strobus*, *Betula alleghaniensis*,
   *Betula papyrifera*, *Picea* (genus).
   A Central European *Fagus sylvatica* inevitably lands on *Fagus grandifolia*,
   a *Picea abies* on *Picea* — at **genus level** that is often usable, at
   species level it is not. Species without a North American counterpart
   (Douglas fir, oak, ash, lime, hornbeam) have no correct output option at all.
2. **Domain shift.** Trained on photogrammetric orthomosaics at 1.9 cm/px,
   applied to compressed single video frames at ~5–8 cm/px. The softmax values
   are systematically overconfident as a result; `entropy` in the CSV is the more
   honest indicator.
3. **The height is not calibrated.** `height_m_pred` comes from the height head
   and is trained on the Quebec height range (mean 14.2 m) and the scale there.
   On foreign data without a DSM the value is at best a ranking, not a
   measurement.
4. **The instances are an error path of their own.** In dense canopies DeepForest
   delivers crown fragments instead of trees — wrong centres shift the crop and
   thereby the classification, without it being DINOvTree's fault.
   `segment_trees.py` largely fixes that, but has limits of its own: the
   monocular depth is trained on ground perspectives, not on nadir from 80 m, and
   in structurally poor stands neighbouring crowns still merge (visible in the
   area coverage of only 71–72 % in `urban` and `dense`).

For solid species identification on Central European stands there is no way
around our own labels: the classification head (`query_token_cls`,
`cross_attn_cls`, `norm_cls`, `classifier` — together ~3 M parameters) can be
retrained on a few hundred annotated crowns with a frozen backbone.

## Job scripts

| Script | Purpose |
|---|---|
| `sbatch/setup_env.sbatch` | venv on /scratch (inherits container torch) + checkpoint, once |
| `sbatch/run_infer_species.sbatch` | full inference run over all folders |
| `sbatch/run_scale_sweep.sbatch` | scale diagnosis on single frames |
| `sbatch/run_segment_hybrid.sbatch` | hybrid SAM + watershed (best result) |
| `sbatch/run_visualize.sbatch` | diagnostic views from the label maps |
| `sbatch/run_refine_crowns.sbatch` | split + merge via depth |
| `sbatch/run_segment_sam.sbatch` | crown delineation with SAM only |
| `sbatch/run_ablate_sam.sbatch` | ablation over the SAM variants |
| `sbatch/run_segment_trees.sbatch` | crown delineation via depth + watershed |
| `sbatch/run_detect_only.sbatch` | DeepForest detections only, for diagnosis |
| `sbatch/run_depth_probe.sbatch` | depth maps / hillshade for visual inspection |

Everything runs through SLURM — never start anything directly on the login node.
