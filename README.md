# TreeClassifier

Single-tree analysis on our own, non-georeferenced drone frames: **which tree
stands where** (crown instance segmentation), **which species it is**, and **how
tall it is**. Everything runs on the SLURM cluster inside the existing
`vesselgpt_7.sif` container.

```
Frame (1920x1080 JPG)
  └─ crown instances            EoMT trained on BAMFORESTS  → one label map per frame
      ├─ species               head trained on FORTRESS     → European species per crown
      └─ height                Depth Pro fine-tuned on FORTRESS → metres above ground
```

This file is the operating guide. For *why* things are built this way and for
every measured number, read the reports:

| Report | Subject |
|---|---|
| [`crownseg/REPORT.md`](crownseg/REPORT.md) | Crown instance segmentation — every experiment, in order |
| [`crownseg/OVERVIEW.md`](crownseg/OVERVIEW.md) | The same on one page |
| [`REPORT_species.md`](REPORT_species.md) | Species classification with DINOvTree, the scale problem, the limits |
| [`REPORT_height_from_images.md`](REPORT_height_from_images.md) | Tree height from images: parallax, monocular depth |
| [`REPORT_depthpro_finetuning.md`](REPORT_depthpro_finetuning.md) | Depth Pro fine-tuned on FORTRESS for metric height |

## The three rules

1. **Never run anything on the login node.** Every script is submitted with
   `sbatch`; the job scripts live in `sbatch/`, `crownseg/sbatch/` and
   `depthft/sbatch/`.
2. **Everything runs inside `vesselgpt_7.sif`.** The venv on `/scratch` is only
   usable *inside* the container — its interpreter points at `/miniconda3`,
   which does not exist outside. Every job wraps its Python call in
   `apptainer exec --nv`.
3. **Never install torch into the venv.** It is inherited from the container
   (2.10.0+cu128, the build that works on the node's RTX PRO 6000 Blackwell,
   `sm_120`). A `pip install torch` would shadow it with a PyPI default build
   that imports fine and then dies on the first kernel.

## One-time setup

```bash
sbatch sbatch/setup_env.sbatch
```

Creates `/scratch/shared/$USER/envs/treeclf` as a venv **with
`--system-site-packages`** (inherits container torch, adds only deepforest,
opencv, omegaconf), downloads the DINOvTree checkpoint, and finishes with a real
forward pass on the GPU. If that smoke test fails, nothing else will work.

## How every job script is operated

All of them follow the same convention: defaults in the script, overridden by
environment variables placed **before** `sbatch`.

```bash
sbatch sbatch/run_infer_species.sbatch                                  # defaults
OUT=results_v2 MAX_TREES=200 sbatch sbatch/run_infer_species.sbatch     # overrides
EXTRA="--min-score 0.2 --crop-mode relative" sbatch sbatch/run_infer_species.sbatch
```

- `EXTRA` is a pass-through for any additional flag of the underlying Python
  script — use it before editing a job script.
- Variables understood by nearly all scripts: `SIF`, `REPO`, `ENV_DIR`,
  `HF_HOME`, `INPUT`/`FRAMES_DIR`, `OUT`, `EXTRA`.
- Every Python script also answers `--help`; run it through the container if you
  want to read the flags:
  ```bash
  apptainer exec /scratch/shared/$USER/containers/vesselgpt_7.sif \
      /scratch/shared/$USER/envs/treeclf/bin/python crownseg/queryseg.py --help
  ```

Watching a job:

```bash
squeue -u $USER
tail -f /scratch/shared/$USER/runs/logs/<jobname>_<jobid>.out
```

Logs land in `/scratch/shared/$USER/runs/logs/%x_%j.{out,err}` — `.out` carries
the progress, `.err` the tracebacks.

## Where things live

| What | Where |
|---|---|
| Container | `/scratch/shared/$USER/containers/vesselgpt_7.sif` |
| Python env | `/scratch/shared/$USER/envs/treeclf` |
| Checkpoints | `/scratch/shared/$USER/data/treeclf/checkpoints/` |
| HF / torch cache | `/scratch/shared/$USER/hf_cache` |
| Job logs | `/scratch/shared/$USER/runs/logs/` |
| Training runs | `/scratch/shared/$USER/runs/` |
| Our own drone frames | `/cold/Mahfuz/chosen_frames/<folder>/frame_*.jpg` |
| BAMFORESTS | `/scratch/shared/$USER/data/bamforests` |
| FORTRESS | `/scratch/shared/$USER/data/fortress` |
| Quebec Trees | `/scratch/shared/$USER/data/quebec_trees` |
| Results | `results*/` in the repo — **gitignored** |

**Result folder convention:** one folder per variant, `results_<name>/`, and the
matching diagnostic views in `results_views_<name>/`. Variants are kept side by
side, never overwritten, so older runs stay comparable.

## Gated models (SAM 3)

`facebook/sam3` is access-restricted. Request access on the model page, create a
read token, then deposit it once — `huggingface_hub` picks that path up on its
own under `HF_HOME` and every script works unchanged afterwards:

```bash
printf 'hf_YOUR_TOKEN' > /scratch/shared/$USER/hf_cache/token
chmod 600 /scratch/shared/$USER/hf_cache/token
```

## The common interface: label maps

Every segmentation method writes the same thing, which is what makes them
comparable and chainable:

```
<out>/<folder>/<frame stem>_labels.png     uint16, 0 = background, 1..N = one crown each
<out>/<folder>/<frame stem>_<method>.jpg   overlay for looking at
```

Anything downstream — species classification, clustering, evaluation,
diagnostic views — takes such a folder as input. `crownseg/eval_labels.py`
scores any of them against the BAMFORESTS ground truth, which is how old runs
can still be ranked against new ones without touching their code.

## The recommended chain (current best)

```bash
# 1. crowns from the frames — EoMT, trained on BAMFORESTS (F1 0.624 on unseen ground)
MODEL=eomt sbatch crownseg/sbatch/run_frames.sbatch
#    -> results_frames_eomt/  (label maps + overlays)
#    -> results_views_eomt/   (diagnostic views, rendered in the same job)

# 2. species per crown — head trained on FORTRESS (European species)
LABELS=results_frames_eomt OUT=results_arten_fortress \
    sbatch crownseg/sbatch/run_apply_head.sbatch
#    -> <folder>/<stem>_arten.csv, <stem>_arten.jpg, alle_kronen.csv
```

`run_frames.sbatch` does prediction and view rendering in one job on purpose, so
label maps and views can never drift apart.

The **scale per folder** is the parameter that decides everything here; see
below. It is passed as `SCALES="dense=1.2 pines=1.2 urban=1.5 ..."` and the
defaults in the job script are the measured/estimated values — read the comment
block in `crownseg/sbatch/run_frames.sbatch` before changing them.

Alternative stage-2 heads on the same label maps:

```bash
LABELS=results_frames_eomt sbatch crownseg/sbatch/run_classify.sbatch   # Quebec head, 14 Canadian classes
LABELS=results_frames_eomt sbatch crownseg/sbatch/run_cluster.sbatch    # unsupervised clustering, no labels needed
```

## Handing it to someone else: the standalone demo

[`demo_sam3_multiscale.py`](demo_sam3_multiscale.py) is a single self-contained
file that segments tree crowns with SAM 3 at several tile scales. It imports
nothing from this repository, needs no container and no cluster — copy the file,
`pip install "transformers>=4.57" torch opencv-python pillow numpy pandas`,
request access to `facebook/sam3` on Hugging Face, and run:

```bash
python demo_sam3_multiscale.py --image frame.jpg --out results/
```

`--image` takes a single file or a folder; a folder is searched recursively and
its structure is mirrored under `--out`, so frames of the same name in different
folders keep their own results. Per frame it writes an overlay JPG, a uint16
label map and a CSV, plus one `all_crowns.csv` for the run. Its defaults are the
configuration that measured best here (prompt `tree`, threshold 0.15, tile levels
2/3/4, no shape filter); `--help` explains every knob, and the docstring carries
the setup, the access instructions and what the input data has to look like —
the one real requirement being scale, not paths: a crown should be roughly
60-200 px across.

To check it on the cluster before passing it on:

```bash
IMAGE=/cold/Mahfuz/chosen_frames/dense/frame_000073.jpg sbatch sbatch/run_demo_sam3.sbatch
```

## Scale — read this before interpreting anything

Every model in this repo carries a scale assumption from its training data
(DINOvTree: 1.9 cm/px crops covering 9.73 m; BAMFORESTS: 1.70 cm/px), and an
ungeoreferenced drone frame has no known scale. Two ways to supply one:

- **From flight altitude and field of view** —
  `GSD = 2 · altitude · tan(HFOV/2) / image width`. Used by `infer_species.py`
  (`--altitudes pines=35 dense=60`, `--hfov-deg`) and `depthft/apply_frames.py`.
  Caveat: **73.7° is a placeholder in the code, not a camera spec.**
  `depthft/kalibrieren.py` back-calculates roughly 48° from frames with known
  altitude; the depthft scripts therefore default to `HFOV=48.0`. Any number
  derived from the 73.7° value inherits that error.
- **Measured from the model's own response** — `crownseg/scale_probe.py` sweeps
  the inference scale and takes the maximum. This is where the `SCALES` defaults
  in `run_frames.sbatch` come from, and it is the better source when the
  altitude is unknown, which it is for most folders.

`scale_sweep.py` demonstrates how much this matters: on the same detections the
top-1 species flips between broadleaf- and conifer-dominated depending on
whether 150 px or 400 px are interpreted as 9.73 m.

## Job catalogue

### Root — species classification and the earlier (heuristic) segmentation

| Job script | Purpose |
|---|---|
| `sbatch/setup_env.sbatch` | one-time env + checkpoint + GPU smoke test |
| `sbatch/run_infer_species.sbatch` | full chain DeepForest → crop → DINOvTree over all folders |
| `sbatch/run_classify_crowns.sbatch` | same, but on segmented crowns instead of detector boxes (`SEGMENTS=`, `CROP_FACTOR=`) |
| `sbatch/run_cluster_crowns.sbatch` | cluster crowns by feature vector (`FEATURES=head\|backbone`, `METHOD=kmeans\|hdbscan`, `CLUSTERS=`) |
| `sbatch/run_scale_sweep.sbatch` | scale diagnosis on single frames (`FRAMES=`, `CROP_SIZES=`, `N_TREES=`) |
| `sbatch/run_detect_only.sbatch` | DeepForest detections only, for diagnosis (`MIN_SCORE=`, `SHOW_SCORES=1`) |
| `sbatch/run_detect_scale_test.sbatch` | does downscaling help the detector? (`SCALES=`) |
| `sbatch/run_segment_sam3.sbatch` | SAM 3, text-prompted (`PROMPT=tree`, `THRESHOLD=`, `TILES=`) |
| `sbatch/run_demo_sam3.sbatch` | run the standalone demo here (`IMAGE=`, `OUT=`, `EXTRA=`) |
| `sbatch/run_segment_sam.sbatch` | SAM 1, automatic mask generation (`CROWN_PX=`) |
| `sbatch/run_segment_hybrid.sbatch` | SAM first, depth watershed on the remainder |
| `sbatch/run_segment_trees.sbatch` | monocular depth + marker watershed (`CROWN_PX=`, `EXTRA="--save-chm"`) |
| `sbatch/run_segment_prompted.sbatch` | depth peaks as point prompts for SAM (`PROMINENCE=`) |
| `sbatch/run_refine_crowns.sbatch` | split and merge instances using depth (`SPLITPROM=`, `EXTRA="--no-merge"`) |
| `sbatch/run_merge_crowns.sbatch` | merge over-split crowns (`SPLIT=`, `COLOR=`, `ROUNDS=`) |
| `sbatch/run_ablate_sam.sbatch` | compare SAM checkpoints under identical rules (`MODELS=`) |
| `sbatch/run_visualize.sbatch` | render the four diagnostic views (`SEGMENTS=`, `VIEWS=`, `FRAMES=`) |
| `sbatch/run_depth_probe.sbatch` | depth maps / hillshade for visual inspection (`MODELS=`) |
| `sbatch/run_stereo_probe.sbatch` | real parallax from consecutive video frames (`FOLDERS=`) |
| `sbatch/run_build_parallax.sbatch` | one measured height map per frame from all partner frames |
| `sbatch/run_distill_height.sbatch` | distil measured parallax into a single-image net (`MODE=train\|predict`) |
| `sbatch/run_crownnet.sbatch` | the original crown net (`MODE=prepare\|train\|predict`) — superseded |
| `sbatch/run_download_bamforests.sbatch` | fetch the dataset (`VARIANT=`) |

`make_viewer.py` builds a local HTML viewer over a results folder (zoom, blend
between original and overlay) — handy for reviewing a run without opening 33
JPEGs.

### `crownseg/` — crown instance segmentation (the part that works)

Training data first:

```bash
sbatch crownseg/sbatch/run_prepare.sbatch          # BAMFORESTS TIFF -> JPEG + polygons, ~2456 tiles, once
sbatch crownseg/sbatch/run_download_quebec.sbatch  # optional second source (DATE=)
sbatch crownseg/sbatch/run_quebec_prepare.sbatch
sbatch crownseg/sbatch/run_fortress_unpack.sbatch  # FORTRESS for the European species head
sbatch crownseg/sbatch/run_fortress.sbatch         # crowns + species labels (SITES=)
```

| Job script | Purpose |
|---|---|
| `run_queryseg.sbatch` | **EoMT / Mask2Former** — the best models (`ARCH=eomt\|mask2former`, `MODE=train\|eval\|predict`) |
| `run_maskrcnn.sbatch` | Mask R-CNN baseline (`MODE=train\|eval\|inspect\|predict`) |
| `run_frames.sbatch` | apply a trained model to our own frames + render views (`MODEL=`, `NAME=`, `SCALES=`, `DEPTH=1`) |
| `run_train_head.sbatch` | train the European species head on FORTRESS crowns |
| `run_apply_head.sbatch` | apply that head to label maps (`LABELS=`, `OUT=`, `KOPF=`, `SCALES=`) |
| `run_classify.sbatch` | species per crown with the Quebec head (`LABELS=`, `SCALES=`) |
| `run_cluster.sbatch` | cluster crowns without labels (`LABELS=`, `CLUSTERS=`) |
| `run_dinocluster.sbatch` | dense DINOv3 patch features, clustered directly (no segmentation) |
| `run_scale_probe.sbatch` | estimate the true image scale per folder from the model response |
| `run_sam3_depth.sbatch` | SAM 3 + Depth Pro, splitting and seeding (`DEPTH=depthpro\|dav2`) |
| `run_sam3_ceiling.sbatch` | upper bound: what could a fine-tuned SAM 3 reach? |
| `run_probe_seeds.sbatch` | are depth peaks usable as extra prompts? (answer: barely) |
| `run_reject.sbatch` | remove non-crowns (meadow, path, roof) from a label map |
| `run_depthcache.sbatch` | precompute depth maps for all tiles (`DEPTH=`) |
| `run_frames_tiefe.sbatch` | depth-only model on our frames |
| `run_schritte.sbatch` | render each method's intermediate steps on one crop — for the report |
| `run_video_parallax.sbatch` | measured height maps from a drone video (`VIDEO=`, `STRIDE=`, `COUNT=`) |

Evaluation helpers that need no GPU job of their own:
`crownseg/eval_labels.py` (any label map vs. ground truth),
`crownseg/metrics.py` (F1@IoU 0.5, AP50, per area),
`crownseg/show_gt.py` (look at what is actually annotated).

**Always report test1 (Hain) and test2 separately.** Hain appears in neither
training nor validation and is the only honest transfer number; test2 only says
how well already-seen areas fit.

### `depthft/` — metric depth and tree height (Depth Pro fine-tuned on FORTRESS)

Run in this order:

```bash
sbatch depthft/sbatch/run_prepare.sbatch      # 47 orthos -> ground raster, ~1 h, once (SITES=)
sbatch depthft/sbatch/run_check.sbatch        # MANDATORY: does the truth match the image?
sbatch depthft/sbatch/run_finetune.sbatch     # EPOCHS= TRAINABLE=decoder|all BATCH= ACCUM=
sbatch depthft/sbatch/run_evaluate.sbatch     # metric, on held-out test sites
sbatch depthft/sbatch/run_apply_frames.sbatch # on our own frames (ALTITUDES=)
sbatch depthft/sbatch/run_export.sbatch       # shippable package for other people
```

`run_check.sbatch` is not optional — a georeferencing offset between orthomosaic
and height model would silently poison the training, and exact zeros in the nDSM
are fill values, not ground.

Products and side tools: `run_karten_export.sbatch` (depth/height maps as
npy/png/jpg), `run_punktwolke.sbatch` (`.ply` / `.las`, anchored to the ground
rather than the camera), `run_wolke3d.sbatch` and `run_viewer3d.sbatch`
(standalone HTML 3D viewer), `run_vergleichsbild.sbatch` (pure vs. fine-tuned
figures), `run_kalibrieren.sbatch` (back-calculate the field of view),
`run_hoehe_pruefen.sbatch` (whole chain against the truth). All of them take
`HFOV=48.0` and `RUN=` (which training run to read).

## Reading the outputs

| File | Content |
|---|---|
| `<stem>_labels.png` | uint16 label map, one id per crown, 0 = background |
| `<stem>_arten.csv` | one row per crown: box, top-3 species with probabilities, predicted height, entropy |
| `alle_kronen.csv` / `all_trees.csv` | all crowns of a run in one table |
| `summary_by_folder.csv` | per-folder aggregate |
| `results_views_<name>/` | four diagnostic views: `instanzen`, `luecken`, `relief`, `trennungen` |

What the four views answer is described in [`REPORT_species.md`](REPORT_species.md)
(section *Diagnostic views*); rendering them takes about two minutes for all 33
frames and needs no model, only the label maps.

Two caveats that apply to every species output: the Quebec head can only emit
its 14 Canadian classes (genus level is often usable, species level is not), and
`height_m_pred` from that head is not calibrated — use the depthft chain if the
height itself matters. `entropy` is the more honest confidence indicator than
the softmax value.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Job hangs, GPU idle, SLURM still says RUNNING | OpenCV thread pool in forked DataLoader workers. `cv2.setNumThreads(0)` is already set in `bamforests.py`; add it in any new data layer. |
| CUDA error on the first kernel, import was fine | A PyPI torch shadowing the container build. Rebuild the venv; never `pip install torch`. |
| `python: not found` / wrong interpreter outside a job | The venv only works inside the container. Go through `apptainer exec --nv`. |
| SAM 3 downloads fail with 401/403 | Missing or unreadable `$HF_HOME/token` — see *Gated models*. |
| Crowns look far too coarse or far too fine | Wrong scale. Re-run `crownseg/scale_probe.py` and set `SCALES=` per folder. |
| Validation loss rises while results improve | Expected for detection models. Select checkpoints by instance F1, not by val loss. |

## Note on language

The documentation is English. Python module docstrings, code comments and many
identifiers (`ordner`, `kronen`, `hoehe`, `tiefe`) and output filenames
(`_arten.csv`, `alle_kronen.csv`) are still German — they are part of the code
and of existing result folders, so renaming them would break older runs.
