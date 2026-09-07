# Height from images — foundations, methods, and what of it is in this pipeline

A report from the ground up, assuming no prior knowledge of camera technology,
stereo or photogrammetry. All examples use the actual numbers of this project.

---

## Contents

1. [The problem: why colour is not enough](#1-the-problem-why-colour-is-not-enough)
2. [How a camera turns a 3D world into a 2D image](#2-how-a-camera-turns-a-3d-world-into-a-2d-image)
3. [Why a single image contains no height](#3-why-a-single-image-contains-no-height)
4. [Route A — guessing height: monocular depth estimation](#4-route-a--guessing-height-monocular-depth-estimation)
5. [Route B — measuring height: parallax](#5-route-b--measuring-height-parallax)
6. [The concrete method: plane + parallax](#6-the-concrete-method-plane--parallax)
7. [From one pair to a map](#7-from-one-pair-to-a-map)
8. [Distillation: moving the measurement into a network](#8-distillation-moving-the-measurement-into-a-network)
9. [What happens with the height map: CHM, treetops, watershed](#9-what-happens-with-the-height-map-chm-treetops-watershed)
10. [Where the models come from: training data and the domain gap](#10-where-the-models-come-from-training-data-and-the-domain-gap)
11. [Failure modes and where the pipeline stands](#11-failure-modes-and-where-the-pipeline-stands)
12. [Glossary](#12-glossary)
13. [References](#13-references)

---

## 1. The problem: why colour is not enough

The goal of the project is to delineate **individual tree crowns** in drone
imagery and then determine their species. The second part is solved (DINOvTree).
The first part is the problem.

Picture a closed canopy from above. Two neighbouring beeches touch. In the RGB
image you see there:

- green on the left, green in the middle, green on the right
- the same leaf structure everywhere
- no edge, no colour difference, no contrast

There is simply no information **in the image** about where one tree ends. A human
struggles with it just as much as an algorithm. No segmentation model in the world
can find a boundary that is not depicted.

In three dimensions, by contrast, the boundary is obvious: every tree has a
treetop, and between two tops there is a **dip**. Seen from the side, the canopy
looks like a chain of hills, not like a slab. The tree boundary is the valley
floor between two hills.

```
        top A             top B
           /\                /\
          /  \     dip      /  \
         /    \    \/      /    \
        /      \______\___/      \
       /                          \
    ===============================  ground
       |<-- tree A -->|<- tree B ->|
             boundary lies here ^
```

That is exactly how forest remote sensing has worked for decades: on a **CHM**
(canopy height model). That is a raster map saying, for every ground point, how
tall the vegetation there is. Treetops = local maxima, boundaries = watersheds
between them. A CHM is normally produced with **LiDAR** — a laser scanner on an
aircraft measures millions of distances.

We have no LiDAR. We have video frames from a drone.

The whole pipeline therefore revolves around one question:

> **How do you get a usable height map out of ordinary RGB images?**

There are two fundamentally different answers — guessing and measuring. Both are
implemented in this repo, and the comparison of the two is the core of the work.

---

## 2. How a camera turns a 3D world into a 2D image

To understand why height is hard, you first have to understand what a camera
actually does. That is simpler than it sounds.

### The pinhole camera model

Think of a closed box with a tiny hole at the front and film at the back. Light
from a point in the world falls through the hole and hits exactly one place on the
film. That is the entire model — modern lenses are more complicated, but
geometrically they behave like this.

```
   World                   Hole            Sensor
                             |
   Point P  ----             |             ----
   (X, Y, Z)    ----         |         ----
                    ----     |     ----
                        -----o-----  <-- image point (u, v)
                    ----     |     ----
                ----         |         ----
            ----             |             ----
                          <- f ->
                        focal length
```

The projection equation is a simple similar-triangles calculation:

```
u = f * X / Z
v = f * Y / Z
```

- `X, Y, Z` are the world coordinates of the point, measured from the camera. `Z`
  is the **depth** — the distance along the viewing direction.
- `u, v` are the image coordinates in pixels.
- `f` is the **focal length expressed in pixels**. That is not a physical length
  but a conversion constant depending on the lens *and* the sensor resolution.

The decisive part is the **division by Z**. Everything gets smaller the further
away it is. That is perspective, in one line.

### Focal length and field of view for our frames

Lenses are usually specified by their **field of view** rather than by `f`. The
relation:

```
f = (image width in px / 2) / tan(HFOV / 2)
```

For this project ([infer_species.py:350](infer_species.py#L350)):

- image width: 1920 px (the frames are 1920×1080)
- horizontal field of view: 73.7°  ← **careful: that is a default estimate in the
  code, not a measured camera specification.** If the real EXIF data are
  available, the value should come from there.

```
f = 960 / tan(36.85°) = 960 / 0.750 = 1280 px
```

### Ground sampling distance (GSD)

For a nadir capture — camera pointing straight down — `Z` for the ground equals
the flight altitude `H`. That lets you compute how many metres a pixel covers.
This is called **GSD** (ground sample distance),
[infer_species.py:133](infer_species.py#L133):

```
ground width = 2 * H * tan(HFOV / 2)
GSD          = ground width / image width in px
```

For the folder `80m`:

```
ground width = 2 * 80 m * 0.750 = 120 m
GSD          = 120 m / 1920 px  = 0.0625 m/px  =  6.25 cm per pixel
```

**What that means in practice:**

| Object | real size | in the image |
|---|---|---|
| tree crown, medium | 8 m diameter | 128 px |
| tree crown, large | 15 m | 240 px |
| a single branch | 20 cm | 3 px |
| park bench | 1.5 m | 24 px |

So the default `--crown-px 100` in [segment_trees.py](segment_trees.py)
corresponds to a crown of a good 6 m diameter. Plausible.

> **A side note on the scale problem:** the DINOvTree checkpoint was trained on
> tiles at 1.9 cm/px ([infer_species.py:41](infer_species.py#L41)). Our frames
> have 6.25 cm/px — a factor of 3.3 coarser. A tree the network saw at 400 px
> width in training is 120 px wide for us. That is why
> [scale_sweep.py](scale_sweep.py) and the GSD conversion when cropping exist. It
> is a topic of its own, but it shows: in this domain scale is the core problem
> everywhere.

---

## 3. Why a single image contains no height

Now the crux. Look at the projection equation again:

```
u = f * X / Z
```

We know `u` (that is the image) and `f`. We want to know `X` and `Z`. That is
**one equation with two unknowns**. It has infinitely many solutions.

Concretely: all of these points land on the same pixel.

```
camera
   o
   |\
   | \      * small tree, 30 m away
   |  \
   |   \
   |    \
   |     \     * medium tree, 60 m away
   |      \
   |       \
   |        \
   |         \      * giant tree, 90 m away
   |          \
```

A 3 m shrub near the camera and a 30 m oak far away are **exactly
indistinguishable** in the image if they lie on the same viewing ray. That is not
a shortcoming of the technology but a mathematical property of the projection:
information is lost, irretrievably.

This ambiguity is called **depth ambiguity**. It is the reason the following two
chapters exist at all.

There are exactly two ways out:

- **Guess.** Bring in additional knowledge about the world — "trees are usually
  between 5 and 40 m tall", "foliage looks like this from this distance". That is
  what a neural network does. → chapter 4
- **Measure.** Add a second observation from a different place. Then there are two
  equations for two unknowns, and the thing is uniquely solvable. → chapter 5

---

## 4. Route A — guessing height: monocular depth estimation

### What a depth model does

A monocular depth model receives a single image and outputs a depth for every
pixel. It can only do that because it has learned statistical relationships from
millions of training images:

- **Known sizes.** A car is about 4.5 m long. If it occupies 200 px, the distance
  can be derived.
- **Occlusion.** What overlaps another object is nearer.
- **Texture gradient.** Grass gets finer and lower in contrast with distance.
- **Perspective lines.** Converging road edges give depth.
- **Haze.** Distant objects become paler and bluer.
- **Shading.** How light falls on a surface reveals its inclination.

These are all **priors** — learned assumptions about how the world usually looks.
None of it is a measurement. The model guesses in a very well-informed way.

### What is used in [depth_probe.py](depth_probe.py)

```python
MODELS = {
    "depth_anything": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "depthpro":       "apple/DepthPro-hf",
}
```

Both are large vision transformers with a decoder. Interesting: **Depth Anything
itself came about through distillation** — a teacher model on 1.5 M labelled
images, then pseudo-labels for 62 M unlabelled images, and a student trained on
those. Remember that for chapter 8; our approach is the same recipe with a
different teacher.

### Why it does not work well here

**First: the wrong domain.** These models were trained on images that people take
— street scenes, interiors, landscapes from the ground. In such images there is a
horizon, a foreground, converging lines, objects of known size. A canopy from 80 m
straight down has none of that. It is a textured green surface without any of the
learned cues. There the model is **out of distribution** — it keeps guessing, but
without a usable basis.

**Second: hallucination in low-contrast areas.** Where a network has no cues, it
invents the most plausible thing — usually a smooth surface. So in exactly the
closed canopy where we would need the boundary, it reliably delivers a gentle
bulge without dips.

**Third — and this is the finest piece of evidence: shadows.**

The network has learned that dark areas are often depressions, niches or
occlusions. A cast shadow on a meadow is, however, geometrically **exactly as high
as the meadow** — namely zero. It nevertheless shows up as structure in the depth
map.

You can look at this directly in
[results_views_tiefe_gegen_parallax/urban/](results_views_tiefe_gegen_parallax/urban/).
The instance view of the park screenshot shows, for many free-standing trees, a
polygon covering the crown **plus its cast shadow**, consistently offset towards
the lower right — i.e. in the direction of the shadow. At 208 instances and 41 %
coverage that is not an isolated case.

Two causes act together here:

1. **SAM segments by contrast.** On a bright meadow, crown and attached shadow
   form *one* dark blob. Its outer edge is the stronger gradient; the boundary
   between the crown and its own shadow is much weaker. SAM takes the stronger
   edge. The compactness filter
   ([segment_sam.py:85](segment_sam.py#L85)) catches free-standing shadow bands,
   but a directly attached shadow still gives a reasonably round shape.
2. **The depth map cannot correct it.** The ground filter in
   [segment_hybrid.py:53-55](segment_hybrid.py#L53-L55) excludes ground via a
   *height percentile*. If the depth map considers the shadow elevated, that
   filter does not bite.

Remember this example — in chapter 6 it resolves itself.

---

## 5. Route B — measuring height: parallax

### The everyday phenomenon

Hold a finger in front of your face and close your left and right eye
alternately. The finger jumps back and forth. A tree on the horizon does not.

That is **parallax**: when the observer changes position, near objects shift more
than far ones. The magnitude of the shift is a direct measure of distance. Your
brain evaluates this constantly — that is stereoscopic vision.

Importantly: that is not an estimate and not a prior. It is a geometric necessity.

### The calculation

Two cameras (or one camera at two places) separated by `B` — the **baseline**. A
point at depth `Z` appears at a slightly different place in the two images. The
difference is called **disparity** `d`:

```
d = f * B / Z          and rearranged:          Z = f * B / d
```

Two images, two equations, two unknowns — the ambiguity from chapter 3 is
resolved. **That is why this is a measurement and not a guess.**

A few consequences worth keeping in mind:

- **Longer baseline = stronger signal.** Twice the separation, twice the
  disparity. Hence the parameter `--pair-stride` in
  [stereo_probe.py](stereo_probe.py): a larger frame separation = a longer
  baseline.
- **But:** a longer baseline = less image overlap and harder matching. A
  trade-off, not a free lunch.
- **No motion = no signal.** If the drone hovers, `B ≈ 0`, so `d ≈ 0`. You measure
  only noise. That is exactly why
  [build_parallax.py:48](build_parallax.py#L48) has `--min-displacement 15` —
  pairs below that are discarded.

### Why we do not do classical stereo

Classical stereo vision requires:

- two **calibrated** cameras (focal length, distortion, principal point exactly
  known)
- the exact relative position and orientation of both cameras
- a **rectification** — undistorting both images so that corresponding points lie
  on the same image row

We have none of that. These are video frames from a drone, without a calibration
protocol, without a known pose, without timestamp synchronisation. The field of
view in the code is an estimate.

This project therefore takes a route that **works without calibration** and gives
up absolute measurements in metres in exchange.

---

## 6. The concrete method: plane + parallax

That is the core of [stereo_probe.py](stereo_probe.py). The approach is old and
well studied — in the literature it is called **plane + parallax (P+P)**,
developed in the 1990s by Irani, Anandan, Kumar and others.

### The basic idea in one sentence

> Work out how the **ground** shifted between two frames, and subtract that. What
> remains can only come from objects that are **not** on the ground — and its
> magnitude grows with height.

The elegant part: you never have to determine the camera motion explicitly. It is
implicit in the ground shift and is cancelled along with it.

### Step 1 — find correspondences (SIFT)

First we need point pairs: "this corner in frame A is the same corner as that one
in frame B".

**SIFT** (scale-invariant feature transform, Lowe 1999/2004) does that in two
parts:

- *Detector:* finds salient places — corners, blobs, structures that stand out
  from their surroundings. A smooth meadow yields nothing; a roof ridge or a
  branch fork does.
- *Descriptor:* describes the neighbourhood of each place as 128 numbers,
  constructed so that the description does not change when the image is rotated,
  scaled or brightened.

Then descriptors are compared between the images. The **ratio test**
([stereo_probe.py:53](stereo_probe.py#L53)) is more important than it looks:

```python
good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
```

For every point from A, the two most similar candidates in B are found. Only if
the best is **clearly** better than the second best (factor 0.75) is the match
accepted. In a forest very many places look very similar — without this test half
the matches would be wrong.

### Step 2 — determine the ground plane (homography + RANSAC)

A **homography** is a 3×3 matrix describing how a *planar surface* maps between
two camera views. It has a remarkable property: for a real plane the mapping is
**exact**, no matter how the camera moved — translation, rotation, tilt, zoom, all
included.

Intuition: photograph a chessboard twice from different angles. The homography is
the transformation that lays one photo perfectly onto the other. For anything
sticking out of the board plane it does not work.

For nadir captures over forest the dominant plane is the **forest floor**. It is
not perfectly flat, but good enough — and it is the reference surface against
which we want to measure height.

**RANSAC** (Fischler & Bolles 1981) is the trick that makes this work despite
wrong matches. Instead of weighting all points equally:

1. Draw 4 point pairs at random and compute a homography from them.
2. Check all remaining pairs: how many fit this hypothesis (error below
   `--ransac-thresh`, 3 px here)? Those are the **inliers**.
3. Repeat hundreds of times, keep the hypothesis with the most inliers.

The result is robust against a substantial share of gross errors — and at the same
time against the tree crowns themselves. **Crowns are outliers for RANSAC**,
because they do not behave like the ground. And those are precisely what we want
to find.

```python
homography, inliers = cv2.findHomography(points_b, points_a, cv2.RANSAC, args.ransac_thresh)
```

### Step 3 — warp

```python
warped = cv2.warpPerspective(gray_b, homography, (gray_a.shape[1], gray_a.shape[0]))
```

Frame B is warped so that its ground coincides with the ground of frame A.
Afterwards:

- **Ground:** at the same place in both images → difference zero
- **Treetops:** still displaced, because they do not behave like the plane
- **Image border:** partly black, because B does not cover the full area of A

The last point is caught in
[stereo_probe.py:105](stereo_probe.py#L105): `valid = warped > 0`.

### Step 4 — measure the residual flow (optical flow)

**Optical flow** answers, for *every single pixel*: where did it move to? The
result is a vector field, two numbers per pixel.

We apply it to the already-warped pair. Whatever motion is still found here is by
construction only the height-induced part.

```python
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
dis.setFinestScale(args.finest_scale)   # 0 = finest level
dis.setPatchSize(args.patch_size)       # 8
flow = dis.calc(gray_a, warped, None)
residual = np.linalg.norm(flow, axis=2)  # magnitude of the residual vector
```

**Why DIS and not Farneback:** Farneback fits a polynomial within a window
(default 41 px here). That smooths — and smears exactly the small-scale crown
detail that matters. DIS works with small patches (8 px) and spatial propagation
and is sub-pixel accurate. That is decisive, as the next calculation shows.

### Step 5 — residual flow is height

Now the formula. Camera at height `H` above the ground, point at height `h` above
the ground, i.e. at depth `Z = H - h`. Baseline `B`.

```
displacement of a ground point:   f * B / H
displacement of a point at h:     f * B / (H - h)

residual flow = difference       =  f * B * h
                                   ------------
                                    H * (H - h)
```

**Worked example with our numbers** (f = 1280 px, H = 80 m, baseline B = 2 m):

| Object | height h | residual flow |
|---|---|---|
| shadow on the meadow | 0 m | **0.00 px** |
| meadow, path, car park | 0 m | **0.00 px** |
| car | 1.5 m | 0.61 px |
| shrub | 5 m | 2.13 px |
| medium tree | 15 m | 7.38 px |
| tall tree | 25 m | 14.55 px |
| roof ridge | 12 m | 5.65 px |

Three things stand out immediately:

**(a) The effect is small.** A 5 m shrub moves by 2 pixels. That is why sub-pixel
accuracy is not a nicety but the precondition — and why DIS.

**(b) The relationship is non-linear.** At h = 25 m the residual flow is not 5×
that of h = 5 m but almost 7×. Tall trees are emphasised disproportionately. For
relative treetop finding that is harmless; for metric statements it would need
correcting.

**(c) The shadow has exactly zero parallax.** And that is the central point.

### Why this solves the shadow problem

A cast shadow lies **on the ground**. So between two frames it shifts exactly like
the ground plane — and that is fully cancelled by the homography. Residual flow
zero.

It does not matter how dark it is. SIFT cares about structure, not brightness.
Optical flow measures displacement, not colour. Geometry does not even know the
concept "dark".

Put more sharply: **the shadow does not travel with the tree.** It sticks to the
ground while the crown stands above it. Between frame A and frame B the two
visibly separate. Precisely that separation is the signal — and a single-image
method cannot see it in principle.

That is the strongest substantive argument for the whole parallax approach.

### What can go wrong

In fairness, the method has clear limits:

| Problem | Cause | Remedy in the code |
|---|---|---|
| Drone hovers | B ≈ 0, no signal | `--min-displacement 15` |
| Too few SIFT points | homogeneous texture, blur | returns `None` below 20 points |
| Ground not visible | in a closed stand RANSAC may fit the canopy plane | — open, see below |
| Wind moves branches | real motion, misread as height | — open |
| Rolling shutter | sensor reads row by row, distorts under fast motion | — open |
| Baseline too long | too little overlap | `--min-overlap 0.6` |

The point "ground not visible" deserves attention: RANSAC finds the plane with the
most inliers. In a genuinely closed stand that could be the **canopy plane**
instead of the ground. Then you measure height relative to the mean canopy — still
usable for treetop finding, but the interpretation changes. It can be checked: do
the inliers lie where ground is visible in the image?

---

## 7. From one pair to a map

[stereo_probe.py](stereo_probe.py) computes one pair. The pipeline needs **one**
map per frame, and one as free of noise as possible. That is what
[build_parallax.py](build_parallax.py) does.

### Problem 1: the scale is arbitrary and changes

The residual flow is proportional to `B` — the camera motion between the two
frames. We do not know it, and it differs for every pair. A pair with 4 m
separation yields values twice as high as one with 2 m, **for an identical scene**.

Averaged unweighted, the longest pair would outvote all the others.

**Solution:** divide every map by the median displacement of its pair. The median
displacement is a proxy for the baseline. Afterwards all maps are on the same —
still unknown — scale.

### Problem 2: individual pairs are noisy

Optical flow makes mistakes, especially in homogeneous areas. **Solution:** for
every frame A, all other frames of the same folder are tried as partners and the
normalised results are averaged. Random errors average out, the real height signal
remains.

### What comes out

```
/scratch/shared/nik/data/treeclf/parallax_cache/
    100__frame_000537.npy
    80m__frame_000297.npy
    dense__frame_000073.npy
    ...
```

One `.npy` file per frame, in the same format as the depth cache — so that the
downstream scripts can use both sources interchangeably
([visualize_crowns.py:210](visualize_crowns.py#L210): `--surface depth|parallax`).

### The current extent — and an important restriction

```
$ ls parallax_cache/ | sed 's/__.*//' | sort | uniq -c
      4 100      4 80m      4 dense      4 dense1
      4 mixed    4 mixed1   5 pines
```

**29 frames from 7 folders.** The folder `urban` is missing entirely — it consists
of four mutually independent Google Earth screenshots from different days, not of
a video sequence. Without a common scene there is no baseline and no parallax.

That is **decisive** when judging the results so far: the shadow errors from
chapter 4 come from `urban` and show exclusively the depth side. The folder name
`results_views_tiefe_gegen_parallax` is misleading there.

29 frames are also a very narrow data basis for the next chapter.

---

## 8. Distillation: moving the measurement into a network

### The remaining problem

Parallax is measured rather than guessed — but it needs **several frames of the
same scene with camera motion in between**. For a single photograph it cannot be
run. For later operation that is a hard restriction.

### What distillation is

Classically (Hinton et al. 2015): a large, slow **teacher model** produces
outputs, and a small **student model** is trained to imitate those outputs. The
student learns not from the original labels but from what the teacher produces.
Goal: almost the same quality at a fraction of the cost.

For us the teacher is **not a network but a method** — the parallax computation.
What is distilled is therefore not compute but a **capability**:

```
before:   needs a video sequence + SIFT + homography + optical flow
after:    needs one image
```

The knowledge moves out of the algorithm and into weights.

That is exactly how the models from chapter 4 were built. MegaDepth (2018) ran
structure-from-motion over internet photo collections and used the results as the
training target for a single-image network. Depth Anything (2024) scaled that to
62 million images. We do the same — only with a teacher from **our own domain**
instead of from generic ground-level images.

### The setup in [distill_height.py](distill_height.py)

**Backbone: DINOv3 ViT-B/16, frozen**

```python
for parameter in backbone.parameters():
    parameter.requires_grad_(False)
backbone.eval()
```

A vision transformer divides the image into tiles of 16×16 pixels (`PATCH = 16`)
and describes every tile by a vector of 768 numbers — a **patch token**. For a
512×512 crop that is 32×32 = 1024 tokens.

You can think of a token as a rich description: "here is conifer texture, medium
contrast, an edge from upper left to lower right, illumination from an angle".
What exactly is in it is learned and not directly interpretable — but it is far
more informative than the raw pixels.

The backbone comes from the DINOvTree checkpoint and is thus already fine-tuned on
tree crowns. It is **not trained further**. Reason: 29 frames would overfit an
86-million-parameter model in seconds.

**Head: small convolutional decoder, ~2 M parameters**

```python
nn.Conv2d(768, 256, 3), GELU,
nn.Conv2d(256, 256, 3), GELU,
Upsample(×2),
nn.Conv2d(256, 128, 3), GELU,
Upsample(×2),
nn.Conv2d(128,  64, 3), GELU,
nn.Conv2d(64,    1, 1),      # → one channel = height
```

The token map (32×32×768) is upscaled step by step and reduced to one channel.
Only these ~2 M parameters are trained. That is few enough that 29 frames plus
augmentation might suffice.

**Training data: random crops**

512×512 crops from the frames, plus rotations by multiples of 90° and mirroring
([distill_height.py:113](distill_height.py#L113)). For nadir captures rotation
augmentation is physically legitimate — there is no "up" in a straight-down image.
From 29 frames one thus gets arbitrarily many training examples, though not
arbitrarily many *independent* ones.

### The decisive trick: a scale-invariant loss

That is the conceptually most important line in the whole script.

```python
loss = F.l1_loss(standardize(prediction), standardize(targets))
```

```python
def standardize(x):
    mean = x.flatten(1).mean(dim=1, keepdim=True)
    std  = x.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-6)
    return ((x.flatten(1) - mean) / std).view_as(x)
```

**Why that is necessary:** parallax has no known unit. Even after the
normalisation in [build_parallax.py](build_parallax.py) a residual factor remains
that depends on the flight situation. Two frames with identical forest structure
can have target values differing by a factor of 3 — only because the drone flew at
a different speed.

An ordinary L1 loss would force the network to guess the flight speed from the
image. That is impossible, and the network would get lost in compromises.

**What standardize does** — a numerical example:

```
target A (slow flight):   [1, 2, 3, 4, 5]      → standardised: [-1.41, -0.71, 0, 0.71, 1.41]
target B (fast flight):   [10, 20, 30, 40, 50] → standardised: [-1.41, -0.71, 0, 0.71, 1.41]
```

After standardisation the two are **identical**. Scale and offset are removed,
only the *shape* of the relief remains.

And exactly that shape is all we need: treetop finding reads local maxima,
watershed reads saddle prominence. Both are insensitive to any monotone
stretching. Absolute metres would not be used anywhere anyway.

This idea comes from Eigen et al. (2014) and has become the standard in depth
estimation via MiDaS (Ranftl et al.).

### Split by folder, not by frame

```python
parser.add_argument("--val-folders", nargs="*", default=["dense", "mixed"])
```

That matters methodologically. Frames from the same flight overlap spatially — in
part they show **the same tree** from a slightly different angle. A random frame
split would put the same trees in training and validation. The validation loss
would look excellent and would say nothing about new stands.

Whole folders are therefore held back. With 7 folders in total that means,
however: **5 folders training, 2 folders validation, 29 frames in total.** That is
very little. The validation number will have a high variance.

### The overall flow

```
  ┌─────────────────────────────────────────────────────────┐
  │  TEACHER — geometric, no neural network                 │
  │                                                         │
  │  video frames  →  SIFT  →  RANSAC homography            │
  │               →  optical flow  →  residual flow         │
  │               →  normalise, average over partners       │
  │                                                         │
  │  Result: measured height map  (needs a video sequence)  │
  └───────────────────────┬─────────────────────────────────┘
                          │  serves as the training target
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  STUDENT — neural network                               │
  │                                                         │
  │  single image  →  DINOv3 ViT-B/16 (frozen)              │
  │               →  convolutional decoder (2 M, trained)   │
  │               →  scale-invariant L1 against the target  │
  │                                                         │
  │  Result: estimated height map  (needs one image)        │
  └─────────────────────────────────────────────────────────┘
```

### The obvious concern

The student cannot become better than its teacher. If the parallax maps are noisy,
the head learns the noise along with everything else. MegaDepth reports exactly
this problem and needed explicit data cleaning.

For comparison: Tolan et al. (2024, Meta) trained **the same architecture** —
DINOv2 frozen plus a convolutional decoder — with **LiDAR** as the teacher and
reach a mean absolute error of 2.8 m. Our teacher is markedly weaker and our data
volume smaller by orders of magnitude. Expectations should be calibrated
accordingly.

---

## 9. What happens with the height map: CHM, treetops, watershed

Whatever the source — depth, parallax or the distilled network — the downstream
processing is the same. It is in [segment_trees.py](segment_trees.py) and follows
the standard forestry procedure.

### Step 1: build a surrogate CHM

```python
def build_pseudo_chm(depth, crown_px, detrend_factor):
    surface = -depth.astype(np.float32)                                  # invert
    trend   = cv2.GaussianBlur(surface, (0, 0), crown_px * detrend_factor)  # trend
    return surface - trend                                               # detrend
```

Three operations:

**Invert.** Depth models output distance. Closer to the camera = higher above the
ground. Change of sign.

**Compute the trend.** A very strong blur (sigma = 100 px × 3 = 300 px) gives the
large-scale shape without any crown detail.

**Subtract the trend.** That removes:
- the tilt of the camera (one image edge is further away than the other)
- terrain slope
- the ground-plane prior of the depth model

That is the image-processing equivalent of **DSM minus DTM** — surface model minus
terrain model gives vegetation height. The standard recipe in forest remote
sensing.

### Step 2: canopy mask

```python
canopy = smoothed > np.percentile(smoothed, gap_percentile)
```

The lowest areas are gaps, paths, ground — those are excluded. **This is where the
shadow error gets through:** if the height source considers the shadow elevated, it
survives this filter.

### Step 3: find treetops

```python
seeds = h_maxima(surface, (high - low) * peak_prominence)
```

Not simply local maxima — **prominence**. The term comes from topography: the
prominence of a peak is the height difference to the lowest col you have to cross
to reach a higher peak.

```
              /\  <- prominent: deep cols on both sides
             /  \
        /\  /    \
       /  \/      \      <- the small rise on the left is
      /  ^         \        a local maximum but not prominent
     /   not        \
```

Why that matters: watershed is a **partition, not a detector**. It divides the
area into exactly as many parts as markers go into it. A pure local maximum is far
too weak a criterion — in flat areas any amount of noise creates arbitrarily many
of them, and hence arbitrarily many pseudo-crowns. `h_maxima` demands a minimum
rise above the surroundings.

The threshold is chosen **relative** to the robust range (5th to 95th percentile).
It therefore does not depend on the arbitrary scale of the height source — the
same reasoning as with the scale-invariant loss.

### Step 4: watershed

The metaphor: imagine the inverted CHM as a landscape — treetops become basins.
Let water flow in at every marker. The basins grow. Where two basins meet, a dam
is built. Those dams are the crown boundaries.

```python
labels = watershed(-smoothed, markers, mask=canopy)
```

The result is a label map: every pixel carries the number of its crown.

### Step 5: shape filter

Crowns are reasonably round and have a plausible size. Segments that are too
small, too large or too elongated are thrown out
([segment_sam.py:85](segment_sam.py#L85)).

### The hybrid

[segment_hybrid.py](segment_hybrid.py) combines the two methods according to their
strengths:

- **SAM** finds precise boundaries where there is edge contrast and leaves out the
  uncertain parts. High precision, coverage only 50–74 %.
- **Watershed** partitions completely, but also tiles over areas where there is no
  tree.

The hybrid lets SAM fix the certain crowns and then runs the watershed
**exclusively on the remaining area**. Every crown carries its origin
(`quelle = sam | watershed`) in the output, so that the two parts can be judged
separately.

---

## 10. Where the models come from: training data and the domain gap

So far this has been about *how* the methods work. This chapter asks what they
were trained *on* — and that explains a large part of the failure modes in the
next chapter.

The pipeline contains five trained models. Only one of them has ever seen a frame
from this project.

### Overview

| Component | Training data | Region | Scale |
|---|---|---|---|
| Height head ([distill_height.py](distill_height.py)) | 29 of our own frames, target = parallax | **our own flights** | 6.25 cm/px |
| DINOvTree-B, species classification | Quebec Trees, 14 classes | Québec, Canada | 1.9 cm/px |
| DeepForest, detection | NEON | USA | ~10 cm/px |
| SAM 1/2/3, segmentation | SA-1B, generic everyday images | worldwide, no aerial imagery | any |
| Depth Anything V2 / DepthPro | mixed depth datasets | ground-level perspectives | any |
| CrownNet ([crownnet.py](crownnet.py)) | BAMFORESTS, 58,228 crowns | **Germany** | 1.70 cm/px |

### The height head — the only model on our own data

- **Target:** the measured parallax maps from `parallax_cache/`
- **Extent:** 29 frames from 7 folders — `pines` (5 frames), `100`, `80m`,
  `dense`, `dense1`, `mixed`, `mixed1` (4 each)
- **Split:** `dense` and `mixed` as validation → **about 21 frames training, 8
  validation**
- **Trained parameters:** only the ~2 M of the convolutional decoder; the backbone
  stays frozen

21 training frames are very few even with crop and rotation augmentation. The
crops of one frame are not independent of each other — the effective sample is
considerably smaller than the number of drawn crops suggests. That is the
bottleneck of this stage, and more video folders would help here more than any
change to the architecture.

### DINOvTree — Quebec Trees

The checkpoint `dinovtreeb_quebectrees.pth` is a DINOv3 ViT-B/16 fine-tuned on
data from **Québec, Canada**. The category file
[quebec_trees_categories.json](third_party/quebec_trees_categories.json) contains
17 entries; after excluding the three supercategories that the paper counts as
annotator uncertainty (`Pinopsida`, `Magnoliopsida`, `Acer L.`, see
`QUEBEC_TREES_EXCLUDE`), **14 classes** remain:

| Group | Classes |
|---|---|
| Conifers | *Thuja occidentalis*, *Abies balsamea*, *Larix laricina*, *Tsuga canadensis*, *Pinus strobus*, *Picea* |
| Broadleaves | *Fagus grandifolia*, *Populus*, *Acer pensylvanicum*, *A. saccharum*, *A. rubrum*, *Betula alleghaniensis*, *B. papyrifera* |
| other | `dead` |

That is a boreal to north-eastern North American species set. For Central European
forest it lacks, among others, Norway spruce (*Picea abies*), Scots pine, oak,
ash, lime and Douglas fir. *Fagus grandifolia* is the American beech, not *Fagus
sylvatica*.

A German stand can therefore at best be mapped onto the nearest Canadian relative
with this head. That is exactly why
[cluster_crowns.py](cluster_crowns.py) exists: it groups crowns by their feature
vectors instead of forcing them into a class set that does not contain the species
present at all.

On top of that comes the scale offset from chapter 2: training at 1.9 cm/px, our
frames at 6.25 cm/px — a factor of 3.3.

### DeepForest — NEON

`weecology/deepforest-tree`, pretrained on data of the **National Ecological
Observatory Network** (USA) at about 10 cm/px. A pure detector: it delivers crown
boxes without a species. The scale is closer to our 6.25 cm/px than the
classifier's, but the vegetation is again North American. How strongly detection
depends on scale is investigated by
[detect_scale_test.py](detect_scale_test.py).

### SAM — SA-1B, and not a single forest

[segment_sam.py](segment_sam.py) uses `facebook/sam-vit-large`,
[ablate_sam.py](ablate_sam.py) compares six variants from SAM 1 to SAM 2.1, and
[segment_sam3.py](segment_sam3.py) uses the text-promptable `facebook/sam3`.

All were trained on **SA-1B** — around 11 million ordinary photographs with 1.1
billion automatically generated masks. Aerial images of forest are contained in it
at best by accident.

That explains the behaviour precisely: **SAM does not know tree crowns.** It finds
regions with a closed, high-contrast boundary. Where a crown has such a boundary,
it works excellently. Where the strongest closed boundary runs around *crown plus
cast shadow*, SAM takes that one — not out of error but because it does exactly
what it was built for. See chapter 4.

### Depth Anything / DepthPro — ground-level perspectives

Both were trained on mixed depth datasets whose common denominator is the human
capture perspective: street scenes, interiors, landscapes. Depth Anything V2 uses
1.5 M labelled plus 62 M unlabelled images — hardly any of them a straight-down
shot from 80 m. Chapter 4 covers the consequences.

### CrownNet — BAMFORESTS, the one geographic match

[crownnet.py](crownnet.py) is the exception in this list. It is trained on
**BAMFORESTS**: 58,228 annotated crowns from German forest, natively at
1.70 cm/px, brought to the scale of our own frames in the code via `--scale`
([crownnet.py:281](crownnet.py#L281)).

That is the only dataset in the pipeline that matches our own imagery
geographically and in species composition — and the only place with a hard
instance accuracy measured against real labels
([crownnet.py:412](crownnet.py#L412)). For judging the segmentation as a whole it
is the most valuable reference point the project has.

### The pattern

```
Your drone frames        ──> height head        (21 frames)   ← the only own data
Québec, Canada           ──> species classification
NEON, USA                ──> detection
SA-1B, everyday images   ──> segmentation
Ground perspectives      ──> depth
German forest            ──> CrownNet           (stage 2, not yet in the main pipeline)
```

Every stage except the height head works on a domain it was not trained for —
another continent, another scale, another perspective, another species
composition.

The domain gap is therefore not *one* problem among several but the pervasive
pattern. And it explains why the effort around parallax can pay off: it is the
only building block of the pipeline that **has no training domain at all**. SIFT,
RANSAC and optical flow work in Québec just as they do in Brandenburg, at
1.9 cm/px just as at 6.25 cm/px, at nadir just as obliquely. Geometry knows no
domain gap.

That is the real reason it is suitable as a teacher — and why a head distilled
from it yields a model in *this* domain instead of one borrowed from a foreign
one.

---

## 11. Failure modes and where the pipeline stands

### Classical vs. learned — the current state

Of 21 scripts, 11 use `torch` directly. Network-free are **exactly the parallax
scripts** ([stereo_probe.py](stereo_probe.py),
[build_parallax.py](build_parallax.py)) plus the pure visualisation and evaluation
tools.

The architecture of the project can be summarised like this:

> **Classical geometry produces truth, neural networks make it scalable and
> semantic.**

```
SIFT / RANSAC / optical flow   ──> height (measured, needs video)   [no network]
                                          │  distillation
                                          ▼
DINOv3 + convolutional head    ──> height (estimated, one image)    [network]
                                          │
SAM / CrownNet / DeepForest    ──> crown instances                  [network]
                                          │
DINOvTree                      ──> tree species                     [network]
```

Only the top row is network-free — but it supplies everything below it with a
training signal. Its quality is the ceiling for the rest.

### Known failure modes

**Shadows captured as part of the crown.** Covered in detail in chapter 4.
Demonstrably affects `urban` (depth source). Prediction: with parallax the error
disappears, because a shadow has zero parallax. **Not yet verified.**

**False separations in the closed canopy.** Two colours on a visually continuous
crown. Diagnosable via the `trennungen` view in
[visualize_crowns.py](visualize_crowns.py), which colours every boundary by the
depth of the saddle: red = flat saddle, the two probably belong together.

**No vegetation filter.** A search across
[segment_hybrid.py](segment_hybrid.py), [segment_sam.py](segment_sam.py),
[segment_sam3.py](segment_sam3.py) and [refine_crowns.py](refine_crowns.py) finds
nothing about greenness, HSV or excess green. Filtering happens only by shape and
height. A brightness/green filter would be a cheap patch against the shadow
problem — but only a patch.

**Scale dependence.** The classification checkpoint saw 1.9 cm/px, our frames have
6.25 cm/px. See [scale_sweep.py](scale_sweep.py).

### What is to be clarified next

1. **Compare parallax against depth on the same segmentation.** There are 7
   folders with parallax data. `visualize_crowns.py --surface parallax` against
   `--surface depth` on `pines` and `dense` would be the direct comparison. The
   shadow argument from chapter 6 would then be testable rather than merely
   plausible.

2. **Quantify the quality of the teacher.** How often does the homography fail,
   how many partner pairs survive the `--min-displacement` filter, how high is the
   median residual flow? These values are already computed in
   [stereo_probe.py:77](stereo_probe.py#L77) but not evaluated in aggregate.

3. **Check what plane RANSAC settles on.** In dense stands it could be the canopy
   plane instead of the ground. Visualise the inlier distribution.

4. **Broaden the data basis.** 29 frames for the distillation are very few. More
   video folders would help most here — more than any architectural improvement.

5. **Compare against Tolan et al. as a baseline.** Their weights are openly
   available and run on our frames. That would give an honest reference instead of
   only Depth Anything.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Baseline** | Distance between two camera positions. Larger = stronger parallax. |
| **Backbone** | The large, pretrained part of a network that supplies general image features. |
| **CHM** | Canopy height model — raster map of vegetation height above ground. |
| **Distillation** | Transferring knowledge from a teacher (model or method) into another model. |
| **Disparity** | Positional difference of the same point between two views, in pixels. |
| **DSM / DTM** | Digital surface / terrain model — with and without vegetation. Difference = CHM. |
| **Detrend** | Subtract the large-scale trend to expose local structure. |
| **GSD** | Ground sample distance — how many metres a pixel covers on the ground. |
| **Homography** | 3×3 matrix describing the mapping of a *plane* between two views. |
| **Inlier** | A data point that fits a model. Opposite: outlier. |
| **LiDAR** | Laser scanner, measures distances directly. Gold standard for a CHM. |
| **monocular** | From a single image. Opposite: stereo / multi-view. |
| **Nadir** | Viewing direction straight down. |
| **Optical flow** | Vector field: where did each pixel move between two images? |
| **Parallax** | Apparent displacement of objects when the observer changes position. |
| **Patch token** | Feature vector of a vision transformer for one 16×16 image tile. |
| **Prominence** | How far a peak rises above the surrounding cols. |
| **RANSAC** | Robust model estimation by random trials and counting inliers. |
| **Rectification** | Warping two images so that correspondences lie on the same row. |
| **SIFT** | Method for finding and describing salient image points. |
| **scale-invariant** | Insensitive to multiplication by a factor. |
| **Vision transformer (ViT)** | Network architecture that processes an image as a sequence of tiles. |
| **Watershed** | Segmentation by the watershed principle, starting from markers. |

---

## 13. References

### On the geometry (chapters 5–7)

- **Plane + parallax, originally:** Irani, Anandan, Kumar et al., 1990s.
  Formalised in Triggs, *Plane + Parallax, Tensors and Factorization*, ECCV 2000 —
  https://lear.inrialpes.fr/people/triggs/pubs/Triggs-eccv00.pdf
- **Modern application, road plane:** *Monocular Road Planar Parallax Estimation*
  — https://arxiv.org/html/2111.11089
  (Homography onto the road, residual flow = height above it. Identical setup,
  only our plane is the forest floor.)
- **Metrically accurate variant:** *DepthP+P* — https://arxiv.org/pdf/2301.02092
- **Satellite imagery:** *Parallax estimation for push-frame satellite imagery* —
  https://arxiv.org/pdf/2102.02301
  (Contains explicitly the proportionality to height *and* baseline — the
  justification for our baseline normalisation.)
- **SIFT:** Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*,
  IJCV 2004.
- **RANSAC:** Fischler & Bolles, *Random Sample Consensus*, CACM 1981.

### On distillation (chapter 8)

- **MegaDepth:** Li & Snavely, *Learning Single-View Depth Prediction from
  Internet Photos*, CVPR 2018 — https://arxiv.org/abs/1804.00607
  *The structurally closest relative:* SfM/MVS as teacher, single-image network as
  student, scale-invariant loss because of the unknown scale.
- **MiDaS:** Ranftl et al., *Towards Robust Monocular Depth Estimation*, TPAMI
  2022 — https://arxiv.org/pdf/2307.14460
  (Scale- and shift-invariant loss as the standard.)
- **Depth Anything:** Yang et al., CVPR 2024 — https://arxiv.org/pdf/2401.10891
  (The model from `depth_probe.py` — itself built by pseudo-label distillation.)
- **Alternative, not chosen:** Zhou et al., *Unsupervised Learning of Depth and
  Ego-Motion from Video*, CVPR 2017 — https://arxiv.org/abs/1704.07813
  (End-to-end from video, without an explicit parallax intermediate step.
  Disadvantage for us: no inspectable intermediate stage.)
- **Scale-invariant loss, originally:** Eigen, Puhrsch, Fergus, *Depth Map
  Prediction from a Single Image using a Multi-Scale Deep Network*, NIPS 2014.

### On the forestry application (chapters 1, 9, 10)

- **Tolan et al., 2024:** *Very high resolution canopy height maps from RGB imagery
  using self-supervised vision transformer and convolutional decoder trained on
  aerial lidar*, Remote Sensing of Environment 300 —
  https://arxiv.org/abs/2304.07213
  **The most important point of comparison.** Almost identical architecture to
  `distill_height.py` (DINOv2 frozen + convolutional decoder), but LiDAR as the
  teacher. 2.8 m MAE. Code and weights open:
  https://github.com/facebookresearch/HighResCanopyHeight
- **UAV photogrammetry as teacher:** *Ultrahigh-resolution boreal forest canopy
  mapping*, RSE 2022 —
  https://www.sciencedirect.com/science/article/pii/S0303243422000125
- **Crown delineation, limits of the method:** *Individual tree crown delineation
  from high-resolution UAV images in broadleaf forest* —
  https://www.sciencedirect.com/science/article/abs/pii/S1574954120301576
  (Important for expectations: works well in conifer stands, markedly worse in
  broadleaf and mixed stands. Our `mixed` folders will be the hard part,
  independently of the height source.)
- **Benchmark:** *Open-Canopy* — https://arxiv.org/pdf/2407.09392

---

*Status: 20 August 2026*
