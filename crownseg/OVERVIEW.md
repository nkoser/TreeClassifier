# Instance segmentation — the whole thing on one page

Short form of what `REPORT.md` covers in detail. Chronological, with the reason
for each next step.

## The starting point

Goal: capture every tree individually, so that stage 2 gets one crop per crown.
Problem: our own drone frames have **no annotations**. Without truth a
segmentation can only be looked at, not judged — and that is exactly where the
earlier work had got stuck. Back then it was rated by crown count, area
coverage and the share of "suspicious separations"; the last one is defined via
saddle prominence in the *estimated* depth, i.e. via the same depth that
produced the separation. A quantity that grades itself.

Hence, first, an external dataset with real crown polygons: **BAMFORESTS**,
2456 tiles, 92,445 hand-digitised crowns, 1.70 cm/px. The split is what matters:
**Hain (test1) occurs in neither training nor validation** and is the only real
transfer test. test2 only measures how well already-seen areas fit.

## Phase 1 — the measuring rig before the experiments

So that not every variant brings its own definition of success:

- **F1 at IoU 0.5** — share of crowns hit at the threshold actually used
- **mean IoU of the hits** — separates "finds little but cleanly" from "finds a lot but imprecisely"
- **AP50** — threshold-independent, only for methods with a per-instance confidence
- **separately per area**, never averaged

All methods write label maps in the same format (`eval_labels.py`), so older runs
can be ranked afterwards as well. The window logic lives in a shared module
(`tiling.py`) — a comparison in which the methods tile differently measures the
tiling as well.

## Phase 2 — wiring up the existing tools

What they all have in common: **no crown annotation as a training target**.
Either pre-trained general-purpose models plus hand-built rules, or — in the
case of crownnet — training on area maps instead of on instances.

| Method | F1 Hain | Note |
|---|---:|---|
| crownnet (interior/edge/centre + watershed) | 0.063 | starting point, trained on area maps |
| depth + SAM 1, promptable | 0.227 | |
| hybrid SAM 1 + watershed | 0.246 | |
| SAM 3, text prompt `tree` | 0.281 → **0.312** | the size prior had been set wrong |
| + Depth Pro for splitting | 0.342 | Depth-Anything only 0.315 |
| + seeding peaks as prompts | 0.348 | the gain comes almost entirely from splitting |

**Conclusion of this phase:** any combination of the existing tools stalls
between 0.23 and 0.35. Two dead ends were cleanly ruled out along the way:
searching for crowns in the area SAM discarded hits **not once** in 291 cases
(meaning: working against the decision of a model that delineates well, at IoU
0.78), and peak prompts yield 12 new crowns instead of the hoped-for 170.

## Phase 3 — training on real annotations

| Architecture | test2 Stadtwald | test2 Tretzendorf | **test1 Hain** | IoU | AP50 |
|---|---:|---:|---:|---:|---:|
| **EoMT (DINOv3)** | 0.688 | 0.698 | **0.624** | 0.773 | 0.507 |
| EoMT, short schedule | 0.721 | 0.687 | 0.554 | 0.773 | 0.406 |
| Mask2Former (Swin) | 0.565 | 0.562 | 0.454 | 0.708 | 0.372 |
| Mask R-CNN | 0.629 | 0.544 | 0.367 | 0.706 | 0.241 |

Mask R-CNN started at 0.144 with 2628 predictions for 780 crowns. The cause was
measurable: Hain has 21.5 crowns per tile against 47.5 in the Stadtwald, median
diameter 392 px against 281 px — the trees are larger, and the model broke them
apart. Three corrections brought 0.367: anchor ladder 32–512 → 64–1024 px (an
842 px crown could not even be proposed by the RPN before), scale jitter upwards
as well, model selection by instance F1 instead of validation loss.

The query-based architectures do not have the anchor problem in the first place:
fixed queries, each with its own mask and its own score, no anchors, no NMS.

**What is remarkable is not the ranking but the drop on the unseen area:** EoMT
falls from 0.72 to 0.62, Mask R-CNN from 0.59 to 0.37. EoMT transfers better,
not just scores better in absolute terms.

Spread between two runs of the same configuration: 0.015 to 0.035. Differences
of that magnitude are not interpretable; the gap to Mask2Former (0.100) is above
it.

## Phase 4 — errors in the measuring itself

Four of them changed numbers, not just code:

1. **The window discarded large crowns.** "Discard everything touching a window
   edge" kills every crown wider than the overlap — in *every* window. Replaced
   by assignment via the centroid, overlap 768 px.
2. **Tile selection took the first N alphabetically.** Tretzendorf thereby fell
   out of test2; the early test2 numbers were pure Stadtwald.
3. **Model selection by validation loss** picked epoch 2 of 25 — in detection the
   loss rises while accuracy is still improving.
4. **AP averaged per tile** overestimates: short curves reach 1.0 easily. Pooled,
   test1 is 0.406 instead of 0.458.

## Phase 5 — our own hypotheses, refuted

On our own frames the crowns look too coarse. Tested and **all rejected**:
inference scale, multi-scale levels, confidence threshold, a finer dataset
(Quebec), variable field of view during training. The predicted crown diameter
stays at 2.3–2.45 m, even at scale ×1.2 versus ×2.8. Whether that is *wrong*
cannot be said without an independent measurement of the real crown sizes.

Also closed: **height does not help segmentation.** Four fusion routes (depth as
a fourth channel, a separate branch with a gate, depth only, depth for splitting)
plus Ruschhaupt et al. with a real photogrammetric CHM arrive independently at
the same result.

## Phase 6 — dense DINOv3 features

Instead of cutting out a crown → one vector: one vector per 16×16 patch,
clustered directly.

- **Species:** NMI 0.43–0.49, purity up to **79 %** at 20 clusters, without
  segmentation and without labels. Controls: colour alone 0.15, position alone
  0.06 — both rule out the obvious spurious explanations.
- **Individual crowns:** F1 0.045–0.142 against 0.624. It separates species and
  stands, not neighbouring trees of the same species.

**So it does not replace segmentation**, but it could carry the species stage.

## Status

Segmentation: **EoMT, F1 0.624** on ground never seen, mean IoU 0.773. For
comparison, the baseline of the BAMFORESTS authors: AP50 69.05/68.89 against our
62/56 — on different test subsets, so not directly comparable, but we are below
it.
