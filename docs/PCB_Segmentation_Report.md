# Synthetic PCB Material Segmentation — Technical Report

*A synthetic dataset and U-Net model that classify copper, solder, substrate, and silkscreen from 6-channel photometric imagery — what it does, and an honest account of what it doesn't.*

**Repository:** PCBDataset · **Report date:** 2026-07-28 · **Production checkpoint:** `unet_best.pt` · **Status:** active development

### At a glance

| | | | |
|---|---|---|---|
| **Board designs** 31 | **Material classes** 8 | **Input channels** 6 | **Model params** 7.76M |
| **Mean IoU** 0.966 | **Pixel accuracy** 0.996 | **Weakest class** solder | **Real-data validation** none yet |

## Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Timeline of improvements](#3-timeline-of-improvements)
4. [Model & training](#4-model--training)
5. [Results](#5-results)
6. [Discussion — threats to validity](#6-discussion--threats-to-validity)
7. [Reproducing this work](#7-reproducing-this-work)
8. [Recommended next steps](#8-recommended-next-steps)

---

## 1. Introduction

The long-term goal is a physical inspection rig — a five-band (R/G/B/IR/UV) LED ring light — paired with a model that segments a captured PCB into its materials: copper, solder, fiberglass, soldermask, silkscreen, components, and copper routed under intact soldermask ("trace"). Labelling a real dataset for this is slow and expensive, so this repo instead **synthesizes** one: real open-hardware KiCad designs become physically-plausible multi-band renders with pixel-perfect ground truth, and a U-Net learns to reproduce that segmentation.

---

## 2. Dataset

### 2.1 Generation pipeline

Every board runs the same pipeline, from a cloned KiCad repo to a verified sample:

`KiCad Gerbers` → `STL/component precheck` → `gerber2blend mesh` → `mesh + UV export` → `STL finalize` → `Mitsuba3 render` → `alignment verify`

Two geometry sources merge for every board: `gerber2blend` turns the Gerber layers into a flat textured mesh with no concept of discrete parts, while KiCad's STEP exporter produces real 3D component models where footprints have them assigned. The two align via a board-body-shell anchor match (≈0.1mm agreement) and render together through [Mitsuba3](https://mitsuba.readthedocs.io), chosen because Blender/Cycles has no physical model of non-visible light — an "IR render" there is a colour swap, not a simulation.

### 2.2 Board sources

All 31 boards are real, permissively-licensed open-hardware designs. Hit rate — footprints resolving to real, populated 3D models — varied sharply by source:

| Source org | Category | Tried | Survived |
|---|---|---:|---:|
| Antmicro | Generic dev/carrier boards | 40 | 10 |
| SparkFun | Sensor breakouts | 13 | 1 |
| Great Scott Gadgets | RF/USB test equipment | 6 | 3 |
| Libre Solar | Solar charge controllers / BMS | 15 | 14 |
| ProtoCentral | Sensor breakouts | 4 | 3 |

Test-equipment and power-electronics designs assign real 3D models far more often than generic dev boards, likely because their footprints pass through a real BOM/procurement process. This bounds diversity structurally: patch sampling multiplies training *examples*, not board *designs* — generalisation is limited by these 31 shapes, not by patch count.

### 2.3 Input channels

Each sample stacks 6 channels — R, G, B, IR, UV, and height — rendered through the identical mesh/camera/UV, so all channels are pixel-aligned by construction. Per-material reflectance is one flat approximate value per class, standing in for what a monochrome sensor under one coloured LED would read.

- **Sourced**: soldermask goes ~80% transparent past ~750nm, so IR sees through it to the copper underneath — cited to two USPTO NIR-inspection patents.
- **Unverified placeholders**: everything else — chosen so classes look different, not because they were measured.

Height is a 16-bit micrometre-precision PNG, rendered through the same camera as the other channels, so it lines up with them pixel-for-pixel.

### 2.4 Ground-truth classes

Ground truth is an 8-class mask built from gerber2blend's own layer rasters, rendered as a self-lit pass through the identical geometry — guaranteeing exact pixel alignment with the band images by construction, not by a second registration step.

| ID | Class |
|---|---|
| 0 | background |
| 1 | copper |
| 2 | solder |
| 3 | fiberglass |
| 4 | soldermask |
| 5 | silkscreen |
| 6 | component |
| 7 | trace |

`copper` is exposed metal, visible in a photo. `trace` is copper routed *under* intact soldermask — real routing, invisible in visible light, distinguished only via IR and ground truth. `component` appears only on boards whose STL came back populated — a board without resolvable 3D models has no component class, never a mislabel.

### 2.5 Known dataset limitations

- **Front face only.** At least one otherwise-good board is excluded because its silkscreen is entirely on the back.
- **Solder occlusion is physically real.** A mounted component hides 95–98% of its own nominal solder joint from a top-down camera — a structural ceiling on visible solder, independent of any rendering issue.
- **No solder-joint geometry** (fillets, menisci, barrel fill) is modelled — see §3, synthesis was attempted and not adopted.
- **Board diversity is registry-bounded** — 31 layouts, not an open-ended distribution (§2.2).
- **No real-world validation exists anywhere in this pipeline** — the single biggest open question (§6).

---

## 3. Timeline of improvements

Chronological record of what was tried, in order, with the metric movement each produced. Numbers before 2026-07-27 are on a noisier ground-truth render and **not directly comparable** to later rows — see the GT-fidelity entry below.

**2026-07-20 — Original baseline** *(10 boards)*
Mean IoU 0.927, solder 0.728.

**2026-07-21 – 22 — Board-level split fix, loss tuning** ✅ *win*
Fixed a random-patch train/val split that leaked board layout into validation; added Dice loss and class-weighted CE. Registry expanded 10→25 boards.

**2026-07-23 — Class-weight cap (1.5×)** ✅ *win*
Uncapped inverse-frequency weighting let rare `solder` dominate the gradient near `silkscreen`. Capping both fixed a boundary problem in `trace` and an imbalance in `silkscreen`. Mean IoU 0.932, solder 0.752.

**2026-07-24 — Solder-fillet synthesis, attempt 1** ⚠️ *inconclusive*
Procedural lead-toe solder meniscus. Result within noise (0.932→0.930, solder +0.003); left open pending the GT-fidelity fix below.

**2026-07-22 — Alternate solder GT source** ❌ *reverted*
Swapped the undersized paste-stencil solder label for gerber2blend's grown solder-dome footprint. Its fixed-pixel dilation bridged adjacent pads on fine-pitch boards; solder and mean IoU both regressed. Reverted.

**2026-07-27 — 3D-model recovery, two rounds** — *flat on held-out*
Registry expanded 25→31 boards. Round 1 (exact-match .wrl/.step + SparkFun path shim) fixed 16/31 boards. Round 2 (upstream-repo exact + pin-count-guarded fuzzy match) recovered 25 more references, closing 53→28 unresolved. Real geometry recovered on over half the registry, but headline mIoU held flat (~0.930) because neither held-out validation board needed the fix.

**2026-07-27 – 28 — GT-rendering fidelity fix** ✅ *largest win*
Ground truth was rendered at 4 samples/px vs. 32 for band images, adding real anti-aliasing noise at class boundaries — 57–65% of "model error" was this noise, not a real failure. Re-scoring the same weights at 32 samples alone: 0.930→0.955. Retraining: 0.955→**0.964**, solder 0.749→**0.873**.

**2026-07-28 — Solder-fillet synthesis, attempt 2** ❌ *closed, negative*
Re-ran the identical, unmodified fillet code against the now-clean GT. Unambiguous this time: solder 0.873→0.863, component 0.991→0.987. Geometry and appearance change are both real, but too thin a pixel band to help. Reverted; closed as a dead end.

**2026-07-28 — Solder-oversample retest** ✅ *promoted, mixed signal*
`--solder-oversample-prob 0.3` on top of the production config, first time combined with the class-weight cap and clean GT. First attempt was killed mid-run; relaunched clean. Result: mean IoU 0.9642→0.9662, solder 0.873→0.876 (still under this project's ~0.01 noise floor, so not a confirmed solder-specific fix), `trace` and `fiberglass` moved further than solder did, no class regressed beyond noise. Promoted on no-downside grounds, not a proven solder win.

---

## 4. Model & training

### 4.1 Architecture

A U-Net: four encoder stages shrink the image while learning "what," a bottleneck, then four decoder stages upsample back to full resolution to mark "where," with skip connections carrying fine detail from each encoder stage across to its matching decoder stage so it isn't lost to downsampling. Each stage is a double 3×3-convolution block (conv → batch norm → ReLU, twice), 32 base channels doubling per level. Built from scratch — **7.76M parameters**, no pretrained backbone — because pretrained encoders are trained on ordinary 3-channel photos, and this input is 6 channels of physical measurements (reflectance and height) with no photographic equivalent to transfer from.

### 4.2 Loss and class balancing

The training signal has two parts: cross-entropy, up-weighted per class by inverse square-root frequency so rare classes like `solder` aren't ignored just because they cover few pixels; and soft Dice loss (weight 0.5, background excluded), which rewards getting each class's overall region shape right rather than just per-pixel correctness. The per-class weights are capped at 1.5× the mean weight — an earlier, uncapped version let the rarest class's weight dominate the gradient near a spatially adjacent class and hurt it.

### 4.3 Split and augmentation

The train/val split is by **board**, never by patch: two boards (`icicle-hdmi-breakout`, `signal-integrity-test-board`) are held out entirely and used only for scoring, so the model is always tested on layouts it has never seen — a patch-level split would leak a board's specific geometry into validation and inflate the score. Augmentation is geometric only — flip and 90° rotation — deliberately excluding colour jitter, since channel values here are physical measurements, not photographic style, so jitter would corrupt the signal rather than regularise it. Training draws 2000 random 256×256 patches per epoch, batch size 8, Adam optimizer, 40 epochs.

### 4.4 Evaluation protocol

Scoring runs full-resolution sliding-window inference (256px tiles, 32px overlap, logits averaged across overlapping tiles before the final class decision) rather than scoring training patches directly, since patch-level stats read a few points higher and aren't a fair comparison. Metrics: per-class IoU (overlap between predicted and actual pixels for that class), mean IoU averaged evenly across all 8 classes regardless of how common each one is, and overall pixel accuracy.

---

## 5. Results

### 5.1 Current production result

Held-out evaluation, both boards pooled:

| Class | IoU |
|---|---:|
| background | 0.9968 |
| copper | 0.9870 |
| solder | 0.8763 |
| fiberglass | 0.9763 |
| soldermask | 0.9943 |
| silkscreen | 0.9677 |
| component | 0.9910 |
| trace | 0.9404 |
| **Mean IoU** | **0.9662** |
| **Pixel accuracy** | **0.9957** |

### 5.2 Where it succeeds

`background`, `soldermask`, `component`, and `copper` all exceed 0.98 IoU — the easy majority of a board's area is reliably distinguished, and the remaining error concentrates where expected (thin classes and boundaries) rather than diffusing everywhere.

### 5.3 Where it struggles

**Solder — 0.873, the clear laggard.** Bounded mainly by physical occlusion: mounted components hide 95–98% of their own solder joint from a top-down camera (§2.5), leaving a thin, boundary-sensitive sliver as the entire visible class. This is believed close to the real occlusion-bounded ceiling for this rendering approach.

**Trace and silkscreen — boundary precision.** `trace` (0.932) is one-to-few pixels wide; its errors sit at the copper/soldermask boundary, not a spectral confusion. Both classes improved substantially from the §4.2 loss fix; residual error is boundary precision at this pixel scale, the same sensitivity solder shows.

---

## 6. Discussion — threats to validity

The single most important caveat: **nothing in this pipeline has been checked against a real, physical PCB.** Every metric measures the model reproducing this project's own synthetic compositing rules — a closed loop between renderer and scorer. Two candidate real-data sources were investigated and rejected: a hyperspectral PCB dataset has component-type labels only, not material layers, and would need a multi-terabyte partial extraction; a high-resolution PCB photo dataset has the same gap. No dataset was found publishing ground truth for this project's 8-class taxonomy.

Secondary caveats: component and IR/UV reflectance are largely unsourced placeholders (§2.3). Board diversity is bounded to 31 registry-selected layouts with a known selection effect by source organisation. Front-face-only imaging excludes at least one otherwise-usable board.

---

## 7. Reproducing this work

| | |
|---|---|
| `09_batch.py` | Regenerate the full dataset and write `out/manifest.json`, the source of truth for a batch run. |
| `train/train.py` | Train the U-Net; production run: `--val-boards icicle-hdmi-breakout,signal-integrity-test-board --dice-weight 0.5 --max-class-weight 1.5`. |
| `train/evaluate.py` | Full-resolution sliding-window evaluation — the numbers in §5. |
| `train/error_heatmap.py` | Per-pixel error maps and density heatmaps. |
| `docs/STATUS.md` | The living decision-history record; this report is a snapshot, that file is current. |

---

## 8. Recommended next steps

1. **Real-data validation.** The highest-value open item — even a small, hand-labelled real-PCB set would give the first genuine sim-to-real number.
2. **Remaining 3D-model references.** 28 references across a handful of boards still unresolved (§3).
3. **Back-face imaging.** Would recover at least one currently-excluded, otherwise-good board.
4. **Further board-diversity search**, targeted at test-equipment/power-electronics sources over generic dev-board houses (§2.2).

---

*Synthetic PCB Material Segmentation — Technical Report · Generated 2026-07-28*
