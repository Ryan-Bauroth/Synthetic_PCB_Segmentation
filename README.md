# PCB Synthetic Material Dataset

Generates a synthetic dataset of PCB images with pixel-level ground-truth material labels.

## Overview

This pipeline takes real KiCad PCB designs and renders them as 6-channel images:
- **5 spectral bands**: R, G, B, IR, UV
- **1 geometric channel**: height map (mm above bench)
- **Ground truth**: 8-class material mask (copper, solder, substrate, soldermask, silkscreen, trace, component, background)

The output is used to train a U-Net semantic segmentation model.

## Setup

```bash
bash 00_install.sh
./.venv311/bin/python 01_verify_env.py
```

## Generate Dataset

```bash
./.venv311/bin/python 09_batch.py                              # all boards
./.venv311/bin/python 09_batch.py --boards scalenode           # single board
./.venv311/bin/python 09_batch.py --workers 8 --samples 16     # faster, lower quality
```

Outputs go to `out/<board>/`:
- `<board>_{R,G,B,IR,UV}.png` — spectral band images
- `<board>_height.png` — height map (16-bit, micrometers)
- `<board>_gt.png` — ground-truth class mask
- `<board>.stl` — populated board mesh

## Train U-Net

```bash
./.venv311/bin/python train/train.py
./.venv311/bin/python train/evaluate.py --checkpoint train/checkpoints/unet_best.pt
./.venv311/bin/python train/error_heatmap.py --checkpoint train/checkpoints/unet_best.pt
```

## File Reference

| File | Purpose |
|------|---------|
| `labels.py` | Class definitions and palette |
| `rig.py` | LED ring light geometry |
| `spectral.py` | Material reflectance model |
| `02_fetch_gerbers.py` | Download KiCad boards |
| `03_run_gerber2blend.py` | Convert Gerbers to mesh |
| `07_render_mitsuba.py` | Render bands + ground truth |
| `09_batch.py` | Run full pipeline |
| `train/` | U-Net training code |

## Material Classes

| ID | Class |
|----|-------|
| 0 | background |
| 1 | copper (exposed) |
| 2 | solder |
| 3 | fiberglass |
| 4 | soldermask |
| 5 | silkscreen |
| 6 | component |
| 7 | trace (copper under mask) |
