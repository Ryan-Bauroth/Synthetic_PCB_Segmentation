# PCB Synthetic Material Dataset

> **Picking this up in a new session?** See `docs/STATUS.md` for a full
> narrative handoff — current state, decision history, known bugs already
> fixed, and recommended next steps. This README covers *how it works*;
> `docs/STATUS.md` covers *where things stand right now*.

> **Fresh clone?** This repo ships **code only** — no generated dataset,
> no trained checkpoint. `boards/`, `out/`, `.cache_repos/`, and
> `train/checkpoints/` are all gitignored because they're fully
> reproducible from the pipeline below, but that means you must run
> **Setup → Generate the dataset → Training a U-Net** yourself before
> anything in `train/` will work. Dataset generation clones ~30 real KiCad
> repos and renders every band per board (see "Heavy downloads" below); a
> full run and a full training run each take real time — budget for both.

Generates a synthetic PCB dataset per board: **5 single-channel images — R,
G, B, IR, UV** — each an evenly-lit reference capture (all 16 ring-light
positions fired together, shadows minimized), a **height map** (mm above
the bench, from the same geometry), plus a **per-pixel ground-truth
material mask** (copper, solder, fiberglass/FR-4, soldermask, silkscreen,
trace, and — where available — **populated 3D components**) and a
**populated STL** (board + placed 3D component models, where the source
board's KiCad footprints have them). The 6 images (5 bands + height) are
also the intended input to a U-Net semantic-segmentation model trained
against the GT — see `train/`.

## How it works
Real board → `kicad-cli` exports Gerbers (`02_fetch_gerbers.py`) →
**STL/component precheck** (`02b_stl_precheck.py`, straight from the cached
`.kicad_pcb`, no Blender): is this board POPULATED (real 3D component
models resolvable)? STL/component data is required for this project — the
height channel and the `component`/`trace` GT classes all depend on it — so
a board that comes back bare is **dropped here**, before the expensive
steps below ever run for it (see "Gating on STL/component viability"). Only
surviving boards continue: `gerber2blend` builds a Blender model and
rasterises each PCB layer to a PNG (`fab/PNG/F_Cu.png`, `F_Mask.png`,
`F_SilkS.png`, `F_Paste.png`, …) → the board mesh + UV are exported once
(`05_export_mesh.py`, the last bpy-dependent step) → `06_export_stl.py`
finalizes the STL (reusing the precheck's cached STEP/STL — no
re-tessellating the full model) and exports a components-only mesh, aligned
into gerber2blend's coordinate space, for rendering. **Mitsuba3** then
renders all 5 bands + the ground truth + a height map through the board
mesh/camera/UV (plus the aligned components), so every image is
pixel-aligned by construction.

### Gating on STL/component viability
`02b_stl_precheck.py` runs `kicad-cli pcb export step --subst-models`
directly from the cached `.kicad_pcb` and applies the same populated
heuristic `06_export_stl.py` always used (Z-extent ≥ 3% of board width) —
the only difference is WHEN it runs: right after Gerber export, before
`03_run_gerber2blend.py`'s Blender conversion or any Mitsuba imaging. A bare
board is dropped from the batch entirely (not imaged with a flat height
map and no components — that would be usable-looking but uninformative
training data). This also caches the board-body anchor shell's world
bounds into `stl_report.json`, so `06_export_stl.py`'s later finalize pass
(once `board_bounds.json` exists) can align the components-only mesh
without re-parsing the full populated STEP scene a second time — real
savings for many-shell boards (OpenCascade tessellation of a 720-shell
board took ~3 minutes in one observed case).

### Populated components in the renders
Two genuinely separate geometry pipelines both describe the same physical
board: gerber2blend's mesh (Gerber-derived, flat, richly textured — no
concept of 3D parts, since Gerbers don't carry that data) and KiCad's STEP
export (has real 3D component models, no fine copper/silkscreen texture).
`06_export_stl.py` merges them for rendering by exporting a **second,
components-only STEP** (`--no-board-body --subst-models`) and aligning it
into gerber2blend's coordinate space. KiCad's STEP export uses its own
absolute origin, unrelated to gerber2blend's board-centred convention, so
alignment isn't automatic — the anchor is the STEP assembly's own board-body
shell (found by name, the `_PCB` suffix KiCad's exporter uses, falling back
to the largest/flattest shell), matched against `board_bounds.json`'s
known-good center. Verified empirically: the board-body shell's own
independently-measured size matches `board_bounds.json`'s Gerber-derived
size to within ~0.1mm, confirming it's the same physical board and giving a
precise, board-anchored offset. Component material is a single flat
placeholder reflectance per band (`spectral.py`'s `"component"` entries) —
real components mix black plastic, metal shielding, ceramic, plated pins;
this is coarse, not materially-informed, but distinguishes "there's a part
here" from bare board. GT gets a 7th class, `component` (id 6), for the
same footprint — only added to the palette-matching candidates when
components are actually in that board's scene (an early version let
anti-aliased edge pixels between two *other* classes spuriously match the
new palette entry on boards with zero component geometry; fixed by scoping
the match set per-render).

### Why Mitsuba, and why 5 bands are "approximate"
Blender/Cycles has no physical model of non-visible light — confirmed via
Blender's own dev-community spectral-rendering thread; any Blender "IR/UV
render" is a manual per-material color swap, not real spectral simulation.
Mitsuba3 *can* do genuine per-wavelength rendering (via `spec_film` + custom
`irregular` spectra, bypassing Mitsuba's own 360–830nm CIE clamp — confirmed
feasible, with prior art in Eradiate). This project ships the simpler,
faster path first: **one consistent renderer (Mitsuba, ordinary RGB
variant) for all 5 bands**, with hand-authored per-material reflectance
values (`spectral.py`) rather than the full spectral machinery — since the
actual reflectance *data* feeding either approach is thin regardless (one
well-sourced NIR fact — copper stays reflective past ~850nm, soldermask
goes ~80% transparent past ~750nm, revealing copper beneath, from a cited
NIR-inspection patent — and no solid UV numbers). **R/G/B are a
simplification too**: flat approximate class colors, not gerber2blend's
richer baked PBR shading (no specular highlights/gloss nuance). See
`spectral.py`'s module docstring for exactly what's sourced vs. placeholder.
The true `spec_film` spectral path is a documented future upgrade if/when
real measured PCB spectra become available.

### Ground truth
gerber2blend does **not** expose copper/soldermask/silkscreen as separate
materials — it bakes them into surface textures. Ground truth is built by
compositing the per-layer PNGs (`spectral.build_class_id_map`) into class
ids, then rendered through the *same* mesh/UV as an emissive (self-lit,
non-reflective) Mitsuba pass — the equivalent of Blender's old
emission-swap trick, guaranteeing pixel alignment with the 5 band images.
GT is invariant to soldermask color (it encodes material *class*, not
appearance) — that's expected, not a bug.

**`copper` vs. `trace`**: `copper` means exposed metal (soldermask open
over `F_Cu`) — what you'd actually see bare. `trace` means copper routed
*under* intact soldermask (soldermask closed over `F_Cu`) — the routing
that's invisible in a normal photo but still real material underneath.
Splitting these lets GT expose full trace geometry as its own class
without pretending soldermask isn't there: the R/G/B/UV band images render
`trace` pixels identically to `soldermask` (no sourced visible-light
transparency claim), so appearance stays physically honest — only GT and
the IR band (which already treats all copper as reflective regardless of
mask, per the sourced ~750nm mask-transparency fact) distinguish covered
copper from bare soldermask.

### Height map
`<board>_height.png` — mm above the bench (`ground_plane()`'s Z, the same
reference the earlier "shadows on the gray surface below the board" design
used), encoded as a **16-bit grayscale PNG in micrometres**
(`round(height_mm * 1000)`, one LSB = 1µm, range 0–65.535mm — comfortably
covers every board's Z-extent seen so far, largest ~19mm). Rendered through
the *same* mesh/camera as the band/GT passes via Mitsuba's `position` AOV,
not a second STL-based rasterization pipeline — same pixel-alignment
guarantee, no extra alignment step. Verified: background/bare-bench pixels
read ~0mm, board-surface classes cluster ~2mm (matches board thickness),
`component` pixels average ~8mm and peak at the STL's real component
height. This is one of the U-Net's 6 input channels (`train/`), not a class
in the material GT — it's a geometric cue, not a material one.

### macOS / Mitsuba gotchas found the hard way
- gerber2blend is Linux-first; three `.pth`-file tweaks in `00_install.sh`
  make it run under the pip `bpy` wheel (preload `bpy` before its submodules
  import `bmesh`; force multiprocessing `fork`, not macOS's default
  `spawn`; install `imagemagick` for its `wand` backend).
- **Mitsuba's `bitmap` texture defaults to sRGB→linear gamma decoding.**
  Our PNGs encode literal reflectance/label values, not photographic
  colors — without `"raw": True` every texture silently darkens (board
  reflectance read ~13x too dark; GT's darker classes like soldermask
  collapsed toward the background class). Always pass `raw=True`.
- **Mitsuba's orthographic `to_world` scale must be ISOTROPIC.** An
  anisotropic scale (different X/Y half-extents) is silently *not* applied
  independently — the film's width:height resolution ratio is what actually
  determines per-axis mm/px. Using an anisotropic scale stretched every
  board vertically almost to the frame edge (near-zero top/bottom margin
  while the left/right margin looked correct) — verified via a control
  render of a known square object, then fixed by scaling both axes by the
  same (width-based) half-extent and letting the film resolution ratio
  handle the aspect.
- Blender's OBJ exporter defaults to a Y-up axis swap — `05_export_mesh.py`
  pins `forward_axis="Y", up_axis="Z"` so the exported mesh stays in the
  same Z-up mm space as `board_bounds.json` and `rig.py`'s light positions.
- Point-light energy needed empirical retuning for Mitsuba's exposure model
  (`--exposure-k`, `--ambient` in `07_render_mitsuba.py`) — unrelated to the
  old bpy pipeline's calibration.
- trimesh's fast vertex-normal path needs `scipy`; without it, every mesh
  shell in a multi-part STEP (e.g. 348 shells) prints a non-fatal but noisy
  fallback traceback. Installed in `00_install.sh`.
- **`trimesh.Scene.geometry` values are in LOCAL mesh space, not world
  space.** Concatenating them directly (an earlier version of this script
  did) ignores the scene graph's per-part placement transforms, collapsing
  every component onto its own local origin instead of its real assembly
  position — this silently produced a WRONG (but plausible-looking) STL for
  every multi-part board: right shell/vertex counts, right-ish Z-extent
  heuristic, completely bogus XY layout (components piled on top of each
  other). Caught by comparing the exported bbox against the known board
  size, not by the shape/count heuristics alone. Fixed with
  `scene.to_geometry()`, which bakes the transforms in first.
- Mitsuba's `area` emitter's `radiance` needs an explicit colour-type dict
  (`{"type": "rgb", "value": [...]}`) — a plain 3-tuple gets interpreted as
  a "vector" (position-like), not a colour, and fails to instantiate.
- **The `aov` integrator's `position` AOV is alpha-premultiplied.** With a
  `box` reconstruction filter, fully-covered pixels give the exact raw
  world-space hit coordinate (verified: a test plane at world Z=3.5 render
  exactly 3.5 at full coverage), but partially-covered edge pixels return
  `value × coverage_fraction` (e.g. 0.875 = 3.5 × 0.25) — would need
  unpremultiplying by the alpha channel in general. Sidestepped here because
  `ground_plane()` already guarantees alpha=1 across the *entire* frame (no
  pixel is ever a true miss), so the raw channel can be read directly.
- **AOV channel indices depend on the film's `pixel_format`.** With
  `pixel_format: "rgb"` (3 base channels, used throughout this project),
  the `position` AOV's xyz lands at channels 3-5, not 4-6 as it would with
  `"rgba"` — verified empirically after an initial off-by-one silently
  read the wrong channel as "height."

## Class legend
| id | class | | id | class |
|----|-------|-|----|-------|
| 0 | background | | 4 | soldermask |
| 1 | copper (exposed) | | 5 | silkscreen |
| 2 | solder | | 6 | component (populated 3D bodies, when available) |
| 3 | fiberglass (FR-4) | | 7 | trace (copper under intact soldermask) |

## Setup (one-time, macOS + Homebrew)
```bash
bash 00_install.sh            # gerbv, inkscape, kicad, python@3.11, bpy + mitsuba venv, gerber2blend
./.venv311/bin/python 01_verify_env.py   # all-PASS check
```
Heavy downloads: Inkscape (~1 GB), KiCad (~2 GB), the `bpy` wheel (~250 MB).
Mitsuba/trimesh/cascadio/scipy/tqdm are small pure-Python wheels.

## Generate the dataset
```bash
./.venv311/bin/python 09_batch.py                     # every registered board, full rig scale
./.venv311/bin/python 09_batch.py --mpp 0.117 --samples 16   # quick low-res validation pass
./.venv311/bin/python 09_batch.py --boards scalenode,cvbs-mipi-bridge --soldermask Black
./.venv311/bin/python 09_batch.py --workers 8         # more concurrent boards per phase
```
Runs in three phases, each processing boards CONCURRENTLY (`--workers`,
default 4; progress bars for all three): **Phase 0** — fetch Gerbers +
STL/component precheck for every registered board; a board with no
resolvable 3D component models is **dropped here**, before the expensive
steps below ever run for it (see "Gating on STL/component viability"
above). **Phase 1** — gerber2blend, gate, mesh-export, STL-finalize, for
surviving boards only. **Phase 2** — render 5 bands + GT + height, verify
alignment, for every board whose core pipeline succeeded. Writes
`out/manifest.json`: per-board soldermask color, STL populated status +
mesh stats, image list, GT classes present, any errors, AND a
`dropped_boards` map (board → reason) for boards Phase 0 rejected — the
single source of truth for what a batch run actually produced.

### Single board (manual / debugging)
```bash
bash run_mvp.sh                          # default board, full rig scale
MPP=0.117 SAMPLES=16 bash run_mvp.sh      # quick low-res validation
bash run_mvp.sh scalenode
# or step by step, using the venv python:
./.venv311/bin/python 02_fetch_gerbers.py     --board scalenode
./.venv311/bin/python 02b_stl_precheck.py     --board scalenode   # GATE: populated?
./.venv311/bin/python 03_run_gerber2blend.py  --board scalenode --soldermask Green
./.venv311/bin/python 04_inspect_blend.py     --board scalenode   # GATE
./.venv311/bin/python 05_export_mesh.py       --board scalenode   # last bpy step
./.venv311/bin/python 06_export_stl.py        --board scalenode   # STL finalize
./.venv311/bin/python 07_render_mitsuba.py    --board scalenode
./.venv311/bin/python 08_verify_alignment.py  --board scalenode
```

## Outputs
Per board, in `out/<board>/` (no serial numbers — one deterministic capture
set per board/soldermask-color combination, named by board):
- `<board>_{R,G,B,IR,UV}.png` — the 5 band images (single-channel)
- `<board>_height.png` — height above the bench, 16-bit PNG in µm (see
  "Height map" above)
- `<board>_gt.png` — the indexed ground-truth mask (pixel value = class id)

And in `boards/<board>/`:
- `<board>.stl` (+ `.step`, `stl_report.json`) — populated board mesh (a
  board with no resolvable 3D component models never gets this far — it's
  dropped at the Phase 0 precheck gate instead of producing a bare STL)
- `fab/mitsuba/components.obj` — the aligned, render-ready components mesh;
  `stl_report.json`'s `render_components_available` reflects whether it
  exists

## Files
| file | role |
|------|------|
| `labels.py` | class ids, palette, layer-name→class rules |
| `rig.py` | 16-position LED ring geometry (elevation/radius/azimuths) |
| `spectral.py` | per-band material reflectance model + composited textures |
| `stl_common.py` | shared helpers for `02b_stl_precheck.py` + `06_export_stl.py` |
| `00_install.sh` | install all tools (idempotent) |
| `01_verify_env.py` | PASS/FAIL env check → `env.json` |
| `02_fetch_gerbers.py` | clone board + `kicad-cli` Gerber export → `boards/<b>/fab/` (also the `BOARDS` registry) |
| `02b_stl_precheck.py` | **gate**: is this board's STL populated? No Blender dependency — runs right after `02_` |
| `03_run_gerber2blend.py` | Gerbers → `.blend` (enables `EFFECTS.SOLDER`, sets soldermask color) |
| `04_inspect_blend.py` | **gate**: dump objects/materials, check per-layer PNGs exist |
| `05_export_mesh.py` | (bpy) export board mesh+UV → OBJ, write `board_bounds.json` |
| `06_export_stl.py` | STL **finalize**: reuses `02b_`'s cached STEP/STL, exports aligned components.obj |
| `07_render_mitsuba.py` | render 5 bands + GT + height via Mitsuba (progress bar) |
| `08_verify_alignment.py` | assert band↔GT↔height alignment + class coverage |
| `09_batch.py` | **generator**: runs 02→08 across boards in three concurrent phases, writes `out/manifest.json` |
| `run_mvp.sh` | run the full pipeline for one board: `run_mvp.sh [board]` |
| `train/` | U-Net semantic segmentation training on the generated dataset (see below) |
| `docs/` | `STATUS.md` (decision history) and `PCB_Segmentation_Report.md` (technical report) |

## Training a U-Net on the dataset
`train/` consumes `out/manifest.json` — only boards with `verify_ok: true`
are used. Input = the 6-channel stack (R, G, B, IR, UV, height); target =
the 8-class GT mask (`labels.CLASSES`). Split is by **board**, not by patch
(patches from the same board would leak layout information into val).

No checkpoint ships with this repo (`train/checkpoints/` is gitignored) —
run `train/train.py` first; it writes `unet_best.pt`/`unet_last.pt` there.

```bash
./.venv311/bin/python train/train.py                       # default split/hyperparams
./.venv311/bin/python train/evaluate.py --checkpoint train/checkpoints/unet_best.pt
./.venv311/bin/python train/error_heatmap.py --checkpoint train/checkpoints/unet_best.pt
```
`error_heatmap.py` visualizes WHERE a checkpoint is wrong on its held-out
boards (not just the aggregate per-class IoU table): `<board>_issue_points.png`
colors every wrong pixel by its true class over a dimmed board image;
`<board>_error_heatmap.png` is a blurred error-density map that shows
clustered problem regions vs. scattered boundary noise.

See `train/dataset.py`'s module docstring for the known limitation this
implies: with 25 board layouts in the registry, patch sampling multiplies
training *examples*, not board *diversity* — generalization to genuinely
new board designs improves with more boards, but is still bounded by how
many distinct layouts exist.

## Not yet implemented
Domain randomization, sim-to-real validation vs. the macro rig, dataset
train/val/test packaging, more boards in the registry (diversity is
currently the training pipeline's main limitation, not the model).
