"""Per-band material reflectance model + composited textures.

Single source of truth for how each material class (see labels.CLASSES)
should appear under each of the 5 capture bands: R, G, B, IR, UV. Shared by
07_render_mitsuba.py (band textures) and implicitly by the ground-truth pass
(build_class_id_map is the same compositing labels.py's GT logic used).

Design (see plan): R/G/B bands are a deliberate simplification vs.
gerber2blend's richer baked PBR shading -- flat approximate reflectance per
material class, reduced to ONE channel (the mono-sensor-under-one-LED-colour
model already used throughout this project), not a full-color render. IR/UV
are the same idea, with values sourced where possible and clearly flagged
as approximate placeholders elsewhere.

Sourcing notes (read before tuning REFLECTANCE):
- IR copper/solder: metals (copper especially) stay strongly reflective well
  past 850nm -- broadband reflector, not just visible-band.
- IR soldermask-over-copper: solder resist becomes ~80%+ transparent past
  ~750nm, so a COVERED copper trace reads almost as bright in IR as an
  EXPOSED pad -- this is why `build_band_texture` has a dedicated IR rule
  keying off "is there copper anywhere underneath" (F_Cu.png), not just
  "is the mask open" (F_Mask.png) the way the visible-light/GT identity map
  does. Source: USPTO patents 9,488,595 / 10,066,927 ("Inspection of
  microelectronic devices using near-infrared light").
- Everything else (fiberglass in IR, ~all of UV, silkscreen absorption) is an
  UNVERIFIED placeholder -- no measured source was found for these. Flagged
  inline. Recalibrate once real IR/UV captures exist (the physical rig's
  IR/UV LED hardware is new / not yet capturing data as of this writing).
- "component" (populated 3D component bodies, from STL/STEP) is the LEAST
  characterized class -- real components are a mix of black plastic
  packages, metal shielding cans, ceramic caps, ENIG-plated connector pins,
  etc. with wildly different reflectance per part. A single flat value here
  is a coarse placeholder standing in for "generic component," not a
  materially-informed estimate. Good enough to visually distinguish
  "there's a part here" from the bare board; not good enough to distinguish
  component sub-materials.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels  # noqa: E402

BANDS = ["R", "G", "B", "IR", "UV"]

# --- visible-light (R/G/B) approximate class colour, 0..1 floats -----------
# Used only to pick each band's single channel (e.g. band "R" reads the
# r-component) -- this is the mono-sensor-under-one-coloured-LED model, not
# a full-colour render.
VISIBLE_COLOR: dict[str, tuple[float, float, float]] = {
    "copper": (0.85, 0.55, 0.25),
    "solder": (0.75, 0.75, 0.78),
    "fiberglass": (0.55, 0.50, 0.30),
    "silkscreen": (0.92, 0.92, 0.90),
    # soldermask default; overridden per-board by read_soldermask_color()
    "soldermask": (0.05, 0.35, 0.15),
    # generic "component" -- coarse placeholder, see module docstring
    "component": (0.25, 0.25, 0.28),
}

# name (as written in blendcfg.yaml) -> approximate visible RGB
SOLDERMASK_COLORS: dict[str, tuple[float, float, float]] = {
    "Green": (0.05, 0.35, 0.15),
    "Black": (0.03, 0.03, 0.03),
    "Blue": (0.05, 0.15, 0.45),
    "Red": (0.45, 0.05, 0.05),
    "White": (0.85, 0.85, 0.80),
}
SILKSCREEN_COLORS: dict[str, tuple[float, float, float]] = {
    "White": (0.92, 0.92, 0.90),
    "Black": (0.05, 0.05, 0.05),
}

# --- IR / UV approximate scalar reflectance per class, 0..1 -----------------
REFLECTANCE: dict[str, dict[str, float]] = {
    "IR": {
        "copper": 0.85,      # sourced: broadband metal reflector past 850nm
        "solder": 0.80,      # sourced (same reasoning, metallic)
        "fiberglass": 0.35,  # UNVERIFIED placeholder
        "soldermask": 0.15,  # base value where mask has nothing conductive beneath
        "silkscreen": 0.30,  # UNVERIFIED placeholder
        "component": 0.30,   # UNVERIFIED coarse placeholder (mixed materials)
    },
    "UV": {
        "copper": 0.35,      # UNVERIFIED number; direction (lower than IR/vis) follows
                             # the general physical trend of metal reflectance
                             # dropping toward UV, not a PCB-specific measurement
        "solder": 0.30,      # UNVERIFIED, same reasoning
        "fiberglass": 0.25,  # UNVERIFIED placeholder
        "soldermask": 0.10,  # UNVERIFIED; assumes strong absorption, consistent
                             # with UV-curable soldermask chemistry
        "silkscreen": 0.12,  # UNVERIFIED; UV fluorescence hypothesis NOT modelled
                             # (no confirmed source found -- see plan notes)
        "component": 0.15,   # UNVERIFIED coarse placeholder (mixed materials)
    },
}


def read_soldermask_color(proj: Path) -> str:
    cfg = proj / "blendcfg.yaml"
    if cfg.exists():
        m = re.search(r"(?ms)^\s*SOLDERMASK:\s*\n?\s*(Black|White|Green|Blue|Red)\b",
                      cfg.read_text())
        if m:
            return m.group(1)
    return "Green"


def read_silkscreen_color(proj: Path) -> str:
    cfg = proj / "blendcfg.yaml"
    if cfg.exists():
        m = re.search(r"(?m)^\s*SILKSCREEN:\s*(Black|White)\b", cfg.read_text())
        if m:
            return m.group(1)
    return "White"


def _layer_masks(proj: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (cu, mask_open, silk, solder) boolean arrays from fab/PNG/*.png."""
    png = proj / "fab" / "PNG"

    def feat(name: str) -> np.ndarray:
        return np.asarray(Image.open(png / f"{name}.png").convert("L")) < 127

    # NOTE: gerber2blend's EFFECTS.SOLDER-grown F_Solder.png (also the
    # footprint of the 3D solder dome it extrudes into board.obj) looks like
    # a better solder-GT source than the raw, deliberately-undersized
    # F_Paste.png stencil -- but it's NOT: its dilation is a fixed-pixel
    # image-processing grow, not a physically-motivated stencil-to-pad
    # correction, and on dense/fine-pitch boards it bridges ADJACENT,
    # separate pads into one connected blob (confirmed: e.g.
    # urti-mainboard has a single "solder" region spanning 4 distinct pads
    # that are separate in F_Paste.png -- real solder joints on adjacent
    # fine-pitch pins don't merge like that, it'd be a short). Tried using
    # F_Solder.png for solder GT this session: retrained/re-evaluated and it
    # REGRESSED solder IoU (0.752->0.732) and mIoU slightly, not just for
    # dense boards but overall -- see docs/STATUS.md. Reverted to F_Paste.png.
    return feat("F_Cu"), feat("F_Mask"), feat("F_SilkS"), feat("F_Paste")


def build_class_id_map(proj: Path) -> np.ndarray:
    """Per-pixel material-identity class id (uint8), used for GT and for the
    visible-light (R/G/B) bands, where soldermask genuinely occludes copper."""
    cu, mask_open, silk, solder = _layer_masks(proj)
    C = labels.CLASSES
    ids = np.full(cu.shape, C["soldermask"], np.uint8)
    ids[mask_open & cu] = C["copper"]
    ids[mask_open & ~cu] = C["fiberglass"]
    ids[cu & ~mask_open] = C["trace"]  # copper routed under intact soldermask
    ids[solder] = C["solder"]
    ids[silk] = C["silkscreen"]
    return ids


def build_band_texture(proj: Path, band: str) -> np.ndarray:
    """Single-channel (H, W) float32 reflectance/appearance map in [0, 1] for
    the given band ("R"/"G"/"B"/"IR"/"UV"). This is what a mono sensor under
    one band's illumination would see reflected off each material."""
    if band not in BANDS:
        raise ValueError(f"unknown band {band!r}, expected one of {BANDS}")

    if band in ("R", "G", "B"):
        colors = dict(VISIBLE_COLOR)
        colors["soldermask"] = SOLDERMASK_COLORS[read_soldermask_color(proj)]
        colors["silkscreen"] = SILKSCREEN_COLORS[read_silkscreen_color(proj)]
        # trace (copper under intact soldermask) has no sourced visible-light
        # transparency claim -- it must look identical to soldermask, not copper.
        colors["trace"] = colors["soldermask"]
        chan = {"R": 0, "G": 1, "B": 2}[band]
        ids = build_class_id_map(proj)
        out = np.zeros(ids.shape, np.float32)
        for cls, cid in labels.CLASSES.items():
            if cls == "background" or cls not in colors:
                continue
            out[ids == cid] = colors[cls][chan]
        return out

    # IR / UV: approximate scalar reflectance, with the IR-specific
    # "soldermask is transparent to copper beneath" override.
    cu, mask_open, silk, solder = _layer_masks(proj)
    R = REFLECTANCE[band]
    out = np.full(cu.shape, R["soldermask"], np.float32)
    if band == "IR":
        out[cu] = R["copper"]                    # covered OR exposed copper
    else:  # UV: no sourced transparency claim, assume mask stays opaque
        out[mask_open & cu] = R["copper"]
    out[mask_open & ~cu] = R["fiberglass"]
    out[solder] = R["solder"]
    out[silk] = R["silkscreen"]
    return out


def save_texture_png(arr: np.ndarray, out_png: Path) -> Path:
    """Save a [0,1] float (H,W) or (H,W,3) array as an 8-bit PNG texture."""
    u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    mode = "L" if u8.ndim == 2 else "RGB"
    Image.fromarray(u8, mode).save(out_png)
    return out_png
