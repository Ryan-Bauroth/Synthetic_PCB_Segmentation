"""Single source of truth for the PCB material segmentation legend.

Every other script imports from here so the class ids, palette, and the
gerber2blend layer -> class mapping never drift apart.

Classes (7 materials + background):
    0 background   4 soldermask
    1 copper       5 silkscreen
    2 solder       6 component (populated 3D component body, from STEP)
    3 fiberglass   7 trace (copper routed under intact soldermask)
      (FR-4)

NB: this module is imported both by regular CPython scripts AND by Blender's
bundled Python (steps 05/06). Keep it dependency-free (stdlib only).

The "component" class only appears in a board's GT/renders when its STL came
back populated (see 06_export_stl.py's stl_report.json) -- boards without
resolvable 3D component models never produce this class, they just don't
have components rendered at all (not mislabeled as another class).

The "trace" class is copper routed under intact soldermask -- sourced
straight from F_Cu.png, independent of mask state (unlike "copper", which
only covers EXPOSED metal). It exists so trace routing is segmentable in
GT even though soldermask visually hides it. Visible-light (R/G/B) band
appearance for "trace" pixels is intentionally identical to "soldermask"
(no sourced visible-light transparency claim -- see spectral.py); only GT
and the IR band distinguish covered copper from bare soldermask.
"""

from __future__ import annotations

# --- class id map -----------------------------------------------------------
CLASSES: dict[str, int] = {
    "background": 0,
    "copper": 1,
    "solder": 2,
    "fiberglass": 3,
    "soldermask": 4,
    "silkscreen": 5,
    "component": 6,
    "trace": 7,
}

ID_TO_NAME: dict[int, str] = {v: k for k, v in CLASSES.items()}
NUM_CLASSES = len(CLASSES)

# --- indexed-PNG palette (id -> RGB) ---------------------------------------
# Chosen to be visually distinct for the overlay/eyeball check.
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),        # background  - black
    1: (184, 115, 51),   # copper      - copper/orange
    2: (192, 192, 200),  # solder      - light grey/silver
    3: (34, 120, 60),    # fiberglass  - FR-4 green
    4: (20, 90, 50),     # soldermask  - dark green
    5: (240, 240, 240),  # silkscreen  - near-white
    6: (90, 90, 140),    # component   - slate blue/purple (visually distinct)
    7: (120, 80, 45),    # trace       - muted/dark copper (covered, not exposed)
}


def flat_palette() -> list[int]:
    """Return the 256*3 flat palette list PIL wants for P-mode images.

    Ids not in PALETTE map to black.
    """
    flat: list[int] = []
    for i in range(256):
        r, g, b = PALETTE.get(i, (0, 0, 0))
        flat.extend((r, g, b))
    return flat


# --- gerber2blend layer/material name -> class ------------------------------
# gerber2blend names objects/materials after PCB layers. We can't know the
# exact strings until 04_inspect_blend.py dumps them, so we match on substrings
# (case-insensitive). Order matters: earlier, more-specific rules win.
#
# `05_setup_scene.py` walks each object's materials and assigns pass_index by
# the first rule whose keyword appears in the object OR material name.
LAYER_KEYWORD_RULES: list[tuple[tuple[str, ...], str]] = [
    (("paste", "solder_paste"), "solder"),       # paste stencil -> solder
    (("silk", "silkscreen"), "silkscreen"),
    (("mask", "soldermask"), "soldermask"),      # must precede bare "solder"
    (("solder",), "solder"),                     # SOLDER-effect geometry
    (("cu", "copper"), "copper"),                # F.Cu / B.Cu / *copper*
    (("substrate", "dielectric", "fr4", "fr-4", "pcb", "board", "edge"),
     "fiberglass"),
]


def classify_name(name: str) -> str | None:
    """Map a Blender object/material name to a class name, or None if unknown."""
    low = name.lower()
    for keywords, cls in LAYER_KEYWORD_RULES:
        if any(k in low for k in keywords):
            return cls
    return None


if __name__ == "__main__":
    # quick self-test / legend print
    print("Class legend:")
    for name, cid in CLASSES.items():
        print(f"  {cid}  {name:<11} rgb={PALETTE[cid]}")
    print("\nKeyword rules (first match wins):")
    for kws, cls in LAYER_KEYWORD_RULES:
        print(f"  {cls:<11} <- {kws}")
