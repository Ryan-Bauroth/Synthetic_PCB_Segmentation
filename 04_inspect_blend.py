#!/usr/bin/env python3
"""Phase-1 GATE: open the generated .blend and dump every object + material so
we can confirm copper / soldermask / silkscreen / substrate / solder exist as
distinct, labelable entities. Writes boards/<name>/blend_report.json.

Run with the bpy venv:
    ./.venv311/bin/python 04_inspect_blend.py --board <name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import labels  # noqa: E402  (local module, stdlib-only)

try:
    import bpy  # noqa: E402
except ImportError:
    sys.exit("Could not import bpy. Run this with ./.venv311/bin/python")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    args = ap.parse_args()

    proj = ROOT / "boards" / args.board
    blends = list((proj / "fab").glob("*.blend"))
    if not blends:
        sys.exit(f"No .blend in {proj/'fab'}. Run 03_run_gerber2blend.py first.")
    blend = blends[0]

    bpy.ops.wm.open_mainfile(filepath=str(blend))

    objects = []
    classes_seen: set[str] = set()
    for obj in bpy.data.objects:
        mats = [ms.material.name for ms in obj.material_slots if ms.material]
        # classify by object name first, then any material name
        cls = labels.classify_name(obj.name)
        if cls is None:
            for mn in mats:
                cls = labels.classify_name(mn)
                if cls:
                    break
        if cls:
            classes_seen.add(cls)
        objects.append({
            "object": obj.name,
            "type": obj.type,
            "materials": mats,
            "pass_index": obj.pass_index,
            "classified_as": cls,
        })

    material_classes = {
        m.name: labels.classify_name(m.name) for m in bpy.data.materials
    }

    # Labeling precondition: gerber2blend does NOT split copper/soldermask/
    # silkscreen into separate materials -- it bakes them as per-layer PNGs in
    # fab/PNG/, which 06_render.py composites into the label mask. So the real
    # gate is that those source layer PNGs exist.
    png = proj / "fab" / "PNG"
    needed = ["F_Cu", "F_Mask", "F_SilkS", "F_Paste"]
    layer_pngs = {n: (png / f"{n}.png").exists() for n in needed}
    missing = sorted(n for n, ok in layer_pngs.items() if not ok)

    report = {
        "board": args.board,
        "blend": str(blend.relative_to(ROOT)),
        "n_objects": len(objects),
        "n_materials": len(bpy.data.materials),
        "objects": objects,
        "material_classes": material_classes,
        "classes_seen": sorted(classes_seen),
        "layer_pngs": layer_pngs,
        "missing_layer_pngs": missing,
    }
    out = proj / "blend_report.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"Objects: {len(objects)}   Materials: {len(bpy.data.materials)}")
    print("Material -> class:")
    for mn, c in material_classes.items():
        print(f"  {mn:<40} -> {c}")
    print("\nPer-layer label sources (fab/PNG):")
    for n, ok in layer_pngs.items():
        print(f"  {n:<10} {'ok' if ok else 'MISSING'}")
    if missing:
        print(f"\nGATE FAIL: missing layer PNGs {missing} -- labels for those")
        print("  classes can't be built. Check gerber export / EFFECTS.SOLDER.")
    else:
        print("\nGATE OK: all per-layer label sources present.")
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
