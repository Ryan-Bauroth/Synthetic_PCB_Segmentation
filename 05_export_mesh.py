#!/usr/bin/env python3
"""Export the gerber2blend board mesh (with its UV map) for Mitsuba to render,
and write board_bounds.json so every later step is pure Python (no bpy).

This is the ONLY remaining bpy step in the pipeline -- gerber2blend's mesh only
exists inside the .blend, so bpy is still needed to get geometry+UV out of it.
Everything downstream (rendering, STL export) runs plain Python in the same venv.

All mesh objects (board + solder-effect geometry) are joined into one mesh so
a single UV-mapped OBJ can carry the whole board into Mitsuba. The board's
existing UV (shared across gerber2blend's material slots -- confirmed by
inspection) is what 07_render_mitsuba.py's composited band textures
(spectral.py, built from the same fab/PNG/*.png layer rasters) are sampled
through, so the Mitsuba render lands in the same UV space gerber2blend used.

Run:  ./.venv311/bin/python 05_export_mesh.py --board <name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import bpy
    import mathutils
except ImportError:
    sys.exit("Could not import bpy. Run with ./.venv311/bin/python")


def board_bounds():
    """World-space bbox center and size over all mesh objects (mm)."""
    mins = mathutils.Vector((1e9, 1e9, 1e9))
    maxs = mathutils.Vector((-1e9, -1e9, -1e9))
    found = False
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        found = True
        for corner in obj.bound_box:
            wc = obj.matrix_world @ mathutils.Vector(corner)
            mins = mathutils.Vector(map(min, mins, wc))
            maxs = mathutils.Vector(map(max, maxs, wc))
    if not found:
        return mathutils.Vector((0, 0, 0)), mathutils.Vector((120, 60, 2))
    return (mins + maxs) / 2, (maxs - mins)


def join_all_mesh_objects():
    """Merge every mesh object into one, in world space, preserving UVs."""
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        sys.exit("No mesh objects found in the .blend")
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    args = ap.parse_args()

    proj = ROOT / "boards" / args.board
    blends = list((proj / "fab").glob("*.blend"))
    if not blends:
        sys.exit(f"No .blend in {proj/'fab'}. Run 03 first.")
    bpy.ops.wm.open_mainfile(filepath=str(blends[0]))

    center, size = board_bounds()

    merged = join_all_mesh_objects()
    uv_maps = [uv.name for uv in merged.data.uv_layers]
    print(f"Merged mesh: {merged.name}  verts={len(merged.data.vertices)}  "
          f"uv_maps={uv_maps}")
    if not uv_maps:
        print("WARNING: merged mesh has no UV map -- band textures won't "
              "align. Check the source .blend.", file=sys.stderr)

    mitsuba_dir = proj / "fab" / "mitsuba"
    mitsuba_dir.mkdir(parents=True, exist_ok=True)
    obj_path = mitsuba_dir / "board.obj"

    bpy.ops.object.select_all(action="DESELECT")
    merged.select_set(True)
    bpy.context.view_layer.objects.active = merged
    # Blender's OBJ exporter defaults to a Y-up axis swap (forward=-Z, up=Y).
    # Keep Z-up / no axis conversion so the mesh stays in the SAME coordinate
    # system as board_bounds.json (from board_bounds(), Blender-native) and
    # rig.py's light_positions() -- both already Z-up mm. Mismatched axes here
    # would silently misalign the Mitsuba camera/lights against the mesh.
    bpy.ops.wm.obj_export(filepath=str(obj_path), export_selected_objects=True,
                          forward_axis="Y", up_axis="Z",
                          export_uv=True, export_normals=True,
                          export_materials=True, export_triangulated_mesh=True)

    bounds = {
        "center_mm": [center.x, center.y, center.z],
        "size_mm": [size.x, size.y, size.z],
    }
    bounds_path = proj / "board_bounds.json"
    bounds_path.write_text(json.dumps(bounds, indent=2))

    print(f"Wrote {obj_path.relative_to(ROOT)}")
    print(f"Wrote {bounds_path.relative_to(ROOT)}")
    print(f"Board center={tuple(round(v,1) for v in center)} "
          f"size={tuple(round(v,1) for v in size)} mm")
    print(f"\nNext:  ./.venv311/bin/python 07_render_mitsuba.py --board {args.board}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
