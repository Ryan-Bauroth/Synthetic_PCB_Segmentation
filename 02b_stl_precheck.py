#!/usr/bin/env python3
"""Cheap early gate: is this board's STL POPULATED (real 3D component models
resolvable), using ONLY the cached .kicad_pcb from 02_fetch_gerbers.py -- no
gerber2blend/Blender dependency at all. Run this BEFORE
03_run_gerber2blend.py so a board with unresolvable components is dropped
before the expensive Blender conversion + Mitsuba imaging, not after.
STL/component data is required for this project (height channel, the
component/trace GT classes) -- a bare board is not usable training data.

Writes boards/<board>/stl_report.json with `populated`, mesh stats, AND the
board-body anchor shell's cached world bounds. Caching that anchor is what
lets 06_export_stl.py's LATER finalize pass (run after 05_export_mesh.py,
once board_bounds.json exists) align the components-only mesh WITHOUT
re-parsing/re-tessellating the full populated STEP scene a second time --
that tessellation is the expensive part for many-shell boards (observed
~3 minutes for a 720-shell board).

Run:  ./.venv311/bin/python 02b_stl_precheck.py --board <name>
Exit 0 if populated, 1 if bare/unusable (caller decides whether to drop it).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import trimesh

import os

from stl_common import find_board_shell, find_cached_pcb, kicad_cli, prepare_step_export_pcb

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    args = ap.parse_args()

    proj = ROOT / "boards" / args.board
    pcb = find_cached_pcb(ROOT, args.board)

    step_path = proj / "fab" / f"{args.board}.step"
    step_path.parent.mkdir(parents=True, exist_ok=True)
    kcli = kicad_cli()
    # recover real 3D geometry kicad-cli would otherwise silently drop -- see
    # stl_common.py's module-level comment for the two mechanisms fixed here.
    # allow_fuzzy=True: validated 2026-07-27 on the 12 boards that had
    # leftover unfixable .wrl refs after the exact-match tiers -- every
    # match preserves pin/position count exactly (see
    # stl_common.find_kicad_packages3d_fuzzy_match's pin-count guard,
    # added after it caught a real wrong-part match), and the ones that
    # add real new geometry were spot-checked (shell-count deltas match
    # expected fixes exactly, one verified by name/dimensions).
    export_pcb, extra_env = prepare_step_export_pcb(ROOT, args.board, pcb, allow_fuzzy=True)
    cmd = [kcli, "pcb", "export", "step", str(export_pcb), "--output", str(step_path),
           "--subst-models", "--force"]
    print("  $", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, **extra_env})
    if proc.returncode != 0 or not step_path.exists():
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"kicad-cli step export failed (exit {proc.returncode})")
    print(f"Wrote {step_path.relative_to(ROOT)}")

    print("Converting STEP -> STL (trimesh/cascadio)...")
    loaded = trimesh.load(str(step_path))
    full_scene = loaded if isinstance(loaded, trimesh.Scene) else None
    if isinstance(loaded, trimesh.Scene):
        n_geoms = len(loaded.geometry)
        mesh = loaded.to_geometry()
    else:
        n_geoms = 1
        mesh = loaded

    stl_path = proj / f"{args.board}.stl"
    mesh.export(str(stl_path))

    bbox = mesh.bounds
    size = bbox[1] - bbox[0]
    span_xy = max(size[0], size[1])
    ratio = float(size[2] / span_xy) if span_xy else 0.0
    # n_geoms > 1 required too: a single-shell STEP has no separable
    # component geometry at all, so a high Z-ratio there is just a small
    # board's bare substrate thickness relative to its width tripping the
    # heuristic, not real populated 3D content (found empirically:
    # pmod-i3c-sensor-board and pdm-microphone-board both cleared the 3%
    # ratio with n_shells=1 and zero exportable components).
    populated = bool(ratio >= 0.03 and n_geoms > 1)

    anchor_bounds = None
    if populated and full_scene is not None:
        anchor = find_board_shell(full_scene)
        if anchor is not None:
            anchor_bounds = anchor[1].tolist()

    print(f"Mesh stats: {n_geoms} source shells/parts, {len(mesh.vertices)} vertices, "
          f"{len(mesh.faces)} faces")
    print(f"Bounding box: {size[0]:.5f} x {size[1]:.5f} x {size[2]:.5f}  "
          f"(Z-ratio {ratio * 100:.1f}%)")
    if not populated:
        print(f"\nNOT POPULATED -- {args.board} looks like a bare board "
              f"(Z-ratio {ratio * 100:.1f}% < 3%). Drop it before the expensive "
              f"gerber2blend/imaging steps -- STL/component data is required "
              f"for this project.", file=sys.stderr)
    elif anchor_bounds is None:
        print("\nWARNING: populated but no board-body anchor shell found -- "
              "the render-time components alignment step will not be able "
              "to run later.", file=sys.stderr)

    report = {
        "board": args.board,
        "stl_path": str(stl_path.relative_to(ROOT)),
        "step_path": str(step_path.relative_to(ROOT)),
        "n_shells": n_geoms,
        "n_vertices": len(mesh.vertices),
        "n_faces": len(mesh.faces),
        "bbox_size": [float(size[0]), float(size[1]), float(size[2])],
        "z_extent_ratio": float(ratio),
        "populated": populated,
        "anchor_world_bounds": anchor_bounds,  # cached for 06_export_stl.py's finalize pass
        "render_components_available": False,  # updated later by finalize, if it succeeds
        "precheck_only": True,
    }
    (proj / "stl_report.json").write_text(json.dumps(report, indent=2))
    return 0 if populated else 1


if __name__ == "__main__":
    raise SystemExit(main())
