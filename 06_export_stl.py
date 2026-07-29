#!/usr/bin/env python3
"""Finalize step for the populated STL (board + placed 3D component models).
Component fidelity depends entirely on whether the source board's KiCad
footprints have 3D models assigned -- this varies board to board, so
02b_stl_precheck.py already reported (and gated on) whether this board is
populated BEFORE the expensive gerber2blend/imaging steps ran at all.

If 02b_stl_precheck.py already ran for this board (the normal path via
09_batch.py), this script REUSES its cached STEP/STL + board-body anchor
bounds instead of re-parsing/re-tessellating the full populated STEP scene
a second time -- that OpenCascade tessellation is the expensive part for
many-shell boards (observed ~3 minutes for a 720-shell board), so doing it
twice per board would waste real time. Only the (always-single-pass)
components-only STEP export + alignment happens here.

Run standalone (e.g. via run_mvp.sh, without 02b having run first) it falls
back to doing the full STEP export + STL conversion itself, same as this
script did before the precheck step existed.

Needs board_bounds.json (05_export_mesh.py) to align the components-only
mesh into gerber2blend's coordinate space: KiCad's STEP export uses its own
absolute origin, unrelated to gerber2blend's board-centred convention. The
alignment anchor is the STEP assembly's own board-body shell (found by
name, "_PCB" suffix -- the KiCad STEP exporter's convention -- falling back
to the largest/flattest shell) matched against board_bounds.json's
known-good center (verified empirically: the board-body shell's own size
independently matches board_bounds.json's Gerber-derived size to within
~0.1mm, confirming it's the same physical board and giving a precise,
board-anchored offset rather than an approximate one from the scattered
component cluster).

Run:  ./.venv311/bin/python 06_export_stl.py --board <name>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

import os

from stl_common import (MM_PER_STEP_UNIT, find_board_shell, find_cached_pcb, kicad_cli,
                        prepare_step_export_pcb)

ROOT = Path(__file__).resolve().parent


def export_components_mesh(kcli: str, pcb: Path, anchor_bounds: list | None, proj: Path,
                           board: str) -> bool:
    """Export a components-only mesh, aligned into gerber2blend's board-centred
    mm coordinate space, for the render pipeline. `anchor_bounds` is the
    board-body shell's cached world bounds (from 02b_stl_precheck.py, or a
    fresh find_board_shell() call in the standalone fallback path) -- no
    need to re-parse the full populated STEP scene here. Returns True on
    success."""
    bounds_path = proj / "board_bounds.json"
    if not bounds_path.exists():
        print("  no board_bounds.json yet (run 05_export_mesh.py first) -- "
              "skipping component mesh for rendering.", file=sys.stderr)
        return False
    if anchor_bounds is None:
        print("  no board-body anchor bounds available -- "
              "skipping component mesh for rendering.", file=sys.stderr)
        return False

    board_world_bounds = np.array(anchor_bounds)
    step_board_center_mm = board_world_bounds.mean(axis=0) * MM_PER_STEP_UNIT

    gerber_bounds = json.loads(bounds_path.read_text())
    gerber_center_mm = np.array(gerber_bounds["center_mm"])
    offset_mm = gerber_center_mm - step_board_center_mm

    comp_step = proj / "fab" / f"{board}_components.step"
    # allow_fuzzy=True: see 02b_stl_precheck.py's call site comment --
    # validated 2026-07-27, pin-count-guarded, spot-checked.
    export_pcb, extra_env = prepare_step_export_pcb(ROOT, board, pcb, allow_fuzzy=True)
    cmd = [kcli, "pcb", "export", "step", str(export_pcb), "--output", str(comp_step),
           "--subst-models", "--no-board-body", "--force"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, **extra_env})
    if proc.returncode != 0 or not comp_step.exists():
        print("  components-only STEP export failed -- skipping.", file=sys.stderr)
        return False

    comp_scene = trimesh.load(str(comp_step))
    if isinstance(comp_scene, trimesh.Scene):
        comp_mesh = comp_scene.to_geometry()
    else:
        comp_mesh = comp_scene
    if not len(comp_mesh.vertices):
        print("  components-only mesh is empty -- skipping.", file=sys.stderr)
        return False

    comp_mesh.apply_scale(MM_PER_STEP_UNIT)          # metres -> mm
    comp_mesh.apply_translation(offset_mm)            # align to gerber2blend space

    mitsuba_dir = proj / "fab" / "mitsuba"
    mitsuba_dir.mkdir(parents=True, exist_ok=True)
    comp_obj = mitsuba_dir / "components.obj"
    comp_mesh.export(str(comp_obj))
    print(f"  wrote {comp_obj.relative_to(ROOT)}  "
          f"({len(comp_mesh.vertices)} verts, offset applied: "
          f"{tuple(round(v,1) for v in offset_mm)} mm)")
    return True


def full_stl_pass(pcb: Path, step_path: Path, stl_path: Path, kcli: str, board: str):
    """Standalone fallback: do the full populated STEP export + STL
    conversion + anchor lookup from scratch (no precheck available).
    Returns (n_shells, mesh, anchor_bounds_or_None)."""
    step_path.parent.mkdir(parents=True, exist_ok=True)
    # allow_fuzzy=True: see 02b_stl_precheck.py's call site comment --
    # validated 2026-07-27, pin-count-guarded, spot-checked.
    export_pcb, extra_env = prepare_step_export_pcb(ROOT, board, pcb, allow_fuzzy=True)
    cmd = [kcli, "pcb", "export", "step", str(export_pcb),
           "--output", str(step_path), "--subst-models", "--force"]
    print("  $", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, **extra_env})
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"kicad-cli step export failed (exit {proc.returncode})")
    if not step_path.exists():
        sys.exit("kicad-cli reported success but no .step file was produced")
    print(f"Wrote {step_path.relative_to(ROOT)}")

    print("Converting STEP -> STL (trimesh/cascadio)...")
    loaded = trimesh.load(str(step_path))
    full_scene = loaded if isinstance(loaded, trimesh.Scene) else None
    if isinstance(loaded, trimesh.Scene):
        n_geoms = len(loaded.geometry)
        # scene.geometry values are in LOCAL mesh space -- concatenating them
        # directly (as an earlier version of this script did) ignores the
        # scene graph's per-part placement transforms, collapsing every
        # component onto its own local origin instead of its real assembly
        # position. to_geometry() bakes those transforms in first. Verified
        # empirically: local-concat bbox for a 348-part board was a bogus
        # 73x60mm; the correctly-assembled bbox is 133x40mm, matching the
        # board's real 128x40mm Gerber-derived footprint.
        mesh = loaded.to_geometry()
    else:
        n_geoms = 1
        mesh = loaded
    mesh.export(str(stl_path))

    anchor_bounds = None
    if full_scene is not None:
        anchor = find_board_shell(full_scene)
        if anchor is not None:
            anchor_bounds = anchor[1]
    return n_geoms, mesh, anchor_bounds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    args = ap.parse_args()

    proj = ROOT / "boards" / args.board
    pcb = find_cached_pcb(ROOT, args.board)

    step_path = proj / "fab" / f"{args.board}.step"
    stl_path = proj / f"{args.board}.stl"
    report_path = proj / "stl_report.json"
    kcli = kicad_cli()

    precheck = None
    if report_path.exists():
        candidate = json.loads(report_path.read_text())
        if candidate.get("precheck_only") and step_path.exists() and stl_path.exists():
            precheck = candidate

    if precheck is not None:
        print(f"Reusing 02b_stl_precheck.py's result for {args.board} "
              f"(skips re-tessellating the full STEP model)")
        n_geoms = precheck["n_shells"]
        size = precheck["bbox_size"]
        ratio = precheck["z_extent_ratio"]
        populated = precheck["populated"]
        anchor_bounds = precheck.get("anchor_world_bounds")
        mesh = trimesh.load(str(stl_path))  # cheap: already-flattened STL, no OpenCascade
        n_vertices, n_faces = len(mesh.vertices), len(mesh.faces)
    else:
        n_geoms, mesh, anchor_bounds_arr = full_stl_pass(pcb, step_path, stl_path, kcli, args.board)
        bbox = mesh.bounds
        size3 = bbox[1] - bbox[0]
        size = [float(size3[0]), float(size3[1]), float(size3[2])]
        span_xy = max(size[0], size[1])
        ratio = float(size[2] / span_xy) if span_xy else 0.0
        # n_geoms > 1 required too -- see 02b_stl_precheck.py's comment:
        # a single-shell STEP has no separable component geometry, so a
        # high Z-ratio there is just bare substrate thickness, not real
        # populated 3D content.
        populated = bool(ratio >= 0.03 and n_geoms > 1)
        anchor_bounds = anchor_bounds_arr.tolist() if anchor_bounds_arr is not None else None
        n_vertices, n_faces = len(mesh.vertices), len(mesh.faces)
        print(f"Mesh stats: {n_geoms} source shells/parts, {n_vertices} vertices, {n_faces} faces")
        print(f"Bounding box: {size[0]:.5f} x {size[1]:.5f} x {size[2]:.5f} "
              f"(Z-ratio {ratio * 100:.1f}%)")
        if not populated:
            print(f"\nWARNING: Z-extent is only {ratio * 100:.1f}% of board width -- "
                  f"this looks like just the bare board with no (or few) populated "
                  f"3D component models. Check whether {args.board}'s KiCad "
                  f"footprints have 3D models assigned (this board may reference "
                  f"an env-var model library not available here).", file=sys.stderr)

    render_components = False
    if populated and anchor_bounds is not None:
        print("\nExporting components-only mesh for the render pipeline...")
        render_components = export_components_mesh(kcli, pcb, anchor_bounds, proj, args.board)

    report = {
        "board": args.board,
        "stl_path": str(stl_path.relative_to(ROOT)),
        "step_path": str(step_path.relative_to(ROOT)),
        "n_shells": n_geoms,
        "n_vertices": n_vertices,
        "n_faces": n_faces,
        "bbox_size": list(size),
        "z_extent_ratio": float(ratio),
        "populated": populated,
        "anchor_world_bounds": anchor_bounds,
        "render_components_available": render_components,
        "precheck_only": False,
    }
    report_path.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
