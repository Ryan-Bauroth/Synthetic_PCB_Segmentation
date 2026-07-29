#!/usr/bin/env python3
"""Synthesize a small, component-aware SMD solder-fillet mesh for a populated
board, approximating the raised meniscus of solder that's actually visible
peeking out from under a component body at the edge of its pad -- something
this pipeline never modeled before (GT `solder` was previously just the flat,
mostly-occluded paste-stencil footprint, see docs/STATUS.md).

Why this approximation, not true per-lead geometry: KiCad/manufacturer 3D
component models are package bodies only, with no lead/pin geometry (verified
this session) -- true per-lead fillet shape isn't derivable from this data.
Instead: derive per-pad polygons from the paste-stencil raster (F_Paste.png),
rasterize the populated component bodies' top-down footprint, and raise a
small tapered wedge in the thin margin band where a pad transitions from
"under the component" to "open" -- tallest (mostly hidden, physically
correct -- that's where a real fillet climbs the lead) right at the
component's edge, tapering to zero at the pad's outer, fully-open edge so it
blends flush with the surrounding board surface.

Deliberately NOT per-blob polygon/contour extraction: instead a coarse
world-mm heightfield grid is sampled from a per-pixel signed-distance-to-
component-edge field (scipy.ndimage.distance_transform_edt, restricted to
pad pixels) -- this handles arbitrary pad/blob shapes uniformly without
needing to trace or triangulate individual polygon contours by hand.

Through-hole (THT) joints are explicitly out of scope (see docs/STATUS.md).

Run:  ./.venv311/bin/python 06b_generate_solder_fillets.py --board <name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import board_geom  # noqa: E402

BAND_WIDTH_MM = 0.20   # how far the wedge extends into the OPEN (visible) pad area
OVERLAP_MM = 0.05      # fixed extra reach UNDER the component, so there's no gap/crack
PEAK_HEIGHT_MM = 0.08  # wedge height at the component edge (tapers to 0 at BAND_WIDTH_MM)
MESH_GRID_MM = 0.05    # world-mm spacing of the generated mesh's vertex grid
MIN_HEIGHT_MM = 1e-3   # skip emitting a quad if no corner clears this


def flat_board_top_z(board_obj: Path) -> float:
    """True flat board-top surface Z (world mm), NOT board_bounds.json's
    bbox-derived center+size/2 -- that bbox is taken over ALL merged mesh
    objects including gerber2blend's own EFFECTS.SOLDER dome bump (named
    "Solder" in board.obj, confirmed via inspection: it reaches noticeably
    higher than the actual flat board surface), so it silently overstates
    "board top" by however tall that dome bump is. gerber2blend's own flat
    top-copper/soldermask surface is consistently named "main_pcb_top"
    across boards -- use ITS max world Z instead, so the new fillet mesh
    sits flush with the real board surface, not floating above it inside
    (or on top of) the existing, separate dome geometry."""
    scene = trimesh.load(str(board_obj), process=False)
    if not isinstance(scene, trimesh.Scene) or "main_pcb_top" not in scene.geometry:
        # Fallback: some boards may not split into named groups -- use the
        # overall mesh's max Z (slightly overstates if a dome exists, but
        # better than crashing).
        merged = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
        return float(merged.bounds[1][2])
    geom = scene.geometry["main_pcb_top"]
    transform = scene.graph.get("main_pcb_top")[0]
    verts_world = trimesh.transform_points(geom.vertices, transform)
    return float(verts_world[:, 2].max())


def rasterize_component_footprint(components_obj: Path, tex_w: int, tex_h: int,
                                  center_mm: np.ndarray, size_mm: np.ndarray) -> np.ndarray:
    """Top-down (XY) boolean occupancy grid, True wherever a component body
    triangle projects, at the same pixel grid as fab/PNG/*.png. Per-triangle
    PIL rasterization (not per-shell/convex-hull -- trimesh.split() on this
    merged, --subst-models-simplified mesh yields tens of thousands of tiny/
    degenerate fragments, not usable per-part shells). Verified fast enough
    in practice (<1s for a 430k-triangle board)."""
    mesh = trimesh.load(str(components_obj), process=False)
    faces = mesh.faces
    verts = mesh.vertices
    tri_xy = verts[faces][:, :, :2]  # (N, 3, 2) world mm
    rows, cols = board_geom.world_to_tex_px(tri_xy[:, :, 0], tri_xy[:, :, 1],
                                            tex_w, tex_h, center_mm, size_mm)
    canvas = Image.new("1", (tex_w, tex_h), 0)
    draw = ImageDraw.Draw(canvas)
    for tri_rows, tri_cols in zip(rows, cols):
        draw.polygon(list(zip(tri_cols.tolist(), tri_rows.tolist())), fill=1)
    return np.array(canvas, dtype=bool)


def compute_height_field(pad_mask: np.ndarray, component_mask: np.ndarray,
                         mm_per_px_row: float, mm_per_px_col: float) -> np.ndarray:
    """Per-pixel wedge height (mm), zero outside the margin band. Signed
    distance (mm) to the pad/component boundary: negative = under the
    component, positive = in the open/visible pad area. Height peaks
    OVERLAP_MM under the component edge and tapers linearly to zero at
    BAND_WIDTH_MM into the open area -- restricted to actual pad pixels so
    fully-exposed pads (no nearby component) get height 0 everywhere
    (distance stays large/positive, clipped away by construction)."""
    covered = pad_mask & component_mask
    sampling = (mm_per_px_row, mm_per_px_col)
    if not covered.any():
        return np.zeros(pad_mask.shape, np.float32)
    dist_open = ndimage.distance_transform_edt(~covered, sampling=sampling)
    dist_covered = ndimage.distance_transform_edt(covered, sampling=sampling)
    signed_dist = dist_open - dist_covered  # >0 outside covered, <0 inside
    span = BAND_WIDTH_MM + OVERLAP_MM
    height = PEAK_HEIGHT_MM * np.clip((BAND_WIDTH_MM - signed_dist) / span, 0.0, 1.0)
    height[~pad_mask] = 0.0
    return height.astype(np.float32)


def build_fillet_mesh(height_field: np.ndarray, tex_w: int, tex_h: int,
                      center_mm: np.ndarray, size_mm: np.ndarray,
                      z_top_mm: float) -> trimesh.Trimesh | None:
    """Coarse world-mm quad-grid mesh, restricted to cells where the sampled
    height clears MIN_HEIGHT_MM at at least one corner -- follows arbitrary
    pad-blob shapes without polygon contour extraction."""
    rows_nz, cols_nz = np.where(height_field > MIN_HEIGHT_MM)
    if len(rows_nz) == 0:
        return None
    xs, ys = board_geom.tex_px_to_world(rows_nz, cols_nz, tex_w, tex_h, center_mm, size_mm)
    pad_mm = BAND_WIDTH_MM + MESH_GRID_MM  # small margin around the active region
    x_lo, x_hi = xs.min() - pad_mm, xs.max() + pad_mm
    y_lo, y_hi = ys.min() - pad_mm, ys.max() + pad_mm

    nx = max(2, int(np.ceil((x_hi - x_lo) / MESH_GRID_MM)) + 1)
    ny = max(2, int(np.ceil((y_hi - y_lo) / MESH_GRID_MM)) + 1)
    grid_x = x_lo + np.arange(nx) * MESH_GRID_MM
    grid_y = y_lo + np.arange(ny) * MESH_GRID_MM
    gx, gy = np.meshgrid(grid_x, grid_y)  # (ny, nx)

    grow, gcol = board_geom.world_to_tex_px(gx, gy, tex_w, tex_h, center_mm, size_mm)
    grow_i = np.clip(np.round(grow).astype(int), 0, tex_h - 1)
    gcol_i = np.clip(np.round(gcol).astype(int), 0, tex_w - 1)
    grid_h = height_field[grow_i, gcol_i]  # (ny, nx), nearest-sample

    verts = np.stack([gx, gy, np.full_like(gx, z_top_mm) + grid_h], axis=-1)  # (ny, nx, 3)
    verts_flat = verts.reshape(-1, 3)

    def vid(r, c):
        return r * nx + c

    faces = []
    corner_h = np.stack([grid_h[:-1, :-1], grid_h[:-1, 1:],
                         grid_h[1:, :-1], grid_h[1:, 1:]], axis=-1)
    active = corner_h.max(axis=-1) > MIN_HEIGHT_MM
    ar, ac = np.where(active)
    for r, c in zip(ar.tolist(), ac.tolist()):
        v00, v01 = vid(r, c), vid(r, c + 1)
        v10, v11 = vid(r + 1, c), vid(r + 1, c + 1)
        # Winding must produce an UPWARD (+Z) normal so the one-sided area
        # emitter (fillet_shape()) faces the top-down camera -- verified
        # empirically: the opposite winding rendered as pure black.
        faces.append([v00, v11, v10])
        faces.append([v00, v01, v11])
    if not faces:
        return None
    mesh = trimesh.Trimesh(vertices=verts_flat, faces=np.array(faces), process=True)
    return mesh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    args = ap.parse_args()

    proj = ROOT / "boards" / args.board
    bounds_path = proj / "board_bounds.json"
    board_obj = proj / "fab" / "mitsuba" / "board.obj"
    components_obj = proj / "fab" / "mitsuba" / "components.obj"
    paste_png = proj / "fab" / "PNG" / "F_Paste.png"
    out_path = proj / "fab" / "mitsuba" / "solder_fillets.obj"

    if not components_obj.exists():
        print(f"No components.obj for {args.board} (bare/unpopulated board) -- "
              f"skipping fillet synthesis.")
        return 0
    if not bounds_path.exists() or not paste_png.exists() or not board_obj.exists():
        sys.exit(f"Missing board_bounds.json/F_Paste.png/board.obj for {args.board}. "
                 f"Run 05_export_mesh.py first.")

    d = json.loads(bounds_path.read_text())
    center_mm = np.array(d["center_mm"], np.float64)
    size_mm = np.array(d["size_mm"], np.float64)
    z_top_mm = flat_board_top_z(board_obj)

    pad_mask = np.asarray(Image.open(paste_png).convert("L")) < 127
    tex_h, tex_w = pad_mask.shape
    if not pad_mask.any():
        print(f"No solder-paste pixels for {args.board} -- skipping fillet synthesis.")
        return 0

    component_mask = rasterize_component_footprint(components_obj, tex_w, tex_h,
                                                    center_mm, size_mm)
    mm_per_px_row, mm_per_px_col = board_geom.mm_per_px(tex_w, tex_h, size_mm)
    height_field = compute_height_field(pad_mask, component_mask,
                                        mm_per_px_row, mm_per_px_col)

    n_active = int((height_field > MIN_HEIGHT_MM).sum())
    print(f"Pad px: {int(pad_mask.sum())}  covered-by-component px: "
          f"{int((pad_mask & component_mask).sum())}  fillet-band px: {n_active}")
    if n_active == 0:
        print(f"No pad/component margin band found for {args.board} -- "
              f"skipping fillet synthesis (no populated pads, or none border a "
              f"component edge).")
        return 0

    mesh = build_fillet_mesh(height_field, tex_w, tex_h, center_mm, size_mm, z_top_mm)
    if mesh is None or len(mesh.faces) == 0:
        print(f"Fillet mesh ended up empty for {args.board} -- skipping.")
        return 0

    mesh.export(str(out_path))
    print(f"Wrote {out_path.relative_to(ROOT)}  "
         f"verts={len(mesh.vertices)} faces={len(mesh.faces)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
