"""fab/PNG pixel <-> board-mm coordinate transform.

Empirically derived and verified (not assumed): a synthetic emissive marker
texture was rendered through the real board mesh/camera and its landing
position read back via the fully-known orthographic camera formula (see
07_render_mitsuba.py's ortho_sensor). Confirmed: the fab/PNG rasters (all
layers share one resolution per board, e.g. F_Cu.png/F_Paste.png/etc, and
this is the SAME resolution build_class_id_map/label_tex.png use) map onto
the board's bounding box (board_bounds.json's center_mm/size_mm) with row 0
= max world Y (top of image), col 0 = min world X (left of image) -- i.e.
the standard image-array convention, uniform scale in both axes.
"""

from __future__ import annotations

import numpy as np


def tex_px_to_world(row, col, tex_w: int, tex_h: int,
                    center_mm: np.ndarray, size_mm: np.ndarray):
    """(row, col) pixel coords (can be arrays) -> (world_x, world_y) mm."""
    x = center_mm[0] - size_mm[0] / 2.0 + (np.asarray(col) + 0.5) * (size_mm[0] / tex_w)
    y = center_mm[1] + size_mm[1] / 2.0 - (np.asarray(row) + 0.5) * (size_mm[1] / tex_h)
    return x, y


def world_to_tex_px(x, y, tex_w: int, tex_h: int,
                    center_mm: np.ndarray, size_mm: np.ndarray):
    """(world_x, world_y) mm (can be arrays) -> (row, col) pixel coords (float)."""
    col = (np.asarray(x) - center_mm[0] + size_mm[0] / 2.0) * (tex_w / size_mm[0]) - 0.5
    row = (center_mm[1] + size_mm[1] / 2.0 - np.asarray(y)) * (tex_h / size_mm[1]) - 0.5
    return row, col


def mm_per_px(tex_w: int, tex_h: int, size_mm: np.ndarray) -> tuple[float, float]:
    """(mm/px in Y i.e. row spacing, mm/px in X i.e. col spacing)."""
    return size_mm[1] / tex_h, size_mm[0] / tex_w
