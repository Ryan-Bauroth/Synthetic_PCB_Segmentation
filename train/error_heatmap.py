#!/usr/bin/env python3
"""Visualize WHERE a trained checkpoint gets predictions wrong on its
held-out val boards -- the aggregate per-class IoU table (evaluate.py)
tells you WHICH classes are weak, this tells you where on the board those
errors actually happen and cluster.

Two images per val board:
- <board>_issue_points.png: every wrong pixel, colored by its TRUE class
  (labels.PALETTE) over a dimmed grayscale board image -- shows exactly
  which class is being missed and where.
- <board>_error_heatmap.png: a box-blurred error-density map (jet-like
  colormap) over the same grayscale board -- smooths past single stray
  mispredicted pixels to show concentrated problem regions.

Run:  ./.venv311/bin/python train/error_heatmap.py --checkpoint train/checkpoints/unet_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import labels  # noqa: E402
from dataset import _load_stack  # noqa: E402
from evaluate import sliding_window_predict  # noqa: E402
from model import UNet  # noqa: E402
from train import pick_device  # noqa: E402
from viz import add_class_legend, add_colorbar_legend  # noqa: E402


def jet_colormap(x: np.ndarray) -> np.ndarray:
    """x in [0,1] (H,W) -> (H,W,3) uint8: blue (low) -> green -> red (high)."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--blur-radius", type=int, default=15,
                    help="box-blur radius (px) for the error-density heatmap")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "eval_out")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    val_boards = ckpt["val_boards"]
    print(f"Checkpoint epoch={ckpt['epoch']} val_mIoU={ckpt['val_miou']:.4f}")
    print(f"Val boards: {val_boards}")

    device = pick_device()
    model = UNet(in_channels=ckpt["in_channels"], num_classes=labels.NUM_CLASSES,
                base=ckpt["base_channels"]).to(device)
    model.load_state_dict(ckpt["model"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for board in val_boards:
        x, y = _load_stack(ROOT, board)
        pred = sliding_window_predict(model, x, device, args.tile, args.overlap)
        wrong = pred != y
        n_wrong = int(wrong.sum())
        pct = 100.0 * n_wrong / wrong.size
        print(f"\n{board}: {n_wrong} wrong / {wrong.size} px ({pct:.2f}%)")

        u, c = np.unique(y[wrong], return_counts=True)
        print("  wrong pixels by TRUE class (what's actually being missed):")
        for cid, n in sorted(zip(u, c), key=lambda t: -t[1]):
            name = labels.ID_TO_NAME.get(int(cid), "?")
            print(f"    {name:<11} {int(n):7d}  ({100.0 * n / max(n_wrong, 1):.1f}% of errors)")

        r_band = (x[0] * 255).astype(np.uint8)  # R channel as grayscale context
        base = np.stack([r_band] * 3, axis=-1).astype(np.float32)

        issue = (base * 0.35).astype(np.uint8)  # dim the board so error colours pop
        error_classes = np.unique(y[wrong])
        for cid in error_classes:
            m = wrong & (y == cid)
            issue[m] = labels.PALETTE.get(int(cid), (255, 0, 255))
        issue_im = add_class_legend(Image.fromarray(issue), error_classes.tolist(),
                                    title="wrong pixels, coloured by TRUE class")
        issue_png = args.out_dir / f"{board}_issue_points.png"
        issue_im.save(issue_png)

        # local error RATE: fraction of wrong pixels in each blur_radius window
        # -- uniform_filter of a 0/1 mask is literally that mean, so this is
        # already a real, directly-interpretable number (0-100%), not an
        # arbitrary relative unit.
        window_px = args.blur_radius * 2 + 1
        density = uniform_filter(wrong.astype(np.float32), size=window_px)
        density_max = float(density.max())
        print(f"  peak local error rate: {density_max * 100:.1f}% of pixels wrong "
              f"in a {window_px}x{window_px}px window")
        color_scale = density / max(density_max, 1e-6)  # spread the colour range for visibility
        heat = jet_colormap(color_scale).astype(np.float32)
        overlay = (base * 0.5 + heat * 0.5).clip(0, 255).astype(np.uint8)
        heat_im = add_colorbar_legend(
            Image.fromarray(overlay), jet_colormap, vmin=0.0, vmax=density_max * 100,
            title=f"local error rate, % wrong pixels per {window_px}x{window_px}px window",
            fmt="{:.1f}%")
        heat_png = args.out_dir / f"{board}_error_heatmap.png"
        heat_im.save(heat_png)

        print(f"  wrote {issue_png.name}, {heat_png.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
