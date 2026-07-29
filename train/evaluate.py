#!/usr/bin/env python3
"""Evaluate a trained checkpoint on its held-out val boards at FULL
resolution (sliding-window tiled inference, not random crops) -- shows real
whole-board behavior, not just patch-level accuracy. Reports the final
per-class IoU table and saves a palette-colored prediction PNG next to the
true GT for visual side-by-side inspection.

Run:  ./.venv311/bin/python train/evaluate.py --checkpoint train/checkpoints/unet_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import labels  # noqa: E402
from dataset import _load_stack  # noqa: E402
from metrics import ConfusionMatrix  # noqa: E402
from model import UNet  # noqa: E402
from train import pick_device  # noqa: E402
from viz import add_class_legend  # noqa: E402


def sliding_window_predict(model: torch.nn.Module, x: np.ndarray, device: torch.device,
                           tile: int = 256, overlap: int = 32) -> np.ndarray:
    """x: (C, H, W) float32. Returns (H, W) predicted class ids.
    Overlapping tiles averaged in logit space to avoid seam artifacts at
    tile borders."""
    c, h, w = x.shape
    stride = tile - overlap
    logits_sum = np.zeros((labels.NUM_CLASSES, h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)

    model.eval()
    with torch.no_grad():
        for ty in ys:
            for tx in xs:
                th, tw = min(tile, h - ty), min(tile, w - tx)
                patch = np.zeros((c, tile, tile), dtype=np.float32)
                patch[:, :th, :tw] = x[:, ty:ty + th, tx:tx + tw]
                inp = torch.from_numpy(patch).unsqueeze(0).to(device)
                out = model(inp)[0].cpu().numpy()  # (num_classes, tile, tile)
                logits_sum[:, ty:ty + th, tx:tx + tw] += out[:, :th, :tw]
                weight_sum[ty:ty + th, tx:tx + tw] += 1.0

    logits_sum /= np.maximum(weight_sum, 1e-6)
    return logits_sum.argmax(0).astype(np.uint8)


def save_colored(idmap: np.ndarray, out_png: Path) -> None:
    im = Image.fromarray(idmap, "P")
    im.putpalette(labels.flat_palette())
    im = add_class_legend(im, title="class legend")
    im.save(out_png)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "eval_out")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    val_boards = ckpt["val_boards"]
    print(f"Checkpoint epoch={ckpt['epoch']} train val_mIoU={ckpt['val_miou']:.4f}")
    print(f"Val boards: {val_boards}")

    device = pick_device()
    model = UNet(in_channels=ckpt["in_channels"], num_classes=labels.NUM_CLASSES,
                base=ckpt["base_channels"]).to(device)
    model.load_state_dict(ckpt["model"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cm = ConfusionMatrix(labels.NUM_CLASSES)
    for board in val_boards:
        x, y = _load_stack(ROOT, board)
        pred = sliding_window_predict(model, x, device, args.tile, args.overlap)
        cm.update(torch.from_numpy(pred), torch.from_numpy(y.astype(np.int64)))

        board_cm = ConfusionMatrix(labels.NUM_CLASSES)
        board_cm.update(torch.from_numpy(pred), torch.from_numpy(y.astype(np.int64)))
        print(f"\n{board}: mIoU={board_cm.mean_iou():.4f} pixel_acc={board_cm.pixel_accuracy():.4f}")

        save_colored(pred, args.out_dir / f"{board}_pred.png")
        save_colored(y, args.out_dir / f"{board}_gt.png")
        print(f"  wrote {args.out_dir / (board + '_pred.png')} (+ _gt.png for comparison)")

    print("\nOverall val report (all held-out boards pooled):")
    print(cm.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
