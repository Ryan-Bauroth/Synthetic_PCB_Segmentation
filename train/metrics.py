"""Confusion-matrix-based segmentation metrics: per-class IoU, mean IoU,
pixel accuracy. Class names come from labels.ID_TO_NAME so reports line up
with the GT legend everywhere else in this project."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import labels  # noqa: E402


class ConfusionMatrix:
    def __init__(self, num_classes: int = labels.NUM_CLASSES):
        self.n = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred, target: any shape of integer class ids, same shape."""
        p = pred.detach().cpu().numpy().reshape(-1)
        t = target.detach().cpu().numpy().reshape(-1)
        valid = (t >= 0) & (t < self.n)
        idx = t[valid] * self.n + p[valid]
        self.mat += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def per_class_iou(self) -> np.ndarray:
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(0) - tp
        fn = self.mat.sum(1) - tp
        denom = tp + fp + fn
        iou = np.full(self.n, np.nan)
        present = denom > 0
        iou[present] = tp[present] / denom[present]
        return iou

    def mean_iou(self) -> float:
        iou = self.per_class_iou()
        return float(np.nanmean(iou))

    def pixel_accuracy(self) -> float:
        total = self.mat.sum()
        return float(np.diag(self.mat).sum() / total) if total else 0.0

    def report(self) -> str:
        iou = self.per_class_iou()
        lines = [f"  pixel accuracy: {self.pixel_accuracy():.4f}",
                f"  mean IoU:       {self.mean_iou():.4f}",
                "  per-class IoU:"]
        for cid in range(self.n):
            name = labels.ID_TO_NAME.get(cid, "?")
            val = "n/a (absent)" if np.isnan(iou[cid]) else f"{iou[cid]:.4f}"
            lines.append(f"    {cid} {name:<11} {val}")
        return "\n".join(lines)
