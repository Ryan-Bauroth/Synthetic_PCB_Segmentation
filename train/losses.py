"""Soft Dice loss for multi-class segmentation from raw logits.

Combined with CrossEntropyLoss in train.py to counteract per-pixel CE's
under-weighting of small/thin classes (e.g. solder) relative to broad,
easy classes (soldermask, fiberglass) that dominate pixel counts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1.0,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Soft Dice loss averaged over classes (equal weight per class).

    logits: (B, C, H, W) raw, un-normalized logits.
    target: (B, H, W) int64 class-id map, values in [0, num_classes).
    smooth: Laplace smoothing in numerator/denominator, avoids 0/0 when a
        class is entirely absent from a batch/patch (common for `solder`
        in a single 256x256 patch).
    ignore_index: if given, that class id is excluded from the mean (used
        to drop background so its near-1.0 Dice score doesn't dilute the
        signal for genuinely hard, sparse classes).
    """
    probs = F.softmax(logits, dim=1)                           # (B, C, H, W)
    target_onehot = F.one_hot(target, num_classes)               # (B, H, W, C)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()    # (B, C, H, W)

    dims = (0, 2, 3)  # sum over batch + spatial, keep per-class
    intersection = (probs * target_onehot).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target_onehot.sum(dim=dims)

    dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)  # (C,)

    if ignore_index is not None:
        keep = [c for c in range(num_classes) if c != ignore_index]
        dice_per_class = dice_per_class[keep]

    return 1.0 - dice_per_class.mean()
