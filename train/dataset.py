"""PCB semantic-segmentation dataset: 6-channel input (R, G, B, IR, UV,
height) -> 8-class GT mask (labels.CLASSES), read from out/manifest.json.

Split is by BOARD, not by patch -- patches cropped from the same board would
leak that board's specific layout into both train and val, giving an
inflated validation score. This does mean the split is coarse: with ~10
board layouts in the registry (02_fetch_gerbers.BOARDS), holding out N_VAL
boards leaves only (10 - N_VAL) distinct layouts to train on.

KNOWN LIMITATION, not hidden: random-crop patch sampling multiplies
training *examples* (thousands of crops/epoch from a handful of full-res
renders), not board *diversity*. The model can only ever have seen the
component/trace/silkscreen layouts of the boards in the train split --
generalization to a genuinely new board design is untested until more
boards are added to the registry and this dataset is regenerated.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import labels  # noqa: E402

BANDS = ["R", "G", "B", "IR", "UV"]
NUM_CHANNELS = len(BANDS) + 1  # + height
HEIGHT_NORM_MM = 30.0  # placeholder scale -- retune once real height ranges
                        # across a larger board registry are characterized;
                        # see README's "Height map" section (largest seen so
                        # far is ~19mm bbox / ~14mm above-bench max).


def board_split(root: Path = ROOT, val_boards: list[str] | None = None,
                n_val: int = 2, seed: int = 42) -> tuple[list[str], list[str]]:
    """Boards with verify_ok=True from out/manifest.json, split by board."""
    manifest = json.loads((root / "out" / "manifest.json").read_text())
    valid = sorted(b for b, info in manifest["boards"].items()
                   if info.get("verify_ok"))
    if not valid:
        raise RuntimeError("No boards with verify_ok=true in out/manifest.json "
                           "-- run 09_batch.py first.")
    if val_boards is not None:
        val = [b for b in val_boards if b in valid]
        train = [b for b in valid if b not in val]
        return train, val
    rng = random.Random(seed)
    shuffled = list(valid)
    rng.shuffle(shuffled)
    n_val = min(n_val, len(shuffled) - 1)  # always leave >=1 board to train on
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def _load_stack(root: Path, board: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (input CHW float32, label HW uint8) for one board, full res."""
    outdir = root / "out" / board
    chans = []
    for b in BANDS:
        arr = np.asarray(Image.open(outdir / f"{board}_{b}.png"), dtype=np.float32) / 255.0
        chans.append(arr)
    h_mm = np.asarray(Image.open(outdir / f"{board}_height.png"), dtype=np.float32) / 1000.0
    chans.append(np.clip(h_mm / HEIGHT_NORM_MM, 0.0, 1.0))
    x = np.stack(chans, axis=0)  # (6, H, W)
    y = np.asarray(Image.open(outdir / f"{board}_gt.png"), dtype=np.uint8)  # (H, W)
    return x, y


class PCBSegDataset(Dataset):
    """Random-crop patch sampler over a set of boards.

    __len__ is a virtual epoch size (patches_per_epoch), not tied to board
    count -- each __getitem__ picks a random board, then a random crop.
    Augmentation is geometric only (flip / 90-degree rotation): band and
    height values are physical measurements, not arbitrary photo colour, so
    colour jitter would corrupt the signal rather than regularize it.
    """

    def __init__(self, boards: list[str], root: Path = ROOT, patch_size: int = 256,
                patches_per_epoch: int = 2000, augment: bool = True,
                solder_oversample_prob: float = 0.0):
        """solder_oversample_prob: fraction of patches (0-1) forced to
        contain at least one `solder` GT pixel, jittered around a random
        solder pixel rather than always centering it (so the model doesn't
        just memorize "solder = center of patch"). 0 = pure uniform random
        cropping (unchanged prior behavior) -- keep at 0 for val/eval
        datasets so validation stays representative, not oversampled."""
        if not boards:
            raise ValueError("PCBSegDataset needs at least one board")
        self.root = root
        self.boards = boards
        self.patch = patch_size
        self.n = patches_per_epoch
        self.augment = augment
        self.solder_oversample_prob = solder_oversample_prob
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._solder_coords_cache: dict[str, np.ndarray] = {}

        # Precompute which boards actually have any `solder` GT pixels at
        # all (a board with no F/B_Paste layer collapses solder into copper,
        # per 02_fetch_gerbers.py's has_paste_layer warning -- oversampling
        # would have nothing to draw from on those boards).
        self._solder_boards: list[str] = []
        if solder_oversample_prob > 0:
            solder_id = labels.CLASSES["solder"]
            for b in boards:
                gt_path = root / "out" / b / f"{b}_gt.png"
                if not gt_path.exists():
                    continue
                y = np.asarray(Image.open(gt_path))
                if (y == solder_id).any():
                    self._solder_boards.append(b)

    def _board(self, board: str) -> tuple[np.ndarray, np.ndarray]:
        if board not in self._cache:
            self._cache[board] = _load_stack(self.root, board)
        return self._cache[board]

    def _solder_coords(self, board: str) -> np.ndarray:
        """(N, 2) array of (row, col) coordinates of `solder` GT pixels,
        cached per board."""
        if board not in self._solder_coords_cache:
            _, y = self._board(board)
            self._solder_coords_cache[board] = np.argwhere(y == labels.CLASSES["solder"])
        return self._solder_coords_cache[board]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        oversample = (self.solder_oversample_prob > 0 and self._solder_boards
                     and random.random() < self.solder_oversample_prob)

        if oversample:
            board = random.choice(self._solder_boards)
            x, y = self._board(board)
            c, h, w = x.shape
            ph, pw = min(self.patch, h), min(self.patch, w)
            coords = self._solder_coords(board)
            r, cc = coords[random.randrange(len(coords))]
            # jitter so the solder pixel lands anywhere in the patch, not
            # always dead-center -- avoids the model learning a positional
            # shortcut instead of the actual visual pattern.
            top = r - random.randint(0, ph - 1)
            left = cc - random.randint(0, pw - 1)
            top = max(0, min(top, h - ph))
            left = max(0, min(left, w - pw))
        else:
            board = random.choice(self.boards)
            x, y = self._board(board)
            c, h, w = x.shape
            ph, pw = min(self.patch, h), min(self.patch, w)
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)

        xp = x[:, top:top + ph, left:left + pw]
        yp = y[top:top + ph, left:left + pw]

        if ph < self.patch or pw < self.patch:
            # board smaller than the patch on at least one axis -- pad with
            # background (class 0 / all-zero input) rather than resampling.
            xpad = np.zeros((c, self.patch, self.patch), dtype=np.float32)
            ypad = np.zeros((self.patch, self.patch), dtype=np.uint8)
            xpad[:, :ph, :pw] = xp
            ypad[:ph, :pw] = yp
            xp, yp = xpad, ypad

        if self.augment:
            if random.random() < 0.5:
                xp = xp[:, :, ::-1].copy()
                yp = yp[:, ::-1].copy()
            if random.random() < 0.5:
                xp = xp[:, ::-1, :].copy()
                yp = yp[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                xp = np.rot90(xp, k, axes=(1, 2)).copy()
                yp = np.rot90(yp, k).copy()

        return torch.from_numpy(xp), torch.from_numpy(yp.astype(np.int64))


def class_weights(boards: list[str], root: Path = ROOT,
                  max_weight: float | None = None) -> torch.Tensor:
    """Inverse-frequency class weights from the training split's GT pixel
    counts. Printed by train.py at startup so the imbalance is visible, not
    silently corrected -- background/soldermask/component dominate over
    e.g. silkscreen/trace by orders of magnitude (see README class-legend
    context).

    max_weight: if given, clip weights to this ceiling (post-normalization,
    where the mean weight is 1.0) before returning. Without a cap, a very
    rare class (e.g. `solder`) can end up with a weight so far above its
    spatial neighbors (e.g. `silkscreen`) that CE's gradient pulls the
    decision boundary toward it whenever a small/ambiguous pixel sits near
    both -- observed empirically as asymmetric solder<-silkscreen
    confusion, see docs/STATUS.md."""
    counts = np.zeros(labels.NUM_CLASSES, dtype=np.int64)
    for board in boards:
        y = np.asarray(Image.open(root / "out" / board / f"{board}_gt.png"))
        u, c = np.unique(y, return_counts=True)
        for cid, n in zip(u, c):
            if cid < labels.NUM_CLASSES:
                counts[cid] += n
    counts = np.maximum(counts, 1)  # avoid div-by-zero for absent classes
    weights = 1.0 / np.sqrt(counts.astype(np.float64))
    weights = weights / weights.mean()
    if max_weight is not None:
        weights = np.clip(weights, None, max_weight)
    return torch.from_numpy(weights.astype(np.float32))
