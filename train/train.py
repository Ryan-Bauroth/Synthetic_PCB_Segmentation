#!/usr/bin/env python3
"""Train the U-Net (train/model.py) on the generated PCB dataset
(out/manifest.json) to reproduce the material GT (labels.py, 8 classes)
from the 6-channel input (R, G, B, IR, UV, height -- train/dataset.py).

Run:  ./.venv311/bin/python train/train.py
      ./.venv311/bin/python train/train.py --val-boards scalenode,cvbs-mipi-bridge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import labels  # noqa: E402
from dataset import PCBSegDataset, NUM_CHANNELS, board_split, class_weights  # noqa: E402
from losses import soft_dice_loss  # noqa: E402
from metrics import ConfusionMatrix  # noqa: E402
from model import UNet  # noqa: E402


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patches-per-epoch", type=int, default=2000)
    ap.add_argument("--val-patches", type=int, default=400)
    ap.add_argument("--val-boards", type=str, default=None,
                    help="comma-separated board names; default: seeded random split")
    ap.add_argument("--n-val", type=int, default=2,
                    help="number of boards to hold out when --val-boards is not given")
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--dice-weight", type=float, default=0.0,
                    help="weight of soft Dice loss added to CE (background "
                         "excluded from the Dice average); 0 disables Dice "
                         "(default: off -- tried, didn't fix `solder`, see "
                         "docs/STATUS.md)")
    ap.add_argument("--solder-oversample-prob", type=float, default=0.0,
                    help="fraction of TRAINING patches forced to contain a "
                         "solder GT pixel (jittered, not centered); 0 "
                         "disables oversampling. Val set is never oversampled.")
    ap.add_argument("--max-class-weight", type=float, default=None,
                    help="cap inverse-frequency CE class weights at this "
                         "ceiling (mean weight is 1.0); None disables "
                         "capping. Prevents a very rare class (e.g. solder) "
                         "from dominating CE's gradient at the expense of "
                         "spatially adjacent classes (e.g. silkscreen), see "
                         "docs/STATUS.md.")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    args = ap.parse_args()

    val_boards = args.val_boards.split(",") if args.val_boards else None
    train_boards, val_boards = board_split(ROOT, val_boards=val_boards, n_val=args.n_val)
    print(f"Train boards ({len(train_boards)}): {train_boards}")
    print(f"Val boards   ({len(val_boards)}): {val_boards}")
    if not val_boards:
        sys.exit("No val boards -- need at least 2 verified boards in out/manifest.json.")

    device = pick_device()
    print(f"Device: {device}")

    train_ds = PCBSegDataset(train_boards, ROOT, args.patch_size, args.patches_per_epoch,
                             augment=True, solder_oversample_prob=args.solder_oversample_prob)
    val_ds = PCBSegDataset(val_boards, ROOT, args.patch_size, args.val_patches, augment=False)
    print(f"Solder-oversample prob (train only): {args.solder_oversample_prob} "
          f"({len(train_ds._solder_boards)}/{len(train_boards)} train boards have solder pixels)")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    uncapped = class_weights(train_boards, ROOT)
    weights = class_weights(train_boards, ROOT, max_weight=args.max_class_weight).to(device)
    print(f"Class weights (inverse-frequency, training split; "
          f"max_class_weight={args.max_class_weight}):")
    for cid, (w_raw, w) in enumerate(zip(uncapped.tolist(), weights.cpu().tolist())):
        flag = "  (capped)" if args.max_class_weight is not None and w_raw > w else ""
        print(f"  {cid} {labels.ID_TO_NAME.get(cid, '?'):<11} {w_raw:.3f} -> {w:.3f}{flag}")
    print(f"Dice loss weight: {args.dice_weight} "
          f"(background excluded from the Dice average)")

    model = UNet(in_channels=NUM_CHANNELS, num_classes=labels.NUM_CLASSES,
                base=args.base_channels).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} [train]", unit="batch")
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            ce = criterion(logits, y)
            dice = soft_dice_loss(logits, y, labels.NUM_CLASSES, ignore_index=0)
            loss = ce + args.dice_weight * dice
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running_loss / len(train_ds)

        model.eval()
        cm = ConfusionMatrix(labels.NUM_CLASSES)
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"epoch {epoch}/{args.epochs} [val]", unit="batch"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                ce = criterion(logits, y)
                dice = soft_dice_loss(logits, y, labels.NUM_CLASSES, ignore_index=0)
                val_loss += (ce + args.dice_weight * dice).item() * x.size(0)
                cm.update(logits.argmax(1), y)
        val_loss /= len(val_ds)
        miou = cm.mean_iou()
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mIoU={miou:.4f}")

        torch.save({"model": model.state_dict(), "epoch": epoch, "val_miou": miou,
                   "train_boards": train_boards, "val_boards": val_boards,
                   "in_channels": NUM_CHANNELS, "base_channels": args.base_channels,
                   "dice_weight": args.dice_weight,
                   "solder_oversample_prob": args.solder_oversample_prob,
                   "max_class_weight": args.max_class_weight},
                  args.out_dir / "unet_last.pt")
        if miou > best_miou:
            best_miou = miou
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_miou": miou,
                       "train_boards": train_boards, "val_boards": val_boards,
                       "in_channels": NUM_CHANNELS, "base_channels": args.base_channels,
                       "dice_weight": args.dice_weight,
                   "solder_oversample_prob": args.solder_oversample_prob,
                   "max_class_weight": args.max_class_weight},
                      args.out_dir / "unet_best.pt")
            print(f"  new best val_mIoU={miou:.4f} -> {args.out_dir / 'unet_best.pt'}")

    print("\nFinal validation report:")
    print(cm.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
