#!/usr/bin/env python3
"""Success proof: confirm the 5 band images (R/G/B/IR/UV) and the ground-truth
mask are all pixel-aligned (same dimensions -- guaranteed by construction
since 07_render_mitsuba.py renders them all through the same mesh/camera/UV,
but verified here rather than assumed) and that the expected material
classes appear in the GT.

Files (per board):  <board>_{R,G,B,IR,UV}.png, <board>_height.png, and
<board>_gt.png

Run:  ./.venv311/bin/python 08_verify_alignment.py --board <name>
Exit 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import labels  # noqa: E402
import spectral  # noqa: E402

# classes we consider mandatory for a valid PCB sample (solder is best-effort)
REQUIRED = {"copper", "fiberglass", "soldermask", "silkscreen", "trace"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    args = ap.parse_args()

    outdir = ROOT / "out" / args.board
    stem = args.board
    band_paths = {b: outdir / f"{stem}_{b}.png" for b in spectral.BANDS}
    gt_path = outdir / f"{stem}_gt.png"
    height_path = outdir / f"{stem}_height.png"
    for p in [*band_paths.values(), gt_path, height_path]:
        if not p.exists():
            sys.exit(f"{p} missing. Run 07_render_mitsuba.py first.")

    sizes = {b: Image.open(p).size for b, p in band_paths.items()}
    sizes["gt"] = Image.open(gt_path).size
    sizes["height"] = Image.open(height_path).size
    ok = True
    if len(set(sizes.values())) != 1:
        print(f"FAIL dimensions differ: {sizes}")
        ok = False
    else:
        print(f"PASS all 7 images share dimensions: {next(iter(sizes.values()))}")

    ids = np.asarray(Image.open(gt_path))
    present = {labels.ID_TO_NAME[i] for i in np.unique(ids)
               if i in labels.ID_TO_NAME}
    print("GT classes present:", sorted(present))
    missing = REQUIRED - present
    if missing:
        print(f"FAIL required classes missing: {sorted(missing)}")
        ok = False
    else:
        print("PASS all required classes present")
    if "solder" not in present:
        print("  note: no 'solder' pixels (paste layer absent or EFFECTS.SOLDER"
              " produced none) -- acceptable.")

    print(f"\nSample: {', '.join(p.name for p in band_paths.values())}  +  "
          f"{height_path.name}  +  {gt_path.name}")
    print("Result:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
