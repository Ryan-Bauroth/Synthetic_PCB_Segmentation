#!/usr/bin/env python3
"""Run gerber2blend on boards/<name>/ to produce fab/<name>.blend with copper,
soldermask, silkscreen, substrate -- and, with EFFECTS.SOLDER enabled, real
solder geometry grown from the paste layers.

Steps:
  1. drop a <name>.kicad_pro marker so gerber2blend derives the blend filename
  2. generate a default blendcfg.yaml (gerber2blend -g) if absent
  3. flip EFFECTS.SOLDER -> True (needs F/B_Paste gerbers)
  4. run gerber2blend
  5. confirm fab/<name>.blend exists

Usage:  python3 03_run_gerber2blend.py --board <name> [--debug]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOARDS_DIR = ROOT / "boards"


def gerber2blend_bin() -> str:
    p = shutil.which("gerber2blend")
    if p:
        return p
    cand = Path.home() / ".local" / "bin" / "gerber2blend"
    if cand.exists():
        return str(cand)
    sys.exit("gerber2blend not found. Run 00_install.sh (pipx install).")


def enable_solder(cfg: Path) -> None:
    """Flip the single-line `SOLDER: False` under EFFECTS to True."""
    text = cfg.read_text()
    new, n = re.subn(r"(?m)^(\s*)SOLDER:\s*False\b", r"\1SOLDER: True", text)
    if n:
        cfg.write_text(new)
        print("  enabled EFFECTS.SOLDER (solder grown from paste layers)")
    else:
        print("  note: could not find 'SOLDER: False' to flip; leaving as-is")


def set_soldermask(cfg: Path, color: str) -> None:
    """Set the SOLDERMASK colour. The template puts the value on the line after
    the `SOLDERMASK:` key (block-mapping style), so patch that token."""
    text = cfg.read_text()
    colors = "Black|White|Green|Blue|Red"
    # value-on-next-line form
    new, n = re.subn(rf"(?m)^(\s*SOLDERMASK:\s*\n\s*)(?:{colors})\b",
                     rf"\g<1>{color}", text)
    if not n:  # inline form fallback
        new, n = re.subn(rf"(?m)^(\s*SOLDERMASK:\s*)(?:{colors})\b",
                         rf"\g<1>{color}", text)
    if n:
        cfg.write_text(new)
        print(f"  set SOLDERMASK: {color}")
    else:
        print(f"  note: could not set SOLDERMASK to {color}; leaving as-is")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--soldermask", default="Green",
                    choices=["Black", "White", "Green", "Blue", "Red"],
                    help="soldermask colour baked into the board texture")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    proj = BOARDS_DIR / args.board
    fab = proj / "fab"
    if not fab.is_dir() or not any(fab.glob("*.gbr")):
        sys.exit(f"No Gerbers in {fab}. Run 02_fetch_gerbers.py first.")

    # 1. marker file so the output is named <name>.blend
    kicad_pro = proj / f"{args.board}.kicad_pro"
    if not kicad_pro.exists():
        kicad_pro.write_text("{}\n")  # gerber2blend only reads the filename

    g2b = gerber2blend_bin()

    # 2. generate default blendcfg.yaml if missing (-R copies template & exits)
    cfg = proj / "blendcfg.yaml"
    if not cfg.exists():
        print("Generating default blendcfg.yaml (gerber2blend -R)")
        subprocess.run([g2b, "-R"], cwd=proj, check=True)

    # 3. enable solder generation + set soldermask colour
    if cfg.exists():
        enable_solder(cfg)
        set_soldermask(cfg, args.soldermask)

    # 4. run
    cmd = [g2b] + (["-d"] if args.debug else [])
    print(f"\nRunning {' '.join(cmd)}  (cwd={proj})")
    env = dict(os.environ)
    # gerber2blend forks a multiprocessing pool (fork forced via .pth so spawn
    # children inherit config); allow forking a process that has bpy/ObjC loaded.
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    proc = subprocess.run(cmd, cwd=proj, env=env)
    if proc.returncode != 0:
        sys.exit(f"gerber2blend exited {proc.returncode}")

    # 5. verify output
    blend = fab / f"{args.board}.blend"
    if not blend.exists():
        found = list(fab.glob("*.blend"))
        if found:
            blend = found[0]
            print(f"Note: blend produced as {blend.name}")
        else:
            sys.exit("gerber2blend finished but no .blend was produced.")
    print(f"\nOK -> {blend.relative_to(ROOT)}")
    print(f"Next:  ./.venv311/bin/python 04_inspect_blend.py --board {args.board}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
