#!/usr/bin/env bash
# Phase 0 installer for the PCB synthetic-material pipeline (macOS / Homebrew).
#
# Design note: gerber2blend depends on the *pip* package `bpy==4.5.4`
# (Blender-as-a-Python-module), so we do NOT install the Blender.app cask.
# Instead we make a Python 3.11 venv with `bpy`, and our mesh-export step
# (05) runs as a plain `python` program that `import bpy`. gerber2blend
# itself is installed via pipx (its own isolated bpy).
#
# The renderer is Mitsuba3 (pure-Python wheel, no bpy needed) -- everything
# from 06 onward (rendering, STL export) is plain Python in the same venv.
#
# Idempotent: each step checks first and skips if present.
# The heavy/uncertain steps (KiCad + Inkscape casks, bpy wheel, gerber2blend)
# are the ones flagged as risks in the plan. Run deliberately.
#
# Usage:  bash 00_install.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv311"

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

if ! have brew; then
  echo "Homebrew not found. Install it first: https://brew.sh" >&2
  exit 1
fi

# --- CLI deps used by gerber2blend ------------------------------------------
say "gerbv (Gerber rasterizer)"
have gerbv && echo "already installed" || brew install gerbv

say "imagemagick + freetype (gerber2blend's 'wand' image backend)"
brew list imagemagick >/dev/null 2>&1 && echo "imagemagick present" \
  || brew install imagemagick freetype

say "inkscape (mesh-fixup step) -- large download (~1GB)"
have inkscape && echo "already installed" || brew install --cask inkscape

say "python@3.11 + pipx"
have python3.11 && echo "python3.11 present" || brew install python@3.11
have pipx       && echo "pipx present"       || brew install pipx
pipx ensurepath || true

# --- KiCad (kicad-cli for Gerber export; pcbnew API for later phases) --------
say "KiCad (kicad-cli) -- large ~2GB"
if have kicad-cli || [ -x "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli" ]; then
  echo "already installed"
else
  brew install --cask kicad
fi

# --- Python 3.11 venv with bpy (our renderer) -------------------------------
say "Python 3.11 venv with bpy==4.5.4 + pillow + numpy  (large bpy wheel ~250MB)"
if [ ! -x "$VENV/bin/python" ]; then
  python3.11 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
# bpy wheel version must match the Blender line gerber2blend targets (4.5.x).
"$VENV/bin/python" -m pip install "bpy==4.5.4" pillow numpy || \
  "$VENV/bin/python" -m pip install "bpy~=4.5" pillow numpy

# --- Mitsuba3 renderer + mesh/STEP tooling (pure Python, same venv) ---------
say "mitsuba + trimesh + cascadio + scipy (renderer + STL/STEP export, pure Python)"
# scipy: trimesh uses it for sparse-matrix vertex-normal computation; without
# it, trimesh falls back to a slower path and prints a noisy (non-fatal)
# traceback per mesh shell -- harmless on a single-shell board, but a
# multi-component STEP (hundreds of shells) spams hundreds of them.
# tqdm: progress bars for the render loop and the batch generator.
"$VENV/bin/python" -m pip install mitsuba trimesh cascadio scipy tqdm

# --- PyTorch (U-Net training, train/) ----------------------------------------
# Pure pip wheel; arm64 macOS build includes the MPS (Metal) backend, no CUDA
# or extra system deps needed on Apple Silicon.
say "torch (U-Net training)"
"$VENV/bin/python" -m pip install torch

# --- gerber2blend -----------------------------------------------------------
say "gerber2blend (Antmicro) -- macOS unproven; watch for errors"
if have gerber2blend; then
  echo "already installed"
else
  pipx install --python "$(command -v python3.11)" \
    'git+https://github.com/antmicro/gerber2blend.git'
fi

# gerber2blend assumes it runs inside a real Blender interpreter, where bmesh /
# mathutils / gpu are preloaded. Under the pip `bpy` wheel they only register
# after `import bpy`, but gerber2blend's submodules import bmesh at top level.
# An executable .pth line preloads bpy at interpreter startup (a sitecustomize.py
# would be shadowed by Homebrew's own). Idempotent.
say "patch gerber2blend venv to preload bpy (fixes 'No module named bmesh')"
G2B_SP="$(pipx environment --value PIPX_LOCAL_VENVS 2>/dev/null)/gerber2blend/lib/python3.11/site-packages"
[ -d "$G2B_SP" ] || G2B_SP="$HOME/.local/pipx/venvs/gerber2blend/lib/python3.11/site-packages"
if [ -d "$G2B_SP" ]; then
  printf 'import bpy\n' > "$G2B_SP/zzz_bpy_preload.pth"
  # macOS defaults multiprocessing to 'spawn', so gerbv_convert's worker pool
  # loses the parent's populated config.blendcfg (KeyError 'SETTINGS'). Force
  # 'fork' so children inherit it (as on Linux, where gerber2blend is developed).
  printf "import multiprocessing; multiprocessing.set_start_method('fork', force=True)\n" \
    > "$G2B_SP/aaa_fork.pth"
  echo "wrote bpy-preload + fork .pth files to $G2B_SP"
else
  echo "WARNING: could not find gerber2blend site-packages to patch" >&2
fi

say "Done. Now run:  ./.venv311/bin/python 01_verify_env.py"
