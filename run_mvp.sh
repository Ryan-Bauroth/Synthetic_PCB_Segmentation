#!/usr/bin/env bash
# Orchestrate the full pipeline for one board: verify env, fetch Gerbers,
# STL/component precheck (gate -- BEFORE the expensive Blender step), convert,
# inspect (gate), export mesh, finalize STL, render 5 bands + GT + height via
# Mitsuba, verify alignment.
#
# Usage:  bash run_mvp.sh [board]   (default: scalenode)
# Env overrides (handy for quick low-res validation before a full-scale run):
#   MPP=0.117 SAMPLES=16 bash run_mvp.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BOARD="${1:-scalenode}"
MPP="${MPP:-0.0209}"
SAMPLES="${SAMPLES:-32}"
PY="$ROOT/.venv311/bin/python"

if [ ! -x "$PY" ]; then
  echo "Missing .venv311. Run:  bash 00_install.sh" >&2
  exit 1
fi

step() { printf '\n\033[1;35m### %s\033[0m\n' "$*"; }

step "01 verify env";        "$PY" "$ROOT/01_verify_env.py"
step "02 fetch gerbers";     "$PY" "$ROOT/02_fetch_gerbers.py" --board "$BOARD"
step "02b STL/component precheck (GATE)"
if ! "$PY" "$ROOT/02b_stl_precheck.py" --board "$BOARD"; then
  echo "Bare board -- no usable STL/component data. Stopping before the" >&2
  echo "expensive gerber2blend/imaging steps (STL/components are required" >&2
  echo "for this project)." >&2
  exit 1
fi
step "03 gerber2blend";      "$PY" "$ROOT/03_run_gerber2blend.py" --board "$BOARD"
step "04 inspect (GATE)";    "$PY" "$ROOT/04_inspect_blend.py" --board "$BOARD"
step "05 export mesh";       "$PY" "$ROOT/05_export_mesh.py" --board "$BOARD"
step "06 export STL (finalize)"; "$PY" "$ROOT/06_export_stl.py" --board "$BOARD"
step "07 render (Mitsuba)";  "$PY" "$ROOT/07_render_mitsuba.py" --board "$BOARD" \
                               --mpp "$MPP" --samples "$SAMPLES"
step "08 verify alignment";  "$PY" "$ROOT/08_verify_alignment.py" --board "$BOARD"

echo
echo "Done. Outputs in out/$BOARD/:"
echo "  ${BOARD}_{R,G,B,IR,UV}.png  (5 band images)"
echo "  ${BOARD}_gt.png             (ground-truth mask)"
echo "  boards/$BOARD/${BOARD}.stl  (populated STL)"
