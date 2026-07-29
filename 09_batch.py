#!/usr/bin/env python3
"""Batch dataset generator: runs the full pipeline across multiple boards.

Three phases, in order, run with boards processed CONCURRENTLY within each
phase (--workers, default 4; each unit of work is a blocking subprocess
call, so a thread pool is sufficient -- no need for multiprocessing):

  Phase 0 (NEW): fetch Gerbers -> STL/component PRECHECK (02b_stl_precheck.py).
                 STL + populated 3D component data is REQUIRED for this
                 project (the height channel, the component/trace GT
                 classes all depend on it) -- a board that comes back bare
                 is DROPPED here, before the expensive Blender
                 (gerber2blend) conversion or Mitsuba imaging ever runs for
                 it. This is a real reordering vs. the old single-phase
                 design: 02b_stl_precheck.py only needs the cached
                 .kicad_pcb, so the populated/bare verdict is available
                 much earlier than the pipeline used to check it.
  Phase 1 (per surviving board): gerber2blend -> gate -> export mesh
                 -> finalize STL (06_export_stl.py reuses Phase 0's cached
                 STEP/STL + anchor bounds instead of re-tessellating the
                 full populated STEP scene a second time).
  Phase 2 (per surviving board): render 5 bands + GT + height (Mitsuba)
                 -> verify alignment.

A board that fails Phase 1's core geometry pipeline (convert/mesh-export) is
marked failed and skipped in Phase 2 -- rendering needs that mesh.

Writes out/manifest.json summarizing every board's result, including a
`dropped_boards` entry (board -> reason) for boards that never made it past
Phase 0.

Run:
  ./.venv311/bin/python 09_batch.py                       # all registered boards
  ./.venv311/bin/python 09_batch.py --boards jetson-nano-baseboard,scalenode
  ./.venv311/bin/python 09_batch.py --soldermask Black --mpp 0.117 --samples 16
  ./.venv311/bin/python 09_batch.py --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from importlib import import_module  # noqa: E402

fetch_mod = import_module("02_fetch_gerbers")
import spectral  # noqa: E402

PY = str(ROOT / ".venv311" / "bin" / "python")


def run_step(script: str, board: str, extra: list[str] | None = None) -> tuple[bool, str]:
    cmd = [PY, str(ROOT / script), "--board", board, *(extra or [])]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    return ok, "\n".join(tail[-15:])


def run_parallel(items: list[str], worker_fn, workers: int, desc: str, on_result=None) -> dict:
    """Run worker_fn(item) for each item using a bounded thread pool (each
    unit of work is a blocking subprocess call, so threads -- not
    multiprocessing -- are enough), with a live tqdm bar updated as results
    land (order is non-deterministic across items; each item's own step
    sequence is still strictly ordered)."""
    results: dict = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker_fn, item): item for item in items}
        bar = tqdm(total=len(items), desc=desc, unit="board")
        for fut in cf.as_completed(futures):
            item = futures[fut]
            res = fut.result()
            results[item] = res
            if on_result:
                bar.write(on_result(item, res))
            bar.update(1)
        bar.close()
    return results


def precheck_one(board: str, log: list[str]) -> dict:
    """Phase 0: fetch Gerbers + the STL/component precheck ONLY -- no
    gerber2blend/Blender. Returns populated=True only if the board has
    usable STL/component data worth spending the rest of the pipeline on."""
    result = {"board": board, "fetch_ok": False, "populated": False,
             "stl": None, "errors": []}

    ok, tail = run_step("02_fetch_gerbers.py", board)
    result["fetch_ok"] = ok
    if not ok:
        result["errors"].append(f"02_fetch_gerbers.py: {tail}")
        log.append(f"[{board}] fetch FAILED\n{tail}")
        return result

    # 02b_stl_precheck.py exits 1 for a bare (but successfully-checked) board
    # -- that's an expected outcome, not a crash; only a missing report means
    # something actually went wrong (e.g. kicad-cli itself failed).
    run_step("02b_stl_precheck.py", board)
    report_path = ROOT / "boards" / board / "stl_report.json"
    if report_path.exists():
        result["stl"] = json.loads(report_path.read_text())
        result["populated"] = bool(result["stl"].get("populated"))
    else:
        result["errors"].append("02b_stl_precheck.py: no stl_report.json written")
        log.append(f"[{board}] STL precheck crashed (no report written)")
    return result


def phase1_finalize(board: str, soldermask: str, log: list[str]) -> dict:
    """gerber2blend -> gate -> mesh export -> STL finalize, for a board
    Phase 0 already confirmed is populated. 06_export_stl.py automatically
    reuses Phase 0's cached STEP/STL + anchor bounds."""
    result = {"board": board, "soldermask": soldermask, "core_ok": False,
             "stl": None, "errors": []}

    steps = [
        ("03_run_gerber2blend.py", ["--soldermask", soldermask]),
        ("04_inspect_blend.py", None),
        ("05_export_mesh.py", None),
    ]
    for script, extra in steps:
        ok, tail = run_step(script, board, extra)
        if not ok:
            result["errors"].append(f"{script}: {tail}")
            log.append(f"[{board}] FAILED at {script}\n{tail}")
            return result  # core pipeline broken -> can't proceed for this board
    result["core_ok"] = True

    ok, tail = run_step("06_export_stl.py", board)  # finalize; reuses Phase 0's cache
    if not ok:
        result["errors"].append(f"06_export_stl.py: {tail}")
        log.append(f"[{board}] STL finalize failed (non-fatal, imaging still proceeds)\n{tail}")
    report_path = ROOT / "boards" / board / "stl_report.json"
    if report_path.exists():
        result["stl"] = json.loads(report_path.read_text())
    return result


def phase2_imaging(board: str, mpp: float, samples: int, gt_samples: int, log: list[str]) -> dict:
    result = {"board": board, "render_ok": False, "verify_ok": False,
             "images": [], "classes": [], "errors": []}

    ok, tail = run_step("07_render_mitsuba.py", board,
                        ["--mpp", str(mpp), "--samples", str(samples),
                         "--gt-samples", str(gt_samples)])
    result["render_ok"] = ok
    if not ok:
        result["errors"].append(f"07_render_mitsuba.py: {tail}")
        log.append(f"[{board}] render FAILED\n{tail}")
        return result

    outdir = ROOT / "out" / board
    result["images"] = sorted(p.name for p in outdir.glob(f"{board}_*.png"))

    ok, tail = run_step("08_verify_alignment.py", board)
    result["verify_ok"] = ok
    if not ok:
        result["errors"].append(f"08_verify_alignment.py: {tail}")
        log.append(f"[{board}] alignment verification FAILED\n{tail}")

    gt_path = outdir / f"{board}_gt.png"
    if gt_path.exists():
        import numpy as np
        from PIL import Image
        import labels
        ids = np.asarray(Image.open(gt_path))
        result["classes"] = sorted(labels.ID_TO_NAME[i] for i in np.unique(ids)
                                   if i in labels.ID_TO_NAME)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default=None,
                    help="comma-separated board keys (default: all registered)")
    ap.add_argument("--soldermask", default="Green",
                    choices=["Black", "White", "Green", "Blue", "Red"])
    ap.add_argument("--mpp", type=float, default=0.0209)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--gt-samples", type=int, default=4,
                    help="sample count for the GT pass only -- default matches "
                         "prior behavior; higher values reduce anti-aliased "
                         "class-boundary noise in GT, see docs/STATUS.md")
    ap.add_argument("--workers", type=int, default=4,
                    help="boards processed concurrently per phase")
    args = ap.parse_args()

    boards = (args.boards.split(",") if args.boards
             else list(fetch_mod.BOARDS.keys()))
    unknown = [b for b in boards if b not in fetch_mod.BOARDS]
    if unknown:
        sys.exit(f"Unknown board(s) {unknown}. Registered: {list(fetch_mod.BOARDS.keys())}")

    log: list[str] = []
    manifest: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "soldermask": args.soldermask, "mpp": args.mpp,
                      "samples": args.samples, "boards": {}}

    print(f"Batch: {len(boards)} candidate board(s) -> {boards}")

    # ---- Phase 0: STL/component precheck -- gate BEFORE the expensive steps --
    def fmt_precheck(board: str, r: dict) -> str:
        if r["populated"]:
            stl = r["stl"]
            return (f"  {board}: POPULATED ({stl['n_shells']} shells, "
                    f"{stl['z_extent_ratio'] * 100:.1f}% Z-ratio) -- proceeding")
        if r["stl"]:
            return (f"  {board}: BARE ({r['stl']['z_extent_ratio'] * 100:.1f}% Z-ratio) "
                    f"-- dropping, no usable STL/component data")
        return f"  {board}: precheck FAILED outright -- dropping ({r['errors']})"

    phase0 = run_parallel(boards, lambda b: precheck_one(b, log), args.workers,
                         "Phase 0: STL/component precheck", fmt_precheck)

    survivors = [b for b in boards if phase0[b]["populated"]]
    dropped_boards: dict[str, str] = {}
    for b in boards:
        if b in survivors:
            continue
        if phase0[b]["stl"]:
            dropped_boards[b] = (f"bare board (Z-ratio "
                                 f"{phase0[b]['stl']['z_extent_ratio'] * 100:.1f}% < 3%)")
        else:
            dropped_boards[b] = "fetch/precheck failed: " + "; ".join(phase0[b]["errors"])

    print(f"\n{len(survivors)}/{len(boards)} boards populated -- proceeding with "
          f"gerber2blend + imaging for those only.")
    if dropped_boards:
        print(f"Dropped (no usable STL/component data): {list(dropped_boards.keys())}")

    # ---- Phase 1: gerber2blend/mesh/STL finalize for survivors only ----------
    def fmt_phase1(board: str, r: dict) -> str:
        if r["core_ok"]:
            return f"  {board}: core pipeline OK"
        return f"  {board}: FAILED (core pipeline) -- skipping imaging"

    phase1_results = run_parallel(survivors, lambda b: phase1_finalize(b, args.soldermask, log),
                                  args.workers, "Phase 1: gerber2blend + STL finalize", fmt_phase1)

    ready = [b for b in survivors if phase1_results[b]["core_ok"]]
    core_failed = [b for b in survivors if b not in ready]

    # ---- Phase 2: imaging for boards whose core pipeline succeeded -----------
    def fmt_phase2(board: str, r: dict) -> str:
        status = "OK" if r["render_ok"] and r["verify_ok"] else "PROBLEM"
        return f"  {board}: {status} -- {len(r['images'])} images, classes={r['classes']}"

    phase2_results = run_parallel(ready, lambda b: phase2_imaging(b, args.mpp, args.samples, args.gt_samples, log),
                                  args.workers, "Phase 2: imaging", fmt_phase2)

    # ---- manifest --------------------------------------------------------------
    for board in survivors:
        p1 = phase1_results[board]
        p2 = phase2_results.get(board)
        manifest["boards"][board] = {
            "soldermask": args.soldermask,
            "core_ok": p1["core_ok"],
            "stl": p1["stl"],
            "imaged": p2 is not None,
            "render_ok": p2["render_ok"] if p2 else False,
            "verify_ok": p2["verify_ok"] if p2 else False,
            "images": p2["images"] if p2 else [],
            "classes": p2["classes"] if p2 else [],
            "errors": p1["errors"] + (p2["errors"] if p2 else []),
        }
    manifest["dropped_boards"] = dropped_boards

    manifest_path = ROOT / "out" / "manifest.json"
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {manifest_path.relative_to(ROOT)}")
    ok_boards = [b for b in survivors if manifest["boards"][b]["render_ok"]
                and manifest["boards"][b]["verify_ok"]]
    print(f"\nSummary: {len(ok_boards)}/{len(boards)} candidate boards fully generated "
          f"(imaged + verified). {len(dropped_boards)} dropped at precheck, "
          f"{len(core_failed)} failed core pipeline.")
    if dropped_boards:
        print(f"Dropped at precheck (no usable STL/component data): {dropped_boards}")
    if core_failed:
        print(f"Failed core pipeline (skipped imaging): {core_failed}")

    if log:
        print(f"\n{len(log)} issue(s) logged -- see out/manifest.json for details.")
    return 0 if len(ok_boards) == len(survivors) else 1


if __name__ == "__main__":
    raise SystemExit(main())
