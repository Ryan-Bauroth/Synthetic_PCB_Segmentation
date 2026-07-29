#!/usr/bin/env python3
"""Phase 0 environment check. Read-only: locates every tool the pipeline needs,
prints a PASS/FAIL table, and writes discovered paths to env.json for later
scripts to consume.

Renderer note: we use the pip `bpy` module inside .venv311 (no Blender.app).

Usage:  python3 01_verify_env.py     (any Python is fine for this check)
Exit code 0 if all *required* tools pass, else 1.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_JSON = ROOT / "env.json"
VENV_PY = ROOT / ".venv311" / "bin" / "python"

# kicad-cli is not symlinked onto PATH by the macOS cask by default.
KICAD_CLI_CANDIDATES = [
    "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
]


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return (out.stdout + out.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def check_cmd(name: str, version_args: list[str], required: bool,
              candidates: list[str] | None = None) -> dict:
    path = shutil.which(name)
    if not path and candidates:
        for c in candidates:
            if Path(c).exists():
                path = c
                break
    if not path:
        return {"required": required, "found": False, "path": None,
                "detail": "not found"}
    detail = run([path, *version_args]).splitlines()[0] if version_args else ""
    return {"required": required, "found": True, "path": path, "detail": detail}


def check_bpy_venv() -> dict:
    """Confirm .venv311 exists and can import bpy; report the Blender version."""
    if not VENV_PY.exists():
        return {"required": True, "found": False, "path": None,
                "detail": ".venv311 missing (run 00_install.sh)"}
    out = run([str(VENV_PY), "-c",
               "import bpy,sys; print(bpy.app.version_string)"])
    ok = out and "<error" not in out and "Traceback" not in out
    return {"required": True, "found": bool(ok), "path": str(VENV_PY),
            "detail": (f"bpy {out}" if ok else f"import failed: {out[:50]}")}


def check_mitsuba() -> dict:
    """Confirm .venv311 can import mitsuba and report an available variant."""
    if not VENV_PY.exists():
        return {"required": True, "found": False, "path": None,
                "detail": ".venv311 missing (run 00_install.sh)"}
    out = run([str(VENV_PY), "-c",
               "import mitsuba as mi; print(mi.__version__, "
               "'|'.join(mi.variants()))"])
    ok = out and "<error" not in out and "Traceback" not in out
    return {"required": True, "found": bool(ok), "path": str(VENV_PY),
            "detail": (f"mitsuba {out}" if ok else f"import failed: {out[:80]}")}


def check_pip_module(module: str) -> dict:
    if not VENV_PY.exists():
        return {"required": True, "found": False, "path": None,
                "detail": ".venv311 missing (run 00_install.sh)"}
    out = run([str(VENV_PY), "-c",
               f"import {module}; print(getattr({module}, '__version__', 'ok'))"])
    ok = out and "<error" not in out and "Traceback" not in out
    return {"required": True, "found": bool(ok), "path": str(VENV_PY),
            "detail": (out if ok else f"import failed: {out[:60]}")}


def main() -> int:
    checks: dict[str, dict] = {
        "bpy (.venv311)": check_bpy_venv(),
        "mitsuba (.venv311)": check_mitsuba(),
        "trimesh (.venv311)": check_pip_module("trimesh"),
        "cascadio (.venv311)": check_pip_module("cascadio"),
        "kicad-cli": check_cmd("kicad-cli", ["version"], required=True,
                               candidates=KICAD_CLI_CANDIDATES),
        "gerber2blend": check_cmd("gerber2blend", ["--help"], required=True),
        "gerbv": check_cmd("gerbv", ["--version"], required=True),
        "inkscape": check_cmd("inkscape", ["--version"], required=True),
        "python3.11": check_cmd("python3.11", ["--version"], required=True),
        "java": check_cmd("java", ["-version"], required=False),
    }

    print(f"{'TOOL':<16}{'STATUS':<8}{'DETAIL'}")
    print("-" * 72)
    all_required_ok = True
    for name, info in checks.items():
        ok = info["found"]
        if info["required"] and not ok:
            all_required_ok = False
        status = "PASS" if ok else ("FAIL" if info["required"] else "warn")
        detail = info.get("detail", "")
        if info.get("path"):
            detail = f"{info['path']}  {detail}".strip()
        print(f"{name:<16}{status:<8}{detail[:56]}")

    ENV_JSON.write_text(json.dumps(checks, indent=2))
    print(f"\nWrote {ENV_JSON.name}")

    if not all_required_ok:
        print("\nSome required tools are missing. Run:  bash 00_install.sh")
        return 1
    print("\nAll required tools present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
