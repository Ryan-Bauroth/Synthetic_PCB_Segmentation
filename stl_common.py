"""Shared helpers for 02b_stl_precheck.py and 06_export_stl.py. Split out so
neither script needs to import the other (numbered-script names aren't
valid Python import targets without importlib gymnastics)."""

from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import trimesh

MM_PER_STEP_UNIT = 1000.0  # KiCad STEP export is in metres

# --- 3D-model recovery: many boards reference 3D models that kicad-cli's STEP
# exporter fails to resolve, NOT because the geometry doesn't exist, but for
# several distinct, fixable reasons found by inspecting real boards' .kicad_pcb
# files directly:
#
# 1. Legacy `.wrl` (VRML) references. kicad-cli's STEP exporter refuses these
#    outright ("Cannot use VRML models when exporting to non-mesh formats"),
#    based purely on the reference string's extension -- regardless of whether
#    the file exists or what format it's actually in. But KiCad's own bundled
#    global library (KISYS3DMOD / KICADn_3DMODEL_DIR -- confirmed empirically
#    to already auto-resolve with NO env var setup needed on this machine, a
#    single modern KiCad install) has long since replaced its .wrl models with
#    .step equivalents AT THE SAME PATH/basename, just the newer extension --
#    these boards were simply saved before that migration. Fix: patch a SCRATCH
#    copy of the .kicad_pcb, rewriting `.wrl` -> `.step` wherever a real .step
#    file exists at that path in the bundled library. Verified on a real board
#    (libresolar-uext-esp32): recovered mesh grew from 1,964 -> 10,964 faces
#    (5.6x), including the board's main ESP32 module, previously entirely
#    missing. This is NOT a niche case -- a scan across the registry found it
#    affects the majority of boards, some very heavily (e.g. cynthion-hardware:
#    273/283 model references were .wrl-with-a-real-.step-equivalent).
#
# 2. `${SPARKFUN_KICAD_LIBRARY}` references. This is a real, public, cloneable
#    GitHub repo (sparkfun/SparkFun-KiCad-Libraries) that just isn't installed/
#    configured -- but the library has ALSO been reorganized since these board
#    files were saved (old path: Footprints/SparkFun-3D-models/<Category>/;
#    current: 3dmodels/<Category>.3dshapes/), so simply cloning it and setting
#    the env var isn't enough either. Fix: clone it once (cached), then build a
#    symlink shim directory reconstructing the OLD expected path layout
#    pointing at the files' real current locations, and set
#    SPARKFUN_KICAD_LIBRARY to that shim when invoking kicad-cli. Verified on
#    sparkfun-spectral-as7343: recovered mesh grew from 328 -> 11,743 faces
#    (2 -> 24 real geometries), paste-pixel component coverage 5.1% -> 54.8%,
#    visually confirmed correctly aligned (LED dome, resistor chips land
#    exactly where the real product photo shows them).
#
# 3. Refs absent from THIS machine's bundled KiCad.app library but present in
#    the full upstream `github.com/KiCad/kicad-packages3d` repo -- found
#    2026-07-27 while investigating a batch of ~50 still-"unfixable" refs
#    left over after fixes 1-2: the local bundled library
#    (kicad_bundled_3d_lib()) turns out to be a PARTIAL snapshot, not the
#    complete official library (confirmed via GitHub API: e.g.
#    `SOIJ-8_5.3x5.3mm_P1.27mm.step`, referenced by urti-mainboard, is
#    genuinely absent locally but exists at the exact same
#    `<Category>.3dshapes/<name>.step` path upstream). Fix: clone the
#    upstream repo (cached, same pattern as the SparkFun clone), and for any
#    ref unresolved locally, check it there too -- exact match, same
#    zero-ambiguity confidence as fixes 1-2, so also applied unconditionally.
#
# 4. Refs genuinely RENAMED upstream between KiCad library versions (e.g.
#    `SOIC-8_5.23x5.23mm_P1.27mm` -> upstream's current
#    `SOIC-8_5.275x5.275mm_P1.27mm`, a dimension correction;
#    `JST_PH_B2B-PH-SM4-TB_1x02-1MP_P2.00mm_Vertical` -> upstream's current
#    `JST_PH_B2B-PH-K_1x02_P2.00mm_Vertical`, a variant-designation rename) --
#    also found 2026-07-27, in the same leftover batch. UNLIKE fixes 1-3,
#    this is NOT an exact match -- it's a same-category, high-similarity
#    fuzzy match (find_kicad_packages3d_fuzzy_match()), which carries a real
#    risk of silently substituting a dimensionally-different real part if
#    applied carelessly. Therefore: applied only when explicitly requested
#    (`allow_fuzzy=True`, opt-in, NOT the default), and every fuzzy
#    substitution is reported separately (`fuzzy_matched`, distinct from the
#    exact-match `wrl_fixed` count) rather than silently merged in, so it
#    stays auditable which boards got an approximate vs. exact fix. Some refs
#    (e.g. `JST_XH_B02B-XH-A_1x02_P2.50mm_Vertical`, and the custom
#    `LibreSolar.3dshapes/DTMSS-27-H` library which has no discoverable
#    public 3D-model repo at all) have no plausible match anywhere and
#    correctly stay unfixable even with fuzzy matching enabled.
#
# Fixes 1-3 recover REAL manufacturer-provided geometry with no ambiguity --
# not an approximation -- so they're applied unconditionally wherever
# detected, no opt-out needed. Fix 4 (fuzzy) is opt-in and always flagged.

WRL_MODEL_RE = re.compile(
    r'(\(model\s+"?)(\$\{[A-Z0-9_]+\}/[^")\s]+?)(\.wrl)("?)')
SPARKFUN_ENV_VAR = "SPARKFUN_KICAD_LIBRARY"
SPARKFUN_REPO = "https://github.com/sparkfun/SparkFun-KiCad-Libraries.git"
# old library layout (referenced by board files) -> new layout (what the repo
# actually ships today, since reorganized) -- both are `Category` names,
# mapped 1:1 by directory listing, not hardcoded per-category.
SPARKFUN_OLD_PREFIX = "Footprints/SparkFun-3D-models"
SPARKFUN_NEW_SUBDIR = "3dmodels"

KICAD_PACKAGES3D_REPO = "https://github.com/KiCad/kicad-packages3d.git"
# Env var we control ourselves (not a real KiCad-recognized name) -- used to
# point resolved-via-upstream-repo refs at our clone, instead of whatever the
# ref's original env var (e.g. ${KICAD6_3DMODEL_DIR}) would otherwise resolve
# to on this machine (the local bundled snapshot, which is what's missing
# the file in the first place). Keeps this fix surgical: only refs we
# actually rewrite to use this var are affected; every other ref's
# resolution is untouched.
KICAD_PACKAGES3D_ENV_VAR = "PCBDATASET_KICAD_PACKAGES3D"


def kicad_bundled_3d_lib() -> Path | None:
    """KiCad's own bundled global 3D model library -- what KISYS3DMOD /
    KICADn_3DMODEL_DIR already auto-resolve to with NO env var setup on this
    machine (confirmed empirically: a board referencing KICAD10_3DMODEL_DIR
    exported 0 model failures with no env vars set at all). Used only to
    check whether a .wrl reference has a real .step file at the same path,
    for patch_wrl_references() below."""
    cand = Path("/Applications/KiCad/KiCad.app/Contents/SharedSupport/3dmodels")
    return cand if cand.is_dir() else None


def ensure_kicad_packages3d_repo(cache_root: Path) -> Path | None:
    """Clone the full upstream official KiCad 3D-model library (cached,
    shallow, once). This machine's bundled KiCad.app library
    (kicad_bundled_3d_lib()) turns out to be a PARTIAL snapshot -- some refs
    genuinely absent there exist at the exact same
    `<Category>.3dshapes/<name>.step` path in this upstream repo (verified
    2026-07-27 via GitHub's API; see the module comment above). Unlike the
    SparkFun shim, no path reshaping is needed -- this repo already uses the
    same layout board files reference. Returns the repo root, or None if the
    clone failed (caller should proceed without this tier -- those specific
    refs just stay unfixable, same as before this fix existed)."""
    repo_dir = cache_root / "_kicad_packages3d"
    if repo_dir.is_dir():
        return repo_dir
    proc = subprocess.run(["git", "clone", "--depth", "1", "--quiet",
                           KICAD_PACKAGES3D_REPO, str(repo_dir)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  WARNING: could not clone {KICAD_PACKAGES3D_REPO}: {proc.stderr.strip()}",
              file=sys.stderr)
        return None
    return repo_dir


def _leading_pin_count(name: str) -> int | None:
    """The first purely-numeric token immediately after the leading
    alphabetic prefix (e.g. "16" in "QFN-16-1EP...", "8" in "SOIC-8_...",
    "02" in "B02B-XH-A..." -- parsed as an int so "02" == "2"). This is
    almost always a real pin count or position count, which must NEVER
    differ between a ref and its fuzzy match -- unlike later dimension
    tokens (5.23mm -> 5.275mm), a pin-count change means a DIFFERENT part,
    not a renamed one. Returns None if no such token is found."""
    m = re.match(r"^[A-Za-z_]+[-_]?(\d+)", name)
    return int(m.group(1)) if m else None


def find_kicad_packages3d_fuzzy_match(rel: str, upstream_repo: Path) -> str | None:
    """Conservative fuzzy match for a ref RENAMED upstream between KiCad
    library versions (a corrected dimension, a variant-suffix rename) --
    NOT a general "find something similar" search. Only looks within the
    ref's own `<Category>.3dshapes/` directory, requires candidates to share
    the ref's leading alphabetic prefix (the part before the first digit --
    e.g. "SOIC-8", "JST_PH_B2B-PH"), then picks the single closest remaining
    candidate by difflib string similarity, but ONLY if it clears a high
    similarity cutoff AND its leading pin/position-count token (see
    _leading_pin_count()) matches the ref's exactly.

    This second check exists because it caught a REAL wrong-match in
    validation (2026-07-27, cynthion-hardware): `QFN-16-1EP_3x3mm_P0.5mm_
    EP1.45x1.45mm` fuzzy-matched to `QFN-12-1EP_3x3mm_P0.5mm_EP1.65x1.65mm`
    on string similarity alone (same body size and pitch, high overall
    similarity) despite being a genuinely DIFFERENT 12-pin part, not a
    rename of the 16-pin one. Pin-count-preserving renames (SOIC-8 dimension
    correction, JST_PH suffix renames, JST_XH leading-zero variants) all
    still pass this check; that one case correctly does not.

    Returns the matched category-relative path (no extension), or None if
    there's no sufficiently-confident candidate -- callers must treat None
    as "leave unfixable," not force a low-confidence guess."""
    category, _, name = rel.partition("/")
    cat_dir = upstream_repo / category
    if not cat_dir.is_dir():
        return None
    prefix_match = re.match(r"^[A-Za-z_]+", name)
    prefix = prefix_match.group(0) if prefix_match else name
    ref_pins = _leading_pin_count(name)
    candidates = [p.stem for p in cat_dir.glob("*.step") if p.stem.startswith(prefix)
                 and _leading_pin_count(p.stem) == ref_pins]
    if not candidates:
        return None
    best = difflib.get_close_matches(name, candidates, n=1, cutoff=0.75)
    if not best:
        return None
    return f"{category}/{best[0]}"


def patch_wrl_references(pcb_path: Path, out_path: Path, bundled_lib: Path | None,
                         upstream_repo: Path | None = None,
                         allow_fuzzy: bool = False) -> dict:
    """Write a patched copy of pcb_path to out_path, rewriting `.wrl` model
    references wherever a resolvable equivalent is found, tried in tiers of
    decreasing confidence:
      1. exact match in KiCad's local bundled library (bundled_lib) --
         extension swapped in place, since this machine's own KiCad install
         already auto-resolves the ref's original env var to this same
         library at kicad-cli runtime.
      2. exact match in the full upstream kicad-packages3d repo
         (upstream_repo), if given -- rewritten to use
         `${KICAD_PACKAGES3D_ENV_VAR}` instead of the ref's original env
         var, since that original var resolves to the LOCAL bundled library
         (which is exactly what's missing the file) -- not our separately
         cloned upstream repo. Surgical: only refs actually resolved this
         way are touched; every other ref's resolution is untouched.
      3. (only if allow_fuzzy) a same-category fuzzy match in upstream_repo
         via find_kicad_packages3d_fuzzy_match() -- also rewritten via
         KICAD_PACKAGES3D_ENV_VAR, but reported separately (`fuzzy_matched`)
         from the exact-match `fixed` count, never silently merged in.
    A board with fixed == 0, unfixable == 0, and no fuzzy_matched entries has
    no `.wrl` references at all (nothing to do); out_path is always written
    regardless, so callers can use it unconditionally.

    Returns {"fixed": int, "unfixable": int,
             "fuzzy_matched": [{"ref": str, "matched": str}, ...],
             "needs_upstream_repo": bool}."""
    text = pcb_path.read_text(errors="ignore")
    counts = {"fixed": 0, "unfixable": 0}
    fuzzy_matched: list[dict] = []
    needs_upstream = {"flag": False}

    def repl(m: re.Match) -> str:
        prefix, ref, ext, suffix = m.groups()
        env_var_match = re.match(r'^\$\{([A-Z0-9_]+)\}/', ref)
        rel = re.sub(r'^\$\{[A-Z0-9_]+\}/', '', ref)

        if bundled_lib is not None and (bundled_lib / f"{rel}.step").exists():
            counts["fixed"] += 1
            return f"{prefix}{ref}.step{suffix}"

        if upstream_repo is not None and (upstream_repo / f"{rel}.step").exists():
            counts["fixed"] += 1
            needs_upstream["flag"] = True
            return f"{prefix}${{{KICAD_PACKAGES3D_ENV_VAR}}}/{rel}.step{suffix}"

        if allow_fuzzy and upstream_repo is not None:
            match = find_kicad_packages3d_fuzzy_match(rel, upstream_repo)
            if match is not None:
                counts["fixed"] += 1
                needs_upstream["flag"] = True
                fuzzy_matched.append({"ref": rel, "matched": match})
                return f"{prefix}${{{KICAD_PACKAGES3D_ENV_VAR}}}/{match}.step{suffix}"

        counts["unfixable"] += 1
        return m.group(0)

    patched = WRL_MODEL_RE.sub(repl, text)
    out_path.write_text(patched)
    return {"fixed": counts["fixed"], "unfixable": counts["unfixable"],
            "fuzzy_matched": fuzzy_matched, "needs_upstream_repo": needs_upstream["flag"]}


def ensure_sparkfun_library_shim(cache_root: Path) -> Path | None:
    """Clone sparkfun/SparkFun-KiCad-Libraries (cached, shallow, once) and
    build a symlink shim reconstructing the OLD path layout board files
    reference (Footprints/SparkFun-3D-models/<Category>/<file>.step) pointing
    at the CURRENT layout's real files (3dmodels/<Category>.3dshapes/<file>).
    Returns the shim root to set SPARKFUN_KICAD_LIBRARY to, or None if the
    clone failed (caller should proceed without it -- SparkFun refs just
    won't resolve, same as before this fix existed)."""
    repo_dir = cache_root / "_sparkfun_kicad_library"
    shim_dir = cache_root / "_sparkfun_kicad_library_shim"
    marker = shim_dir / ".built"
    if marker.exists():
        return shim_dir

    if not repo_dir.exists():
        proc = subprocess.run(["git", "clone", "--depth", "1", "--quiet",
                               SPARKFUN_REPO, str(repo_dir)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  WARNING: could not clone {SPARKFUN_REPO}: {proc.stderr.strip()}",
                  file=sys.stderr)
            return None

    models_dir = repo_dir / SPARKFUN_NEW_SUBDIR
    if not models_dir.is_dir():
        print(f"  WARNING: {models_dir} not found in cloned SparkFun library "
              f"(layout may have changed again) -- skipping shim.", file=sys.stderr)
        return None

    old_root = shim_dir / SPARKFUN_OLD_PREFIX
    old_root.mkdir(parents=True, exist_ok=True)
    for cat_dir in models_dir.iterdir():
        if not cat_dir.is_dir() or not cat_dir.name.endswith(".3dshapes"):
            continue
        category = cat_dir.name[: -len(".3dshapes")]
        dest_cat = old_root / category
        dest_cat.mkdir(parents=True, exist_ok=True)
        for f in cat_dir.iterdir():
            link = dest_cat / f.name
            if not link.exists():
                link.symlink_to(f)

    marker.write_text("ok")
    return shim_dir


def prepare_step_export_pcb(root: Path, board: str, pcb_path: Path,
                            allow_fuzzy: bool = False) -> tuple[Path, dict]:
    """One-stop prep for a kicad-cli STEP export. Chains the recovery tiers
    described in the module comment above: (1) exact match in KiCad's local
    bundled library; (2) exact match in the full upstream kicad-packages3d
    repo, cloned LAZILY -- only if tier 1 leaves at least one ref
    unresolved, since it's a much larger one-time clone than the SparkFun
    shim, not worth paying for boards tier 1 already covers completely; (3)
    (only if allow_fuzzy=True) a conservative same-category fuzzy match
    against that same upstream repo, for refs renamed between KiCad library
    versions -- reported separately (`fuzzy_matched`), never silently
    trusted like tiers 1-2. Also, if the board references
    SPARKFUN_KICAD_LIBRARY, ensures that library's shim is built. Returns
    (pcb_path_to_export, extra_env_vars) -- callers pass the returned path
    to `kicad-cli pcb export step` and merge the returned env vars into the
    subprocess environment. Cached per-board so repeat runs (precheck then
    finalize) don't redo the patch/clone work."""
    cache_dir = root / ".cache_repos" / board
    patched_path = cache_dir / f"{pcb_path.stem}_3dfix.kicad_pcb"
    report_path = cache_dir / "_3dfix_report.json"

    bundled_lib = kicad_bundled_3d_lib()
    if report_path.exists() and patched_path.exists():
        report = json.loads(report_path.read_text())
    else:
        text = pcb_path.read_text(errors="ignore")
        needs_sparkfun = SPARKFUN_ENV_VAR in text

        if bundled_lib is not None:
            result = patch_wrl_references(pcb_path, patched_path, bundled_lib)
        else:
            shutil.copy(pcb_path, patched_path)
            result = {"fixed": 0, "unfixable": 0, "fuzzy_matched": [],
                      "needs_upstream_repo": False}

        if result["unfixable"] > 0:
            upstream_repo = ensure_kicad_packages3d_repo(root / ".cache_repos")
            if upstream_repo is not None:
                result = patch_wrl_references(pcb_path, patched_path, bundled_lib,
                                              upstream_repo=upstream_repo,
                                              allow_fuzzy=allow_fuzzy)

        report = {"needs_sparkfun": needs_sparkfun,
                  "wrl_fixed": result["fixed"],
                  "wrl_unfixable": result["unfixable"],
                  "fuzzy_matched": result["fuzzy_matched"],
                  "needs_upstream_repo": result["needs_upstream_repo"]}
        report_path.write_text(json.dumps(report))
        if result["fixed"] or result["unfixable"] or needs_sparkfun:
            n_fuzzy = len(result["fuzzy_matched"])
            print(f"  3D-model recovery for {board}: {result['fixed']} .wrl->.step fixed "
                  f"({n_fuzzy} via fuzzy match), {result['unfixable']} unfixable, "
                  f"sparkfun_lib_needed={needs_sparkfun}")

    env: dict[str, str] = {}
    if report["needs_sparkfun"]:
        shim = ensure_sparkfun_library_shim(root / ".cache_repos")
        if shim is not None:
            env[SPARKFUN_ENV_VAR] = str(shim)
    if report.get("needs_upstream_repo"):
        upstream_repo = ensure_kicad_packages3d_repo(root / ".cache_repos")
        if upstream_repo is not None:
            env[KICAD_PACKAGES3D_ENV_VAR] = str(upstream_repo)
    return patched_path, env


def find_cached_pcb(root: Path, board: str) -> Path:
    """Locate the cached .kicad_pcb for `board`. Prefers the exact relative
    path 02_fetch_gerbers.py already resolved and recorded in meta.json's
    "kicad_pcb" field (authoritative for multi-board repos, e.g. GreatFET's
    neighbor-board monorepo, where a plain rglob() first-match would be
    arbitrary) -- falls back to rglob()'s first match for older meta.json
    files written before this field existed, or if meta.json is missing."""
    cache_dir = root / ".cache_repos" / board
    meta_path = root / "boards" / board / "meta.json"
    if meta_path.exists():
        rel = json.loads(meta_path.read_text()).get("kicad_pcb")
        if rel:
            candidate = cache_dir / rel
            if candidate.exists():
                return candidate
    # rglob (recursive), not glob -- some repos (e.g. SparkFun's) nest the
    # .kicad_pcb under a Hardware/ subdirectory rather than the repo root.
    cached_pcbs = list(cache_dir.rglob("*.kicad_pcb"))
    if not cached_pcbs:
        sys.exit(f"No cached .kicad_pcb for {board}. Run 02_fetch_gerbers.py first.")
    return cached_pcbs[0]


def kicad_cli() -> str:
    p = shutil.which("kicad-cli")
    if p:
        return p
    cand = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"
    if Path(cand).exists():
        return cand
    sys.exit("kicad-cli not found. Run 00_install.sh then 01_verify_env.py.")


def find_board_shell(scene: trimesh.Scene):
    """Locate the board-body shell in a full (board+components) STEP scene.
    Returns (geometry_name, world_bounds) or None if nothing plausible found."""
    candidates = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        g = scene.geometry.get(geom_name)
        if not isinstance(g, trimesh.Trimesh) or not len(g.vertices):
            continue
        world = g.copy()
        world.apply_transform(transform)
        candidates.append((geom_name, world.bounds))

    if not candidates:
        return None

    named = [c for c in candidates if c[0].lower().endswith("_pcb")
            or c[0].lower() == "pcb"]
    if named:
        return named[0]

    # fallback: largest XY footprint, flattest Z (the board slab, not a part)
    def flatness_score(c):
        size = c[1][1] - c[1][0]
        xy_area = size[0] * size[1]
        z = max(size[2], 1e-9)
        return xy_area / z  # large area, small Z -> highest score
    candidates.sort(key=flatness_score, reverse=True)
    return candidates[0]
