#!/usr/bin/env python3
"""Acquire a real, permissively-licensed board and export its Gerbers into
boards/<name>/fab/ using kicad-cli -- the exact flow gerber2blend's own CI uses,
so the filenames (``*-F_Cu.gbr`` etc.) match gerber2blend's default patterns.

Default board: Antmicro's jetson-orin-baseboard (Apache-2.0), the board
gerber2blend is CI-tested against, so it is known-good for first light.

Usage:
    python3 02_fetch_gerbers.py                 # default board
    python3 02_fetch_gerbers.py --board <key>   # pick from BOARDS
    python3 02_fetch_gerbers.py --list
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOARDS_DIR = ROOT / "boards"
CACHE_DIR = ROOT / ".cache_repos"
ENV_JSON = ROOT / "env.json"

# name -> {git url, license, note}. All Apache-2.0 Antmicro open hardware.
BOARDS: dict[str, dict] = {
    "scalenode": {
        "git": "https://github.com/antmicro/scalenode.git",
        "license": "Apache-2.0",
        "note": "Antmicro Raspberry Pi CM4 clustering baseboard.",
    },
    "snapdragon-845-baseboard": {
        "git": "https://github.com/antmicro/snapdragon-845-baseboard.git",
        "license": "Apache-2.0",
        "note": "Antmicro open-source Snapdragon 845 baseboard.",
    },
    "signal-integrity-test-board": {
        "git": "https://github.com/antmicro/signal-integrity-test-board.git",
        "license": "Apache-2.0",
        "note": "Antmicro signal integrity test board.",
    },
    "raspberry-pi-cm4-csi-adapter": {
        "git": "https://github.com/antmicro/raspberry-pi-cm4-csi-adapter.git",
        "license": "Apache-2.0",
        "note": "Antmicro Raspberry Pi CM4 CSI adapter.",
    },
    "framos-csi-adapter": {
        "git": "https://github.com/antmicro/framos-csi-adapter.git",
        "license": "Apache-2.0",
        "note": "Antmicro FRAMOS CSI adapter.",
    },
    "dual-ipex-csi-interposer": {
        "git": "https://github.com/antmicro/dual-ipex-csi-interposer.git",
        "license": "Apache-2.0",
        "note": "Antmicro dual IPEX CSI interposer.",
    },
    "cvbs-mipi-bridge": {
        "git": "https://github.com/antmicro/cvbs-mipi-bridge.git",
        "license": "Apache-2.0",
        "note": "Antmicro CVBS to MIPI CSI-2 bridge.",
    },
    "snickerdoodle-hdmi-breakout": {
        "git": "https://github.com/antmicro/snickerdoodle-hdmi-breakout.git",
        "license": "Apache-2.0",
        "note": "Antmicro snickerdoodle HDMI breakout.",
    },
    "icicle-hdmi-breakout": {
        "git": "https://github.com/antmicro/icicle-hdmi-breakout.git",
        "license": "Apache-2.0",
        "note": "Antmicro PolarFire Icicle HDMI breakout.",
    },
    "sdi-mipi-bridge-hw": {
        "git": "https://github.com/antmicro/sdi-mipi-bridge-hw.git",
        "license": "Apache-2.0",
        "note": "Antmicro open hardware 3G SDI into MIPI CSI-2 converter.",
    },
    # --- candidates added for board-diversity expansion; gated by
    # 02b_stl_precheck.py BEFORE the expensive gerber2blend/imaging steps --
    # confirmed-bare/unresolvable candidates are removed from this dict
    # entirely (not left around as dead weight that gets re-fetched every
    # run) once precheck has ruled on them; disk artifacts (.cache_repos/,
    # boards/<name>/) for dropped boards are deleted too. Tried and
    # confirmed UNUSABLE so far (do not re-add without a reason to think
    # the upstream repo changed): jetson-orin-baseboard (private,
    # unresolvable ${ANT3DMDL} 3D-model library), jetson-nano-baseboard,
    # hdmi-mipi-bridge, ov9281-camera-board, m2-pcie-adapter,
    # thunderbolt-pcie-adapter, artix-dc-scm, lpddr4-test-board (all bare,
    # Z-ratio 1.0-2.1%, no resolvable 3D component models), and
    # google-coral-baseboard / zynq-video-board (repo has no top-level
    # .kicad_pcb -- different layout than expected, not investigated
    # further) / kria-k26-devboard (precheck crashed outright).
    # --- round 2 (found via GitHub's search API, org:antmicro, keyword
    # search across name/description, after round 1 above turned up 0/10
    # survivors): polarfire-som, jetson-agx-thor-baseboard,
    # cm4-baseboard-lvds-adapter, kintex-410t-devboard, cm4-baseboard,
    # com-express-7-baseboard, snapdragon-625-baseboard,
    # dc-scm-osm-l-carrier-card, dc-scm-breakout-board, stm32h7-hdmi-board,
    # poe-usb-c-pd-converter, job-ict-bottom-board,
    # sdi-mipi-video-converter-hw, hdmi-edid-debug-board,
    # audio-latency-tester-board -- all bare (Z-ratio 0.7-3.0%, no
    # resolvable 3D component models). parallella-lcd-baseboard /
    # apalis-smart-vision-baseboard -- repo has no top-level .kicad_pcb,
    # not investigated further. pmod-i3c-sensor-board / pdm-microphone-board
    # -- initially misread as "populated" by the Z-ratio heuristic alone
    # (4.5% / 13.7%), but n_shells=1 for both: no separable component
    # geometry at all, just a small board's bare substrate thickness
    # tripping the ratio (this is WHY the populated check now also requires
    # n_geoms > 1, see 02b_stl_precheck.py).
    #
    # ov5640-dual-camera-board is a DIFFERENT kind of exclusion, worth
    # calling out separately: it's genuinely populated (136 shells, 28.5%
    # Z-ratio, real separable components) and its core pipeline runs fine,
    # but 08_verify_alignment.py fails it -- its front (F_SilkS) layer is
    # completely blank; all its silkscreen is on the BACK face
    # (B_SilkS has ~92k dark px). This project's whole pipeline only
    # images/labels the front face, so this board can't produce a valid
    # sample today. Not a bad board -- a real board blocked by a genuine,
    # separate pipeline limitation (front-face-only imaging). Worth
    # re-trying if back-face imaging is ever added.
    # --- round 3: a different open-hardware source (SparkFun, not Antmicro)
    # after round 1+2 (30 Antmicro candidates) found 0 net-new survivors.
    # SparkFun maintains a mature KiCad library with 3D models
    # (SparkFun-KiCad-Libraries), but confirmed via GitHub contents API that
    # MUCH of their product hardware is still Eagle-format (.brd/.sch), not
    # KiCad -- e.g. SparkFun_Thing_Plus_RP2350 and SparkFun_Thing_Plus_RA6M5
    # are both Eagle.
    # License: SparkFun hardware is typically CC BY-SA 4.0 (per-repo,
    # not individually re-verified here the way Antmicro's Apache-2.0 was).
    #
    # NB found while checking this batch: 02b_stl_precheck.py/06_export_stl.py
    # originally used a non-recursive glob() for the cached .kicad_pcb
    # (unlike 02_fetch_gerbers.py's rglob()) and silently mis-flagged
    # several genuinely-KiCad SparkFun repos (nested under Hardware/) as
    # "crashed" instead of checking them -- fixed to rglob() too. Only ONE
    # of 13 SparkFun candidates tried turned out genuinely usable:
    "sparkfun-spectral-as7343": {
        "git": "https://github.com/sparkfun/SparkFun_Spectral_Sensor_Breakout_AS7343_Qwiic.git",
        "license": "CC BY-SA 4.0",
        "note": "SparkFun 14-channel multi-spectral sensor breakout (AS7343). "
                "Small (1inx1in) but genuinely populated: 3 shells, 10.6% Z-ratio.",
    },
    # Tried and confirmed unusable (do not re-add without reason to think
    # upstream changed): sparkfun-iot-redboard-rp2350 (real KiCad, but
    # single-shell/2.2% Z-ratio -- footprints have no 3D models assigned),
    # sparkfun-qwiic-power-meter-acs37800, sparkfun-openscale-iot,
    # sparkfun-qwiic-boost-ap3012k, sparkfun-qwiic-adc-ads1219,
    # sparkfun-qwiic-dac-mcp4725 (same: real KiCad, single-shell/bare) --
    # and sparkfun-qwiic-radar-xm125, sparkfun-qwiic-presence-sths34pf80,
    # sparkfun-ultrasonic-distance-qwiic, sparkfun-qwiic-power-switch,
    # sparkfun-qwiic-hall-tmag5273, sparkfun-qwiic-buzzer (Eagle-format,
    # no .kicad_pcb anywhere in the repo).
    #
    # --- round 4: two new source orgs, chosen because they're real hardware
    # companies (RF/USB test equipment, solar charge controllers) that tend
    # to assign real 3D component models -- confirmed via GitHub's git-trees
    # API (recursive tree listing) that each repo below has a top-level or
    # nested .kicad_pcb before spending a clone+precheck on it. Verdict: by
    # far the best hit rate of any round so far -- 14/16 tried are populated
    # AND fully image+verify (vs. round 1-3's ~1/43). Two dropped as
    # genuinely bare, disk artifacts removed: amalthea-hardware (Z-ratio
    # 2.7% < 3%, 1 shell) and libresolar-rpi-can (Z-ratio 2.3%, 1 shell) --
    # do not re-add without reason to think upstream changed. Also found and
    # fixed a real bug along the way: several older Libre Solar boards
    # (mppt-1210-hus, mppt-2420-rc/hpx/lc, pwm-2420-lus) have their F.Cu/
    # B.Cu copper layers custom-renamed to "Top"/"Bottom" in board setup --
    # kicad-cli's gerber export uses the current layer name, not the
    # canonical designator, so these emitted "*-Top.gbr"/"*-Bottom.gbr"
    # instead of "*-F_Cu.gbr"/"*-B_Cu.gbr", crashing gerber2blend's
    # solder-paste compositing step (same class of issue as the F_SilkS/
    # F_Silkscreen alias below). Fixed with an analogous rename, see below.
    "cynthion-hardware": {
        "git": "https://github.com/greatscottgadgets/cynthion-hardware.git",
        "license": "CERN-OHL-P-2.0",
        "note": "Great Scott Gadgets Cynthion USB analysis multitool mainboard.",
    },
    # amalthea-hardware, libresolar-rpi-can: tried, confirmed bare (see
    # round-4 note above), removed + disk artifacts deleted.
    "urti-mainboard": {
        "git": "https://github.com/greatscottgadgets/urti-mainboard.git",
        "license": "CERN-OHL-P-2.0",
        "note": "Great Scott Gadgets URTI mainboard.",
    },
    "hackrf-pro": {
        "git": "https://github.com/greatscottgadgets/hackrf-pro.git",
        "license": "CERN-OHL-P-2.0",
        "note": "Great Scott Gadgets HackRF Pro (praline) SDR board.",
    },
    "libresolar-uext-esp32": {
        "git": "https://github.com/LibreSolar/uext-esp32.git",
        "license": "CC BY-SA 4.0",
        "note": "Libre Solar UEXT ESP32 extension board.",
    },
    "libresolar-uext-esp32-c3": {
        "git": "https://github.com/LibreSolar/uext-esp32-c3.git",
        "license": "CERN-OHL-W-2.0",
        "note": "Libre Solar UEXT ESP32-C3 extension board.",
    },
    "libresolar-uext-lora": {
        "git": "https://github.com/LibreSolar/uext-lora.git",
        "license": "CC BY-SA 4.0",
        "note": "Libre Solar UEXT LoRa extension board.",
    },
    "libresolar-uext-gsm": {
        "git": "https://github.com/LibreSolar/uext-gsm.git",
        "license": "CC BY-SA 4.0",
        "note": "Libre Solar UEXT GSM extension board.",
    },
    "libresolar-bms-8s50-ic": {
        "git": "https://github.com/LibreSolar/bms-8s50-ic.git",
        "license": "CC BY-SA 4.0",
        "note": "Libre Solar 8S50 battery management system board.",
    },
    "libresolar-mppt-1210-hus": {
        "git": "https://github.com/LibreSolar/mppt-1210-hus.git",
        "license": "NOASSERTION",
        "note": "Libre Solar MPPT 1210 HUS solar charge controller.",
    },
    "libresolar-mppt-2420-rc": {
        "git": "https://github.com/LibreSolar/mppt-2420-rc.git",
        "license": "CC BY-SA 4.0",
        "note": "Libre Solar MPPT 2420 RC solar charge controller.",
    },
    "libresolar-mppt-2420-hpx": {
        "git": "https://github.com/LibreSolar/mppt-2420-hpx.git",
        "license": "CC BY-SA 4.0",
        "note": "Libre Solar MPPT 2420 HPX solar charge controller.",
    },
    "libresolar-mppt-2420-lc": {
        "git": "https://github.com/LibreSolar/mppt-2420-lc.git",
        "license": "CC BY-SA 4.0",
        "note": "Libre Solar MPPT 2420 LC (20A) solar charge controller.",
    },
    "libresolar-pwm-2420-lus": {
        "git": "https://github.com/LibreSolar/pwm-2420-lus.git",
        "license": "CC BY-SA 4.0",
        "note": "Libre Solar PWM 2420 LUS solar charge controller.",
    },
    "libresolar-bms-5s50-sc": {
        "git": "https://github.com/LibreSolar/bms-5s50-sc.git",
        "license": "CC BY-SA 4.0",
        "note": "Libre Solar 5S50 BMS -- repo has 2 boards "
                "(control-pcb, power-pcb); rglob() picks power-pcb/"
                "Switch-N-Sense.kicad_pcb, confirmed working.",
    },
    # --- round 5 (2026-07-24): more boards from the two proven high-hit-rate
    # orgs (GSG, Libre Solar) not tried in round 4, plus a new org
    # (ProtoCentral -- sensor-breakout hardware house, CERN-OHL-P-2.0,
    # explicitly ships "KiCad hardware design files"). Candidates
    # pre-filtered via GitHub's git-trees API for .kicad_pcb presence,
    # same methodology as round 4. GreatFET's neighbor-board monorepo
    # (greatfet-hardware) holds many small boards in one repo -- added
    # `pcb_subpath` support to this registry + `stl_common.find_cached_pcb()`
    # so each neighbor board can be fetched deterministically instead of
    # relying on rglob()'s arbitrary first match (that support stays even
    # though every board that needed it this round turned out bare, below).
    # Tried and confirmed UNUSABLE this round (bare, Z-ratio 1.5-2.1%, no
    # resolvable 3D component models; disk artifacts removed): GSG's two
    # standalone RF boards (greatscottgadgets-lna1109, -lna3030) and its
    # tycho board -- all 3 GSG hardware repos NOT already in this registry
    # turn out bare, unlike GSG's flagship boards (Cynthion, URTI, HackRF
    # Pro) which are populated; and all 5 GreatFET neighbor expansion boards
    # tried (azalea, begonia, crocus, daffodil, edelweiss) -- consistently
    # bare, suggesting the WHOLE GreatFET neighbor-board family lacks 3D
    # component models (not worth trying more of the remaining ~8 neighbors
    # without a reason to think a specific one differs). Also
    # protocentral-sensything (2.1%, bare) -- but ProtoCentral's other 3
    # boards below all survived, so this looks like a per-board exception,
    # not an org-wide pattern like GSG/GreatFET above.
    "libresolar-data-manager": {
        "git": "https://github.com/LibreSolar/data-manager.git",
        "license": "CC BY-SA 4.0",
        "note": "Libre Solar CAN/UART to WiFi gateway + SD card data logger.",
    },
    "libresolar-bms-c1": {
        "git": "https://github.com/LibreSolar/bms-c1.git",
        "license": "NOASSERTION",
        "note": "Libre Solar 16s / 100A battery management system.",
    },
    "libresolar-mppt-2420-hc": {
        "git": "https://github.com/LibreSolar/mppt-2420-hc.git",
        "license": "NOASSERTION",
        "note": "Libre Solar MPPT charge controller with HS load switch + CAN bus.",
    },
    "protocentral-fdc2214": {
        "git": "https://github.com/Protocentral/protocentral_fdc2214_hardware.git",
        "license": "CERN-OHL-P-2.0",
        "note": "ProtoCentral FDC2214 4-channel capacitance-to-digital breakout.",
    },
    "protocentral-tmf8829": {
        "git": "https://github.com/Protocentral/protocentral_tmf8829_hardware.git",
        "license": "CERN-OHL-P-2.0",
        "note": "ProtoCentral TMF8829 48x32 dToF depth sensor breakout.",
    },
    "protocentral-st1vafe3bx": {
        "git": "https://github.com/Protocentral/protocentral_st1vafe3bx_hardware.git",
        "license": "CERN-OHL-P-2.0",
        "note": "ProtoCentral ST1VAFE3BX biopotential sensor breakout.",
    },
}
DEFAULT_BOARD = "scalenode"


def kicad_cli() -> str:
    """Locate kicad-cli, preferring the path 01_verify_env.py discovered."""
    if ENV_JSON.exists():
        info = json.loads(ENV_JSON.read_text()).get("kicad-cli", {})
        if info.get("path"):
            return info["path"]
    p = shutil.which("kicad-cli")
    if p:
        return p
    cand = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"
    if Path(cand).exists():
        return cand
    sys.exit("kicad-cli not found. Run 00_install.sh then 01_verify_env.py.")


def run(cmd: list[str], **kw) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=DEFAULT_BOARD)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in BOARDS.items():
            star = " (default)" if k == DEFAULT_BOARD else ""
            print(f"{k}{star}\n    {v['git']}  [{v['license']}]\n    {v['note']}")
        return 0

    if args.board not in BOARDS:
        sys.exit(f"Unknown board '{args.board}'. Use --list.")
    spec = BOARDS[args.board]
    kcli = kicad_cli()

    CACHE_DIR.mkdir(exist_ok=True)
    repo = CACHE_DIR / args.board
    if repo.exists():
        print(f"Repo already cloned: {repo}")
    else:
        run(["git", "clone", "--depth", "1", "--quiet", spec["git"], str(repo)])

    # pcb_subpath: some repos hold multiple boards (e.g. GreatFET's neighbor-
    # board monorepo) -- an explicit subpath picks the right one deterministically
    # instead of relying on rglob()'s arbitrary first match.
    if "pcb_subpath" in spec:
        pcb = repo / spec["pcb_subpath"]
        if not pcb.exists():
            sys.exit(f"pcb_subpath {spec['pcb_subpath']!r} not found under {repo}")
    else:
        pcbs = list(repo.rglob("*.kicad_pcb"))
        if not pcbs:
            sys.exit(f"No .kicad_pcb found under {repo}")
        pcb = pcbs[0]
    print(f"Using board file: {pcb.relative_to(repo)}")

    fab = BOARDS_DIR / args.board / "fab"
    fab.mkdir(parents=True, exist_ok=True)

    # --no-protel-ext -> emit KiCad-default *.gbr names gerber2blend expects.
    run([kcli, "pcb", "export", "gerbers", "--no-protel-ext",
         "-o", str(fab) + "/", str(pcb)])
    run([kcli, "pcb", "export", "drill", "--format", "gerber",
         "--excellon-separate-th", "-o", str(fab) + "/", str(pcb)])

    # Boards last saved in older KiCad versions export "*-F_SilkS.gbr" /
    # "*-B_SilkS.gbr"; gerber2blend's default blendcfg only globs for the
    # newer "*-F_Silkscreen.gbr" / "*-B_Silkscreen.gbr" names, so without
    # this alias it silently treats silkscreen as absent and later fails
    # packing the .blend (looks for a PNG that never got generated).
    for gbr in fab.glob("*-F_SilkS.gbr"):
        gbr.replace(gbr.with_name(gbr.name.replace("F_SilkS.gbr", "F_Silkscreen.gbr")))
    for gbr in fab.glob("*-B_SilkS.gbr"):
        gbr.replace(gbr.with_name(gbr.name.replace("B_SilkS.gbr", "B_Silkscreen.gbr")))

    # Some boards (seen in several older Libre Solar .kicad_pcb files) have
    # the F.Cu/B.Cu copper layers renamed to custom board-setup names ("Top"/
    # "Bottom") -- kicad-cli's gerber export uses the CURRENT layer name, not
    # the canonical F_Cu/B_Cu designator, so these emit "*-Top.gbr"/
    # "*-Bottom.gbr" instead. gerber2blend's gerbv_convert.py hardcodes the
    # F_Cu/B_Cu filenames when compositing the solder-paste layer, so without
    # this alias it crashes looking for a PNG that was never named that.
    for gbr in fab.glob("*-Top.gbr"):
        gbr.replace(gbr.with_name(gbr.name.replace("Top.gbr", "F_Cu.gbr")))
    for gbr in fab.glob("*-Bottom.gbr"):
        gbr.replace(gbr.with_name(gbr.name.replace("Bottom.gbr", "B_Cu.gbr")))

    gerbers = sorted(p.name for p in fab.glob("*.gbr"))
    has_paste = any("Paste" in g for g in gerbers)

    meta = {
        "board": args.board,
        "source": spec["git"],
        "license": spec["license"],
        "kicad_pcb": str(pcb.relative_to(repo)),
        "gerbers": gerbers,
        "has_paste_layer": has_paste,
        "note": spec["note"],
    }
    (BOARDS_DIR / args.board / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nExported {len(gerbers)} Gerber files to {fab.relative_to(ROOT)}")
    print("Paste layer present (needed for solder class):", has_paste)
    if not has_paste:
        print("  WARNING: no F/B_Paste -> solder will collapse into copper.")
    print(f"Wrote {(BOARDS_DIR / args.board / 'meta.json').relative_to(ROOT)}")
    print(f"\nNext:  python3 03_run_gerber2blend.py --board {args.board}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
