# Session Status / Handoff

**Purpose of this file:** a complete context dump so a future Claude
session (with no memory of this one) can pick up exactly where this left
off — what exists, why it exists, what's been tried and failed, and what's
next. Originally written 2026-07-20, updated 2026-07-21 (round-4 board
search + retrain), updated 2026-07-23 (`solder` loss/sampling experiments
+ root-cause finding; `trace`/`silkscreen` fix; solder-JOINT-geometry
investigation + F_Solder experiment, reverted; SMD solder-fillet synthesis
implemented + validated on 3 boards, PAUSED before dataset-wide rollout),
updated 2026-07-24 (fillet synthesis rolled out dataset-wide, retrained,
evaluated — result INCONCLUSIVE, not a clean win or a clean loss; reverted
live dataset back to the paste-only production state; real-data validation
identified as the actual blocker on judgment calls like this one — see §6
item 2's final bullet and §7 item 2), updated 2026-07-27 (backfilling two
rounds of work that happened in code on 2026-07-24 but were never
chronicled here: round-5 registry expansion 25→31 boards, and a
component-3D-model-recovery fix that sat implemented-but-unexercised until
this session validated it on 2 boards, then rolled it out dataset-wide,
retrained, and evaluated — see §2, §3 items 11-12, §4's new bullet),
updated again same day 2026-07-27 (chased the ~50 refs left "unfixable"
after that fix: found and fixed two more recovery tiers — an upstream
`kicad-packages3d` exact-match tier, plus a pin-count-guarded fuzzy-match
tier for refs renamed between KiCad library versions, the guard added
after it caught a real wrong-part substitution in validation — rolled out,
retrained again, see §2's second retrain entry and §3 item 13),
**updated a THIRD time same day 2026-07-27 — DONE, promoted to
production, see §2's final entry and §3 item 14 for full detail**:
diagnosed and fixed a much bigger issue than either 3D-model round above —
the GT render pass used only `samples=4` (vs 32 for the band images), and
a controlled before/after comparison proved 57-65% of what looked like
"model boundary error" was actually GT anti-aliasing noise, not real model
failure. Added a `--gt-samples` flag, re-rendered the full 31-board dataset
at `--gt-samples 32`, retrained, evaluated: **mIoU 0.9300 → 0.9642**
(`solder` 0.749→0.873, `trace` 0.872→0.932) — the single largest quality
improvement found this session, from a data-generation fix alone, no
loss/architecture changes. `solder`'s previously-documented structural
occlusion ceiling (§6 item 2) is still real (a mounted component genuinely
hides the joint beneath it) — this fix removed a large *additional*
measurement-noise component stacked on top of that real ceiling, it didn't
eliminate the ceiling itself. **Updated again 2026-07-28**: re-tested the
solder-fillet-synthesis experiment (paused inconclusive on 2026-07-24)
against this now-clean dataset, since its own original write-up blamed
part of its ambiguity on the exact GT-noise mechanism just fixed — this
time got a clean, definitive answer: **negative** (`solder` regressed
0.8725→0.8627, `component` regressed 0.9913→0.9873, same boundary
anti-aliasing mechanism as before, just no longer masked by measurement
noise). Reverted, checkpoint restored and re-verified exact
(mIoU 0.9642/solder 0.8725) — see §6 item 2's re-test bullet and §7 item 2.
**This closes the solder-fillet question for good — do not re-attempt it.**
A comprehensive research-paper-style report covering the whole project
(dataset, model, results, honest limitations) was also written this
session for anyone needing the full picture without reading this entire
file — ask the user for the artifact link if it's needed again, it isn't
saved anywhere in this repo. If you're reading this cold: read this whole
file before touching anything, then check `git log`/file timestamps to see
if anything has changed since (NB: this repo has no git — use file
mtimes).

**Note:** an earlier version of this repo had a stray `MODEL_README.md`
belonging to a completely different, unrelated project (photometric-stereo
height reconstruction, 611-board dataset, `scripts/`, `paper/cvpr/`, none
of which exist in this repo). It has been removed (2026-07-29 cleanup).
`README.md` is this project's real doc.

---

## 1. What this project is

A synthetic PCB dataset generator + a U-Net semantic segmentation trainer.
Pipeline: real open-hardware KiCad boards → Gerbers → gerber2blend (mesh +
textures) → Mitsuba3 (physically-based renderer) → 6-channel image stack
(R, G, B, IR, UV, height) + an 8-class per-pixel ground-truth material mask
→ trained into a U-Net that reproduces that segmentation.

`README.md` is the authoritative, up-to-date user-facing doc — read it
first for "how it works." This file is the *narrative*/*decision history*
layer README doesn't carry.

## 2. Current state (as of this writing, 2026-07-27)

- **Dataset**: 31 boards, all in `out/manifest.json` with `verify_ok: true`,
  8-class GT each (background, copper, solder, fiberglass, soldermask,
  silkscreen, component, trace). Rendered at `mpp=0.045, samples=32`.
  Registry (`02_fetch_gerbers.py`'s `BOARDS` dict) has exactly these 31 —
  grew 11 → 25 in the round-4 board search (§3 item 10), then 25 → 31 in a
  round-5 search (§3 item 11, 6 new ProtoCentral/Libre Solar boards) that
  happened in code but was never logged here until now. **All 31 boards
  regenerated 2026-07-27** through the component-3D-model-recovery fix
  (§3 item 12) — 31/31 populated, imaged, verified, 0 dropped.
- **Component geometry accuracy — real, substantial fix landed 2026-07-27**
  (implemented 2026-07-24, validated + rolled out 2026-07-27, see §3 item
  12): 16 of 31 boards (52%) had their populated STEP export silently
  missing real component geometry before this fix, some very heavily —
  `urti-mainboard` 402 model refs recovered, `hackrf-pro` 294,
  `cynthion-hardware` 273, plus 2 boards (`sparkfun-spectral-as7343`,
  `protocentral-st1vafe3bx`) that needed the SparkFun-library shim. Neither
  of the two boards this project uses for matched-val-board comparisons
  (`icicle-hdmi-breakout`, `signal-integrity-test-board`) was itself
  affected (both show `wrl_fixed: 0`) — so the model-quality numbers below
  are flat by construction, NOT evidence the fix didn't matter. The real
  win is dataset accuracy on the other 16 boards' `component`/`trace`/
  height channels, now training on genuinely-recovered 3D geometry instead
  of silently-bare/partial component meshes.
- **Second round, same day: chased the ~50 refs still "unfixable" after
  the fix above** (see §3 item 13 for full mechanism). Two more recovery
  tiers added to `stl_common.py`: (1) an exact-match check against the
  full upstream `github.com/KiCad/kicad-packages3d` repo (this machine's
  bundled KiCad.app library turns out to be a partial snapshot) — safe,
  zero-ambiguity, recovered 16 more refs; (2) a fuzzy-match tier for refs
  genuinely renamed upstream between KiCad library versions (dimension
  corrections, variant-suffix renames) — recovered 9 more refs, but ONLY
  after a pin/position-count guard was added mid-validation: the first
  version of this matcher caught a real wrong-part substitution
  (`QFN-16-1EP..._EP1.45x1.45mm` → a genuinely different 12-pin package,
  `QFN-12-1EP..._EP1.65x1.65mm`, on pure string-similarity alone) before
  the guard rejected it. Registry-wide unfixable count: 53 → 37 (tier 1) →
  28 (both tiers). All 31 boards regenerated again and retrained (see
  below); production checkpoint backed up first as
  `unet_best_pre_kicad3dfix.pt`.
- **Trained model — RETRAINED and EVALUATED this session.** Retrained on
  all 25 boards TWICE, to separate "more data changed results" from
  "val-board choice changed results":
  1. `train/checkpoints/unet_best_25board_randomval.pt` — default seeded
     random split, which happened to hold out 2 NEW tiny Libre Solar boards
     (`libresolar-pwm-2420-lus`, `libresolar-uext-lora`), not the original
     baseline's val boards. Full-res eval: mIoU 0.888, pixel accuracy 0.989
     (not directly comparable to the baseline — different val boards).
  2. `train/checkpoints/unet_best_25board_ce_only.pt` — retrained with
     `--val-boards icicle-hdmi-breakout,signal-integrity-test-board`,
     matching the original 10-board baseline's held-out boards exactly.
     Full-res eval: mIoU 0.923, pixel accuracy 0.991 — essentially flat vs.
     the 10-board baseline's 0.927, with `solder` (the known weak class)
     improving +0.017. This was production until superseded below.
  3. **`train/checkpoints/unet_best.pt` (CURRENT PRODUCTION, also saved as
     `unet_best_25board_capweight_dice05.pt`)** — same 23/2 board split,
     retrained with a capped class-weight ceiling (`--max-class-weight
     1.5`) + Dice loss (`--dice-weight 0.5`) to fix `trace`/`silkscreen`
     (see the loss/sampling experiments below and §6 item 3 for the full
     mechanism). Full-res eval: **mIoU 0.932**, pixel accuracy 0.992 — a
     broad improvement over the CE-only 25-board run above, with no
     regressions on any class. **This is the checkpoint to use going
     forward** and the one the per-class table below is built from.
  The prior 10-board checkpoint is preserved at
  `train/checkpoints/unet_best_10board.pt`.
- Per-class IoU, full-res eval (`train/evaluate.py`), SAME 2 held-out
  boards (`icicle-hdmi-breakout` + `signal-integrity-test-board`) in both
  columns — this is the real before/after of adding 14 boards (both
  columns CE-only; see further down this section for the current
  production numbers after the capped-weight + Dice loss change):

  | class | 10-board baseline | 25-board retrain (matched val) | delta |
  |---|---|---|---|
  | background | 0.996 | 0.996 | flat |
  | copper | 0.975 | 0.963 | -0.012 |
  | component | 0.979 | 0.982 | +0.003 |
  | soldermask | 0.989 | 0.989 | flat |
  | fiberglass | 0.943 | 0.941 | -0.002 |
  | silkscreen | 0.925 | 0.922 | -0.003 |
  | trace | 0.884 | 0.849 | -0.035 |
  | solder | 0.728 | **0.745** | **+0.017** |
  | **mean IoU** | **0.927** | **0.923** | -0.004 |
  | pixel accuracy | 0.991 | 0.991 | flat |

  **Read on this**: essentially flat overall (mean IoU within noise, pixel
  accuracy identical to 3 decimal places) — adding 14 new, meaningfully
  different boards did NOT hurt performance on the original held-out pair,
  which is itself a good sign for generalization (new board diversity isn't
  fighting the model). `solder`, the known weakest class, improved
  (+0.017) — a small step in the right direction but not a dramatic fix;
  still the clear laggard by a wide margin. `trace` dropped the most
  (-0.035) — worth watching in a future retrain/eval round but not
  investigated further this session (single data point, could be
  noise from just 2 held-out boards).

  (Scratch numbers from the random-val run, NOT comparable board-for-board
  since the val set differs — kept only to show the random split isn't
  pathological: mIoU 0.888, with `fiberglass` 0.640 and `copper` 0.884 on
  those two small, low-fiberglass-exposure boards specifically.)

- **`solder` loss/sampling experiments — DONE this session, both negative
  results.** Tried two targeted fixes for `solder` (matched val boards,
  same eval methodology as above):

  | variant | mIoU | solder IoU |
  |---|---|---|
  | CE-only (`unet_best_25board_ce_only.pt`) | 0.923 | 0.745 |
  | + Dice loss (`dice_weight=0.5`) | 0.930 | 0.748 |
  | + Dice loss + solder-oversample patches (`oversample_prob=0.3`) | — | 0.750 |

  Both correctly implemented and verified working as intended, but neither
  meaningfully moved `solder` (0.745 → 0.748 → 0.750, vs. other classes
  gaining more broadly under Dice loss). Root cause found — see §6 item 2 —
  this is a structural ceiling, not a training-signal problem, so further
  loss/sampling tuning specifically targeting `solder` isn't worth
  pursuing. Checkpoints kept on disk for reference:
  `unet_best_25board_dice05.pt`, `unet_best_25board_dice05_oversample03.pt`.

- **`trace`/`silkscreen` fix — DONE this session, clear win, NOW
  PRODUCTION.** Root-caused via an Explore-agent confusion-matrix + code
  trace (see §6 item 3 for the mechanism): `trace`'s errors were a
  boundary-precision problem (thin class, 1-a-few px wide, losing area at
  the `mask_open` transition to `copper`/`fiberglass`), and `silkscreen`'s
  were a CE class-weight-imbalance problem (`solder`'s inverse-frequency
  weight, 1.80, dominated nearby ambiguous pixels that should have gone to
  `silkscreen`, weight 1.25). Fix: (1) capped CE class weights at 1.5
  (`train/dataset.py`'s `class_weights(..., max_weight=1.5)`, new
  `--max-class-weight` flag in `train.py`, printed pre/post-cap table at
  startup) — clips `solder` 1.80→1.50 and `fiberglass` 2.05→1.50; (2)
  re-enabled Dice loss (`--dice-weight 0.5`, same mechanism as the earlier
  `solder` experiment, but effective here since `trace`'s problem is
  genuinely boundary/region-overlap-shaped, unlike `solder`'s occlusion
  ceiling). Retrained on the same 23 train / 2 matched val boards:

  | class | CE-only baseline | capweight(1.5)+dice05 | delta |
  |---|---|---|---|
  | background | 0.995 | 0.995 | flat |
  | copper | 0.963 | 0.971 | +0.008 |
  | solder | 0.745 | 0.752 | +0.007 |
  | fiberglass | 0.941 | 0.956 | +0.015 |
  | soldermask | 0.989 | 0.990 | +0.001 |
  | silkscreen | 0.922 | **0.928** | **+0.006** |
  | component | 0.982 | 0.983 | +0.001 |
  | trace | 0.849 | **0.881** | **+0.032** |
  | **mean IoU** | **0.923** | **0.932** | **+0.009** |

  Broad win, no regressions anywhere — both target classes improved (as
  predicted by the mechanism), `solder` even ticked up slightly instead of
  regressing from the weight cap, and overall mIoU improved too. **This
  checkpoint is now `train/checkpoints/unet_best.pt` (production)**, saved
  also as `unet_best_25board_capweight_dice05.pt`. The prior CE-only
  production checkpoint is preserved as `unet_best_25board_ce_only.pt`.
  `train/train.py`'s defaults are UNCHANGED (`--dice-weight 0.0`,
  `--max-class-weight None`) — this run used explicit flags
  (`--dice-weight 0.5 --max-class-weight 1.5`); a future session
  regenerating the production checkpoint from scratch needs to pass those
  flags explicitly, they are not the CLI defaults.

- **RETRAINED AGAIN 2026-07-27, on the 31-board component-model-recovery
  dataset (§3 item 12), same production config
  (`--val-boards icicle-hdmi-breakout,signal-integrity-test-board
  --dice-weight 0.5 --max-class-weight 1.5`), NOW PRODUCTION.** Old
  25-board checkpoint preserved as
  `train/checkpoints/unet_best_pre_3dfix_rollout.pt` before overwriting.
  Full-res eval (`train/evaluate.py`), same matched val boards:

  | class | 25-board (pre-3D-fix) | 31-board (post-3D-fix) | delta |
  |---|---|---|---|
  | background | 0.9952 | 0.9957 | +0.0005 |
  | copper | 0.9712 | 0.9748 | +0.0036 |
  | solder | 0.7521 | 0.7516 | -0.0005 |
  | fiberglass | 0.9564 | 0.9495 | -0.0069 |
  | soldermask | 0.9904 | 0.9898 | -0.0006 |
  | silkscreen | 0.9278 | 0.9265 | -0.0013 |
  | component | 0.9834 | 0.9843 | +0.0009 |
  | trace | 0.8807 | 0.8781 | -0.0026 |
  | **mean IoU** | **0.9321** | **0.9313** | **-0.0008** |
  | pixel accuracy | 0.9922 | 0.9919 | -0.0003 |

  **Essentially flat, as expected — and NOT a verdict on the fix.** Every
  delta is within this project's own established noise floor (past
  experiments treated 0.003-0.01 swings as negligible). Critically, neither
  held-out val board (`icicle-hdmi-breakout`, `signal-integrity-test-board`)
  had ANY `.wrl`/SparkFun model references needing repair
  (`_3dfix_report.json` shows `wrl_fixed: 0` for both) — so this comparison
  structurally cannot show the fix's benefit; it only shows that adding 6
  more boards + fixing geometry elsewhere didn't hurt generalization to
  these 2 untouched boards, which is itself a reasonable sanity check, same
  pattern as the flat 10→25-board comparison earlier in this file. The
  actual accuracy win is on the 16 affected boards' own `component`/`trace`/
  height channels (see §2), not measurable via this specific val pair.
  **Promoted to production** on the reasoning that the underlying dataset
  is strictly more accurate (real recovered geometry, not silently-missing
  meshes) with no measured downside — same judgment call this file made for
  the flat 10→25-board retrain.

- **RETRAINED A THIRD TIME 2026-07-27 (same day, second round), on the
  kicad-packages3d-tier + fuzzy-match-tier dataset (§3 item 13), same
  production config, NOW PRODUCTION.** Prior checkpoint preserved as
  `train/checkpoints/unet_best_pre_kicad3dfix.pt`. Full-res eval, same
  matched val boards:

  | class | 31-board (post-3D-fix round 1) | 31-board (post-kicad3d+fuzzy) | delta |
  |---|---|---|---|
  | background | 0.9957 | 0.9960 | +0.0003 |
  | copper | 0.9748 | 0.9688 | -0.0060 |
  | solder | 0.7516 | 0.7486 | -0.0030 |
  | fiberglass | 0.9495 | 0.9520 | +0.0025 |
  | soldermask | 0.9898 | 0.9900 | +0.0002 |
  | silkscreen | 0.9265 | 0.9280 | +0.0015 |
  | component | 0.9843 | 0.9847 | +0.0004 |
  | trace | 0.8781 | 0.8716 | -0.0065 |
  | **mean IoU** | **0.9313** | **0.9300** | **-0.0013** |
  | pixel accuracy | 0.9919 | 0.9920 | +0.0001 |

  **Same story as before, for the same reason**: again essentially flat,
  again within this project's established noise floor, and again NOT
  evidence the fix didn't help — `icicle-hdmi-breakout` and
  `signal-integrity-test-board` had ZERO `.wrl` references needing repair
  in the first place (`wrl_fixed: 0, wrl_unfixable: 0` from the very start),
  so this comparison was never going to move on these two boards regardless
  of how much the other 29 boards' geometry improved. **Promoted to
  production** on the same reasoning as the prior round: strictly more
  accurate underlying data (25 more real refs recovered registry-wide, one
  wrong-part substitution caught and blocked before it could ever reach
  production), no measured downside.

- **GT rendering-noise fix — IN PROGRESS, same day 2026-07-27 (third
  round), see §3 item 14.** The GT pass in `07_render_mitsuba.py` used a
  hardcoded `samples=4` (vs. `samples=32` for the band images) — a
  controlled diagnostic (2 boards, samples=4 production GT vs. a fresh
  samples=64 render) proved this causes real anti-aliased class-boundary
  noise: ~99.8-99.9% of GT pixels that flip class between the two sample
  counts sit exactly on a class boundary, and **57-65% of the model's
  "wrong" pixels were against a GT pixel that itself flipped class at
  higher sample count** — i.e. a majority of visible boundary "error" was
  GT noise, not model failure. Added `--gt-samples` (default 4, unchanged
  behavior) to `07_render_mitsuba.py` and `09_batch.py`, re-rendered the
  full 31-board dataset at `--gt-samples 32`. **Re-scoring the OLD,
  unretrained checkpoint (`unet_best_pre_gtsamples_fix.pt`) against the
  cleaner GT alone** (no training changed):

  | class | old (samples=4 GT) | re-scored only (samples=32 GT) | delta |
  |---|---|---|---|
  | background | 0.9960 | 0.9969 | +0.0009 |
  | copper | 0.9688 | 0.9743 | +0.0055 |
  | solder | 0.7486 | **0.8470** | **+0.0984** |
  | fiberglass | 0.9520 | 0.9643 | +0.0123 |
  | soldermask | 0.9900 | 0.9927 | +0.0027 |
  | silkscreen | 0.9280 | 0.9648 | +0.0368 |
  | component | 0.9847 | 0.9896 | +0.0049 |
  | trace | 0.8716 | 0.9070 | +0.0354 |
  | **mean IoU** | **0.9300** | **0.9546** | **+0.0246** |

  **Read carefully — this does NOT mean `solder`'s previously-documented
  structural occlusion ceiling (§6 item 2) was wrong.** That mechanism (a
  mounted component genuinely hiding the solder joint beneath it) is still
  real and still limits how much `solder` area is even visible to score
  against. What this shows is that ON TOP OF that real ceiling, the
  low-sample GT was ALSO adding a large amount of pure measurement noise —
  the true model performance was being substantially understated by both
  effects stacked together, and this fix removes the noise component while
  leaving the physical occlusion limit untouched. A retrain against this
  cleaner dataset (production config, prior checkpoint backed up first as
  `unet_best_pre_gtsamples_fix.pt`) was kicked off immediately after this
  measurement — **check `train/checkpoints/unet_best.pt`'s timestamp
  against that backup to see whether it finished; if so, this entry needs
  a fourth row/column added with the retrained (not just re-scored) result
  before this is considered done.

  **DONE — retrained, evaluated, promoted to production.** Retraining
  against the cleaner dataset pushed accuracy further than just re-scoring
  the old checkpoint did, confirming the model genuinely benefits from
  training on the cleaner boundary signal (not merely being judged more
  favorably by it):

  | class | original (samples=4 GT) | re-scored old ckpt | **retrained** |
  |---|---|---|---|
  | background | 0.9960 | 0.9969 | 0.9971 |
  | copper | 0.9688 | 0.9743 | **0.9882** |
  | solder | 0.7486 | 0.8470 | **0.8725** |
  | fiberglass | 0.9520 | 0.9643 | 0.9694 |
  | soldermask | 0.9900 | 0.9927 | 0.9935 |
  | silkscreen | 0.9280 | 0.9648 | 0.9695 |
  | component | 0.9847 | 0.9896 | 0.9913 |
  | trace | 0.8716 | 0.9070 | **0.9322** |
  | **mean IoU** | **0.9300** | **0.9546** | **0.9642** |
  | pixel accuracy | 0.9920 | 0.9942 | 0.9953 |

  **This is the single largest quality improvement found this session**,
  and it came entirely from a data-generation fix (GT sample count), not
  any loss/architecture change — `solder` +0.124 absolute, `trace` +0.061,
  `copper` +0.019, no regressions anywhere. `unet_best.pt` is this
  checkpoint; prior checkpoint preserved as
  `unet_best_pre_gtsamples_fix.pt`. This item and §3 item 14 are both
  CLOSED — nothing further pending from this round.**

## 3. How we got here (chronological narrative)

1. **Original ask**: synthetic PCB material-segmentation dataset from real
   Gerbers, matching a physical capture rig (16-light ring, R/G/B/IR/UV
   LEDs, 60° elevation, 385mm radius — see `rig.py`).
2. **Major pivot**: user realized Blender/Cycles can't render non-visible
   light (IR/UV) — switched the whole render backend to **Mitsuba3**
   (physically-based, real spectral capability). Also switched the
   deliverable from a 48-image angle-stack to **5 flat band images**
   (R,G,B,IR,UV, all 16 lights combined, shadows minimized) **+ a
   populated STL** (component 3D models via KiCad STEP export).
3. Built the whole `00`-`09` numbered pipeline (see README's file table).
   Found and fixed several real Mitsuba/trimesh bugs along the way
   (documented in README's "gotchas" section: orthographic scale must be
   isotropic, `bitmap` texture gamma decoding, `trimesh.Scene` local-vs-
   world-space transforms, AOV alpha-premultiply, AOV channel indices
   depend on `pixel_format`, emitter radiance needs an explicit colour
   dict).
4. **Component rendering**: originally the STL and the Mitsuba renders
   were two disconnected geometry pipelines (Gerber-derived flat mesh vs.
   KiCad's STEP with real 3D parts). Merged them: `06_export_stl.py`
   aligns a components-only STEP export into gerber2blend's coordinate
   space via a board-body-shell anchor match, verified to ~0.1mm agreement.
5. **`trace` GT class**: user noticed GT didn't show trace routing under
   intact soldermask (only `copper`= exposed metal). Added class 7
   `trace` = covered copper, sourced from `F_Cu.png` regardless of mask
   state; visible-light bands render it identically to `soldermask`
   (no sourced visible-light transparency claim) — only GT + IR
   distinguish it.
6. **Height channel**: user wants U-Net trained on all 5 bands **+ a
   height map from the STL**. Added a 6th channel: `07_render_mitsuba.py`
   renders height via Mitsuba's `position` AOV through the SAME mesh/
   camera as everything else (not a separate STL rasterization pass) —
   16-bit PNG in micrometres, zeroed at the bench (`ground_plane()`'s Z).
7. **U-Net training pipeline** (`train/`): dataset loader (board-level
   split, random-crop patches, geometric-only augmentation), a
   from-scratch 4-level U-Net (6→8 channels), inverse-frequency class
   weighting, full-res sliding-window eval, error-heatmap visualization
   with legends. Trained, got the numbers in §2.
8. **Gating reorg**: user pointed out STL/component viability should be
   checked BEFORE the expensive gerber2blend/Mitsuba steps, not after —
   split `06_export_stl.py` into `02b_stl_precheck.py` (cheap, right after
   Gerber fetch, no Blender dependency) + `06_export_stl.py` (finalize,
   reuses precheck's cached STEP/STL + anchor bounds). `09_batch.py`
   restructured into 3 phases, all parallelized (`ThreadPoolExecutor`,
   `--workers`).
9. **Board diversity search** (3 rounds, all documented in
   `02_fetch_gerbers.py`'s comments): tried 43 new candidate boards across
   Antmicro (30) and SparkFun (13). **Only 1 survived**
   (`sparkfun-spectral-as7343`). Along the way found and fixed two real
   bugs (see §4) and one real, separate pipeline limitation (§6).
10. **Round 4 board diversity search** (2026-07-21): switched to two
    genuinely different source orgs never tried before — **Great Scott
    Gadgets** (RF/USB test equipment: HackRF, Cynthion, URTI) and **Libre
    Solar** (solar charge controllers / BMS boards). Candidates were
    pre-filtered via GitHub's git-trees API (recursive tree listing) to
    confirm a `.kicad_pcb` exists before spending a clone+precheck on them
    — 16 candidates tried, **14 survived** (populated AND fully
    image+verify), by far the best hit rate of any round (vs. ~1/43 for
    rounds 1-3). Dataset grew **11 → 25 boards**. Found and fixed one real
    bug: several older Libre Solar boards (`mppt-1210-hus`,
    `mppt-2420-rc/hpx/lc`, `pwm-2420-lus`) have their F.Cu/B.Cu copper
    layers custom-renamed to "Top"/"Bottom" in KiCad board setup —
    `kicad-cli`'s gerber export uses the CURRENT layer name, not the
    canonical designator, so these emitted `*-Top.gbr`/`*-Bottom.gbr`
    instead of `*-F_Cu.gbr`/`*-B_Cu.gbr`, crashing gerber2blend's
    solder-paste compositing step. Fixed with a rename alias in
    `02_fetch_gerbers.py` (same pattern as the existing F_SilkS ->
    F_Silkscreen alias), then re-verified all 5 affected boards end-to-end.
    Two candidates confirmed genuinely bare and dropped per §5's policy:
    `amalthea-hardware` (Z-ratio 2.7%, 1 shell), `libresolar-rpi-can`
    (Z-ratio 2.3%, 1 shell). Retrained the U-Net on the new 25-board
    dataset (see §2, §7).
11. **Round 5 board search (2026-07-24, code exists but was never logged
    here until now — backfilled 2026-07-27).** Per `02_fetch_gerbers.py`'s
    own round-5 comment block: more boards from the two proven high-hit-rate
    orgs (GSG, Libre Solar) not tried in round 4, plus a new org
    (ProtoCentral, sensor-breakout hardware, CERN-OHL-P-2.0). 6 survived:
    `libresolar-data-manager`, `libresolar-bms-c1`,
    `libresolar-mppt-2420-hc`, `protocentral-fdc2214`,
    `protocentral-tmf8829`, `protocentral-st1vafe3bx`. Confirmed bare and
    dropped: GSG's `lna1109`/`lna3030`/`tycho` and 5 GreatFET neighbor-board
    monorepo boards (suggesting the whole GreatFET neighbor family lacks 3D
    models), plus `protocentral-sensything` (a per-board exception, not an
    org-wide pattern like GSG/GreatFET). Registry grew 25 → 31.
12. **Component-3D-model-recovery fix (implemented 2026-07-24, validated +
    rolled out 2026-07-27).** Root cause (see §4's new bullet): many
    boards' `kicad-cli pcb export step` was silently dropping real component
    geometry for two fixable reasons — legacy `.wrl` references KiCad's
    exporter refuses outright even when a real `.step` equivalent already
    exists at the same path in KiCad's bundled library, and unresolved
    `${SPARKFUN_KICAD_LIBRARY}` references (needs a clone + a path-shim,
    since the upstream repo reorganized its layout since these board files
    were saved). Fixed in `stl_common.py` (`patch_wrl_references`,
    `ensure_sparkfun_library_shim`, `prepare_step_export_pcb`) and wired
    into every STEP-export call site in `02b_stl_precheck.py` and
    `06_export_stl.py` — but that 2026-07-24 session ended with the fix
    implemented and import-clean, **never actually run**. This session
    (2026-07-27): validated on 2 known cases first
    (`libresolar-uext-esp32`: 3 `.wrl` fixed → mesh 1,964→12,704 faces;
    `sparkfun-spectral-as7343`: SparkFun shim built fresh, 324 symlinks →
    23 shells/12,947 faces, paste-pixel component coverage 49.7%, both
    close to or better than the 2026-07-24 session's standalone lab
    numbers), then rolled out to the full 31-board registry
    (`09_batch.py --mpp 0.045 --samples 32`, 31/31 populated+imaged+
    verified, 0 dropped), then retrained (see §2's retrain entry). 16/31
    boards (52%) had real fixes applied, several heavily (`urti-mainboard`
    402 refs, `hackrf-pro` 294, `cynthion-hardware` 273) — full per-board
    fix counts in `.cache_repos/*/_3dfix_report.json`.
13. **Chasing the remaining ~50 unfixable refs (same day, second round,
    2026-07-27).** After item 12's fix, `_3dfix_report.json` still showed
    53 unresolved refs across 12 boards. Investigated (read-only, via
    GitHub's API) what they actually were before writing any more code —
    found they split into three cases: (a) refs genuinely absent from this
    machine's bundled KiCad.app library but present at the exact same path
    in the full upstream `github.com/KiCad/kicad-packages3d` repo (e.g.
    `SOIJ-8_5.3x5.3mm_P1.27mm`) — the local bundle is evidently a partial
    snapshot, not the complete official library; (b) refs renamed upstream
    between KiCad library versions (e.g. `SOIC-8_5.23x5.23mm_P1.27mm` →
    upstream's current `SOIC-8_5.275x5.275mm_P1.27mm`, a dimension
    correction; `JST_PH_B2B-PH-SM4-TB_1x02-1MP_P2.00mm_Vertical` →
    `JST_PH_B2B-PH-K_1x02_P2.00mm_Vertical`, a variant-suffix rename); (c)
    refs with no plausible match anywhere (e.g. the custom
    `LibreSolar.3dshapes/DTMSS-27-H` library, which has no discoverable
    public 3D-model repo — checked `LibreSolar/kicad-footprints`, footprints
    only, no geometry).

    Implemented case (a) as a new exact-match tier in `stl_common.py`
    (`ensure_kicad_packages3d_repo()`, cloned lazily only when tier 1 leaves
    something unresolved — the full repo is a ~5.8GB clone, much bigger
    than the SparkFun shim, not worth paying unconditionally) —
    zero-ambiguity, applied unconditionally like the existing fixes.
    Recovered 16 refs registry-wide (53→37 unfixable), verified via
    shell-count deltas matching `wrl_fixed` deltas exactly on all 12
    affected boards, no regressions.

    Implemented case (b) as an opt-in fuzzy-match tier
    (`find_kicad_packages3d_fuzzy_match()`) — same-category candidates
    sharing the ref's alphabetic prefix, picked by difflib similarity above
    a 0.75 cutoff. **First version caught a real bug in its own validation,
    not a hypothetical one**: it fuzzy-matched
    `QFN-16-1EP_3x3mm_P0.5mm_EP1.45x1.45mm` (on `cynthion-hardware`) to
    `QFN-12-1EP_3x3mm_P0.5mm_EP1.65x1.65mm` — WRONG, a genuinely different
    12-pin package, close enough in string similarity (shared body size,
    shared pitch) to clear the cutoff anyway. Fixed by adding a hard
    pre-filter (`_leading_pin_count()`): the first numeric token right
    after the alphabetic prefix (almost always a real pin/position count)
    must match EXACTLY between ref and candidate, not just score well on
    overall similarity — dimension tokens later in the name may still
    legitimately differ (that's what makes it a correction, not a different
    part), but the pin count never may. Re-validated after the fix: all
    remaining matches preserve pin count exactly (SOIC-8 dimension
    correction ×2 boards, JST_PH suffix rename ×3 refs on one board,
    JST_XH leading-zero variant ×3 boards) — one was spot-checked by
    name/dimensions in the actual exported STEP geometry (`JST_B2B_XH_A`,
    5.75×7.4×10.4mm, a plausible real vertical 2-pin header), and
    shell-count deltas matched expected fix counts exactly on every
    affected board. One interesting side-finding: `cynthion-hardware`'s 2
    fuzzy matches turned out to be on `hide`-flagged (inactive) model
    blocks with a working non-hidden sibling reference already providing
    the real geometry — so those two guesses were provably harmless
    regardless of match quality, a genuine KiCad footprint construct
    (legacy/alternate model references kept alongside an active one), not
    a bug.

    Rolled out registry-wide (53→28 unfixable after both tiers), retrained
    (see §2's third checkpoint entry), same essentially-flat result on the
    matched val boards for the same reason as round 1 (neither val board had
    any refs needing repair to begin with).
14. **GT rendering-noise investigation (same day, third round, 2026-07-27)
    — DONE, see §2's final entry for the full mechanism/numbers.**
    Prompted by the user noticing errors concentrate at feature boundaries
    in the red-diff visualizations from this session (predicted
    segmentation with wrong pixels painted red — see the `pred_red/*.png`
    outputs). Investigated the GT rendering path itself, not just the model,
    before assuming more loss/architecture tuning was the answer: found
    `render_gt()` used `samples=4` vs. band images' `samples=32`, and
    proved via controlled comparison (samples=4 vs. samples=64 GT renders,
    2 boards) that this causes real, sizeable anti-aliasing noise at class
    boundaries — 57-65% of the model's apparent "wrong" pixels were against
    a GT reference that itself was unstable at low sample count. Fixed by
    parameterizing the sample count (`--gt-samples`, `07_render_mitsuba.py`
    + `09_batch.py`, default unchanged) and re-rendering the full registry
    at `--gt-samples 32`. Retrained and evaluated: **mIoU 0.9300 → 0.9642**
    (`solder` 0.749→0.873, `trace` 0.872→0.932) — the single largest quality
    improvement found this session, and it required zero model/loss
    changes, only a data-generation fix.

## 4. Bugs found and fixed this session (don't reintroduce these)

- **Orthographic camera scale must be isotropic** (Mitsuba). Anisotropic
  `to_world` scale is silently not applied per-axis independently.
- **`bitmap` texture gamma decoding** — always pass `raw: True`, our PNGs
  encode literal reflectance/label values, not photographic colour.
- **`trimesh.Scene.geometry` is in LOCAL mesh space** — must use
  `scene.to_geometry()` to bake in world transforms, not raw
  `trimesh.util.concatenate()`. This silently produced WRONG (but
  plausible-looking) STLs for every multi-part board.
- **AOV alpha-premultiply + `pixel_format`-dependent channel indices**
  (Mitsuba) — see README's gotchas list for exact details.
- **GT palette collision** when a new class is added — scope the
  nearest-palette-match candidate set to classes actually present in that
  render, not the full palette.
- **The "populated" STL heuristic had a false-positive hole**: Z-ratio
  ≥3% alone can trip on a small single-shell board's bare substrate
  thickness, with ZERO actual components. Fixed by also requiring
  `n_geoms > 1` (`02b_stl_precheck.py`, `06_export_stl.py`). Caught via
  `pmod-i3c-sensor-board`/`pdm-microphone-board` during the SparkFun
  search — both looked "populated" until this fix, both were bare.
- **Non-recursive `glob()` vs. `rglob()` for cached `.kicad_pcb`** —
  `02b_stl_precheck.py`/`06_export_stl.py` used non-recursive `glob()`,
  missing repos (e.g. SparkFun's) that nest the file under a `Hardware/`
  subdirectory rather than the repo root. `02_fetch_gerbers.py` already
  used `rglob()` correctly. This silently mis-flagged genuinely-good
  boards as "crashed." Fixed to `rglob()` in both files. This is WHY
  `sparkfun-spectral-as7343` was initially miscategorized before being
  correctly identified as the one real win of the SparkFun search.
- **zsh word-splitting**: `for b in $SOME_SPACE_SEPARATED_STRING` does
  NOT word-split in zsh (unlike bash) — silently iterates once with the
  whole string as one token. Use `arr=(a b c); for b in "${arr[@]}"`.
  Bit us once during disk cleanup (a `rm -rf` no-op'd on a bogus
  multi-word path instead of deleting anything).
- **Custom-renamed F.Cu/B.Cu layers export as `*-Top.gbr`/`*-Bottom.gbr`**,
  not `*-F_Cu.gbr`/`*-B_Cu.gbr` (found on several older Libre Solar
  boards, round 4). `kicad-cli`'s gerber export uses the CURRENT
  board-setup layer name, not the canonical F.Cu/B.Cu designator, so a
  board whose author renamed the display name (e.g. to "Top"/"Bottom")
  breaks gerber2blend's hardcoded F_Cu/B_Cu solder-paste compositing.
  Fixed with a rename alias in `02_fetch_gerbers.py` (mirrors the existing
  F_SilkS -> F_Silkscreen alias for the same reason: older/nonstandard
  layer naming vs. gerber2blend's fixed expectations).
- **macOS has no `timeout` command by default** (unlike Linux) — a bash
  script using `timeout N cmd` fails with "command not found" (exit 127),
  not a real timeout. Either install GNU coreutils' `gtimeout` or just
  drop the wrapper if the command isn't expected to hang.
- **`kicad-cli pcb export step` silently drops real component geometry**
  for two distinct, both-fixable reasons (found 2026-07-24, rolled out
  2026-07-27, see §3 item 12): (1) legacy `.wrl` (VRML) model references —
  the STEP exporter refuses these outright based purely on the extension,
  even when a real `.step` file already exists at the same path in KiCad's
  bundled library (these boards just predate KiCad's own .wrl→.step
  migration); (2) `${SPARKFUN_KICAD_LIBRARY}`-style env-var references to a
  library that's a real, cloneable repo but has since reorganized its
  directory layout, so a naive clone+env-var-set still doesn't resolve.
  Both fixed in `stl_common.py`'s `prepare_step_export_pcb()`, applied
  unconditionally (no opt-out) since they recover real, not approximated,
  manufacturer geometry. Affects roughly half of a typical multi-org
  registry — check `.cache_repos/*/_3dfix_report.json` on any new board
  before assuming its populated STEP export is complete.
- **Fuzzy-matching renamed 3D-model refs by string similarity alone WILL
  pick a wrong, different-pin-count part** (found 2026-07-27, see §3 item
  13): a `QFN-16` reference nearly got silently substituted with a
  `QFN-12` upstream candidate because their names otherwise share body size
  and pitch, scoring high on plain difflib similarity. Any future fuzzy/
  approximate-matching logic over part names MUST separately verify a
  pin/position-count-like token matches exactly — don't rely on overall
  string similarity as the sole gate. See `stl_common.py`'s
  `_leading_pin_count()`.

## 5. Registry / disk hygiene policy (established this session)

When a candidate board is tried and confirmed unusable (bare, Eagle-format,
fetch failure, etc.), **remove it from `02_fetch_gerbers.py`'s `BOARDS`
dict entirely** (not left as dead weight that gets re-fetched every run) —
document WHY in a comment block near where it would have been, so a future
session doesn't waste time re-trying the same dead end. **Also delete its
on-disk leftovers**: `boards/<name>/`, `.cache_repos/<name>/`,
`out/<name>/` if any exist. Keep `boards/`, `.cache_repos/`, `out/`, and the
registry in exact 1:1 sync with what's actually in the dataset — verified
multiple times this session with a simple `ls` comparison.

Exception: boards excluded for a reason OTHER than "just bad" get a more
detailed comment explaining the distinct reason, since they might be worth
revisiting later if the blocking issue gets fixed:
- `jetson-orin-baseboard` — private, unresolvable `${ANT3DMDL}` 3D-model
  library env var (from earlier in the project, before this handoff's
  scope).
- `ov5640-dual-camera-board` — genuinely populated (136 shells, 28.5%
  Z-ratio), but its silkscreen is 100% on the BACK face and this pipeline
  only images/labels the FRONT face. Worth re-trying if back-face imaging
  is ever added (see §6).

## 6. Known limitations (real, not yet addressed)

1. **Front-face-only imaging.** The whole pipeline only images/labels one
   face of the board. At least one real, otherwise-good board
   (`ov5640-dual-camera-board`) is blocked purely by this. Adding
   back-face rendering would be genuine new capability, not a bug fix —
   scope it as its own task if pursued (need a second camera pass, a
   decision on whether back-face becomes a second 6-channel+GT sample or
   gets composited somehow, and probably a second `ground_plane()`
   orientation).
2. **`solder` is the weakest class** (IoU ~0.745 vs 0.85+ everywhere else)
   — **root cause found this session, resolved (structural, not fixable by
   training changes).** `solder`'s GT starts from the flat Gerber
   paste-stencil layer (`spectral.build_class_id_map()`), but the final
   rendered GT reflects TRUE 3D occlusion: `06_export_stl.py`'s aligned
   `components.obj` is composited as a real occluding mesh in
   `07_render_mitsuba.py`, and populated component bodies sit almost
   exactly where their own solder joints are. Measuring flat (pre-occlusion)
   vs. rendered (post-occlusion) solder pixel counts on the two held-out
   boards: `icicle-hdmi-breakout` 55,928 → 2,921 px (5.2% survives),
   `signal-integrity-test-board` 589,280 → 11,739 px (2.0% survives). A
   confusion-matrix breakdown confirms this mechanism: `solder`'s errors are
   dominated by confusion with `component` (9.6-11.3%), not `copper`
   (0-1.7% — ruling out an earlier spectral-similarity hypothesis based on
   IR/UV reflectance closeness in `spectral.py`'s `REFLECTANCE` dict).
   **This occlusion is physically correct** (a mounted component genuinely
   hides the solder joint beneath it from a top-down camera), not a
   labeling bug — but it means the visible `solder` class is an inherently
   thin, boundary-sensitive sliver (2-5% of its own nominal area), governed
   by the ~0.1mm alignment tolerance between the flat board mesh and the
   separately-aligned components mesh (§3 item 4). That sets a hard
   structural ceiling that per-pixel loss reweighting or patch oversampling
   cannot move — confirmed empirically, see §2's Dice/oversample results
   (0.745 → 0.748 → 0.750, negligible). Not recommended to pursue further
   via training changes; see §7 item 2 for the actual lever if this is ever
   revisited.
   - **Follow-up: no solder-JOINT geometry exists anywhere in this
     pipeline, confirmed this session.** The user asked whether the
     synthetic data fails to represent real solder CONNECTIONS (the
     fillet/meniscus visible at a lead, not just an occluded flat paste
     footprint) — confirmed true. `06_export_stl.py`'s component 3D models
     (from `kicad-cli pcb export step --subst-models`) are package BODIES
     only — KiCad/manufacturer library models essentially never include
     separate lead/pin geometry. `solder` GT has only ever been a flat 2D
     paste-stencil footprint (itself already undersized ~80-90% vs. the
     real pad, to prevent bridging) with zero fillet/meniscus shape
     modeled. Through-hole (THT) barrel-fill joints — often the most
     visually obvious solder on a real board — aren't modeled at all
     either (drill-layer data is exported and used for board-mesh holes,
     never for solder GT). This is a genuine geometry-synthesis gap, not a
     parameter to tune.
   - **Tried: swap `solder` GT source from `F_Paste.png` to
     `F_Solder.png`, REVERTED — real, mechanistic negative result.**
     Discovered `03_run_gerber2blend.py` already enables gerber2blend's
     `EFFECTS.SOLDER`, which grows the raw paste mask ~5-55% (varies a lot
     per board) into `F_Solder.png` and extrudes a matching 3D "solder
     dome" already merged into `board.obj` — but `spectral.py` was still
     reading the smaller `F_Paste.png`, so that dome's outer ring was
     mislabeled as soldermask/copper in every GT and band image. Looked
     like a clean, free bug fix, so it was tried: switched
     `spectral._layer_masks()` to prefer `F_Solder.png`, regenerated all 25
     boards' GT/bands, retrained (same capweight+dice05 config as
     production). Result: **regressed** — solder IoU 0.752→0.732, mIoU
     0.932→0.927, with `copper`/`fiberglass`/`trace` also drifting down
     slightly. Root-caused why: `F_Solder.png`'s growth is a generic
     fixed-pixel morphological dilation, not a physically-motivated
     stencil-to-pad correction — on dense/fine-pitch boards it BRIDGES
     adjacent, separate pads into one connected blob (confirmed via
     `scipy.ndimage.label` blob-count comparison: `urti-mainboard` lost 107
     pads to merging, `hackrf-pro` 96, `cynthion-hardware` 45; sparser
     boards like the two val boards had zero merges — explaining why the
     regression showed up broadly rather than just on `solder`). Real
     solder joints on adjacent fine-pitch pins don't merge like that (it'd
     be a short) — this is pure image-processing artifact, introducing
     label noise, not more-accurate GT. **Reverted**: `spectral.py` is back
     to `F_Paste.png` (see its `_layer_masks()` docstring/comment for the
     full writeup), all 25 boards' `out/` GT/bands regenerated back to the
     original paste-based state (verified: production checkpoint
     `unet_best.pt` re-evaluates to the exact same mIoU 0.932/solder 0.752
     as before this experiment). **Takeaway for next time**: gerber2blend's
     `F_Solder.png`/`EFFECTS.SOLDER` output should not be trusted as a
     solder-GT source without per-board pad-density-aware handling; any
     future fillet-synthesis work (see below) needs its own pad-boundary
     logic, not a shortcut through this existing-but-flawed asset.
   - **IMPLEMENTED this session (not yet rolled out dataset-wide or
     retrained — paused mid-verification for the day, see below):**
     procedural SMD lead-toe fillet synthesis.
     - **`board_geom.py` (NEW)**: `fab/PNG/*.png` pixel <-> board-mm affine
       transform. Empirically derived AND verified (not assumed) by
       rendering synthetic marker blobs through the real board mesh/camera
       and reading back their landing position via the orthographic
       camera's fully-known formula — confirmed the transform to within
       ~1-2 texture px (camera mpp vs. texture's own finer native
       resolution accounts for the residual). Convention: row 0 = max
       world Y (top of image), col 0 = min world X, uniform scale.
     - **`06b_generate_solder_fillets.py` (NEW)**: for each populated
       board, rasterizes `components.obj`'s top-down triangle footprint
       (plain per-triangle `PIL.ImageDraw.polygon`, NOT per-shell —
       `trimesh.split()` on this merged, `--subst-models`-simplified mesh
       yields ~46,000 tiny/degenerate fragments on a dense board, unusable;
       per-triangle rasterization is simpler AND fast: <1s even at 430k
       triangles, verified on the largest board in the corpus). Computes a
       per-pixel signed-distance-to-component-edge field
       (`scipy.ndimage.distance_transform_edt`, restricted to
       `F_Paste.png`'s pad pixels — NOT `F_Solder.png`, per the dead end
       above) and a tapered wedge height field from it (peak height at a
       small fixed overlap UNDER the component edge, zero at
       `BAND_WIDTH_MM` into the open/visible pad area). Builds a coarse
       (0.05mm) world-mm quad-grid mesh restricted to cells clearing a
       minimum height, avoiding any need for per-blob polygon/contour
       extraction — handles arbitrary pad-blob shapes uniformly. Exports
       `fab/mitsuba/solder_fillets.obj`.
     - **Two real bugs found and fixed while validating on
       `dual-ipex-csi-interposer`** (both would have silently produced
       wrong/invisible geometry across the whole dataset if not caught):
       (1) the wedge's Z origin was first computed from
       `board_bounds.json`'s bbox-derived `center_mm[2] + size_mm[2]/2` —
       WRONG, because that bbox is taken over ALL merged mesh objects in
       `board.obj`, including gerber2blend's own (still-enabled,
       independent of the F_Solder GT dead end) `EFFECTS.SOLDER` dome bump
       (named `"Solder"` in `board.obj`) — so it silently overstated "board
       top" by the dome's peak height, floating the new fillet mesh well
       above the true surface almost everywhere. Fixed: `board.obj` loads
       as a `trimesh.Scene` with consistently-named sub-geometry across
       boards (`main_pcb_top`, `main_pcb_bot`, `main_pcb_edge_bare/gold`,
       `Solder`) — now reads `main_pcb_top`'s own max world Z instead
       (`06b_generate_solder_fillets.py::flat_board_top_z()`). (2) the
       generated quad mesh's triangle winding was backwards, so the
       one-sided area-emitter material (`fillet_shape()`, mirroring
       `component_shape()`) faced away from the top-down camera — rendered
       as pure black / invisible, and the GT solder pixel count didn't
       move at all, which is what caught it. Fixed the winding order
       (verified via a standalone emissive-only render: 0 → 28,135
       nonzero px on the test board once corrected).
     - **`07_render_mitsuba.py`**: added `fillet_shape()` (near-copy of
       `component_shape()`, keyed to `solder`'s reflectance/palette
       instead of `component`'s) and wired `scene_dict["fillets"]` into
       `render_band`/`render_gt`/`render_height`, gated on
       `solder_fillets.obj` existing — same optionality pattern as
       `components_obj`.
     - **Validated on 3 boards** (`dual-ipex-csi-interposer` — 67 shells;
       `urti-mainboard` — dense/fine-pitch; `snapdragon-845-baseboard` —
       largest in the corpus, 430k component triangles): all rendered
       successfully, fast (<5s for fillet generation even on the largest
       board), no crashes. On `dual-ipex-csi-interposer`, rendered `solder`
       GT pixel count went **3,521 → 6,801 (+93%)** — a large, real gain
       in visible solder area, unlike Part A's F_Solder attempt. Minor
       observed side effect worth another look before trusting fully:
       `component`'s pixel count also shifted slightly (+1,650 px on that
       board, ~0.2% of total) — plausible cause is anti-aliasing at the
       new fillet mesh's own boundary edges landing nearest to the
       `component` palette colour at low GT sample counts (a
       previously-documented failure mode in this codebase for boundary
       pixels), not yet root-caused or ruled out as something worse.
     - **DONE 2026-07-24: full rollout, retrain, evaluate — result
       INCONCLUSIVE, not promoted, live dataset REVERTED.** Root-caused the
       `component` pixel-count side effect first (see previous bullet): the
       new "component" pixels are ~99.7% within 1-2px of the new fillet
       geometry, scattered across 647 tiny blobs (mean 2.6px, half
       single-pixel), converted from soldermask/fiberglass — this is GT
       boundary anti-aliasing at `render_gt()`'s fixed `samples=4`, the
       same documented failure mode already called out in
       `07_render_mitsuba.py`'s comments, not a geometry bug. Proceeded:
       ran `06b_generate_solder_fillets.py` + `07_render_mitsuba.py`
       (mpp=0.045, samples=32) across all 25 boards (24/25 populated boards
       got fillet geometry; `libresolar-mppt-2420-rc` has no paste layer at
       all, a pre-existing condition, not new), all 25 re-verified PASS.
       Retrained with the exact production config (`--val-boards
       icicle-hdmi-breakout,signal-integrity-test-board --dice-weight 0.5
       --max-class-weight 1.5`) on the fillet-augmented dataset, evaluated
       full-res on the matched held-out boards:

       | class | production (no fillets) | fillet-augmented | delta |
       |---|---|---|---|
       | background | 0.995 | 0.995 | flat |
       | copper | 0.971 | 0.975 | +0.004 |
       | solder | 0.752 | 0.755 | +0.003 |
       | fiberglass | 0.956 | 0.945 | -0.011 |
       | soldermask | 0.990 | 0.989 | -0.001 |
       | silkscreen | 0.928 | 0.925 | -0.003 |
       | component | 0.983 | 0.979 | -0.004 |
       | trace | 0.881 | 0.874 | -0.007 |
       | **mean IoU** | **0.932** | **0.930** | **-0.002** |

       **Read on this carefully — it is INCONCLUSIVE, not a clean negative
       result, for three concrete reasons**, each worth remembering before
       any future session is tempted to write this off as "tried, made
       things worse" the way the F_Solder experiment genuinely was:
       (1) a -0.002 mIoU delta is *inside this project's own established
       noise floor* — the solder Dice/oversample experiments earlier this
       session (§2) treated 0.745→0.748→0.750 as "negligible," and this
       result is the same magnitude; (2) the two mIoU numbers are scored
       against DIFFERENT ground truths — the fillet-augmented GT has
       roughly double the solder area (more thin, boundary-sensitive
       pixels), a harder and more physically-complete target, so a
       slightly lower score on it is not evidence the geometry is wrong;
       (3) the largest single regression (`fiberglass` -0.011) is in a
       class the fillet mesh doesn't even touch, which is a strong signal
       of run-to-run training noise (single run, no repeats) rather than a
       real causal effect. The `component` regression (-0.004) is the one
       piece that IS mechanistically understood (the anti-aliasing
       artifact above) and is plausibly fixable by excluding fillet-
       boundary pixels from the palette-match candidate set the way
       `component` itself already is on bare boards (see `render_gt()`'s
       `excluded` set) — not attempted, since the aggregate result didn't
       justify chasing it further this session.
       **The actual, deeper blocker surfaced by this exercise**: every
       number in this project (0.932, 0.752, all of it) measures the U-Net
       reproducing `spectral.py`'s OWN compositing rules, not real-board
       performance — there is no real-data ground truth anywhere in this
       pipeline to adjudicate "did more-physically-complete solder GT
       actually help." Investigated real-data options this session (web
       search, not yet acted on): **PCB-Vision** (hifexplo/PCBVision,
       Zenodo, CC BY-NC-SA) is real hyperspectral PCB data (400-1000nm
       VNIR, 224 bands) but is an 11.4GB download that expands to 3.4TB
       (~64GB/scene — this machine has ~1.2TB free, so full extraction is
       impossible and even one scene is a heavy partial-unzip), AND its
       published masks are component-type (IC/capacitor/connector) not
       material-layer, so extracting real copper/solder/FR-4/soldermask
       reflectance from it would mean color-segmenting the RGB by hand as
       a proxy for material identity — real spectra, but only as accurate
       as that heuristic. **FPIC/FICS-PCB** (trust-hub.org) is real
       high-res PCB photography but its public segmentation labels are
       component TYPES (resistor/capacitor/IC/etc. via PCBSegClassNet), no
       overlap with this project's material taxonomy at all. Neither is a
       clean drop-in fix. No dataset found anywhere publishes ground truth
       for this project's exact 8-class material taxonomy — it appears to
       be genuinely unique to this project. **Decision, made explicitly
       this session rather than defaulted into: did NOT chase either real-
       data path further for now** (see §7 item 2) — logged here so a
       future session doesn't have to re-discover the same dead ends.
       **Not promoted.** `train/checkpoints/unet_best.pt` (production,
       mIoU 0.932/solder 0.752) was never touched — the retrain wrote to a
       separate scratch `--out-dir`. **Live dataset REVERTED**: all 25
       boards' `solder_fillets.obj` deleted and all 25 re-rendered back to
       the paste-only state (25/25 re-verified PASS); production
       checkpoint re-evaluated on the reverted dataset and reproduced
       mIoU 0.9321/solder 0.7521 (matches the documented 0.932/0.752 to
       rounding — confirms the revert is clean, same verification pattern
       used for the earlier F_Solder revert). `06b_generate_solder_fillets.py`
       and `board_geom.py` are kept in the tree (not deleted) since the
       geometry-synthesis approach itself isn't disproven, just
       unadjudicated — see §7 item 2 for what would actually resolve it.
       `06b` is still NOT wired into `09_batch.py`'s `phase1_finalize()` —
       intentionally left that way, since it isn't part of the production
       pipeline right now.
   - **RE-TESTED 2026-07-28 against the GT-fidelity-fixed dataset (§2's
     GT-samples entry) — this time DEFINITIVELY NEGATIVE, not
     inconclusive.** The 2026-07-24 "inconclusive" verdict above was always
     suspect because it was measured against the noisy `samples=4` GT — and
     that experiment's own root-cause notes explicitly blamed part of its
     ambiguity on GT anti-aliasing at fillet boundaries, exactly the
     mechanism the GT-samples fix later corrected. So it was re-run
     unmodified against the clean dataset to get an honest answer. Single-
     board validation first (`dual-ipex-csi-interposer`, matching the
     original board): fillet geometry reproduces cleanly (`solder` pixels
     +88%, vs. the original session's +93% on this same board — same real
     mechanism, not a fluke), confirmed via height-map diff (max +0.052mm,
     matching the design's 0.08mm peak-height parameter) and an isolated
     3D view of the mesh, both shown to the user directly (STL exports) for
     visual sign-off before the full rollout. Then rolled out to all 31
     boards, retrained with the identical production config. Result vs. the
     current production baseline (mIoU 0.9642, solder 0.8725, component
     0.9913):

     | class | baseline (no fillets) | with fillets | delta |
     |---|---|---|---|
     | background | 0.9971 | 0.9972 | +0.0001 |
     | copper | 0.9882 | 0.9885 | +0.0003 |
     | solder | 0.8725 | **0.8627** | **-0.0098** |
     | fiberglass | 0.9694 | 0.9750 | +0.0056 |
     | soldermask | 0.9935 | 0.9935 | 0.0000 |
     | silkscreen | 0.9695 | 0.9680 | -0.0015 |
     | component | 0.9913 | **0.9873** | **-0.0040** |
     | trace | 0.9322 | 0.9484 | +0.0162 |
     | **mean IoU** | **0.9642** | **0.9651** | **+0.0009** |

     Overall mIoU ticks up marginally (within noise), but **`solder` — the
     one class this synthesis targets — regressed**, and `component`
     regressed again via the exact same boundary-anti-aliasing mechanism
     identified in the 2026-07-24 attempt. This is now a clean, unambiguous
     negative result, not muddied by measurement noise: the fillet
     geometry's appearance signal (real, and fairly strong where it
     applies — up to 192/255 per-channel colour shift, confirmed via lit
     RGB before/after renders) is confined to too thin a pixel band to
     help the model, consistent with `trace`/`silkscreen`'s own known
     boundary-precision ceiling. **Reverted**: all 31 boards'
     `solder_fillets.obj` deleted, dataset re-rendered back to paste-only
     (31/31 re-verified PASS), production checkpoint restored from
     `unet_best_pre_fillets_retest.pt` and re-evaluated, reproducing
     mIoU 0.9642/solder 0.8725 exactly (confirms the revert is clean, same
     verification pattern used for both earlier reverts). **This question
     is now closed for good** — do not re-attempt fillet synthesis again
     without a fundamentally different geometry/loss approach; the
     appearance-signal-too-thin diagnosis, not a measurement artifact, is
     the real limiter. One process note: the fillet-trained checkpoint
     itself (epoch 39, the one scored above) was not preserved under a
     distinct filename before `unet_best.pt` was overwritten during
     cleanup — not recoverable, but every number from it is fully recorded
     here and the experiment is trivially reproducible from the unmodified
     `06b` script if ever revisited.
3. **`trace` and `silkscreen` weakness — RESOLVED this session (fixable
   training problems, unlike `solder`).** An Explore-agent confusion-matrix
   analysis on the CE-only 25-board checkpoint (matched val boards) found
   two distinct, genuinely fixable mechanisms:
   - `trace` (was IoU 0.849): 87.2% correct, errors dominated by `copper`
     (6.3%) and `fiberglass` (4.8%), NOT `soldermask` (1.4%) despite
     `trace`/`soldermask` being visually identical in R/G/B
     (`spectral.py`'s `build_band_texture()` sets
     `colors["trace"] = colors["soldermask"]`) — the IR-only separating
     signal (trace gets full copper IR reflectance 0.85 vs soldermask's
     0.15) is working fine. The real problem is boundary precision at the
     `mask_open` transition (`_layer_masks()` in `spectral.py`): `trace` is
     only 1-a-few px wide, so a boundary shifted by a pixel flips a large
     fraction of a thin trace segment into the adjacent class. This also
     explains the previously-unexplained `trace` -0.035 regression noted
     in §2's 10-vs-25-board table (new boards shifted mask-boundary
     statistics slightly, and this class is disproportionately sensitive
     to that).
   - `silkscreen` (was IoU 0.922): 94.2% correct, errors almost entirely
     to `solder` (5.8%, asymmetric — solder→silkscreen only 2.3% the other
     way). NOT a band-reflectance issue (IR: silkscreen 0.30 vs solder
     0.80, well separated). Cause: CE class-weight imbalance —
     `solder`'s inverse-sqrt-frequency weight (1.80, the highest of any
     class) dominated `silkscreen`'s (1.25) whenever a small/bright
     ambiguous pixel sat near both (fine silkscreen text often sits
     directly next to solder-paste pads).
   Fix (see §2's capweight+dice05 checkpoint): capped CE class weights at
   1.5 (`train/dataset.py`'s `class_weights(max_weight=...)`, new
   `--max-class-weight` flag) + re-enabled Dice loss (`--dice-weight 0.5`)
   — this time effective, since `trace`'s problem is a genuine
   region-overlap/boundary issue that Dice targets directly, unlike
   `solder`'s occlusion ceiling. Result: `trace` 0.849→0.881 (+0.032),
   `silkscreen` 0.922→0.928 (+0.006), mIoU 0.923→0.932 (+0.009), no
   regressions on any class. Now the production checkpoint.
4. **25 board layouts total now** (up from 11) — round 4 (§3 item 10)
   substantially improved this. Patch sampling still multiplies training
   *examples*, not board *design* diversity, so this is directional
   progress, not a solved problem. Four source orgs searched now: Antmicro
   and SparkFun (43 tried, 1 survivor — genuinely low hit rate, likely
   near-exhausted for those two), Great Scott Gadgets and Libre Solar (16
   tried, 14 survivors — evidently a MUCH better fit for this pipeline's
   requirements, being real hardware companies with dense 3D-modeled
   BOMs). Takeaway: hit rate depends heavily on the SOURCE ORG's hardware
   category (test equipment / power electronics >> generic dev boards),
   not just "try more repos" — worth keeping in mind if searching further.
5. **`ov5640` aside**, the "populated" checks are per-repo heuristics
   (Z-ratio + shell count) — they're a good proxy, not a certainty. Always
   worth a sanity spot-check (see `02b_stl_precheck.py`'s printed mesh
   stats) rather than trusting the boolean blindly on a truly novel board
   source.

## 7. Recommended next steps (in likely priority order)

1. **Retrain — DONE this session**, twice over, with a real matched-val
   comparison against the 10-board baseline — see §2 for the full
   per-class table. Result: essentially flat overall (mean IoU 0.927 →
   0.923), `solder` improved +0.017.
2. **`solder` class — training-side avenues CLOSED, geometry-synthesis
   avenue CLOSED (definitively negative on re-test), real blocker is
   validation, not another idea.** Dice loss + solder-oversampled patch
   sampling both honestly negative (0.745 → 0.748 → 0.750, negligible).
   `F_Solder.png` swap tried and reverted, genuinely negative
   (adjacent-pad bridging, §6 item 2). The procedural SMD lead-toe
   fillet-synthesis pipeline (`06b_generate_solder_fillets.py`) was tried
   TWICE: first 2026-07-24 against noisy GT (inconclusive, mIoU
   0.932→0.930, solder +0.003 — see §6 item 2's original bullet), then
   re-tested 2026-07-28 against the GT-fidelity-fixed dataset with the
   identical, unmodified code (§6 item 2's re-test bullet) — this time
   `solder` itself regressed (0.8725→0.8627) alongside the same
   `component` boundary-anti-aliasing regression as before, a clean,
   unambiguous negative now that measurement noise isn't muddying it.
   **Do not re-attempt fillet synthesis again without a fundamentally
   different geometry/loss approach** — the appearance signal is real but
   too thin a pixel band to help, not a measurement artifact. Not
   promoted either time; live dataset reverted to paste-only both times;
   `06b`/`board_geom.py` kept in the tree for reference, still not wired
   into the production pipeline. **The code/geometry side of this is done
   for good now — don't rebuild it.** What's actually missing is a way to tell whether
   more-physically-complete solder GT helps: every metric in this project
   measures the U-Net reproducing its own synthetic compositing rules
   (`spectral.py`), not real-board performance. Two real-data avenues were
   scoped and explicitly NOT pursued this session (see §6 item 2): PCB-
   Vision (real hyperspectral, but 3.4TB uncompressed with no material-
   layer labels — would need heuristic color-segmentation of the RGB to
   extract material spectra) and FPIC/FICS-PCB (real photos, but labels
   are component-type not material-layer — no taxonomy overlap). Neither
   is a clean win; the honest fallback if this is ever revisited is
   hand-labeling a small real-PCB set into this project's 8 classes for a
   first, even approximate, sim-to-real number — without that, further
   solder-GT tuning of any kind (fillets or otherwise) can't be
   adjudicated better than noise.

   **2026-07-28 update — solder-oversample re-tested against clean GT,
   combined with the class-weight cap for the first time.** The original
   2026-07-23/24 oversample test (0.750, see table above) predates both
   the clean GT and the capped-weight fix. Re-ran
   `--solder-oversample-prob 0.3` on top of the full current production
   config (`--dice-weight 0.5 --max-class-weight 1.5`, clean GT). One
   background run was killed mid-training (~epoch 20/40) by an external
   interruption; `unet_best.pt` had already been partially overwritten by
   that run's mid-training snapshot (epoch 27, worse than production) —
   caught via the standard MD5-verification-before-trusting-a-checkpoint
   habit, restored from the pre-experiment backup
   (`unet_best_pre_solder_oversample_retest.pt`), and re-launched clean.
   Full-res result (held-out boards pooled), vs. the 0.9642/0.8725
   baseline:

   | class | baseline | +oversample 0.3 | delta |
   |---|---|---|---|
   | background | 0.9971 | 0.9968 | −0.0003 |
   | copper | 0.9882 | 0.9870 | −0.0012 |
   | solder | 0.8725 | 0.8763 | +0.0038 |
   | fiberglass | 0.9694 | 0.9763 | +0.0069 |
   | soldermask | 0.9935 | 0.9943 | +0.0008 |
   | silkscreen | 0.9695 | 0.9677 | −0.0018 |
   | component | 0.9913 | 0.9910 | −0.0003 |
   | trace | 0.9322 | 0.9404 | +0.0082 |
   | **mean IoU** | **0.9642** | **0.9662** | **+0.0020** |
   | pixel accuracy | 0.9953 | 0.9957 | +0.0004 |

   Honest read: `solder` itself moved +0.0038, still short of this
   project's ~0.01 noise floor — the original oversampling hypothesis
   (directly fixing solder) is **not confirmed**, same verdict as 2023's
   test. But every class moved together in one direction with no
   individual regression exceeding noise (all three down-ticks ≤0.002,
   on classes already >0.98), and `trace`/`fiberglass` moved further than
   `solder` did. Net effect reads as a small, broadly-distributed
   improvement rather than a solder-specific fix or noise. **Promoted** —
   `unet_best.pt` now holds this checkpoint (epoch 37, full-res mIoU
   0.9662) — on the basis of no downside, not a confirmed solder win.
   `solder` remains the weakest class and training-side avenues remain
   effectively exhausted; validation is still the real blocker (above).

   **Reverted by user request shortly after promotion.** `unet_best.pt`
   restored to the pre-oversample-retest checkpoint (epoch 40, mIoU
   0.9642, solder 0.8725), verified via MD5 match against
   `unet_best_pre_solder_oversample_retest.pt`. The 0.9662 oversample
   checkpoint above is not currently in production but remains fully
   reproducible from this entry if revisited.
3. **`trace`/`silkscreen` classes — DONE and CLOSED this session, fixed.**
   Root-caused (§6 item 3) and fixed via capped class weights
   (`--max-class-weight 1.5`) + Dice loss (`--dice-weight 0.5`): `trace`
   0.849→0.881, `silkscreen` 0.922→0.928, mIoU 0.923→0.932, no
   regressions. This checkpoint is now production (`unet_best.pt`). If
   `trace` is still worth pushing further later: a boundary-aware loss
   term (boundary IoU/Hausdorff) targeting the `mask_open` transition
   specifically, or finer sliding-window overlap at eval time, are the
   next candidate levers — not attempted this session since the combined
   capweight+Dice fix already closed most of the gap.
4. If more diversity is still wanted: round 4 (§3 item 10) showed hit rate
   depends heavily on source-org hardware category — test-equipment/power-
   electronics companies (Great Scott Gadgets, Libre Solar) had a ~90% hit
   rate vs. ~2% for generic dev-board orgs (Antmicro, SparkFun). Similar
   orgs worth trying next: other CERN-OHL/Apache-licensed test-equipment or
   power-electronics hardware companies. Alternatively, invest in back-face
   imaging to unlock boards like `ov5640-dual-camera-board` that are
   blocked by a real, fixable limitation rather than genuinely bad data.
5. Longer-term / not yet scoped at all: domain randomization (soldermask
   color variety across the training set, not just per-board fixed),
   sim-to-real validation against the physical rig once it's capturing
   real data, dataset train/val/test packaging for distribution.

## 8. Quick command reference

```bash
# regenerate/verify the whole dataset (idempotent, gated, parallel)
./.venv311/bin/python 09_batch.py --mpp 0.045 --samples 32 --workers 4

# train / evaluate / visualize errors
# NOTE: to reproduce the current PRODUCTION checkpoint (unet_best.pt),
# --dice-weight and --max-class-weight must be passed explicitly -- the
# CLI defaults are both "off" (see §2/§6 item 3 for why):
./.venv311/bin/python train/train.py \
  --val-boards icicle-hdmi-breakout,signal-integrity-test-board \
  --dice-weight 0.5 --max-class-weight 1.5
./.venv311/bin/python train/evaluate.py --checkpoint train/checkpoints/unet_best.pt
./.venv311/bin/python train/error_heatmap.py --checkpoint train/checkpoints/unet_best.pt
# prior checkpoints kept for comparison:
#   train/checkpoints/unet_best_10board.pt            (pre-round-4, 10 boards)
#   train/checkpoints/unet_best_25board_ce_only.pt    (25 boards, CE-only)

# try a new candidate board (cheap gate first, no Blender)
# 1. add it to 02_fetch_gerbers.py's BOARDS dict
# 2. ./.venv311/bin/python 09_batch.py --boards <name> --mpp 0.045 --samples 32
#    (NOTE: --boards subsets OVERWRITE out/manifest.json to only include
#    those boards -- always follow up with a full-registry run, no
#    --boards flag, to restore a complete/correct manifest)
```
