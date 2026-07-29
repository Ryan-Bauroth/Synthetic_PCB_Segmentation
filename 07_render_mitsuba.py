"""Render a board's 5-band capture set (R, G, B, IR, UV) plus its ground-truth
mask, using Mitsuba3. Pure Python -- no bpy dependency (mesh + UV were already
exported by 05_export_mesh.py).

Design (see plan):
- ONE render per band (not 16 angle-resolved captures): all 16 ring positions
  (rig.light_positions()) lit simultaneously with a NEUTRAL/white light, plus a
  low-radiance ambient fill -- this is the "shadows disappear, just a colour
  image" requirement. Material differentiation comes entirely from each band's
  reflectance texture (spectral.py), not from light colour, so the light
  itself doesn't need to be "coloured" the way the old bpy pipeline did.
- Ground truth is rendered through the SAME mesh/camera/UV as an emissive
  (self-lit, non-reflective) pass reading the class-id palette texture --
  the Mitsuba equivalent of the old Blender emission-swap trick, guaranteeing
  pixel alignment with the 5 band images by construction.
- A height map is ALSO rendered through the same mesh/camera, via Mitsuba's
  "position" AOV rather than a separate STL rasterization pass -- same
  alignment guarantee, no second geometry pipeline. Stored as a 16-bit PNG
  in micrometres, zeroed at ground_plane()'s Z (mm above the bench).

Orthographic camera scale semantics were verified empirically: Mitsuba's
unscaled orthographic window is [-1,1] x [-1,1] (extent 2), so a to_world
scale of (framed_x/2, framed_y/2, 1) gives a world window exactly
framed_x x framed_y wide/tall.

Run:  ./.venv311/bin/python 07_render_mitsuba.py --board <name>
                                                  [--mpp 0.117] [--samples 32]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import labels  # noqa: E402
import rig  # noqa: E402
import spectral  # noqa: E402

import mitsuba as mi  # noqa: E402

mi.set_variant("scalar_rgb")


def load_bounds(proj: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads((proj / "board_bounds.json").read_text())
    return np.array(d["center_mm"], np.float64), np.array(d["size_mm"], np.float64)


def ortho_sensor(center, framed_x, framed_y, width_px, height_px, samples):
    # Mitsuba's orthographic to_world scale must be ISOTROPIC (verified
    # empirically: an anisotropic scale.y is NOT applied independently --
    # the film's width/height resolution ratio is what actually determines
    # the per-axis mm/px mapping). Using framed_x/2 on both axes with a
    # width_px:height_px ratio matching framed_x:framed_y gives the correct,
    # undistorted physical framing on both axes.
    cam_z = center[2] + max(rig.ring_height_mm() * 2, 500.0)
    to_world = (mi.ScalarTransform4f.look_at(
        origin=[center[0], center[1], cam_z],
        target=[center[0], center[1], center[2]],
        up=[0, 1, 0])
        @ mi.ScalarTransform4f.scale([framed_x / 2.0, framed_x / 2.0, 1.0]))
    return {
        "type": "orthographic",
        "near_clip": 1.0,
        "far_clip": cam_z * 3.0,
        "to_world": to_world,
        "film": {"type": "hdrfilm", "width": int(width_px), "height": int(height_px),
                 "pixel_format": "rgb"},
        "sampler": {"type": "independent", "sample_count": samples},
    }


def ring_lights(center, exposure_k: float) -> dict:
    """16 neutral point lights at the ring positions, offset by board centre.
    Material differentiation comes from the band's reflectance texture, not
    light colour -- so every band uses the same white light. Energy is
    divided by N_LIGHTS since all 16 fire simultaneously (avoid ~16x
    overexposure vs. the old one-light-at-a-time model)."""
    total = exposure_k * rig.relative_intensity()
    per_light = total / rig.N_LIGHTS
    lights = {}
    for i, (x, y, z) in enumerate(rig.light_positions()):
        lights[f"led{i:02d}"] = {
            "type": "point",
            "position": [center[0] + x, center[1] + y, center[2] + z],
            "intensity": per_light,
        }
    return lights


def ground_plane(center, size, framed_x, framed_y, gray: float, emissive: bool):
    span = max(framed_x, framed_y) * 1.5
    z_bottom = center[2] - size[2] / 2.0
    shape = {
        "type": "rectangle",
        "to_world": (mi.ScalarTransform4f.translate([center[0], center[1], z_bottom])
                    @ mi.ScalarTransform4f.scale([span / 2.0, span / 2.0, 1.0])),
        "bsdf": {"type": "diffuse", "reflectance": 0.0 if emissive else gray},
    }
    return shape


def board_shape(obj_path: Path, texture_png: Path | None, emissive: bool):
    # raw=True is required: our PNGs encode literal reflectance/label values,
    # not photographic colour -- without it Mitsuba's bitmap texture assumes
    # sRGB and gamma-decodes to linear, silently darkening everything (found
    # empirically: board reflectance read ~13x too dark, and the GT palette's
    # darker classes like soldermask (20,90,50) collapsed toward background
    # (0,0,0) after the extra darkening, causing systematic misclassification).
    bsdf = {"type": "diffuse", "reflectance": 0.0} if emissive else \
        {"type": "diffuse", "reflectance": {"type": "bitmap", "filename": str(texture_png), "raw": True}}
    shape = {"type": "obj", "filename": str(obj_path), "face_normals": True, "bsdf": bsdf}
    if emissive and texture_png is not None:
        shape["emitter"] = {"type": "area",
                            "radiance": {"type": "bitmap", "filename": str(texture_png), "raw": True}}
    return shape


def component_shape(obj_path: Path, band: str | None, emissive: bool):
    """Populated 3D component bodies (from 06_export_stl.py's aligned
    components.obj), if present. Flat/uniform material -- we don't have
    per-component colour/material data, see spectral.py's module docstring.
    band=None + emissive=True is the GT pass (flat "component" class colour)."""
    if emissive:
        r, g, b = labels.PALETTE[labels.CLASSES["component"]]
        color = (r / 255, g / 255, b / 255)
        bsdf = {"type": "diffuse", "reflectance": 0.0}
    else:
        if band in ("R", "G", "B"):
            chan = {"R": 0, "G": 1, "B": 2}[band]
            val = spectral.VISIBLE_COLOR["component"][chan]
        else:
            val = spectral.REFLECTANCE[band]["component"]
        color = (val, val, val)
        bsdf = {"type": "diffuse", "reflectance": val}
    shape = {"type": "obj", "filename": str(obj_path), "face_normals": True, "bsdf": bsdf}
    if emissive:
        # Mitsuba needs an explicit colour-type dict here -- a plain 3-tuple
        # gets interpreted as a "vector" (position-like), not a colour.
        shape["emitter"] = {"type": "area", "radiance": {"type": "rgb", "value": list(color)}}
    return shape


def fillet_shape(obj_path: Path, band: str | None, emissive: bool):
    """Synthesized SMD solder-fillet wedges (06b_generate_solder_fillets.py),
    if present. Same flat/uniform-material pattern as component_shape(), but
    keyed to the `solder` class/reflectance -- this geometry IS solder, not a
    placeholder for it."""
    if emissive:
        r, g, b = labels.PALETTE[labels.CLASSES["solder"]]
        color = (r / 255, g / 255, b / 255)
        bsdf = {"type": "diffuse", "reflectance": 0.0}
    else:
        if band in ("R", "G", "B"):
            chan = {"R": 0, "G": 1, "B": 2}[band]
            val = spectral.VISIBLE_COLOR["solder"][chan]
        else:
            val = spectral.REFLECTANCE[band]["solder"]
        color = (val, val, val)
        bsdf = {"type": "diffuse", "reflectance": val}
    shape = {"type": "obj", "filename": str(obj_path), "face_normals": True, "bsdf": bsdf}
    if emissive:
        shape["emitter"] = {"type": "area", "radiance": {"type": "rgb", "value": list(color)}}
    return shape


def render_band(obj_path, band_png, center, size, framed_x, framed_y, width_px,
                height_px, samples, exposure_k, ambient, plane_gray, band,
                components_obj: Path | None, fillets_obj: Path | None) -> np.ndarray:
    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 3},
        "sensor": ortho_sensor(center, framed_x, framed_y, width_px, height_px, samples),
        "ambient": {"type": "constant", "radiance": ambient},
        "board": board_shape(obj_path, band_png, emissive=False),
        "ground": ground_plane(center, size, framed_x, framed_y, plane_gray, emissive=False),
        **{f"light_{k}": v for k, v in ring_lights(center, exposure_k).items()},
    }
    if components_obj is not None:
        scene_dict["components"] = component_shape(components_obj, band, emissive=False)
    if fillets_obj is not None:
        scene_dict["fillets"] = fillet_shape(fillets_obj, band, emissive=False)
    scene = mi.load_dict(scene_dict)
    img = np.array(mi.render(scene))
    return (img.mean(axis=2).clip(0, 1) * 255).astype(np.uint8)


def render_height(obj_path, center, size, framed_x, framed_y, width_px, height_px,
                  components_obj: Path | None, fillets_obj: Path | None = None) -> np.ndarray:
    """World-space height above the bench (ground_plane's Z), in mm, as a
    uint16 micrometre map. Uses Mitsuba's "position" AOV through the SAME
    mesh/camera as the band/GT renders, so it's pixel-aligned by
    construction and free of a second (STL-based) geometry/alignment
    pipeline. Verified empirically: with a box reconstruction filter, AOV
    channels are alpha-premultiplied, but ground_plane() already guarantees
    alpha=1 across the entire frame (there's always a hit), so the raw Z
    channel can be read directly with no unpremultiply step."""
    flat_bsdf = {"type": "diffuse", "reflectance": 0.0}  # colour is discarded; only geometry matters
    sensor = ortho_sensor(center, framed_x, framed_y, width_px, height_px, samples=4)
    # box filter -> exact per-pixel world position at fully-covered pixels,
    # rather than a filter-blurred blend across neighbours.
    sensor["film"]["rfilter"] = {"type": "box"}
    scene_dict = {
        "type": "scene",
        "integrator": {"type": "aov", "aovs": "h:position",
                       "my_image": {"type": "direct"}},
        "sensor": sensor,
        "board": {"type": "obj", "filename": str(obj_path), "face_normals": True, "bsdf": flat_bsdf},
        "ground": ground_plane(center, size, framed_x, framed_y, 0.0, emissive=False),
    }
    if components_obj is not None:
        scene_dict["components"] = {"type": "obj", "filename": str(components_obj),
                                    "face_normals": True, "bsdf": flat_bsdf}
    if fillets_obj is not None:
        scene_dict["fillets"] = {"type": "obj", "filename": str(fillets_obj),
                                 "face_normals": True, "bsdf": flat_bsdf}
    scene = mi.load_dict(scene_dict)
    img = np.array(mi.render(scene))
    z_bottom = center[2] - size[2] / 2.0
    # sensor film is pixel_format="rgb" (3 channels) -> position AOV's xyz
    # occupies channels 3-5; verified empirically (not rgba's 4-6).
    height_mm = img[:, :, 5] - z_bottom
    height_um = np.clip(np.round(height_mm * 1000.0), 0, 65535).astype(np.uint16)
    return height_um


def render_gt(obj_path, label_png, center, size, framed_x, framed_y, width_px, height_px,
             components_obj: Path | None, fillets_obj: Path | None = None,
             samples: int = 4):
    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 1},
        "sensor": ortho_sensor(center, framed_x, framed_y, width_px, height_px, samples=samples),
        "board": board_shape(obj_path, label_png, emissive=True),
        # non-emissive (reflectance=0) -> reads pure black = background palette
        # colour, same as leaving it out entirely, but keeps geometry/occlusion
        # consistent with the band renders (e.g. component overhang shadows).
        "ground": ground_plane(center, size, framed_x, framed_y, 0.0, emissive=True),
    }
    if components_obj is not None:
        scene_dict["components"] = component_shape(components_obj, None, emissive=True)
    if fillets_obj is not None:
        scene_dict["fillets"] = fillet_shape(fillets_obj, None, emissive=True)
    scene = mi.load_dict(scene_dict)
    img = np.array(mi.render(scene))
    rgb = np.clip(img, 0, 1) * 255
    # nearest palette colour -> class id (int32 to avoid overflow, learned earlier).
    # Only match against classes actually present in this scene: at low sample
    # counts, anti-aliased edge pixels between two real classes can land closer
    # to an unrelated palette colour than to either true neighbour (found
    # empirically -- "component" spuriously appeared as isolated single-pixel
    # noise along class boundaries on boards with NO component geometry at all,
    # simply because the palette grew to 7 entries).
    excluded = set() if components_obj is not None else {labels.CLASSES["component"]}
    ids_list = sorted(i for i in labels.PALETTE if i not in excluded)
    pal = np.array([labels.PALETTE[i] for i in ids_list], dtype=np.int32)
    d = ((rgb.astype(np.int32)[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(3)
    idmap = np.array(ids_list, np.uint8)[d.argmin(2)].astype(np.uint8)
    return idmap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--mpp", type=float, default=0.0209,
                    help="millimetres per pixel (real rig = 0.0209)")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="fraction of board size framed on each side")
    ap.add_argument("--plane-gray", type=float, default=0.4)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--gt-samples", type=int, default=4,
                    help="sample count for the GT pass only (default matches "
                         "prior behavior -- low sample counts risk noisy "
                         "anti-aliased class-boundary pixels, see docs/STATUS.md)")
    ap.add_argument("--exposure-k", type=float, default=4.0,
                    help="total ring energy = k * D^2, split across 16 lights "
                         "(empirically tuned so a 0.4-reflectance ground plane "
                         "lands mid-tone, not saturated)")
    ap.add_argument("--ambient", type=float, default=0.05,
                    help="low-radiance fill light so self-shadows mostly "
                         "disappear (per the 'just a colour image' requirement)")
    args = ap.parse_args()

    proj = ROOT / "boards" / args.board
    obj_path = proj / "fab" / "mitsuba" / "board.obj"
    bounds_path = proj / "board_bounds.json"
    if not obj_path.exists() or not bounds_path.exists():
        sys.exit(f"Missing mesh/bounds for {args.board}. Run 05_export_mesh.py first.")

    center, size = load_bounds(proj)
    framed_x = size[0] * (1 + 2 * args.margin)
    framed_y = size[1] * (1 + 2 * args.margin)
    width_px = max(1, round(framed_x / args.mpp))
    height_px = max(1, round(framed_y / args.mpp))
    print(f"Board centre={tuple(round(v,1) for v in center)} size={tuple(round(v,1) for v in size)} mm")
    print(f"Framed {framed_x:.1f}x{framed_y:.1f} mm -> render {width_px}x{height_px} "
          f"@ mpp={args.mpp}")

    components_obj = proj / "fab" / "mitsuba" / "components.obj"
    if not components_obj.exists():
        components_obj = None
    print(f"Populated components: {'yes -> ' + str(components_obj) if components_obj else 'no (bare board or not exported)'}")

    fillets_obj = proj / "fab" / "mitsuba" / "solder_fillets.obj"
    if not fillets_obj.exists():
        fillets_obj = None
    print(f"Solder fillets: {'yes -> ' + str(fillets_obj) if fillets_obj else 'no (06b not run, or no fillet band found)'}")

    outdir = ROOT / "out" / args.board
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.board
    mitsuba_dir = proj / "fab" / "mitsuba"

    # 5 band renders + 1 GT render + 1 height render -> one progress bar over all 7.
    steps = list(spectral.BANDS) + ["GT", "Height"]
    bar = tqdm(steps, desc="rendering", unit="pass")
    for band in bar:
        if band == "Height":
            bar.set_description("rendering height")
            height_um = render_height(obj_path, center, size, framed_x, framed_y,
                                      width_px, height_px, components_obj, fillets_obj)
            height_png = outdir / f"{stem}_height.png"
            Image.fromarray(height_um).save(height_png)
            bar.write(f"  wrote {height_png.relative_to(ROOT)}  "
                      f"max={height_um.max()/1000.0:.2f}mm")
        elif band != "GT":
            bar.set_description(f"rendering {band}")
            tex = spectral.build_band_texture(proj, band)
            tex_png = mitsuba_dir / f"band_{band}.png"
            spectral.save_texture_png(tex, tex_png)
            mono = render_band(obj_path, tex_png, center, size, framed_x, framed_y,
                               width_px, height_px, args.samples, args.exposure_k,
                               args.ambient, args.plane_gray, band, components_obj,
                               fillets_obj)
            out_png = outdir / f"{stem}_{band}.png"
            Image.fromarray(mono, "L").save(out_png)
            bar.write(f"  wrote {out_png.relative_to(ROOT)}  mean={mono.mean():.1f}")
        else:
            bar.set_description("rendering ground truth")
            class_map = spectral.build_class_id_map(proj)
            label_png = mitsuba_dir / "label_tex.png"
            pal = np.zeros((*class_map.shape, 3), np.uint8)
            for cid, rgb_val in labels.PALETTE.items():
                pal[class_map == cid] = rgb_val
            Image.fromarray(pal, "RGB").save(label_png)

            idmap = render_gt(obj_path, label_png, center, size, framed_x, framed_y,
                              width_px, height_px, components_obj, fillets_obj,
                              samples=args.gt_samples)
            gt_png = outdir / f"{stem}_gt.png"
            im = Image.fromarray(idmap, "P")
            im.putpalette(labels.flat_palette())
            im.save(gt_png)
            u, c = np.unique(idmap, return_counts=True)
            bar.write(f"  wrote {gt_png.relative_to(ROOT)}")
            bar.write("  GT class pixel counts:")
            for cid, n in zip(u, c):
                bar.write(f"    {int(cid)} {labels.ID_TO_NAME.get(int(cid),'?'):<11} {int(n)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
