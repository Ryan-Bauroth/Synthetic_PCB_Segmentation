"""Shared legend/key helpers for the visualization outputs (evaluate.py,
error_heatmap.py) -- every colour-coded PNG this project produces should be
readable without cross-referencing labels.py by hand."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import labels  # noqa: E402

SWATCH = 18
PAD = 8
ROW_H = 24
FONT = ImageFont.load_default()


def _text_w(draw: ImageDraw.ImageDraw, text: str) -> int:
    return int(draw.textlength(text, font=FONT))


def add_class_legend(im: Image.Image, class_ids: list[int] | None = None,
                     title: str | None = None) -> Image.Image:
    """Append a bottom panel with a colour-swatch + class-name key.
    class_ids defaults to every class in labels.PALETTE."""
    if class_ids is None:
        class_ids = sorted(labels.PALETTE)
    entries = [(labels.PALETTE[c], labels.ID_TO_NAME.get(c, str(c))) for c in class_ids]
    return _append_legend(im, entries, title)


def _append_legend(im: Image.Image, entries: list[tuple[tuple[int, int, int], str]],
                   title: str | None) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    probe = ImageDraw.Draw(im)

    # wrap entries into rows that fit the image width
    item_ws = [SWATCH + 4 + _text_w(probe, label) + PAD * 2 for _, label in entries]
    rows: list[list[int]] = [[]]
    row_w = 0
    for i, iw in enumerate(item_ws):
        if row_w + iw > w and rows[-1]:
            rows.append([])
            row_w = 0
        rows[-1].append(i)
        row_w += iw

    title_h = ROW_H if title else 0
    panel_h = title_h + len(rows) * ROW_H + PAD
    panel = Image.new("RGB", (w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    if title:
        draw.text((PAD, 4), title, fill=(0, 0, 0), font=FONT)

    y = title_h + PAD // 2
    for row in rows:
        x = PAD
        for i in row:
            color, label = entries[i]
            draw.rectangle([x, y, x + SWATCH, y + SWATCH], fill=color, outline=(0, 0, 0))
            draw.text((x + SWATCH + 4, y + 2), label, fill=(0, 0, 0), font=FONT)
            x += item_ws[i]
        y += ROW_H

    combo = Image.new("RGB", (w, h + panel_h), (255, 255, 255))
    combo.paste(im, (0, 0))
    combo.paste(panel, (0, h))
    return combo


def add_colorbar_legend(im: Image.Image, colormap_fn, vmin: float, vmax: float,
                        title: str, fmt: str = "{:.1f}%", n_ticks: int = 6) -> Image.Image:
    """Append a horizontal gradient colour bar with actual numeric tick
    labels (not just "low"/"high" text) spanning [vmin, vmax] -- e.g. the
    error-density heatmap's real local-error-rate percentage at each colour,
    so the map shows how much error, not just where."""
    im = im.convert("RGB")
    w, h = im.size

    bar_h = 18
    tick_h = 14
    gradient = np.tile(np.linspace(0, 1, w, dtype=np.float32), (bar_h, 1))
    bar = Image.fromarray(colormap_fn(gradient))

    panel_h = ROW_H + bar_h + tick_h + PAD * 2
    panel = Image.new("RGB", (w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.text((PAD, 2), title, fill=(0, 0, 0), font=FONT)
    bar_y = ROW_H
    panel.paste(bar, (0, bar_y))

    ticks_y = bar_y + bar_h
    for i in range(n_ticks):
        frac = i / (n_ticks - 1)
        val = vmin + frac * (vmax - vmin)
        x = int(frac * (w - 1))
        draw.line([x, ticks_y, x, ticks_y + 4], fill=(0, 0, 0))
        label = fmt.format(val)
        lw = _text_w(draw, label)
        tx = min(max(x - lw // 2, 0), w - lw)
        draw.text((tx, ticks_y + 5), label, fill=(0, 0, 0), font=FONT)

    combo = Image.new("RGB", (w, h + panel_h), (255, 255, 255))
    combo.paste(im, (0, 0))
    combo.paste(panel, (0, h))
    return combo
