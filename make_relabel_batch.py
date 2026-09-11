#!/usr/bin/env python3
"""Re-issue the chips that carry brush strokes, on a bigger canvas.

    python make_relabel_batch.py

The first labelling pass used a brush roughly 26 px wide on a 1024 px canvas
against targets whose median area is 7 px at native 512 -- an equivalent disc of
about 3 px. The result records *where* fire is, accurately, and discards its
shape. Cluster-level agreement came out at 87%; pixel IoU came out at 0.12, and
that 0.12 is a measurement of brush width, not of judgement.

Only the chips that actually carry strokes need redoing. Chips judged empty
involved no brush at all and stay valid, which is why this batch is 56 chips and
not 233.

Two things change:

* **4x canvas instead of 2x.** Still exact nearest-neighbour pixel replication,
  so no information is invented, but a typical hotspot now spans ~12 px instead
  of ~6 and a fine brush becomes controllable rather than heroic.
* **Filenames are preserved exactly.** They still carry the original sequence
  number, so `label_batch/_scoring/manifest.csv` resolves them unchanged and
  `import_labels.py` folds this pass into the same `hand_masks/`. The importer
  reads each mask's own canvas size, so the coarse pass and this one coexist.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TARGET_MEDIAN_PX = 7        # median detector cluster area at native 512


def disc_diameter(area_px: float) -> float:
    return 2.0 * (area_px / np.pi) ** 0.5


def build_guide(dataset: Path, out: Path, scale: int) -> None:
    """A ruler: a real hotspot at this scale, beside discs of known width.

    Built from a *training* chip, so it calibrates the brush without showing an
    answer for anything in this batch.
    """
    # A training chip, and specifically one whose hotspots are bright, so the
    # ruler is calibrated against a target that actually looks like a target.
    rows = [r for r in csv.DictReader((dataset / "summary.csv").open())
            if r["fire"] == "Antelope" and int(r["fire_px"]) >= 60]
    pick, best = None, -1.0
    for r in rows:
        raw = np.asarray(Image.open(dataset / "masks" / r["file_name"]).convert("L"))
        red = np.asarray(Image.open(dataset / "images" / r["file_name"]).convert("RGB"))[..., 0]
        ys, xs = np.nonzero(raw == 1)
        if len(ys) < 40:
            continue
        cy, cx = int(np.median(ys)), int(np.median(xs))
        if (raw[max(0, cy - 32):cy + 32, max(0, cx - 32):cx + 32] == 255).any():
            continue
        score = float(red[ys, xs].mean())
        if score > best:
            best, pick = score, (r["file_name"], cy, cx)
    if pick is None:
        return
    fn, cy, cx = pick

    R = 28                                     # source px half-window
    y0 = min(max(0, cy - R), 512 - 2 * R); x0 = min(max(0, cx - R), 512 - 2 * R)
    img = np.asarray(Image.open(dataset / "images" / fn).convert("RGB"))[y0:y0 + 2 * R, x0:x0 + 2 * R]
    lab = np.asarray(Image.open(dataset / "masks" / fn).convert("L"))[y0:y0 + 2 * R, x0:x0 + 2 * R] == 1
    # Outline, not fill: these composites are already magenta where they are hot,
    # so a magenta fill would be invisible on exactly the pixels that matter.
    grow = lab.copy()
    grow[1:, :] |= lab[:-1, :]; grow[:-1, :] |= lab[1:, :]
    grow[:, 1:] |= lab[:, :-1]; grow[:, :-1] |= lab[:, 1:]
    ann = img.copy(); ann[grow & ~lab] = (60, 255, 140)

    side = 2 * R * scale
    left = Image.fromarray(img).resize((side, side), Image.NEAREST)
    mid = Image.fromarray(ann).resize((side, side), Image.NEAREST)

    # brush ruler, in canvas pixels
    right = Image.new("RGB", (side, side), (18, 22, 28))
    d = ImageDraw.Draw(right)
    good = disc_diameter(TARGET_MEDIAN_PX * scale * scale)
    marks = [(6, "6 px"), (round(good), f"{round(good)} px  <- aim here"),
             (16, "16 px"), (26, "26 px"), (52, "52 px  <- your last pass")]
    y = 26
    for dia, cap in marks:
        d.ellipse([30 - dia / 2, y - dia / 2, 30 + dia / 2, y + dia / 2], fill=(216, 27, 122))
        d.text((64, y - 6), cap, fill=(228, 234, 242))
        y += max(dia, 18) + 16

    GAP, LBL = 10, 18
    sheet = Image.new("RGB", (3 * side + 2 * GAP, side + LBL), (12, 15, 20))
    for i, (im, cap) in enumerate(((left, f"a hotspot at {scale}x"),
                                   (mid, "outlined: the pixels to brush"),
                                   (right, "brush widths, to scale"))):
        sheet.paste(im, (i * (side + GAP), 0))
        ImageDraw.Draw(sheet).text((i * (side + GAP) + 4, side + 2), cap, fill=(228, 234, 242))
    sheet.save(out / "brush_size_guide.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("active_fire_dataset"))
    ap.add_argument("--batch", type=Path, default=Path("label_batch"))
    ap.add_argument("--out", type=Path, default=Path("relabel_batch"))
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()

    listing = args.batch / "_scoring" / "redo_with_smaller_brush.txt"
    if not listing.exists():
        raise SystemExit(f"missing {listing}")
    wanted = [ln.split("\t")[1] for ln in listing.read_text().splitlines() if "\t" in ln]
    manifest = {r["label_file"]: r for r in csv.DictReader(
        (args.batch / "_scoring" / "manifest.csv").open())}

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "images").mkdir(parents=True)

    size = 512 * args.scale
    for name in wanted:
        src = manifest[name]["source_file"]
        im = Image.open(args.dataset / "images" / src).convert("RGB")
        im.resize((size, size), Image.NEAREST).save(args.out / "images" / name, optimize=True)

    shutil.copy(args.batch / "labeling_config.xml", args.out / "labeling_config.xml")
    build_guide(args.dataset, args.out, args.scale)

    good = disc_diameter(TARGET_MEDIAN_PX * args.scale * args.scale)
    (args.out / "README.md").write_text(f"""# Relabel batch - {len(wanted)} chips, {size}x{size}

These are the only chips from the first pass that carry brush strokes. The other
177 you judged empty are already correct and are not reissued.

Filenames are unchanged, so `import_labels.py` folds this pass straight back
into the same `hand_masks/`.

## Brush size - the one thing to change

**Set the brush to about {round(good)} px.** Last pass it was ~52 px at this
scale, which is why regions came out {int(round((52/good)**2))}x too large in area.

`brush_size_guide.png` shows a real hotspot at {args.scale}x beside discs drawn
to scale, so you can match the brush by eye instead of by number.

The rule: **paint the bright core, not its glow.** A hotspot has a saturated
orange-white centre with red bleeding around it. The centre is the label. If
your stroke extends past where the brightness visibly falls off, it is too wide.
Isolated specks want a smaller brush still - go to 6 px for those.

Large saturated fronts are the exception: they really are hundreds of pixels
across, so use whatever width traces them. The error last time was on the small
targets, not the big ones.

## Then

Export twice into one folder as before - **Brush labels to PNG** (unzipped) plus
**JSON** - and run:

    python import_labels.py --export path/to/that/folder --batch label_batch

That overwrites just these {len(wanted)} masks. To score the full set afterwards:

    python import_labels.py --score-only --batch label_batch
""")
    print(f"wrote {len(wanted)} chips to {args.out}/images at {size}x{size}")
    print(f"recommended brush: ~{round(good)} px  (was ~{52} px equivalent last pass)")
    print(f"guide: {args.out}/brush_size_guide.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
