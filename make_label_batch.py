#!/usr/bin/env python3
"""Prepare the held-out test chips for hand labelling in Label Studio.

    python make_label_batch.py                 # build label_batch/
    python make_label_batch.py --scale 1       # native 512 px instead of 2x

Why this exists: the masks distributed with the dataset came from the SWIR2
threshold detector, so the reported test IoU measures agreement with that
detector rather than with reality. Hand labels on the four held-out fires turn
that caveat into a number.

Three decisions are baked in, and each one is a decision about *bias*:

* **Chips are ordered by a fire-stratified shuffle, and the order carries no
  information about the detector's opinion.** Stratifying on whether a chip has
  detector-fire would have leaked that opinion into the sequence; an annotator
  who noticed the pattern would be labelling the detector. Chips are shuffled
  within each fire and then dealt round-robin, so *any* prefix of the sequence
  is a valid stratified sample of the test split and labelling can stop at any
  point without invalidating the estimate.
* **Negatives are included.** Labelling only the chips the detector flagged
  would make the detector's misses structurally invisible -- precisely the error
  that hand labelling is supposed to expose.
* **No pre-annotations are supplied.** Loading the detector's mask as a starting
  point would roughly halve the work and anchor every judgement to the thing
  being audited. The labels have to be drawn blind to be worth anything.

Chips are upscaled 2x nearest-neighbour by default. That invents no information
-- it is exact pixel replication -- but the median fire cluster in this split is
6 px, and brushing a 6 px target is far more accurate at 2x. `import_labels.py`
maps the result back down.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

TEST_FIRES = {"Slater", "Hopkins", "CALDWELL", "Windy"}

CONFIG_XML = """<View>
  <Header value="Mark pixels showing ACTIVE FLAME only"/>
  <Image name="chip" value="$image" zoom="true" zoomControl="true"
         brightnessControl="true" contrastControl="true"/>
  <BrushLabels name="mask" toName="chip">
    <Label value="active_fire" background="#D81B7A"/>
    <Label value="uncertain" background="#E8B33A"/>
  </BrushLabels>
</View>
"""

README = """# Hand-labelling batch - four held-out fires

{n} chips from the test split ({fires}), at {size}x{size} px
({scale}x nearest-neighbour upscale of the distributed 512 px chips).

## What you are deciding

These are Sentinel-2 false-colour composites: **R = B12 (SWIR2), G = B11 (SWIR1),
B = B8A (NIR)**. In this rendering:

| Appearance | What it is | Label it? |
|---|---|---|
| Saturated orange / yellow / white **core**, small and sharp-edged | Active flame | **YES** - `active_fire` |
| Brown, olive, or near-black, matte, often large | Fresh burn scar | No |
| Diffuse white or grey, soft-edged, drifting, casts no hard shadow | Smoke | No |
| Bright white, textured, with a dark shadow offset | Cloud | No |
| Uniform tan or pale grey, follows terrain | Bare rock, dry soil | No |
| Bright spot on water or a roof, bright in *every* channel | Specular glint | No |
| Pure black at a straight frame edge | Outside the acquisition footprint | No - leave blank |

![what to look for](what_to_look_for.png)

`what_to_look_for.png` shows the three scales you will meet, at 4x, with the
pixels to brush on the right. It is built from **training** chips, not from
anything in `images/`, so it calibrates the class without showing you an answer.
Note the largest example: the label covers the saturated core, not the red glow
bleeding around it.

The single discriminator is that **flame emits** at 2.2 um rather than reflecting
it. Emission drives the red channel far past anything a passive surface can
reach, so flame reads as an orange-to-white core that is much brighter in red
than in blue. Fresh char is also dark in NIR and brightish in SWIR2 -- that is
the one confusion worth being careful about -- but char is dull and spatially
broad, while flame is intense and concentrated.

**When you genuinely cannot tell, use `uncertain`.** Obscured by cloud, too faint
to call, ambiguous against bright rock. Those pixels are excluded from scoring
rather than forced into a guess, which is far better than a coin flip -- a forced
guess is indistinguishable from a real judgement once it is in the file.

## How to work

1. Create a Label Studio project, paste `labeling_config.xml` into
   *Settings -> Labeling interface -> Code*.
2. Import everything in `images/`.
3. **Sort tasks by filename** and work top to bottom. The numeric prefix is a
   stratified random order, so stopping at any point still leaves an unbiased
   sample. Do not skip around, and do not label the easy ones first -- that is
   what breaks the sample.
4. Zoom in hard. Most targets are a few pixels; at 1:1 you will miss them.
   Brightness and contrast controls are enabled in the config and are safe to
   use freely -- they change what you can see, not what the pixels are.
5. Use a small brush. Err toward the visible core rather than its glow.

## How far to go

| After | Chips | Gives you |
|---|---|---|
| Batch 1 | 60 | A usable first estimate; enough to see whether the detector is systematically off |
| Batch 2 | 120 | Tight enough to quote in the paper |
| All | {n} | The full test split; no sampling caveat at all |

Roughly 60% of chips contain nothing and take seconds.

## When you are done

Export **twice**, into the same folder:

1. **Brush labels to PNG** -- unzip it.
2. **JSON** -- drop the `.json` next to the unzipped PNGs.

Both are needed. The PNGs hold the strokes (the JSON stores them as Label
Studio's own RLE, which the importer does not decode). The JSON holds two things
the PNGs cannot: which chips you judged *empty* -- those produce no PNG at all,
yet "I looked and there is nothing here" is exactly what catches a detector false
positive -- and which class each stroke was, so `uncertain` is not read as fire.

    python ../import_labels.py --export path/to/that/folder --batch .

That maps masks back to 512 px, writes them to `hand_masks/`, and reports
agreement between your labels and the detector's, per fire and overall.

## Do not open yet

`_scoring/manifest.csv` records which chip is which, **including the detector's
fire-pixel count**. Reading it before you label would anchor you to the answer.
`../import_labels.py` needs it; you do not.
"""


def build(dataset: Path, out: Path, scale: int, seed: int) -> int:
    rows = list(csv.DictReader((dataset / "summary.csv").open()))
    test = [r for r in rows if r["fire"] in TEST_FIRES]
    if not test:
        raise SystemExit("no test-split chips found; check TEST_FIRES against summary.csv")

    # Shuffle within each fire, then deal round-robin. Stratifies by fire without
    # letting fire_px influence position -- see the module docstring.
    rng = np.random.default_rng(seed)
    per_fire = {}
    for fire in sorted(TEST_FIRES):
        sub = sorted([r for r in test if r["fire"] == fire], key=lambda r: r["file_name"])
        rng.shuffle(sub)
        per_fire[fire] = sub

    order = []
    for i in range(max(len(v) for v in per_fire.values())):
        for fire in sorted(per_fire):
            if i < len(per_fire[fire]):
                order.append(per_fire[fire][i])

    images = out / "images"
    scoring = out / "_scoring"
    # The calibration figure is built separately and is not reproduced here, so
    # carry it across the rebuild rather than deleting work that cannot be
    # regenerated by this script.
    keep = {}
    for extra in ("what_to_look_for.png",):
        if (out / extra).exists():
            keep[extra] = (out / extra).read_bytes()
    if (out / "hand_masks").exists():
        raise SystemExit(
            f"{out}/hand_masks exists -- rebuilding would renumber the chips and orphan "
            f"those labels. Move or delete it first if you really mean to start over.")
    if out.exists():
        shutil.rmtree(out)
    images.mkdir(parents=True)
    scoring.mkdir(parents=True)
    for name, blob in keep.items():
        (out / name).write_bytes(blob)

    size = 512 * scale
    manifest = []
    for seq, r in enumerate(order, 1):
        stem = Path(r["file_name"]).stem
        name = f"{seq:03d}_{stem}.png"
        im = Image.open(dataset / "images" / r["file_name"]).convert("RGB")
        if scale != 1:
            im = im.resize((size, size), Image.NEAREST)
        im.save(images / name, optimize=True)
        manifest.append({
            "seq": seq, "label_file": name, "source_file": r["file_name"],
            "fire": r["fire"], "date": r["date"],
            "detector_fire_px": r["fire_px"], "nodata_pct": r["nodata_pct"],
        })

    with (scoring / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)

    (out / "labeling_config.xml").write_text(CONFIG_XML)
    (out / "README.md").write_text(README.format(
        n=len(order), fires=", ".join(sorted(TEST_FIRES)), size=size, scale=scale))

    counts = {f: len(v) for f, v in per_fire.items()}
    print(f"wrote {len(order)} chips to {images}/  at {size}x{size}")
    print("  per fire: " + ", ".join(f"{f} {n}" for f, n in sorted(counts.items())))
    print(f"  first 60 draws: " + ", ".join(
        f"{f} {sum(1 for r in order[:60] if r['fire'] == f)}" for f in sorted(TEST_FIRES)))
    hidden = sum(1 for r in order if int(r["fire_px"]) > 0)
    print(f"  ({hidden} of {len(order)} carry detector fire -- recorded in _scoring/, not in the filenames)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("active_fire_dataset"))
    ap.add_argument("--out", type=Path, default=Path("label_batch"))
    ap.add_argument("--scale", type=int, default=2, help="nearest-neighbour upscale factor")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return build(args.dataset, args.out, args.scale, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
