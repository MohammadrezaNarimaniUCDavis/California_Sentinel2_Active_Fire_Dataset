#!/usr/bin/env python3
"""Build an active-fire segmentation dataset: chips plus per-pixel masks.

    python build_fire_dataset.py                    # 25 California fires
    python build_fire_dataset.py --fires 3 --scenes 3   # quick pilot

Each fire contributes the Sentinel-2 passes that caught it actively burning.
Flame is detected with the SWIR2 hotspot test from `src/indices.py` -- validated
on the Palisades fire, where it reads 121 px on ignition day, decays, and returns
exactly zero on every frame after containment.

The label is a **mask**, one byte per pixel. Boxes were a lossy re-encoding of
this same mask: a flame front is a long thin curve, so its axis-aligned box is
mostly not-fire, and 85% of the boxes it produced sat at an 8 px floor that
existed only to keep them clickable. A mask says exactly which pixels burn.

Chips are rendered R=B12, G=B11, B=B8A -- no band beyond the three the detector
already reads. Flame saturates to orange-white, healthy vegetation reads blue
(NIR is high, both SWIR bands low), and burn scar is brown. A hotspot is
unmistakable, which is the point: the burn-scar task this project started on
turned on a careful olive-versus-green judgement, and this one does not.

Everything runs at 20 m, the native resolution of all three bands.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio.features
from affine import Affine
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src import catalog, chip, fires, indices  # noqa: E402

TILE = 512  # px, at 20 m = 10.24 km across
RES = 20.0

# Window read per fire per scene, centred on the perimeter. 1536 px = 30.7 km.
# Large enough to hold the active front of even a campaign fire, small enough
# that 25 fires stay inside a sane download. The biggest fires here span 115 km,
# so their flanks fall outside; the front is what carries the label.
WINDOW = 1536

# Rendering. B12 is clipped high so that flame -- which drives it far past any
# passive surface -- saturates to white instead of clipping the scar with it.
STRETCH = {"swir22": (0.0, 0.60), "swir16": (0.0, 0.50), "nir08": (0.0, 0.45)}

# Mask encoding, matching src/indices.py.
BACKGROUND, ACTIVE_FIRE, IGNORE = 0, 1, 255

# Cap on empty tiles kept, as a multiple of the tiles that contain fire. Negatives
# matter -- the model must learn that bright bare rock is not a fire -- but an
# unbounded sweep would be 95% empty sky.
NEGATIVE_RATIO = 1.0


def log(msg: str) -> None:
    print(msg, flush=True)


def select_scenes(scenes: list, count: int) -> list:
    """Sample evenly across the burn window rather than taking the first N.

    Taking the earliest passes is what an obvious implementation does and it is
    badly wrong here. A large fire starts at one edge of the area it eventually
    burns, so its front is nowhere near the final perimeter's centroid for the
    first few weeks. Dixie is the case in point: across its 12 passes the front
    shows up on 12-27 August, and the first three passes -- 18, 23 and 28 July --
    contain exactly zero hot pixels. Sampling the whole window catches the peak
    wherever in the window it happens to fall.
    """
    if len(scenes) <= count:
        return scenes
    step = (len(scenes) - 1) / (count - 1) if count > 1 else 1
    return [scenes[round(i * step)] for i in range(count)]


def fire_grid(fire: fires.Fire, epsg: int | None = None,
              size: int = WINDOW, res: float = RES) -> chip.Grid:
    """A read window over the fire, in the projection the *scene* uses.

    Not one grid per fire. Five of the 25 fires straddle the UTM 10N/11N
    boundary, so their passes arrive in both projections and a grid fixed to the
    fire's own zone threw away every scene from the other one -- 30 scenes, 11%
    of the set. Detection chips are independent samples with no cross-date
    correspondence to preserve, so each scene can simply be cut in its own
    projection. (The Palisades burn-scar chips could not do this: comparing NBR
    between dates requires one shared pixel grid.)
    """
    lon, lat = fire.center
    return chip.Grid.centered_on(lon, lat, size, res, epsg=epsg or fire.utm_epsg)


def render(bands: dict[str, np.ndarray]) -> np.ndarray:
    channels = [
        np.clip((bands[b] - lo) / (hi - lo), 0, 1)
        for b, (lo, hi) in (
            ("swir22", STRETCH["swir22"]),
            ("swir16", STRETCH["swir16"]),
            ("nir08", STRETCH["nir08"]),
        )
    ]
    return (np.stack(channels, -1) * 255).round().astype(np.uint8)


def process(fire: fires.Fire, scene, grid: chip.Grid, out: Path) -> list[dict]:
    """One scene of one fire: read, detect, tile, render, box."""
    bands = {
        b: chip.read_reflectance(scene.hrefs[b], grid, *scene.scales[b], out_size=grid.size)
        for b in indices.INDEX_BANDS
    }
    usable = indices.has_data(bands)
    hot = indices.detect_fire(**bands) & usable
    rgb = render(bands)

    # 0 background, 1 fire, 255 nodata. Nodata is ignore rather than background:
    # outside the scene footprint the ground is not "not burning", it is unseen,
    # and a loss that counts it as negative is being taught on made-up pixels.
    full = np.where(hot, ACTIVE_FIRE, BACKGROUND).astype(np.uint8)
    full[~usable] = IGNORE

    records = []
    for ty in range(0, grid.size - TILE + 1, TILE):
        for tx in range(0, grid.size - TILE + 1, TILE):
            sub_mask = full[ty : ty + TILE, tx : tx + TILE]
            sub_rgb = rgb[ty : ty + TILE, tx : tx + TILE]
            nodata = float((sub_mask == IGNORE).mean())
            if nodata > 0.5:
                continue  # mostly outside the scene footprint
            name = f"{fire.slug}_{scene.date.isoformat()}_r{ty // TILE}c{tx // TILE}.png"
            records.append(
                {
                    "file_name": name,
                    "fire": fire.name,
                    "date": scene.date.isoformat(),
                    "day_of_burn": (scene.date - fire.discovered).days,
                    "fire_px": int((sub_mask == ACTIVE_FIRE).sum()),
                    "nodata_pct": round(nodata * 100, 3),
                    "mask": sub_mask,
                    "rgb": sub_rgb,
                }
            )
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fires", type=int, default=25)
    ap.add_argument("--scenes", type=int, default=14,
                    help="max passes per fire, sampled evenly across the burn window")
    ap.add_argument("--out", type=Path, default=Path("active_fire_dataset"))
    ap.add_argument("--state", default="US-CA")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=180,
                    help="seconds allowed per scene before the fire's remaining reads are abandoned")
    args = ap.parse_args()

    log(f"Fetching the {args.fires} largest {args.state} fires since 2019...")
    found = fires.fetch(args.fires, state=args.state, cache=Path("data/ca_fires.geojson"))
    log(f"  {len(found)} fires, {sum(f.acres for f in found):,.0f} acres total")

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    for n, fire in enumerate(found, 1):
        start, end = fire.burn_window
        try:
            scenes = catalog.search(*fire.center, start, end, max_cloud=95)
        except Exception as exc:
            log(f"[{n}/{len(found)}] {fire.name}: search failed ({type(exc).__name__})")
            continue
        scenes = select_scenes([s for s in scenes if "nir08" in s.hrefs], args.scenes)
        grids = {s.item_id: fire_grid(fire, s.epsg) for s in scenes}
        log(f"[{n}/{len(found)}] {fire.name} ({fire.acres:,.0f} ac) "
            f"{start}..{end}: {len(scenes)} passes")
        if not scenes:
            continue

        budget = args.timeout * max(len(scenes), 1)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, fire, s, grids[s.item_id], args.out): s for s in scenes}
            try:
                for future in as_completed(futures, timeout=budget):
                    scene = futures[future]
                    try:
                        all_records.extend(future.result())
                    except Exception as exc:
                        log(f"    skip {scene.date} ({type(exc).__name__}: {exc})")
            except TimeoutError:
                stuck = [s.date for f, s in futures.items() if not f.done()]
                log(f"    TIMED OUT after {budget}s, abandoning {len(stuck)} scene(s): {stuck}")
                for f in futures:
                    f.cancel()

        hot = sum(1 for r in all_records if r["fire_px"])
        log(f"    running total: {hot} tiles with fire, {len(all_records)} tiles seen")

    positives = [r for r in all_records if r["fire_px"]]
    negatives = [r for r in all_records if not r["fire_px"]]
    keep_neg = negatives[:: max(1, len(negatives) // max(int(len(positives) * NEGATIVE_RATIO), 1))]
    kept = sorted(positives + keep_neg, key=lambda r: r["file_name"])
    log(f"\n{len(positives)} tiles with fire, keeping {len(keep_neg)} of "
        f"{len(negatives)} empty tiles as negatives")

    (args.out / "masks").mkdir(exist_ok=True)
    for r in kept:
        Image.fromarray(r["rgb"]).save(args.out / "images" / r["file_name"])
        Image.fromarray(r["mask"], mode="L").save(args.out / "masks" / r["file_name"])

    with (args.out / "summary.csv").open("w", newline="") as fh:
        cols = ["file_name", "fire", "date", "day_of_burn", "fire_px", "nodata_pct"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in kept:
            w.writerow({k: r[k] for k in cols})

    total_fire = sum(r["fire_px"] for r in kept)
    log(f"\nDone: {len(kept)} chips, {total_fire:,} fire pixels, "
        f"{len({r['fire'] for r in positives})} fires contributed fire")
    log(f"  {args.out}/images/   {args.out}/masks/   summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
