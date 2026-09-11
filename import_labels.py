#!/usr/bin/env python3
"""Fold a Label Studio brush export back into dataset-native masks and score it.

    python import_labels.py --export path/to/unzipped_export

Reads the PNG masks Label Studio writes for brush annotations, maps each back to
its 512 px source chip, and writes `hand_masks/` in the dataset's own encoding
(0 background, 1 active_fire, 255 ignore). Then it reports how far the hand
labels sit from the SWIR2 detector that produced the distributed masks.

That last number is the point of the exercise. Every score in the data article is
agreement with the detector; this is the first measurement of whether the
detector is right.

**Export twice from Label Studio, into one folder: "Brush labels to PNG"
(unzipped) and "JSON".** Both are required, and neither substitutes for the
other. The PNGs carry the strokes -- Label Studio's JSON stores them as a bespoke
run-length encoding this script does not decode. The JSON carries two things the
PNGs cannot: which chips you annotated as *empty* (those produce no PNG at all,
yet "I looked and there is no fire" is exactly the judgement that catches a
detector false positive), and which class each stroke belongs to (the PNG
filenames do not reliably say, so without the JSON `uncertain` would silently
become `active_fire`).

Downsampling from the 2x labelling canvas uses *any-fire-wins*: a 512 px pixel is
fire if any of the four 1024 px pixels covering it was painted fire. Averaging
would erode exactly the one- and two-pixel clusters that matter most here, and
`uncertain` takes precedence over `active_fire` so a hedged pixel is never
silently promoted to a confident one.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

BACKGROUND, ACTIVE_FIRE, IGNORE = 0, 1, 255


def log(m: str) -> None:
    print(m, flush=True)


def load_manifest(batch: Path) -> list[dict]:
    path = batch / "_scoring" / "manifest.csv"
    if not path.exists():
        raise SystemExit(f"no manifest at {path}. Run make_label_batch.py first.")
    return list(csv.DictReader(path.open()))


def _brushlabels(node) -> list[str]:
    """Every brushlabels value anywhere under a JSON node."""
    found = []
    if isinstance(node, dict):
        if isinstance(node.get("brushlabels"), list):
            found += [str(x) for x in node["brushlabels"]]
        for v in node.values():
            found += _brushlabels(v)
    elif isinstance(node, list):
        for v in node:
            found += _brushlabels(v)
    return found


def read_tasks(export: Path, manifest: list[dict]) -> dict[str, dict]:
    """label_file -> {task id, labels used, whether it was actually annotated}.

    The JSON export is not optional, for two reasons that both silently corrupt
    the score if it is missing.

    **A chip you judged empty produces no PNG at all.** "I looked and there is no
    fire here" is a real annotation, and on a chip the detector flagged it is
    exactly a false positive *by* the detector. Scoring only the chips that
    yielded a PNG would drop every one of those and inflate the detector's
    precision.

    **The PNG filenames do not reliably say which class a mask belongs to.**
    Without the JSON, `uncertain` strokes would be merged into `active_fire`,
    turning "I could not tell" into "this is burning".
    """
    stems = [(Path(r["label_file"]).stem, r["label_file"]) for r in manifest]
    out: dict[str, dict] = {}
    for jf in sorted(export.rglob("*.json")):
        try:
            blob = json.loads(jf.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        for t in blob if isinstance(blob, list) else [blob]:
            if not isinstance(t, dict):
                continue
            tid = t.get("id", t.get("task_id", t.get("inner_id")))
            data = t.get("data") if isinstance(t.get("data"), dict) else t
            img = next((str(v) for v in data.values()
                        if isinstance(v, str) and v.lower().endswith(".png")), None)
            hit = next((lf for s, lf in stems if img and s in img), None)
            if hit is None:
                continue
            anns = t.get("annotations")
            if anns is None:            # JSON-MIN: regions hang off the tag name
                anns = [v for v in t.values() if isinstance(v, list)]
            ordered = _brushlabels(anns)      # region order, as exported
            labels = sorted(set(ordered))
            # An annotation entry exists even when the annotator drew nothing;
            # that is the "looked, saw none" verdict we must keep.
            annotated = bool(anns) or t.get("annotations") == []
            prev = out.get(hit)
            if prev is None or (labels and not prev["labels"]):
                out[hit] = {"id": str(tid), "labels": labels,
                            "ordered": ordered, "annotated": annotated}
    return out


def match_exports(export: Path, manifest: list[dict], tasks: dict[str, dict]) -> dict[str, list[Path]]:
    """Group exported PNGs by the chip they belong to.

    Two schemes are tried, because Label Studio's export naming varies by version
    and storage backend: the source filename embedded in the PNG name, and
    failing that a `task-<id>-` prefix resolved through a JSON export in the same
    folder.
    """
    pngs = sorted(p for p in export.rglob("*.png"))
    if not pngs:
        if not any(export.rglob("*.json")):
            raise SystemExit(f"no PNG masks and no JSON under {export}")
        log("note: no PNG masks found -- treating every annotated chip as empty. "
            "If you drew anything, re-export as 'Brush labels to PNG' into this folder.")

    stems = [(Path(r["label_file"]).stem, r["label_file"]) for r in manifest]
    ids = {v["id"]: k for k, v in tasks.items()}
    by_chip: dict[str, list[Path]] = defaultdict(list)
    unmatched, by_name, by_id = [], 0, 0
    for p in pngs:
        hit = next((lf for s, lf in stems if s in p.name), None)
        if hit:
            by_name += 1
        else:
            m = re.search(r"task[-_](\d+)", p.name)
            hit = ids.get(m.group(1)) if m else None
            if hit:
                by_id += 1
        if hit:
            by_chip[hit].append(p)
        else:
            unmatched.append(p.name)

    log(f"matched {by_name + by_id} of {len(pngs)} exported PNG(s)"
        + (f" ({by_name} by filename, {by_id} by task id)" if by_id else ""))
    if unmatched:
        log(f"warning: {len(unmatched)} unmatched, e.g. {unmatched[:3]}")
        if not ids:
            log("  hint: export the JSON alongside the PNGs so task ids resolve to "
                "filenames, then re-run against the folder holding both.")
    return by_chip


def fold(paths: list[Path], labels: list[str],
         ordered: list[str] | None = None) -> tuple[np.ndarray, bool]:
    """Merge one chip's exported PNGs into a 512 px mask.

    `labels` is what the JSON says this task actually used. It resolves the case
    the filenames cannot: when a chip carries only one class, every PNG for it
    belongs to that class regardless of how the file was named.
    """
    fire = np.zeros((512, 512), bool)
    unsure = np.zeros((512, 512), bool)
    only = labels[0] if len(labels) == 1 else None
    # Label Studio numbers the exported PNGs in the order the regions appear in
    # the annotation, so when the filename is silent that index still identifies
    # the class -- provided the two counts agree.
    ordered = ordered or []
    indexed = len(paths) == len(ordered) and len(labels) > 1
    ambiguous = False
    for p in paths:
        a = np.asarray(Image.open(p).convert("L"))
        # Each mask carries its own canvas size; deriving the scale per file lets
        # a coarse first pass and a finer relabel live in one hand_masks/.
        k = max(1, a.shape[0] // 512)
        if a.shape[0] != 512 * k or a.shape[1] != 512 * k:
            a = np.asarray(Image.fromarray(a).resize((512 * k, 512 * k), Image.NEAREST))
        painted = a > 0
        if k > 1:                      # any-fire-wins down to 512
            painted = painted.reshape(512, k, 512, k).any((1, 3))
        low = p.name.lower()
        if "uncertain" in low:
            unsure |= painted
        elif "active_fire" in low:
            fire |= painted
        elif only == "uncertain":
            unsure |= painted
        elif only == "active_fire":
            fire |= painted
        elif indexed:
            m = re.search(r"[-_](\d+)\.png$", p.name)
            idx = int(m.group(1)) if m else None
            cls = ordered[idx] if idx is not None and idx < len(ordered) else None
            if cls == "uncertain":
                unsure |= painted
            elif cls == "active_fire":
                fire |= painted
            else:
                fire |= painted
                ambiguous = True
        else:
            fire |= painted           # both classes used, filename says neither
            ambiguous = True

    out = np.full((512, 512), BACKGROUND, np.uint8)
    out[fire] = ACTIVE_FIRE
    out[unsure] = IGNORE          # hedged beats confident
    return out, ambiguous


def _components(m: np.ndarray) -> list[list[tuple[int, int]]]:
    """8-connected components, iterated over the sparse foreground."""
    seen = np.zeros_like(m)
    out = []
    h, w = m.shape
    for y0, x0 in zip(*np.nonzero(m)):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        out.append(pix)
    return out


def score(hand: dict[str, np.ndarray], manifest: list[dict], dataset: Path) -> None:
    per_fire: dict[str, dict] = defaultdict(
        lambda: dict(i=0, u=0, h=0, d=0, miss=0, n=0, unc=0, conf=0, unmk=0, blob=0, extra=0))
    rows = {r["label_file"]: r for r in manifest}
    hand_sizes, det_sizes = [], []

    for name, hm in hand.items():
        r = rows[name]
        det = np.asarray(Image.open(dataset / "masks" / r["source_file"]).convert("L"))
        keep = (hm != IGNORE) & (det != IGNORE)      # exclude nodata and 'uncertain'
        h = (hm == ACTIVE_FIRE) & keep
        d = (det == ACTIVE_FIRE) & keep
        s = per_fire[r["fire"]]
        s["i"] += int((h & d).sum()); s["u"] += int((h | d).sum())
        s["h"] += int(h.sum()); s["d"] += int(d.sum())
        s["n"] += 1
        s["unc"] += int((hm == IGNORE).sum())
        if h.any() and not d.any():
            s["miss"] += 1                            # detector saw nothing, you did

        # Cluster level. Pixel IoU conflates *where* the fire is with *how wide*
        # it was painted; agreement between connected components does not, so it
        # survives a brush that is the wrong size.
        for c in _components(d):
            ys, xs = zip(*c)
            det_sizes.append(len(c))
            if h[ys, xs].any():
                s["conf"] += 1
            else:
                s["unmk"] += 1
        for c in _components(h):
            ys, xs = zip(*c)
            hand_sizes.append(len(c))
            s["blob"] += 1
            if not d[ys, xs].any():
                s["extra"] += 1

    # Sampling fractions differ per fire whenever labelling stopped part-way, so
    # a pooled ratio would silently over-weight the fires that got further.
    total_by_fire = Counter(r["fire"] for r in manifest)

    log(f"\nPIXEL level  (sensitive to brush width -- read the calibration note below)")
    log(f"{'fire':10} {'chips':>6} {'of':>5} {'yours':>9} {'detector':>9} {'IoU':>7} "
        f"{'recall':>7} {'prec':>7}")
    tot = defaultdict(int)
    wi = wu = 0.0
    for fire in sorted(per_fire):
        s = per_fire[fire]
        for k in s:
            tot[k] += s[k]
        frac = s["n"] / total_by_fire[fire]
        wi += s["i"] / frac; wu += s["u"] / frac
        iou = s["i"] / s["u"] if s["u"] else float("nan")
        rec = s["i"] / s["h"] if s["h"] else float("nan")
        pre = s["i"] / s["d"] if s["d"] else float("nan")
        log(f"{fire:10} {s['n']:>6} {total_by_fire[fire]:>5} {s['h']:>9,} {s['d']:>9,} "
            f"{iou:>7.4f} {rec:>7.3f} {pre:>7.3f}")
    iou = tot["i"] / tot["u"] if tot["u"] else float("nan")
    rec = tot["i"] / tot["h"] if tot["h"] else float("nan")
    pre = tot["i"] / tot["d"] if tot["d"] else float("nan")
    log(f"{'ALL':10} {tot['n']:>6} {len(manifest):>5} {tot['h']:>9,} {tot['d']:>9,} "
        f"{iou:>7.4f} {rec:>7.3f} {pre:>7.3f}")
    if wu:
        log(f"{'reweighted':10} {'':>6} {'':>5} {'':>9} {'':>9} {wi / wu:>7.4f}"
            f"   <- per-fire sampling fractions equalised")

    cd = tot["conf"] + tot["unmk"]
    log(f"\nCLUSTER level  (brush-width independent)")
    log(f"  detector clusters you confirmed        {tot['conf']:>6}"
        + (f"   ({tot['conf'] / cd:.1%})" if cd else ""))
    log(f"  detector clusters you did not mark     {tot['unmk']:>6}"
        + (f"   ({tot['unmk'] / cd:.1%})" if cd else ""))
    log(f"  your regions with no detector cluster  {tot['extra']:>6}"
        + (f"   ({tot['extra'] / tot['blob']:.1%} of yours)" if tot["blob"] else "")
        + "   <- candidate detector misses")

    if hand_sizes and det_sizes:
        mh, md = float(np.median(hand_sizes)), float(np.median(det_sizes))
        dh, dd = 2 * (mh / np.pi) ** 0.5, 2 * (md / np.pi) ** 0.5
        log(f"\nBRUSH CALIBRATION")
        log(f"  median region: yours {mh:.0f} px vs detector {md:.0f} px "
            f"(equivalent disc {dh:.1f} px vs {dd:.1f} px)")
        if mh > 3 * md:
            log(f"  *** Your regions are {mh / md:.0f}x the area of the detector's. That is a brush")
            log(f"      wider than the targets, not a disagreement about where fire is -- your")
            log(f"      cluster-level confirmation rate is {tot['conf'] / cd:.0%}. The pixel IoU above")
            log(f"      understates you badly; treat the cluster numbers as the real result.")
            log(f"      To fix, relabel with a finer brush -- see relabel_batch/.")

    log(f"\nof {tot['n']} chips scored, {sum(1 for m in hand.values() if not (m == ACTIVE_FIRE).any())} "
        f"were judged to hold no fire")
    if tot["miss"]:
        log(f"{tot['miss']} chip(s) had fire for you and none for the detector -- the blind spot "
            f"that labelling only its positives could never surface.")
    if tot["unc"]:
        log(f"{tot['unc']:,} pixel(s) marked uncertain and excluded from every number above.")
    log(f"\nCoverage {tot['n']}/{len(manifest)} ({tot['n'] / len(manifest):.0%}). Because the batch "
        f"order is a random shuffle within each fire,")
    log("any contiguous run of it is an unbiased sample of that fire -- use the reweighted row "
        "for a whole-split estimate.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, default=None,
                    help="folder holding BOTH the unzipped PNG export and the JSON export")
    ap.add_argument("--batch", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--dataset", type=Path, default=Path("active_fire_dataset"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--score-only", action="store_true",
                    help="skip importing; score whatever already sits in hand_masks/")
    args = ap.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"dataset not found at {args.dataset}; pass --dataset")
    manifest = load_manifest(args.batch)

    if not args.score_only and args.export is None:
        ap.error("--export is required unless --score-only is given")
    if args.score_only:
        out = args.out or (args.batch / "hand_masks")
        by_src = {r["source_file"]: r["label_file"] for r in manifest}
        hand = {by_src[p.name]: np.asarray(Image.open(p).convert("L"))
                for p in sorted(out.glob("*.png")) if p.name in by_src}
        if not hand:
            raise SystemExit(f"no masks in {out} matching the manifest")
        log(f"scoring {len(hand)} mask(s) already in {out}/")
        score(hand, manifest, args.dataset)
        return 0

    tasks = read_tasks(args.export, manifest)
    if not tasks:
        raise SystemExit(
            f"no JSON export matched this batch under {args.export}.\n"
            "Export twice from Label Studio, into the same folder:\n"
            "  1. 'Brush labels to PNG', unzipped\n"
            "  2. 'JSON'\n"
            "The JSON is required: without it, chips you judged empty are invisible "
            "(they produce no PNG) and 'uncertain' cannot be told from 'active_fire'.")

    grouped = match_exports(args.export, manifest, tasks)
    out = args.out or (args.batch / "hand_masks")
    out.mkdir(parents=True, exist_ok=True)
    rows = {r["label_file"]: r for r in manifest}

    hand, ambiguous, empty = {}, [], 0
    # Drive from the JSON, not from the PNGs: a chip annotated as having no fire
    # is a judgement we must score, and it left no PNG behind.
    for name, meta in tasks.items():
        if not meta["annotated"]:
            continue
        paths = grouped.get(name, [])
        m, amb = fold(paths, meta["labels"], meta.get("ordered"))
        if amb:
            ambiguous.append(name)
        if not paths:
            empty += 1
        hand[name] = m
        Image.fromarray(m).save(out / rows[name]["source_file"])

    stray = set(grouped) - set(hand)
    if stray:
        log(f"warning: {len(stray)} chip(s) had PNG masks but no annotation in the JSON; skipped")
    if ambiguous:
        log(f"warning: {len(ambiguous)} chip(s) used both classes but the PNG names identify "
            f"neither; their strokes were all read as active_fire. Check: {ambiguous[:3]}")
    log(f"wrote {len(hand)} mask(s) to {out}/, named to match {args.dataset}/masks/")
    log(f"  of these, {empty} chip(s) you judged to contain no fire at all")
    score(hand, manifest, args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
