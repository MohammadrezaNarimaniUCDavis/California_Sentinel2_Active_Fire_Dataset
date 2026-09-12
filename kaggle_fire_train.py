#!/usr/bin/env python3
"""Segment active fire in Sentinel-2 chips. Self-contained, Kaggle-ready.

    python kaggle_fire_train.py                       # find data, split, train, test
    python kaggle_fire_train.py --no-train            # build and check the split only
    python kaggle_fire_train.py --aug-preview 12      # write augmented samples, no torch needed
    python kaggle_fire_train.py --epochs 60 --encoder resnet50
    python kaggle_fire_train.py --predict some/chips --checkpoint best_fire_unet.pt

Pasteable into a notebook cell -- it does not auto-run there, because in a
notebook `__name__` is already "__main__" and a bottom-of-file
`raise SystemExit(main())` fires on paste, which IPython renders as a wall of
its own traceback rather than the message. Call `main([...])` instead.

Input is `active_fire_dataset/images/`, target is `active_fire_dataset/masks/`
(0 background, 1 active_fire, 255 nodata). Chips are 512 px at 20 m, rendered
R=B12 (SWIR2), G=B11 (SWIR1), B=B8A (NIR).

The split holds out **whole fires**. Two dates of one fire share terrain, fuel
and weather, so splitting chips at random lets a model memorise a landscape and
call it detection. Separate fires are genuinely independent scenes.

Reported metric is **IoU on the fire class**. Not accuracy: fire is 0.07% of all
pixels, so predicting "no fire" everywhere scores 99.93%.

Fire's rarity is the whole problem, and three things here answer it:
  * **oversampling** -- chips holding fire are drawn more often than empty ones;
  * **copy-paste** -- flame fronts from other chips are composited into the one
    being trained on, which is the only augmentation that manufactures genuinely
    new positive *pixels* rather than re-orienting the ones already present;
  * **Dice** alongside weighted cross-entropy, because CE is an average over
    pixels and 0.07% of them cannot outvote the rest whatever the weight.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Held out for the final number. Mid-sized and varied rather than largest, and
# the same four exported for hand labelling -- so once that pass lands, the test
# set is human ground truth and the score stops measuring agreement with the
# detector that produced the training labels.
TEST_FIRES = {"Slater", "Hopkins", "CALDWELL", "Windy"}
# Distinct again, so validation cannot leak into test.
VAL_FIRES = {"Castle", "McCash", "Bobcat"}

BACKGROUND, ACTIVE_FIRE, NODATA = 0, 1, 255
IGNORE_INDEX = -100  # torch's own default for CrossEntropyLoss
NUM_CLASSES = 2
TILE = 512

SEARCH_ROOTS = ("/kaggle/input", "/content/drive/MyDrive", "/content", ".")
OUT_ROOTS = ("/kaggle/working", "/content", ".")

# Swept on val, then the winner is applied once to test. A single 0.5 argmax is
# the wrong operating point for a class this rare: the decision boundary that
# maximises accuracy and the one that maximises fire IoU are nowhere near
# each other.
THRESHOLDS = np.round(np.concatenate([np.arange(0.05, 0.95, 0.05),
                                      [0.95, 0.97, 0.98, 0.99, 0.995]]), 3)


def log(msg: str) -> None:
    print(msg, flush=True)


def in_notebook() -> bool:
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


def notebook_argv(argv=None):
    return argv if argv is not None else ([] if in_notebook() else sys.argv[1:])


def default_out() -> str:
    for path in OUT_ROOTS:
        if os.path.isdir(path) and os.access(path, os.W_OK):
            return path
    return "."


def find_dataset(root: str | None = None) -> Path:
    roots = [root] if root else [r for r in SEARCH_ROOTS if os.path.isdir(r)]
    for base in roots:
        for candidate in sorted(glob.glob(f"{base}/**/summary.csv", recursive=True)):
            parent = Path(candidate).parent
            if (parent / "images").is_dir() and (parent / "masks").is_dir():
                return parent
    raise FileNotFoundError(
        f"no dataset with images/ and masks/ under {roots}. Pass --dataset "
        f"(in a notebook: main(['--dataset', '/kaggle/input/...']))."
    )


def split_of(fire: str) -> str:
    return "test" if fire in TEST_FIRES else "val" if fire in VAL_FIRES else "train"


def read_manifest(dataset: Path) -> list[dict]:
    rows = list(csv.DictReader((dataset / "summary.csv").open()))
    for r in rows:
        r["split"] = split_of(r["fire"])
        r["fire_px"] = int(r["fire_px"])
    return rows


def load_pair(dataset: Path, file_name: str):
    """(image uint8 HWC, raw mask uint8 HW in {0, 1, 255})."""
    image = np.asarray(Image.open(dataset / "images" / file_name).convert("RGB"), dtype=np.uint8)
    raw = np.asarray(Image.open(dataset / "masks" / file_name).convert("L"), dtype=np.uint8)
    return image, raw


def channel_stats(rows: list[dict], dataset: Path, limit: int = 250):
    """Per-channel mean/std over the *train* split only, so val and test leak nothing."""
    train = [r for r in rows if r["split"] == "train"]
    sample = train[:: max(1, len(train) // limit)]
    total, total_sq, n = np.zeros(3), np.zeros(3), 0
    for r in sample:
        a = load_pair(dataset, r["file_name"])[0].astype(np.float64) / 255.0
        n += a.shape[0] * a.shape[1]
        total += a.sum((0, 1))
        total_sq += (a ** 2).sum((0, 1))
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


# --------------------------------------------------------------------------
# Augmentation. Pure numpy on uint8, so it is testable and previewable without
# torch installed -- see --aug-preview.
# --------------------------------------------------------------------------

class AugConfig:
    """Knobs, with the defaults that make sense for this dataset."""

    def __init__(self, scale=(0.55, 1.35), copy_paste=0.5, copy_paste_max=3,
                 paste_halo=2, gain=0.12, noise=0.02, dihedral=True):
        self.scale = scale
        self.copy_paste = copy_paste
        self.copy_paste_max = copy_paste_max
        self.paste_halo = paste_halo
        self.gain = gain
        self.noise = noise
        self.dihedral = dihedral

    @classmethod
    def from_args(cls, args):
        if args.aug == "none":
            return cls(scale=(1.0, 1.0), copy_paste=0.0, gain=0.0, noise=0.0, dihedral=False)
        if args.aug == "basic":  # what the first version of this script did
            return cls(scale=(1.0, 1.0), copy_paste=0.0, gain=0.0, noise=0.0, dihedral=True)
        return cls(scale=(args.scale_min, args.scale_max), copy_paste=args.copy_paste,
                   copy_paste_max=args.copy_paste_max, gain=args.gain, noise=args.noise)

    def __repr__(self):
        return (f"scale {self.scale[0]:.2f}-{self.scale[1]:.2f}  copy-paste p={self.copy_paste:.2f} "
                f"(<={self.copy_paste_max} donors, halo {self.paste_halo}px)  "
                f"gain +-{self.gain:.2f}  noise {self.noise:.3f}  dihedral {self.dihedral}")


def dihedral(image, raw, rng):
    """The eight exact moves of the square: no pixel is invented, so the label
    stays aligned. Anything interpolating the *mask* would average class 1
    against the ignore value and produce labels that mean nothing."""
    k = int(rng.integers(4))
    if k:
        image, raw = np.rot90(image, k, (0, 1)), np.rot90(raw, k, (0, 1))
    if rng.random() < 0.5:
        image, raw = image[:, ::-1], raw[:, ::-1]
    if rng.random() < 0.5:
        image, raw = image[::-1], raw[::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(raw)


def scale_crop(image, raw, rng, lo, hi):
    """Zoom by cropping (in) or reflect-padding (out), then resample to size.

    The mask resamples NEAREST and only ever NEAREST, so it stays a label map.
    The image resamples BILINEAR, which is fine -- it is a continuous quantity.
    When the chip holds fire the crop window is aimed at a fire pixel most of
    the time; a blind crop at 0.55 scale drops a small hotspot four times in
    five, which would quietly turn positives into mislabelled negatives.
    """
    if lo == hi == 1.0:
        return image, raw
    oh, ow = raw.shape
    s = float(rng.uniform(lo, hi))
    ch, cw = int(round(oh * s)), int(round(ow * s))
    ch, cw = max(64, ch), max(64, cw)

    pad_y, pad_x = max(0, ch - oh), max(0, cw - ow)
    if pad_y or pad_x:
        py, px = (pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)
        image = np.pad(image, (py, px, (0, 0)), mode="reflect")
        raw = np.pad(raw, (py, px), mode="reflect")
    h, w = raw.shape

    fire = np.argwhere(raw == ACTIVE_FIRE)
    if len(fire) and rng.random() < 0.85:
        fy, fx = fire[int(rng.integers(len(fire)))]
        y0 = int(np.clip(fy - rng.integers(ch), 0, h - ch))
        x0 = int(np.clip(fx - rng.integers(cw), 0, w - cw))
    else:
        y0, x0 = int(rng.integers(h - ch + 1)), int(rng.integers(w - cw + 1))

    image, raw = image[y0:y0 + ch, x0:x0 + cw], raw[y0:y0 + ch, x0:x0 + cw]
    if (ch, cw) != (oh, ow):
        image = np.asarray(Image.fromarray(image).resize((ow, oh), Image.BILINEAR), dtype=np.uint8)
        small = np.asarray(Image.fromarray(raw).resize((ow, oh), Image.NEAREST), dtype=np.uint8)
        if ch > oh:
            # Zooming *out*, NEAREST samples one source pixel per destination
            # pixel and steps straight over hotspots a few pixels across --
            # measured at 3% of fire chips, each one silently demoted to a
            # mislabelled negative. Resample the fire plane by area instead, so
            # a destination pixel covering any fire is fire. Not a dilation:
            # one source fire pixel still yields one destination fire pixel.
            small = small.copy()
            for value, keep in ((NODATA, 0.5), (ACTIVE_FIRE, 0.0)):  # fire last: it wins ties
                cover = np.asarray(Image.fromarray(((raw == value) * 255).astype(np.uint8))
                                   .resize((ow, oh), Image.BOX), dtype=np.uint8)
                small[cover > keep * 255] = value
        raw = small
    return image, raw


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Separable binary dilation by r px. Two passes of 2r+1 shifts, not (2r+1)^2."""
    if r <= 0:
        return mask
    out = mask
    for axis in (0, 1):
        acc = out
        for d in range(1, r + 1):
            acc = acc | np.roll(out, d, axis) | np.roll(out, -d, axis)
        out = acc
    return out


def copy_paste(image, raw, donors, rng, cfg):
    """Composite flame fronts from other chips into this one.

    This is the augmentation that matters. Flips and crops rearrange the 285k
    fire pixels the train split already has; copy-paste combines them, so a
    chip can carry fronts from three different fires at once and the model sees
    orders of magnitude more distinct fire/background arrangements than exist
    on disk.

    A halo of surrounding pixels comes across with each front but is *not*
    labelled fire. Pasting the hotspot alone would leave a hard rim of foreign
    terrain exactly on the class boundary -- a cue the model could learn instead
    of learning flame.

    The halo is *feathered* rather than hard-replaced, for the same reason. A
    hard halo trades one seam for another: donor surroundings are usually fresh
    burn scar, which is dark, so a hard edge against unburned ground draws a
    dark ring around every pasted front. A ring is far easier to detect than
    flame, and a model offered both will take the ring. Ramping the blend to
    zero over the halo leaves no edge to find.
    """
    n = 1 + int(rng.integers(cfg.copy_paste_max)) if cfg.copy_paste_max > 1 else 1
    for _ in range(n):
        d_img, d_raw = donors[int(rng.integers(len(donors)))]
        d_img, d_raw = dihedral(d_img, d_raw, rng)
        dy, dx = int(rng.integers(d_raw.shape[0])), int(rng.integers(d_raw.shape[1]))
        d_img = np.roll(d_img, (dy, dx), (0, 1))
        d_raw = np.roll(d_raw, (dy, dx), (0, 1))

        core = d_raw == ACTIVE_FIRE
        if not core.any():
            continue
        valid = (d_raw != NODATA) & (raw != NODATA)  # never paste into either nodata
        core = core & valid
        if not core.any():
            continue

        alpha = core.astype(np.float32)
        prev = core
        for d in range(1, cfg.paste_halo + 1):
            grown = dilate(prev, 1)
            alpha[grown & ~prev] = 1.0 - d / (cfg.paste_halo + 1.0)
            prev = grown
        alpha *= valid
        a = alpha[..., None]
        image = (a * d_img + (1.0 - a) * image).round().astype(np.uint8)
        raw = np.where(core, np.uint8(ACTIVE_FIRE), raw)
    return image, raw


def photometric(image_f, rng, cfg):
    """Gain is applied to all three channels *equally*.

    Per-channel jitter is off limits: a hotspot is defined by SWIR2 brightness,
    which is the red channel, so shifting channels independently edits the
    physical quantity the label encodes. A common gain only restretches the
    composite, which is a real source of variation between renders.
    """
    if cfg.gain:
        image_f *= float(rng.uniform(1.0 - cfg.gain, 1.0 + cfg.gain))
    if cfg.noise:
        image_f += rng.normal(0.0, cfg.noise, image_f.shape).astype(np.float32)
    return np.clip(image_f, 0.0, 1.0, out=image_f)


def augment(image, raw, donors, rng, cfg):
    """uint8 in, uint8 in, (float32 image in [0,1], uint8 raw) out."""
    image, raw = scale_crop(image, raw, rng, *cfg.scale)
    if cfg.dihedral:
        image, raw = dihedral(image, raw, rng)
    if donors and cfg.copy_paste and rng.random() < cfg.copy_paste:
        image, raw = copy_paste(image, raw, donors, rng, cfg)
    return photometric(image.astype(np.float32) / 255.0, rng, cfg), raw


def to_label(raw: np.ndarray) -> np.ndarray:
    label = np.full(raw.shape, IGNORE_INDEX, dtype=np.int64)
    label[raw == BACKGROUND] = 0
    label[raw == ACTIVE_FIRE] = 1
    return label


class FireChips:
    """Yields (image CHW float32 normalised, mask HW int64 in {0, 1, IGNORE_INDEX})."""

    def __init__(self, rows, dataset: Path, split: str, mean, std,
                 cfg: AugConfig | None = None, seed=0, donor_pool=64):
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no chips in split {split!r}")
        self.dataset, self.mean, self.std = Path(dataset), mean, std
        self.cfg, self.seed = cfg, seed
        self._rng = None
        self._donors = None
        self._donor_rows = ([r for r in self.rows if r["fire_px"] > 0][:donor_pool]
                            if cfg and cfg.copy_paste else [])

    def __len__(self):
        return len(self.rows)

    def rng(self):
        # Seeded per worker, not per dataset. A single rng built in __init__ is
        # copied intact into every forked worker, so with two workers half the
        # "random" augmentations are exact duplicates of the other half.
        if self._rng is None:
            try:
                import torch.utils.data as tud
                info = tud.get_worker_info()
                wid = info.id if info is not None else 0
            except Exception:
                wid = 0
            self._rng = np.random.default_rng([self.seed, wid, os.getpid()])
        return self._rng

    def donors(self):
        # Held in RAM: 64 chips at 512x512x4 bytes is ~50 MB, and re-reading a
        # donor PNG per sample would make copy-paste the slowest thing in the
        # loop by a wide margin.
        if self._donors is None:
            self._donors = [load_pair(self.dataset, r["file_name"]) for r in self._donor_rows]
        return self._donors

    def __getitem__(self, i):
        import torch

        row = self.rows[i]
        image, raw = load_pair(self.dataset, row["file_name"])
        if self.cfg is not None:
            image, raw = augment(image, raw, self.donors(), self.rng(), self.cfg)
        else:
            image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return (torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                torch.from_numpy(to_label(raw)))


def make_sampler(dataset, args):
    """Draw fire-bearing chips more often, and draw more of them per epoch.

    63% of train chips are empty. Left alone, most gradient steps see no fire at
    all. --pos-weight tilts the draw; --repeat sets how many draws make an
    epoch, which is the knob for "more data" -- every draw is independently
    augmented, so 4x repeat is 4x distinct examples, not 4x the same one.
    """
    import torch

    w = np.array([args.pos_weight if r["fire_px"] > 0 else 1.0 for r in dataset.rows])
    n = int(round(len(dataset) * args.repeat))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(w, dtype=torch.double), num_samples=n, replacement=True)
    return sampler, w / w.sum(), n


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

RESNET_CHANNELS = {"resnet18": (64, 64, 128, 256, 512), "resnet34": (64, 64, 128, 256, 512),
                   "resnet50": (64, 256, 512, 1024, 2048), "resnet101": (64, 256, 512, 1024, 2048)}


def build_unet(classes: int, encoder: str, pretrained: bool = True):
    """U-Net, preferring segmentation_models_pytorch but never requiring it.

    smp is absent from many Kaggle images, and aborting a run after the data is
    already prepared is a poor trade for one optional dependency. The fallback
    is the same architecture built from torchvision, which ships with torch.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    try:
        import segmentation_models_pytorch as smp
        try:
            model = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None,
                             in_channels=3, classes=classes)
            log(f"model: smp.Unet({encoder}, {'imagenet' if pretrained else 'random'})")
            return model
        except Exception as exc:
            log(f"note: smp could not build/fetch ({exc.__class__.__name__}); using torchvision")
    except ImportError:
        pass

    import torchvision

    if encoder not in RESNET_CHANNELS:
        log(f"note: torchvision fallback has no {encoder}; using resnet34")
        encoder = "resnet34"
    c0, c1, c2, c3, c4 = RESNET_CHANNELS[encoder]

    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))

        def forward(self, x):
            return self.f(x)

    class ResNetUNet(nn.Module):
        def __init__(self, classes, backbone):
            super().__init__()
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
            self.pool = backbone.maxpool
            self.l1, self.l2 = backbone.layer1, backbone.layer2
            self.l3, self.l4 = backbone.layer3, backbone.layer4
            self.d4, self.d3 = Block(c4 + c3, 256), Block(256 + c2, 128)
            self.d2, self.d1 = Block(128 + c1, 64), Block(64 + c0, 64)
            self.head = nn.Sequential(Block(64, 32), nn.Conv2d(32, classes, 1))

        def encoder_modules(self):
            return [self.stem, self.l1, self.l2, self.l3, self.l4]

        def forward(self, x):
            s0 = self.stem(x)
            s1 = self.l1(self.pool(s0))
            s2 = self.l2(s1)
            s3 = self.l3(s2)
            s4 = self.l4(s3)
            up = lambda z: F.interpolate(z, scale_factor=2, mode="nearest")
            u = self.d4(torch.cat([up(s4), s3], 1))
            u = self.d3(torch.cat([up(u), s2], 1))
            u = self.d2(torch.cat([up(u), s1], 1))
            u = self.d1(torch.cat([up(u), s0], 1))
            return self.head(up(u))

    weights = None
    if pretrained:
        try:
            enum = getattr(torchvision.models, f"ResNet{encoder[6:]}_Weights")
            weights = enum.IMAGENET1K_V1
            getattr(torchvision.models, encoder)(weights=weights)
        except Exception as exc:
            log(f"note: no pretrained weights ({exc.__class__.__name__}). Turn on "
                f"Settings -> Internet for a pretrained encoder.")
            weights = None
    log(f"model: torchvision {encoder}-UNet ({'imagenet' if weights else 'random'} init)")
    return ResNetUNet(classes, getattr(torchvision.models, encoder)(weights=weights))


def encoder_parameters(model):
    """Whatever the backbone turned out to be, the pretrained half of it."""
    if hasattr(model, "encoder"):
        return list(model.encoder.parameters())
    if hasattr(model, "encoder_modules"):
        return [p for m in model.encoder_modules() for p in m.parameters()]
    return []


class FireLoss:
    """Weighted CE plus soft Dice on the fire class.

    CE is a mean over pixels; at 0.07% positives, the fire term is a rounding
    error in that mean no matter how the weight is set, and pushing the weight
    high enough to matter destabilises the background term instead. Dice is a
    ratio over the whole batch, so a batch with 400 fire pixels and one with
    40,000 contribute comparably. The two together train far better than either.
    """

    def __init__(self, weights, dice_weight=1.0, focal_gamma=0.0):
        import torch.nn as nn
        self.ce = nn.CrossEntropyLoss(weight=weights, reduction="none")
        self.dice_weight, self.focal_gamma = dice_weight, focal_gamma

    def __call__(self, logits, target):
        import torch

        valid = target != IGNORE_INDEX
        ce = self.ce(logits, target)  # ignore_index=-100 already zeroes the rest
        if self.focal_gamma:
            with torch.no_grad():
                pt = torch.softmax(logits, 1).gather(
                    1, target.clamp_min(0).unsqueeze(1)).squeeze(1)
            ce = ce * (1.0 - pt).pow(self.focal_gamma)
        loss = ce.sum() / valid.sum().clamp_min(1)

        if self.dice_weight:
            prob = torch.softmax(logits.float(), 1)[:, 1] * valid
            tgt = (target == 1).float()
            inter = (prob * tgt).sum()
            loss = loss + self.dice_weight * (1.0 - (2 * inter + 1.0) / (prob.sum() + tgt.sum() + 1.0))
        return loss


def evaluate(model, loader, device, loss_fn=None, thresholds=THRESHOLDS, fixed=None):
    """Dataset-wide fire IoU, accumulated over chips rather than averaged per chip.

    Per-chip averaging would let a chip holding four fire pixels count as much
    as one holding a whole front, and would be undefined on the ~60% of chips
    with no fire at all.
    """
    import torch

    model.eval()
    grid = np.array([fixed]) if fixed is not None else np.asarray(thresholds)
    inter = np.zeros(len(grid))
    union = np.zeros(len(grid))
    tp = np.zeros(len(grid))
    pred_pos = np.zeros(len(grid))
    total = batches = 0.0
    with torch.no_grad():
        for image, mask in loader:
            image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            logits = model(image)
            if loss_fn is not None:
                total += float(loss_fn(logits, mask))
                batches += 1
            prob = torch.softmax(logits.float(), 1)[:, 1]
            keep = mask != IGNORE_INDEX
            t = (mask == 1) & keep
            t_sum = float(t.sum())
            for j, thr in enumerate(grid):
                p = (prob >= float(thr)) & keep
                i_ = float((p & t).sum())
                inter[j] += i_
                union[j] += float(p.sum()) + t_sum - i_
                tp[j] += i_
                pred_pos[j] += float(p.sum())
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    best = int(np.nanargmax(iou)) if np.isfinite(iou).any() else 0
    truth = inter[best] + (union[best] - inter[best] - (pred_pos[best] - tp[best]))
    return {
        "iou": float(iou[best]),
        "threshold": float(grid[best]),
        "precision": float(tp[best] / pred_pos[best]) if pred_pos[best] else float("nan"),
        "recall": float(tp[best] / truth) if truth else float("nan"),
        "loss": total / batches if batches else float("nan"),
        "curve": {float(t): float(v) for t, v in zip(grid, iou)},
    }


def overlay(image_u8: np.ndarray, pred: np.ndarray, truth: np.ndarray | None = None) -> Image.Image:
    """Prediction in magenta, truth outline in green. For eyeballing, never input."""
    out = image_u8.copy()
    out[pred.astype(bool)] = (0.4 * out[pred.astype(bool)] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
    if truth is not None:
        edge = dilate(truth.astype(bool), 1) & ~truth.astype(bool)
        out[edge] = np.array([0, 255, 0], dtype=np.uint8)
    return Image.fromarray(out)


def train(rows, dataset: Path, mean, std, args) -> int:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import torchvision
    log(f"torch {torch.__version__}  torchvision {torchvision.__version__}  "
        f"cuda {torch.version.cuda if device == 'cuda' else 'n/a'}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"\ndevice: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
        + f"  seed={args.seed}")

    cfg = AugConfig.from_args(args)
    log(f"augmentation: {cfg}")

    train_set = FireChips(rows, dataset, "train", mean, std, cfg=cfg, seed=args.seed,
                          donor_pool=args.donor_pool)
    sampler, probs, per_epoch = make_sampler(train_set, args)
    kw = dict(batch_size=args.batch_size, num_workers=args.workers,
              pin_memory=(device == "cuda"), persistent_workers=args.workers > 0)
    loaders = {"train": DataLoader(train_set, sampler=sampler, drop_last=True, **kw)}
    for split in ("val", "test"):
        loaders[split] = DataLoader(
            FireChips(rows, dataset, split, mean, std, cfg=None, seed=args.seed),
            shuffle=False, drop_last=False, **kw)

    n_pos = sum(1 for r in train_set.rows if r["fire_px"] > 0)
    log(f"train draws: {per_epoch}/epoch from {len(train_set)} chips "
        f"({args.repeat:g}x repeat), {n_pos} of them fire-bearing, oversampled {args.pos_weight:g}x "
        f"-> {probs[[r['fire_px'] > 0 for r in train_set.rows]].sum() * 100:.1f}% of draws hold fire")

    model = build_unet(NUM_CLASSES, args.encoder, pretrained=not args.scratch).to(device)
    enc = encoder_parameters(model)
    if args.freeze_epochs and enc:
        for p in enc:
            p.requires_grad_(False)
        log(f"encoder frozen for {args.freeze_epochs} epoch(s): the decoder is random and its "
            f"early gradients would otherwise wreck the pretrained features")

    # The class weight is computed against the *sampled* fire fraction, not the
    # on-disk one -- oversampling has already raised it, and double-counting the
    # correction would over-weight fire by the same factor twice.
    fire_frac = float(np.dot(probs, [r["fire_px"] for r in train_set.rows])) / (TILE * TILE)
    # Background is pinned at 1.0 and never rescaled. Normalising the pair to a
    # fixed sum is what actually destroys the background term: at 1/frac = 1350
    # even a cap of 200 comes back as [0.010, 1.990], which is the "paint
    # everything as fire" failure the cap was supposed to prevent. And the
    # fire weight is sqrt-tempered rather than raw 1/frequency, because Dice is
    # already carrying the imbalance -- stacking a 1350x CE weight on top of it
    # just makes the first few epochs diverge.
    fire_w = float(np.clip(math.sqrt(1.0 / max(fire_frac, 1e-9)), 1.0, args.max_weight_ratio))
    if args.fire_weight:
        fire_w = args.fire_weight
    weights = torch.tensor([1.0, fire_w], device=device, dtype=torch.float32)
    log(f"class weights (background, fire): [1.000, {fire_w:.2f}]"
        f"   (fire is {fire_frac * 100:.4f}% of sampled train pixels)")

    loss_fn = FireLoss(weights, dice_weight=args.dice_weight, focal_gamma=args.focal_gamma)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch, "amp") else \
        torch.cuda.amp.GradScaler(enabled=amp)
    if amp:
        log("mixed precision: on")

    best, best_thr, bad = float("-inf"), 0.5, 0
    os.makedirs(args.out, exist_ok=True)
    checkpoint = os.path.join(args.out, "best_fire_unet.pt")
    history = os.path.join(args.out, "history.csv")
    with open(history, "w", newline="") as fh:
        csv.writer(fh).writerow(["epoch", "train_loss", "val_loss", "val_iou", "threshold",
                                 "precision", "recall", "lr"])

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs and enc and epoch == args.freeze_epochs + 1:
            for p in enc:
                p.requires_grad_(True)
            opt.add_param_group({"params": enc, "lr": args.lr * args.encoder_lr_scale,
                                 "weight_decay": 1e-4})
            log(f"encoder unfrozen at lr {args.lr * args.encoder_lr_scale:.2e}")

        model.train()
        epoch_lr = opt.param_groups[0]["lr"]  # the lr this epoch ran at, not the next one's
        running, steps = 0.0, 0
        for image, mask in loaders["train"]:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                loss = loss_fn(model(image), mask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach())
            steps += 1
        sched.step()

        m = evaluate(model, loaders["val"], device, loss_fn)
        improved = math.isfinite(m["iou"]) and m["iou"] > best
        flag = ""
        if improved or not os.path.exists(checkpoint):
            best, best_thr = (m["iou"], m["threshold"]) if improved else (best, best_thr)
            flag = "  <- best, saved" if improved else "  <- saved"
            bad = 0
            torch.save({"model": model.state_dict(), "val_iou": m["iou"], "epoch": epoch,
                        "threshold": m["threshold"], "encoder": args.encoder,
                        "mean": mean.tolist(), "std": std.tolist()}, checkpoint)
        else:
            bad += 1
        log(f"epoch {epoch:>3}/{args.epochs}  train loss {running / max(steps, 1):.4f}"
            f"  val loss {m['loss']:.4f}  val fire IoU {m['iou']:.4f} @thr {m['threshold']:.2f}"
            f"  P {m['precision']:.3f} R {m['recall']:.3f}{flag}")
        with open(history, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch, running / max(steps, 1), m["loss"], m["iou"],
                                     m["threshold"], m["precision"], m["recall"], epoch_lr])
        if args.patience and bad >= args.patience:
            log(f"no val improvement in {bad} epochs; stopping early")
            break

    if not os.path.exists(checkpoint):
        log("no checkpoint written")
        return 1
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    log(f"\nbest val fire IoU {best:.4f} at threshold {best_thr:.2f} (epoch {state['epoch']})")

    t = evaluate(model, loaders["test"], device, fixed=best_thr)
    log(f"TEST fire IoU {t['iou']:.4f}  P {t['precision']:.3f}  R {t['recall']:.3f}"
        f"  at the val-chosen threshold {best_thr:.2f}")
    log(f"       held-out fires: {', '.join(sorted(TEST_FIRES))}")
    t_free = evaluate(model, loaders["test"], device)
    log(f"       (oracle threshold {t_free['threshold']:.2f} would give {t_free['iou']:.4f} -- "
        f"reported for the gap only, it is not a legitimate score)")

    dump_samples(model, rows, dataset, mean, std, best_thr, device,
                 os.path.join(args.out, "samples"), args.samples)
    log("\nThese masks came from the SWIR hotspot detector, so this measures agreement")
    log("with it. Re-run against hand-corrected masks for a number about reality.")
    return 0


def dump_samples(model, rows, dataset, mean, std, thr, device, out_dir, n):
    """Overlays for the fire-bearing test chips: the score says how much, these say how."""
    import torch

    if n <= 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    picks = [r for r in rows if r["split"] == "test" and r["fire_px"] > 0]
    picks = sorted(picks, key=lambda r: -r["fire_px"])[:: max(1, len(picks) // max(n, 1))][:n]
    model.eval()
    with torch.no_grad():
        for r in picks:
            image_u8, raw = load_pair(Path(dataset), r["file_name"])
            x = ((image_u8.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
            prob = torch.softmax(model(torch.from_numpy(x[None]).to(device)).float(), 1)[0, 1]
            pred = (prob.cpu().numpy() >= thr)
            overlay(image_u8, pred, raw == ACTIVE_FIRE).save(
                os.path.join(out_dir, f"pred_{r['file_name']}"))
    log(f"wrote {len(picks)} overlays to {out_dir}/ (magenta = predicted, green outline = label)")


def predict(args) -> int:
    """Run a trained checkpoint over loose chips and write mask PNGs."""
    import torch

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mean = np.asarray(state["mean"], dtype=np.float32)
    std = np.asarray(state["std"], dtype=np.float32)
    thr = args.threshold if args.threshold is not None else state.get("threshold", 0.5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_unet(NUM_CLASSES, state.get("encoder", args.encoder), pretrained=False)
    model.load_state_dict(state["model"])
    model = model.to(device).eval()
    log(f"checkpoint: epoch {state.get('epoch')} val IoU {state.get('val_iou', float('nan')):.4f}, "
        f"threshold {thr:.2f}")

    src = Path(args.predict)
    files = sorted(src.glob("*.png")) if src.is_dir() else [src]
    if not files:
        log(f"no PNGs under {src}")
        return 1
    out_dir = Path(args.predict_out or os.path.join(args.out or default_out(), "predictions"))
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for f in files:
            image_u8 = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
            x = ((image_u8.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
            prob = torch.softmax(model(torch.from_numpy(x[None]).to(device)).float(), 1)[0, 1]
            pred = prob.cpu().numpy() >= thr
            Image.fromarray((pred.astype(np.uint8) * ACTIVE_FIRE)).save(out_dir / "masks" / f.name)
            overlay(image_u8, pred).save(out_dir / "overlays" / f.name)
            log(f"{f.name}: {int(pred.sum()):>6} fire px")
    log(f"\nmasks (0/1) in {out_dir}/masks, overlays in {out_dir}/overlays")
    return 0


def aug_preview(rows, dataset: Path, args) -> int:
    """Write augmented chips to disk so the augmentation can be looked at.

    Needs numpy and pillow only -- no torch -- so it works before a GPU session.
    """
    cfg = AugConfig.from_args(args)
    log(f"augmentation: {cfg}")
    train_rows = [r for r in rows if r["split"] == "train"]
    seeds = [r for r in train_rows if r["fire_px"] > 0]
    donors = [load_pair(dataset, r["file_name"])
              for r in seeds[: args.donor_pool]] if cfg.copy_paste else []
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out or default_out()) / "aug_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    picked = [seeds[int(rng.integers(len(seeds)))] for _ in range(args.aug_preview)]
    before = after = 0
    for k, row in enumerate(picked):
        image, raw = load_pair(dataset, row["file_name"])
        before += int((raw == ACTIVE_FIRE).sum())
        aug_img, aug_raw = augment(image, raw, donors, rng, cfg)
        after += int((aug_raw == ACTIVE_FIRE).sum())
        Image.fromarray((aug_img * 255).astype(np.uint8)).save(out_dir / f"{k:02d}_{row['file_name']}")
        overlay((aug_img * 255).astype(np.uint8), aug_raw == ACTIVE_FIRE).save(
            out_dir / f"{k:02d}_label_{row['file_name']}")
    log(f"wrote {2 * len(picked)} PNGs to {out_dir}/")
    log(f"fire pixels: {before:,} on disk -> {after:,} after augmentation "
        f"({after / max(before, 1):.2f}x)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scratch", action="store_true", help="skip pretrained encoder weights")
    ap.add_argument("--freeze-epochs", type=int, default=2, help="epochs with the encoder frozen")
    ap.add_argument("--encoder-lr-scale", type=float, default=0.1,
                    help="encoder lr as a fraction of --lr once unfrozen")
    ap.add_argument("--patience", type=int, default=0, help="early stop after N epochs without val gain")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")

    g = ap.add_argument_group("data volume")
    g.add_argument("--repeat", type=float, default=4.0, help="augmented draws per chip per epoch")
    g.add_argument("--pos-weight", type=float, default=3.0, help="oversampling of fire-bearing chips")

    g = ap.add_argument_group("augmentation")
    g.add_argument("--aug", choices=("none", "basic", "heavy"), default="heavy")
    g.add_argument("--scale-min", type=float, default=0.55)
    g.add_argument("--scale-max", type=float, default=1.35)
    g.add_argument("--copy-paste", type=float, default=0.5, help="probability per sample")
    g.add_argument("--copy-paste-max", type=int, default=3, help="donor chips per paste")
    g.add_argument("--donor-pool", type=int, default=64, help="fire chips held in RAM as donors")
    g.add_argument("--gain", type=float, default=0.12)
    g.add_argument("--noise", type=float, default=0.02)

    g = ap.add_argument_group("loss")
    g.add_argument("--dice-weight", type=float, default=1.0, help="0 disables Dice")
    g.add_argument("--focal-gamma", type=float, default=0.0)
    g.add_argument("--max-weight-ratio", type=float, default=50.0,
                   help="cap on the auto sqrt-tempered fire weight")
    g.add_argument("--fire-weight", type=float, default=0.0,
                   help="set the fire class weight outright, ignoring the auto rule")

    g = ap.add_argument_group("modes")
    g.add_argument("--no-train", action="store_true")
    g.add_argument("--aug-preview", type=int, default=0, metavar="N",
                   help="write N augmented samples and exit (no torch needed)")
    g.add_argument("--samples", type=int, default=8, help="test overlays to write after training")
    g.add_argument("--predict", default=None, metavar="PATH", help="PNG or directory of PNGs")
    g.add_argument("--checkpoint", default="best_fire_unet.pt")
    g.add_argument("--predict-out", default=None)
    g.add_argument("--threshold", type=float, default=None, help="override the stored threshold")
    args = ap.parse_args(notebook_argv(argv))

    if args.predict:
        return predict(args)

    dataset = Path(args.dataset) if args.dataset else find_dataset()
    args.out = args.out or default_out()
    log(f"dataset: {dataset}\n")

    rows = read_manifest(dataset)
    counts = collections.Counter(r["split"] for r in rows)
    log(f"{'split':6} {'chips':>6} {'w/fire':>7} {'fire px':>9}  fires")
    for split in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == split]
        names = sorted({r["fire"] for r in sub})
        log(f"{split:6} {len(sub):>6} {sum(1 for r in sub if r['fire_px']):>7} "
            f"{sum(r['fire_px'] for r in sub):>9,}  {len(names)}"
            + (f" ({', '.join(names)})" if split != "train" else ""))
    if not counts["train"]:
        log("\nTrain split is empty; check the fire names in TEST_FIRES/VAL_FIRES.")
        return 1
    overlap = (TEST_FIRES & VAL_FIRES) or (
        {r["fire"] for r in rows if r["split"] == "train"} & (TEST_FIRES | VAL_FIRES))
    if overlap:
        raise AssertionError(f"fire in more than one split: {overlap}")
    log("no fire appears in more than one split: OK")

    if args.aug_preview:
        return aug_preview(rows, dataset, args)

    mean, std = channel_stats(rows, dataset)
    log(f"train-split normalisation  mean {np.round(mean, 4).tolist()}  std {np.round(std, 4).tolist()}")
    if args.no_train:
        return 0
    return train(rows, dataset, mean, std, args)


if __name__ == "__main__":
    if in_notebook():
        # Resolve the dataset now and print a line that can actually be run. The
        # previous banner showed a '<slug>' placeholder, which is exactly the
        # kind of thing that gets pasted verbatim into the next cell.
        try:
            print(f"Loaded. Dataset found at {find_dataset()}\n"
                  f"Now run:\n    main(['--epochs', '40'])")
        except FileNotFoundError:
            print("Loaded, but no dataset was found under /kaggle/input.\n"
                  "Find yours, then pass it explicitly:\n"
                  "    !find /kaggle/input -name summary.csv\n"
                  "    main(['--dataset', '<the directory holding summary.csv>', '--epochs', '40'])")
    else:
        raise SystemExit(main())
