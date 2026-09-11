"""Active-fire detection from Sentinel-2 reflectance.

Runs at 20 m, the native resolution of B8A, B11 and B12.

Murphy et al. (2016), "HOTMAP: Global hot target detection at moderate spatial
resolution", RSE 177 -- flaming combustion emits strongly at 2.2 um, so B12
rises far above B11 while the NIR stays dark. Two departures from the paper,
both forced by what the measurements on this imagery actually showed:

* Murphy's B12/B8A >= 1.4 is kept but raised to 4.0. The false alarms are
  specular glint off glasshouse, metal roofing and water, which is bright in
  *every* band -- measured B12/B8A of 0.76 to 1.33 -- and slips past a 1.4
  threshold by accident. Flame is not spectrally flat: the same ratio runs 8 to
  10 even where B8A is clearly elevated. Where B8A falls to zero or below the
  quotient is treated as infinite, which is correct rather than a special case:
  no NIR at all cannot be a reflective surface.

  An absolute *ceiling* on B8A was tried here first and was badly wrong. Tuned
  to 0.10 on the Palisades fire, it inverted the detector on large fires: an
  intense front emits enough at 0.865 um to push B8A to 0.10-0.20, so the
  ceiling rejected the saturated cores (B12 = 1.44) while keeping their cooler
  margins. On the Caldor fire it lost 72% of the bright pixels and on YORK 91%.
* The brightness floor is 0.35, not 0.15. Fresh char is itself dark in NIR and
  bright in SWIR2, so at 0.15 the test cannot tell a flame from the scar it just
  made -- it reported over a thousand "active fire" pixels four weeks after
  containment. Only thermal emission pushes SWIR2 past what a passive surface
  reflects. On the Palisades fire the corrected test reads 121 px on ignition
  day, decays, and returns exactly zero on every frame after containment.
"""

from __future__ import annotations

import numpy as np
import rasterio.features

from . import chip

INDEX_RES = 20.0
INDEX_BANDS = ("nir08", "swir16", "swir22")

# Confirmed flame: absolutely bright at 2.2 um, and far brighter there than in
# the NIR. 4.0 sits well above every glint cluster measured (max 1.33) and well
# below flame (8-10), so the gap is wide rather than finely tuned.
FIRE_B12_MIN = 0.35
FIRE_NIR_RATIO = 4.0

# Growth tier: a looser test that counts only where it touches a confirmed pixel,
# recovering the cooler skirt of a front without admitting isolated hits.
FIRE_B12_POTENTIAL = 0.25
FIRE_NIR_RATIO_POTENTIAL = 2.0

# B12/B11 is deliberately not tested. On the hottest fronts B11 saturates too,
# which drags the ratio under any useful threshold and discards exactly the
# pixels that are most certainly burning -- 5,290 of them on one Caldor tile.
FIRE_MIN_PIXELS = 2

# Mask encoding, shared with the training pipeline.
BACKGROUND = 0
ACTIVE_FIRE = 1
IGNORE = 255


def read_bands(scene, grid: chip.Grid) -> dict[str, np.ndarray]:
    """Read B8A/B11/B12 over the chip window at their native 20 m."""
    size = int(round(grid.size * grid.resolution / INDEX_RES))
    return {
        band: chip.read_reflectance(scene.hrefs[band], grid, *scene.scales[band], out_size=size)
        for band in INDEX_BANDS
    }

def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num/den, with a floor on the denominator so 0/0 does not become fire."""
    return num / np.maximum(den, 1e-6)

def nir_ratio(nir08: np.ndarray, swir22: np.ndarray) -> np.ndarray:
    """B12/B8A, treating a non-positive NIR as an infinite ratio.

    That is the physically right answer, not a guard: a surface reflecting
    nothing at 0.865 um is not reflecting sunlight, so the glint hypothesis is
    already excluded. Clamping instead would make the hottest pixels -- where
    B8A is driven to zero or slightly negative -- look like the least fiery.
    """
    return np.where(nir08 > 1e-6, swir22 / np.maximum(nir08, 1e-6), np.inf)


def detect_fire(nir08: np.ndarray, swir16: np.ndarray, swir22: np.ndarray) -> np.ndarray:
    """Hotspot mask: SWIR2 far above the NIR, grown into its cooler margins."""
    ratio = nir_ratio(nir08, swir22)
    confirmed = (swir22 >= FIRE_B12_MIN) & (ratio >= FIRE_NIR_RATIO)
    potential = (swir22 >= FIRE_B12_POTENTIAL) & (ratio >= FIRE_NIR_RATIO_POTENTIAL)

    grown = confirmed.copy()
    for _ in range(2):
        grown |= dilate(grown) & potential
    return sieve(grown, FIRE_MIN_PIXELS)


def has_data(bands: dict[str, np.ndarray]) -> np.ndarray:
    """Pixels that carry a real measurement, cloud or not.

    The test is `!= 0`, not `> 0`. Processing baseline 04.00+ carries a -0.1
    reflectance offset, so dark water sits marginally *below* zero while
    `chip.read_reflectance` pins true nodata to exactly 0.0. Screening on `> 0`
    discards the ocean -- a third of this coastal frame -- as unusable, which
    would mark it ignore and teach the model nothing about water.
    """
    ok = np.ones(next(iter(bands.values())).shape, dtype=bool)
    for band in bands.values():
        ok &= band != 0.0
    return ok

def dilate(mask: np.ndarray) -> np.ndarray:
    """4-connected dilation by one pixel."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out

def sieve(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Drop connected foreground regions smaller than `min_pixels`."""
    if min_pixels <= 1 or not mask.any():
        return mask
    kept = rasterio.features.sieve(mask.astype(np.uint8), size=min_pixels, connectivity=8)
    return kept.astype(bool)
