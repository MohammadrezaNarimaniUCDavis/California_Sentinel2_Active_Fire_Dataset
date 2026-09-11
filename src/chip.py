"""Chip grid definition, band reads, cloud screening and false-colour render.

Every chip in the dataset is cut from the *same* map-projected window, snapped
to the pixel grid. That is what makes the 100 outputs co-registered: a polygon
drawn on one date lands on the same ground on every other date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

# UTM zone 11N -- the projection Sentinel-2 tile 11SLT (which contains the
# Palisades) is delivered in. Chips are only cut from scenes in this CRS so no
# reprojection ever has to touch the pixels.
CHIP_EPSG = 32611

# SCL classes that mean "this pixel is not usable imagery".
# 0 nodata, 1 saturated/defective, 3 cloud shadow, 8 cloud medium probability,
# 9 cloud high probability, 10 thin cirrus.
SCL_UNUSABLE = (0, 1, 3, 8, 9, 10)

# Fixed dataset-wide reflectance clip. NOT per-image percentiles: autoscaling
# each frame would make a burning scene and a quiet scene equally bright and
# destroy the cross-date comparability the labelling task depends on.
STRETCH = {
    "swir22": (0.00, 0.35),  # R
    "swir16": (0.00, 0.35),  # G
    "red": (0.00, 0.22),  # B
}

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
    "AWS_NO_SIGN_REQUEST": "YES",
    # Without these a stalled connection blocks the reading thread forever.
    # A 25-fire build hung for 79 minutes on a single scene: 1m41s of CPU
    # against 1h24m of wall clock, no log output, nothing to kill but the whole
    # run. Retry a few times, then give up and let the scene be skipped.
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_CONNECTTIMEOUT": "20",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


@dataclass(frozen=True)
class Grid:
    """A fixed, pixel-snapped map window shared by every chip."""

    epsg: int
    size: int  # pixels per side
    resolution: float  # metres per pixel
    left: float
    bottom: float
    right: float
    top: float

    @classmethod
    def centered_on(
        cls, lon: float, lat: float, size: int, resolution: float, epsg: int = CHIP_EPSG
    ) -> "Grid":
        (x,), (y,) = warp_transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
        # Snap the centre to the pixel grid so the window edges fall exactly on
        # source pixel boundaries and no resampling shift is introduced.
        x = round(x / resolution) * resolution
        y = round(y / resolution) * resolution
        half = size * resolution / 2
        return cls(epsg, size, resolution, x - half, y - half, x + half, y + half)

    @property
    def extent_km(self) -> float:
        return self.size * self.resolution / 1000


def _window(src, grid: Grid):
    return from_bounds(grid.left, grid.bottom, grid.right, grid.top, transform=src.transform)


def read_scl(href: str, grid: Grid) -> np.ndarray:
    """Read the scene classification layer at its native 20 m over the chip window.

    Read at native resolution rather than the chip's 10 m: the cloud screen only
    needs class proportions, and this is a quarter of the bytes.
    """
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        if src.crs.to_epsg() != grid.epsg:
            raise ValueError(f"scene CRS {src.crs.to_epsg()} != chip CRS {grid.epsg}")
        native = int(round(grid.size * grid.resolution / abs(src.transform.a)))
        return src.read(
            1,
            window=_window(src, grid),
            out_shape=(native, native),
            resampling=Resampling.nearest,
            boundless=True,
            fill_value=0,
        )


def clear_fraction(scl: np.ndarray) -> float:
    """Fraction of the chip that is usable imagery.

    Dense smoke is routinely classified as cloud by SCL. That is accepted: a
    chip a quarter obscured by smoke is not labelable either way.
    """
    return float(np.isin(scl, SCL_UNUSABLE, invert=True).mean())


def read_reflectance(
    href: str, grid: Grid, scale: float, offset: float, out_size: int | None = None
) -> np.ndarray:
    """Read one band over the chip window, resampled to `out_size` (default: the chip grid).

    B11/B12 are natively 20 m and get bilinearly upsampled to the 10 m grid.
    That invents no real SWIR detail; it exists so all three channels share one
    array shape.

    `out_size` exists for the annotation pass, which works at the 20 m native
    resolution of B8A/B11/B12 rather than inventing 10 m detail it would then
    have to threshold.
    """
    size = grid.size if out_size is None else out_size
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        if src.crs.to_epsg() != grid.epsg:
            raise ValueError(f"scene CRS {src.crs.to_epsg()} != chip CRS {grid.epsg}")
        raw = src.read(
            1,
            window=_window(src, grid),
            out_shape=(size, size),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        ).astype(np.float32)
    # DN 0 is the nodata sentinel; the affine transform would map it to a
    # negative reflectance, so pin it to zero instead of letting it stretch.
    nodata = raw == 0
    out = raw * scale + offset
    out[nodata] = 0.0
    return out


def _stretch(band: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((band - lo) / (hi - lo), 0.0, 1.0)


def render(red: np.ndarray, swir16: np.ndarray, swir22: np.ndarray) -> np.ndarray:
    """Compose the SWIR fire composite: R=B12, G=B11, B=B4, as 8-bit RGB.

    Active flame saturates R and G, fresh burn scar reads deep red-brown,
    healthy vegetation green, water and shadow near black.
    """
    channels = [
        _stretch(swir22, *STRETCH["swir22"]),
        _stretch(swir16, *STRETCH["swir16"]),
        _stretch(red, *STRETCH["red"]),
    ]
    return (np.stack(channels, axis=-1) * 255.0).round().astype(np.uint8)
