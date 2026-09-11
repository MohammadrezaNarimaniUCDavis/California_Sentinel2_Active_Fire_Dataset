"""Sentinel-2 scene discovery and the date-selection policy."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable

from pystac_client import Client

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ("red", "swir16", "swir22")

# Bands the annotation pass needs on top of the render bands. B8A (NIR narrow,
# 20 m) is the numerator of NBR and the third test in the Murphy hotspot rule.
# It is fetched opportunistically: requiring it would change which scenes pass
# `search`, and the 100 chips already on disk have to stay reproducible.
EXTRA_BANDS = ("nir08",)


@dataclass
class Scene:
    """One Sentinel-2 acquisition, reduced to what the pipeline needs."""

    item_id: str
    date: dt.date
    cloud: float
    hrefs: dict[str, str]
    scales: dict[str, tuple[float, float]]
    epsg: int | None = None
    clear: float = field(default=0.0)

    @property
    def scl_href(self) -> str:
        return self.hrefs["scl"]


def _band_scale(asset: dict) -> tuple[float, float]:
    """Reflectance scale/offset for an asset, from its STAC raster:bands entry.

    Sentinel-2 processing baseline 04.00+ carries a -1000 DN offset. Reading it
    from the item rather than hardcoding it keeps older scenes correct too.
    """
    bands = asset.get("raster:bands") or [{}]
    return float(bands[0].get("scale", 0.0001)), float(bands[0].get("offset", 0.0))


def search(
    lon: float,
    lat: float,
    start: dt.date,
    end: dt.date,
    max_cloud: float = 60.0,
) -> list[Scene]:
    """Return one Scene per acquisition date, cloudiest duplicates dropped.

    Scene-level cloud cover is a weak filter -- an S2 tile is 110 km across, so
    it says little about a 20 km chip. It only exists here to avoid spending
    network reads on hopeless scenes; the real screen is per-chip, on SCL.
    """
    client = Client.open(EARTH_SEARCH)
    items = client.search(
        collections=[COLLECTION],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        query={"eo:cloud_cover": {"lte": max_cloud}},
    ).item_collection()

    best: dict[dt.date, Scene] = {}
    for item in items:
        assets = item.assets
        if not all(b in assets for b in (*BANDS, "scl")):
            continue
        date = item.datetime.date()
        cloud = float(item.properties.get("eo:cloud_cover", 100.0))
        if date in best and best[date].cloud <= cloud:
            continue
        # A scene's own UTM zone. Sentinel-2 tiles either side of a zone
        # boundary are delivered in different projections, so a fire sitting on
        # one gets scenes in both. Carrying the CRS lets the caller build a grid
        # that matches rather than discarding the scene.
        code = item.properties.get("proj:code") or item.properties.get("proj:epsg")
        epsg = int(str(code).rsplit(":", 1)[-1]) if code else None
        best[date] = Scene(
            item_id=item.id,
            date=date,
            cloud=cloud,
            epsg=epsg,
            hrefs={
                k: assets[k].href
                for k in (*BANDS, "scl", *EXTRA_BANDS)
                if k in assets
            },
            scales={
                b: _band_scale(assets[b].to_dict())
                for b in (*BANDS, *EXTRA_BANDS)
                if b in assets
            },
        )
    return sorted(best.values(), key=lambda s: s.date)


def select(
    scenes: Iterable[Scene],
    count: int,
    priority_months: tuple[str, ...],
) -> list[Scene]:
    """Choose `count` scenes: priority months first, then max temporal spread.

    Greedy maximum spread repeatedly takes whichever remaining date is farthest
    in time from everything already chosen. That gives even coverage of the
    whole timeline instead of clustering wherever the weather happened to be
    good.
    """
    pool = sorted(scenes, key=lambda s: s.date)
    chosen = [s for s in pool if s.date.strftime("%Y-%m") in priority_months]
    taken = {s.item_id for s in chosen}

    if len(chosen) > count:
        # More priority scenes than slots: spread within them rather than
        # truncating and losing the tail of the burn window.
        pool, chosen = chosen, [chosen[0]]
        taken = {chosen[0].item_id}

    remaining = [s for s in pool if s.item_id not in taken]
    while len(chosen) < count and remaining:
        far = max(remaining, key=lambda s: min(abs((s.date - c.date).days) for c in chosen))
        chosen.append(far)
        remaining = [s for s in remaining if s.item_id != far.item_id]

    return sorted(chosen, key=lambda s: s.date)
