"""Discover many fires and their active-burn windows from NIFC WFIGS.

Fetches the N largest fires in a state and period, with the dates that bound
each one's burning, because that window is what makes an active-fire dataset
possible at all.

The Palisades chips contained 165 pixels of flame across 100 frames -- Sentinel-2
revisits every ~5 days and that fire was out in three weeks, so it was caught
burning twice. The fires here burned for 20 to 100+ days each, so each one
contributes many passes with a live front.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

WFIGS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)

# Fires below this are not worth a download: too small to be caught mid-burn by a
# 5-day revisit, and their perimeters are often single-pass sketches.
MIN_ACRES = 20_000

# Cap on the burn window. Containment dates run long -- one fire here reports 363
# days -- and the flaming front is over well before then. 60 days past discovery
# is generous for even a large campaign fire.
MAX_BURN_DAYS = 60

# Fallback when containment is missing (it often is for older records).
ASSUMED_BURN_DAYS = 45


@dataclass
class Fire:
    """One fire, reduced to what the chip pipeline needs."""

    name: str
    acres: float
    discovered: dt.date
    contained: dt.date | None
    west: float
    south: float
    east: float
    north: float
    geometry: dict | None = field(default=None, repr=False)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.west + self.east) / 2, (self.south + self.north) / 2)

    @property
    def span_km(self) -> tuple[float, float]:
        lat = (self.south + self.north) / 2
        km_lon = 111.32 * abs(math.cos(math.radians(lat)))
        return ((self.east - self.west) * km_lon, (self.north - self.south) * 110.57)

    @property
    def burn_window(self) -> tuple[dt.date, dt.date]:
        """Dates within which a live flame front is plausible."""
        end = self.contained or (self.discovered + dt.timedelta(days=ASSUMED_BURN_DAYS))
        end = min(end, self.discovered + dt.timedelta(days=MAX_BURN_DAYS))
        return self.discovered, max(end, self.discovered + dt.timedelta(days=7))

    @property
    def utm_epsg(self) -> int:
        """UTM zone for this fire. California straddles 10N and 11N, so a chip
        grid fixed to one zone would force a reprojection on half the state."""
        lon, lat = self.center
        zone = int((lon + 180) / 6) + 1
        return (32600 if lat >= 0 else 32700) + zone

    @property
    def slug(self) -> str:
        return "".join(c if c.isalnum() else "_" for c in self.name.lower()).strip("_")


def _flatten(coords) -> list[list[float]]:
    if coords and isinstance(coords[0], (int, float)):
        return [coords]
    out: list[list[float]] = []
    for part in coords:
        out.extend(_flatten(part))
    return out


def _as_date(millis) -> dt.date | None:
    if not millis:
        return None
    return dt.datetime.fromtimestamp(millis / 1000, dt.timezone.utc).date()


def fetch(
    count: int = 25,
    state: str = "US-CA",
    since: dt.date = dt.date(2019, 1, 1),
    min_acres: float = MIN_ACRES,
    cache: Path | None = None,
    timeout: int = 120,
) -> list[Fire]:
    """The `count` largest fires matching the filters, geometry included.

    Sorted by acreage so the download budget goes to the fires most likely to be
    caught burning. Duplicate names are dropped, keeping the largest polygon:
    WFIGS accumulates progressive perimeters and the final one is the full extent.
    """
    if cache is not None and cache.exists():
        payload = json.loads(cache.read_text())
    else:
        query = urllib.parse.urlencode(
            {
                "where": (
                    f"attr_POOState='{state}' AND poly_GISAcres>{min_acres} AND "
                    f"attr_FireDiscoveryDateTime>DATE '{since.isoformat()}'"
                ),
                "outFields": (
                    "poly_IncidentName,attr_FireDiscoveryDateTime,poly_GISAcres,"
                    "attr_ContainmentDateTime"
                ),
                "returnGeometry": "true",
                "outSR": "4326",
                "orderByFields": "poly_GISAcres DESC",
                "resultRecordCount": max(count * 3, 60),
                "f": "geojson",
            }
        )
        with urllib.request.urlopen(f"{WFIGS_URL}?{query}", timeout=timeout) as response:
            payload = json.loads(response.read())
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload))

    best: dict[str, Fire] = {}
    for feature in payload.get("features") or []:
        props = feature["properties"]
        name = props.get("poly_IncidentName")
        acres = props.get("poly_GISAcres") or 0.0
        discovered = _as_date(props.get("attr_FireDiscoveryDateTime"))
        if not name or not discovered or not feature.get("geometry"):
            continue
        if name in best and best[name].acres >= acres:
            continue
        points = _flatten(feature["geometry"]["coordinates"])
        lons = [p[0] for p in points]
        lats = [p[1] for p in points]
        best[name] = Fire(
            name=name,
            acres=acres,
            discovered=discovered,
            contained=_as_date(props.get("attr_ContainmentDateTime")),
            west=min(lons),
            south=min(lats),
            east=max(lons),
            north=max(lats),
            geometry=feature["geometry"],
        )

    return sorted(best.values(), key=lambda f: -f.acres)[:count]
