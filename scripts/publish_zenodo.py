#!/usr/bin/env python3
"""Package and publish the active-fire dataset to Zenodo.

Reads ZENODO_TOKEN from the environment. Never commit the token.
"""

from __future__ import annotations

import json
import os
import time
import zipfile
import urllib.request
from pathlib import Path

TOKEN = os.environ.get("ZENODO_TOKEN", "").strip()
BASE = "https://zenodo.org/api"
PKG = Path(r"c:\mnarimani\1-UCDavis\9-Github\_zenodo_active_fire_package")
ZIP = Path(r"c:\mnarimani\1-UCDavis\9-Github\California_S2_Active_Fire_Dataset_v1.0.zip")
OUTDIR = Path(r"c:\mnarimani\1-UCDavis\9-Github\California_Sentinel2_Active_Fire_Dataset")


def api(method: str, url: str, data=None):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = resp.read()
        return json.loads(payload.decode("utf-8")) if payload else {}


def main() -> None:
    if not TOKEN:
        raise SystemExit("Set ZENODO_TOKEN in the environment.")

    print("Creating zip...")
    t0 = time.time()
    if ZIP.exists():
        ZIP.unlink()
    files = [p for p in PKG.rglob("*") if p.is_file()]
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for i, p in enumerate(files, 1):
            arc = p.relative_to(PKG).as_posix()
            zf.write(p, arcname=f"California_S2_Active_Fire_Dataset_v1.0/{arc}")
            if i % 500 == 0 or i == len(files):
                print(f"  zipped {i}/{len(files)}")
    print(f"zip done in {time.time() - t0:.0f}s size={ZIP.stat().st_size / 1e6:.1f} MB")

    print("Creating deposition...")
    dep = api("POST", f"{BASE}/deposit/depositions", data={})
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print("deposition", dep_id)

    metadata = {
        "metadata": {
            "title": "California Sentinel-2 Active-Fire Segmentation Dataset",
            "upload_type": "dataset",
            "description": (
                "<p>Per-pixel active-flame image–mask chips from Sentinel-2 for 25 California wildfires "
                "(2,148 chips, 512×512 at 20 m). Masks from a modified HOTMAP SWIR rule; leak-free fire holdout "
                "(18/3/4 train/val/test). Includes <code>summary.csv</code>, <code>partitions.csv</code>, and 233 "
                "analyst review masks (54% of the test split).</p>"
                "<p>Companion code and manuscript figures: "
                "https://github.com/MohammadrezaNarimaniUCDavis/California_Sentinel2_Active_Fire_Dataset</p>"
            ),
            "creators": [
                {
                    "name": "Mitra, Shreyan",
                    "affiliation": "California High School, San Ramon, CA 94583, USA",
                },
                {
                    "name": "Narimani, Mohammadreza",
                    "affiliation": (
                        "Department of Biological and Agricultural Engineering, "
                        "University of California, Davis, Davis, CA 95616, USA"
                    ),
                },
                {
                    "name": "Farajpoor, Parastoo",
                    "affiliation": (
                        "Department of Biological and Agricultural Engineering, "
                        "University of California, Davis, Davis, CA 95616, USA"
                    ),
                },
            ],
            "keywords": [
                "active fire",
                "Sentinel-2",
                "SWIR",
                "semantic segmentation",
                "weak supervision",
                "California wildfire",
                "remote sensing",
            ],
            "license": "cc-by-4.0",
            "access_right": "open",
            "related_identifiers": [
                {
                    "identifier": (
                        "https://github.com/MohammadrezaNarimaniUCDavis/"
                        "California_Sentinel2_Active_Fire_Dataset"
                    ),
                    "relation": "isSupplementTo",
                    "resource_type": "software",
                    "scheme": "url",
                }
            ],
            "version": "1.0.0",
        }
    }
    api("PUT", f"{BASE}/deposit/depositions/{dep_id}", data=metadata)
    print("metadata set")

    print("Uploading zip...")
    t0 = time.time()
    url = f"{bucket}/{ZIP.name}"
    with ZIP.open("rb") as fh:
        data = fh.read()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/octet-stream",
        },
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        up = json.loads(resp.read().decode("utf-8"))
    print(f"upload ok in {time.time() - t0:.0f}s size={up.get('size')}")

    print("Publishing...")
    pub = api("POST", f"{BASE}/deposit/depositions/{dep_id}/actions/publish")
    doi = pub.get("doi")
    conceptdoi = pub.get("conceptdoi")
    html = pub.get("links", {}).get("html") or pub.get("links", {}).get("record_html")
    print("PUBLISHED", doi, html)
    info = {
        "doi": doi,
        "conceptdoi": conceptdoi,
        "url": html,
        "doi_url": f"https://doi.org/{doi}" if doi else None,
        "deposition_id": dep_id,
    }
    OUTDIR.joinpath("docs").mkdir(parents=True, exist_ok=True)
    OUTDIR.joinpath("docs", "ZENODO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("saved docs/ZENODO.json")


if __name__ == "__main__":
    main()
