# California Sentinel-2 Active-Fire Segmentation Dataset

**Leak-free, SWIR-labelled image–mask chips for active flaming combustion across 25 California wildfires.**

| | |
|---|---|
| Chips | **2,148** (512×512 px @ 20 m) |
| Fires | **25** (2020–2026) |
| Split | **18 / 3 / 4** train / val / test **by fire** |
| Baseline | ResNet-34 U-Net · test fire IoU **0.837** |
| Data | [Zenodo DOI 10.5281/zenodo.22713948](https://doi.org/10.5281/zenodo.22713948) |

Authors: **Shreyan Mitra**¹, **Mohammadreza Narimani**²\*, **Parastoo Farajpoor**²  
¹ California High School, San Ramon, CA 94583, USA  
² Department of Biological and Agricultural Engineering, University of California, Davis, Davis, CA 95616, USA  
\* Corresponding author

---

## What this is

Per-pixel **active flame** labels (not burn scar, not smoke) on Sentinel-2 false-colour chips:

**R = B12 (SWIR2) · G = B11 (SWIR1) · B = B8A (NIR)**

Flame *emits* near 2.2 µm, so B12 rises far above what passive surfaces reflect while B8A stays dark. Masks are produced by a modified HOTMAP SWIR rule (`src/indices.py`), with an independent analyst review of 54% of the test split.

## Repository contents

```
src/                     core library (catalog, fires, chips, SWIR rule)
build_fire_dataset.py    rebuild the corpus from public STAC + NIFC
kaggle_fire_train.py     leak-free split, train, evaluate, predict
make_label_batch.py      export test chips for Label Studio
import_labels.py         import analyst masks + score agreement
make_relabel_batch.py    optional larger-canvas relabel export
label_batch/             labelling protocol + 233 hand masks
figures/                 manuscript Figs. 1–9 (PNG/PDF)
docs/                    data dictionary + reproducibility notes
```

The full image/mask corpus (~0.9 GB) is **not** stored in git. Download it from Zenodo (link above) or rebuild with `build_fire_dataset.py`.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt

# Point training at the Zenodo extract (images/, masks/, summary.csv):
# place them under ./active_fire_dataset/

python kaggle_fire_train.py --no-train   # print split stats (no GPU)
python kaggle_fire_train.py --epochs 40  # full baseline (GPU recommended)
```

On Kaggle, paste `kaggle_fire_train.py` into a GPU notebook with Internet enabled (ImageNet weights) and run `main(['--epochs', '40'])`.

## Design choices (short)

- **Fire holdout, not random chips** — dates of one fire share terrain/fuel/weather.
- **Fire-class IoU only** — positives are ~0.08% of pixels; accuracy is misleading.
- **No-data ≠ background** — outside the footprint is class 255, not “not burning”.
- **Fixed render stretch** — no per-image auto-level (would erase emission contrast).

## Citation

```
Mitra, S., Narimani, M., & Farajpoor, P. (2026). California Sentinel-2 Active-Fire
Segmentation Dataset (v1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.22713948
```

Also see `CITATION.cff`.

## License

- **Code:** MIT (`LICENSE`)
- **Data:** CC BY 4.0 (see Zenodo record). Sentinel-2 via Copernicus open data; perimeters via NIFC WFIGS.
