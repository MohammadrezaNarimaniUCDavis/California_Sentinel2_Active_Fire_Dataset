# Data dictionary

## `summary.csv` (one row per chip)

| Column | Meaning |
|--------|---------|
| `file_name` | PNG basename shared by `images/` and `masks/` |
| `fire` | Incident name |
| `date` | Acquisition date (YYYY-MM-DD) |
| `day_of_burn` | Days since discovery (integer) |
| `fire_px` | Count of mask pixels with value 1 |
| `nodata_pct` | Percent of pixels with value 255 |

## Mask encoding

| Value | Class |
|------:|-------|
| 0 | Background |
| 1 | Active fire |
| 255 | No-data (outside footprint) |

## Image encoding

RGB PNG, 512×512, 20 m/pixel.

| Channel | Band | Role |
|---------|------|------|
| R | B12 (SWIR2, 2.19 µm) | Flame emission |
| G | B11 (SWIR1, 1.61 µm) | Context |
| B | B8A (NIR, 0.865 µm) | Contrast vs flame |

Fixed dataset-wide linear stretch (upper reflectance caps 0.60 / 0.50 / 0.45).

## Leak-free split (by fire name)

| Split | Fires |
|-------|-------|
| Test | Slater, Hopkins, CALDWELL, Windy |
| Validation | Castle, McCash, Bobcat |
| Train | all remaining fires in the deposit |

Assigned in `kaggle_fire_train.py` (`TEST_FIRES`, `VAL_FIRES`). Also listed in Zenodo `partitions.csv`.
