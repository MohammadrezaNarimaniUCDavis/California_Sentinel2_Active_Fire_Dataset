# Reproducibility

## Environment

```bash
python -m venv .venv
pip install -r requirements.txt
```

Training notes and the Kaggle docker digest for the published baseline are in
`requirements-train.txt` and `docs/REFERENCE_RUN.md`.

GPU is recommended for the 40-epoch baseline.

## Incident selection (wildfire only)

`src/fires.py` queries NIFC WFIGS with `attr_IncidentTypeCategory='WF'`
(prescribed-fire `RX` excluded). All 25 published incidents were verified as
`WF` (see `docs/incident_type_verification.csv`).

## Data

1. Download the Zenodo archive and extract so that:
   ```
   active_fire_dataset/
     images/
     masks/
     summary.csv
   ```
2. Or rebuild from public sources (hours, several GB download):
   ```bash
   python build_fire_dataset.py
   ```

## Baseline

```bash
python kaggle_fire_train.py --no-train          # verify split
python kaggle_fire_train.py --epochs 40         # train + evaluate
```

The archived reference checkpoint used the **torchvision ResNet-34 U-Net
fallback** (not `segmentation_models_pytorch`). See `docs/REFERENCE_RUN.md`.

Expected reference operating point (validation-swept threshold = 0.99):

- Val fire IoU ≈ 0.876 (best epoch 35)
- Test fire IoU ≈ 0.837 · precision ≈ 0.897 · recall ≈ 0.926

Validation history: `docs/reference_run/history.csv`.

## Analyst review

Protocol: `label_batch/README.md`  
Hand masks: `label_batch/hand_masks/` (233 chips; also on Zenodo under `analyst_review/`)

```bash
python import_labels.py --help
```

## Figures

Figures in `figures/` are the locked manuscript figures (PNG + PDF).
