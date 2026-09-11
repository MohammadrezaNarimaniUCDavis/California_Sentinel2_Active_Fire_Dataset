# Reproducibility

## Environment

```bash
python -m venv .venv
pip install -r requirements.txt
```

Training additionally needs `torch`, `torchvision`, and (for metrics helpers) a recent NumPy/Pillow stack. GPU is recommended for the 40-epoch baseline.

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

Expected reference operating point (validation-swept threshold ≈ 0.99):

- Val fire IoU ≈ 0.876 (best epoch ~35)
- Test fire IoU ≈ 0.837 · precision ≈ 0.897 · recall ≈ 0.926

## Analyst review

Protocol: `label_batch/README.md`  
Hand masks: `label_batch/hand_masks/` (233 chips; also on Zenodo under `analyst_review/`)

```bash
python import_labels.py --help
```

## Figures

Manuscript figures in `figures/` correspond to the Data in Brief draft in `manuscript/`.
