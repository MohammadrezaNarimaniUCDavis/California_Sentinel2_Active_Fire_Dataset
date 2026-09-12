# Reference baseline run (reported test fire IoU 0.837)

This note documents the exact reference training run cited in the Data in Brief
manuscript (Figs. 6–8; Table 4). It does **not** retrain the model.

## Operating point

| Item | Value |
|------|-------|
| Checkpoint epoch | 35 |
| Validation fire IoU | 0.876 |
| Selected threshold | 0.99 |
| Test fire IoU / P / R | 0.837 / 0.897 / 0.926 |
| Seed | 0 |
| Epochs / batch size | 40 / 8 |
| Encoder | ResNet-34 |

Validation history for this run is archived at
[`docs/reference_run/history.csv`](reference_run/history.csv).

## Backend actually used

Inspection of the saved checkpoint
`kaggle_notebooks/wildfire-training-ml-output/best_fire_unet.pt` shows weight
names `stem.*`, `l1.*`–`l4.*`, `d4.*`–`d1.*` — the **torchvision ResNetUNet
fallback**, not `segmentation_models_pytorch`.

ImageNet initialization was **requested** (`encoder_weights` / torchvision
`IMAGENET1K_V1` path) on a Kaggle kernel with Internet enabled. The reported
run therefore used torchvision ImageNet-pretrained ResNet-34 weights when the
download succeeded (as intended by the public script defaults).

The training script still *prefers* `segmentation_models_pytorch` when that
package is available. Installing smp can change the architecture path and is
**not** guaranteed to reproduce these exact scores. For a like-for-like
reproduction, use the torchvision path (or ensure smp is absent) and the
settings below.

## Hardware and host environment

| Item | Value |
|------|-------|
| Host | Kaggle Notebook GPU |
| GPU | Nvidia Tesla T4 (`machine_shape` in kernel metadata) |
| Internet | enabled (ImageNet weights) |
| Docker image digest | `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` |
| Kernel | `shreyanmitra5/wildfire-training-ml` |

Exact `torch` / `torchvision` / CUDA package versions from that Kaggle image
were not written to a freeze file at the time of the run. Future runs log
`torch.__version__`, `torchvision.__version__`, CUDA build, GPU name, and the
chosen backend at startup (`kaggle_fire_train.py`).

## How to re-run

```bash
python kaggle_fire_train.py --epochs 40 --seed 0
# On Kaggle: main(['--epochs', '40']) with Internet + GPU enabled
```

Place the Zenodo extract under `./active_fire_dataset/` (or the Kaggle input
path used in the notebook). Do not expect bit-identical floating-point IoU
across hosts; the published numbers are those of the archived checkpoint /
history above.
