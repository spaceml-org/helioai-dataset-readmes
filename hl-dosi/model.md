# 1. Model Description

`hl-dosi` delivers LSTM-based recurrent models that forecast the deep-space radiation environment during Solar Proton Events (SPEs). Given a multi-hour context window of solar drivers — GOES-16 soft X-ray flux, BioSentinel BPD dose rate, and (optionally) SDO/AIA + HMI imagery — the model autoregressively predicts the BPD dose-rate trajectory over the following hours, with per-step uncertainty estimated via Monte Carlo dropout.

The published checkpoint corresponds to the `RadRecurrentWithSDO` architecture, which combines:

1. A CNN **SDO embedding** (the `SDOEmbedding` module) that compresses each 6-channel 512×512 SDOML-lite frame into a 1024-dimensional vector.
2. A **context LSTM** that ingests the concatenation of the SDO embedding and the in-situ time-series (XRS + BPD) over a history window.
3. A **prediction LSTM** that, conditioned on the context LSTM's hidden state, autoregressively generates future BPD dose-rate samples.

Dropout is left active at inference time so that repeated forward passes produce a sample-based uncertainty band rather than a single point forecast.

In addition to the high-level summary provided below, the full source code used to train and evaluate the model is contained in the project [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-hl-radiation-ml) (`scripts/models.py`, `scripts/run.py`).

## 1.1 RadRecurrentWithSDO — flagship checkpoint (1.4 GB)
- **AWS PATH**: `s3://nasa-radiant-data/helioai-datasets/hl-dosi/models/epoch-002-model-20240905.pth`
- **Type**: PyTorch state dict including model hyperparameters (`model_data_dim`, `model_sdo_dim`, `model_hidden_dim`, etc.) alongside the weights. Loadable with `torch.load(..., map_location=...)` followed by `RadRecurrentWithSDO(**meta).load_state_dict(state)`.
- **Input**:
    - Context window: a sequence of `(sdo_embedding, [xrs, bpd])` tuples at 15-minute cadence over the history window. SDO frames are 6×512×512 (AIA 131/171/193/304 + HMI Bx/By); time-series are scaled to the training normalization shipped in `scripts/datasets.py`.
    - Prediction horizon length (number of 15-minute steps to roll out).
- **Output**: a tensor of predicted BPD dose-rate samples over the horizon; with MC-dropout enabled, repeated calls to `model.predict(...)` form an empirical predictive distribution.
- **Training data**: SDOML-lite-biosentinel + GOES-16 XRS + RadLab BPD, 2022-11 to 2024-05. SPE events held out for testing include `biosentinel01` (58 pfu), `biosentinel07` (620 pfu), `biosentinel19` (116 pfu); "seen" test events include `biosentinel04`, `biosentinel15`, `biosentinel18`.
- This is the checkpoint used by the inference demo notebook in §1.3.

## 1.2 RadRecurrentWithSDO — `elena-bpd` alternative (1.4 GB)
- **AWS PATH**: `s3://nasa-radiant-data/helioai-datasets/hl-dosi/models/epoch-009-model-elena-bpd.pth`
- **Type**: same architectural family as §1.1, but a separately trained checkpoint. Note that the metadata keys differ (`model_data_dim_context` / `model_data_dim_prediction` instead of the single `model_data_dim` used by the 20240905 checkpoint), so loading requires the corresponding branch of the `RadRecurrentWithSDO` constructor.
- Provided for comparison and reproducibility; the demo notebook does **not** use this checkpoint.

## 1.3 Inference Notebook
- Source: [`public/demo.ipynb`](https://github.com/FrontierDevelopmentLab/2024-hl-radiation-ml/blob/main/public/demo.ipynb) in the main GitHub repository.
- A self-contained Google Colab notebook that installs dependencies, clones the repo, downloads the flagship checkpoint and a single-event subset of the data (the four SDOML-lite shards covering `biosentinel07`, plus GOES CSVs and the RadLab DuckDB), runs autoregressive inference with Monte Carlo dropout, and plots the predicted BPD dose-rate trajectory against the ground truth.

Open directly in Colab: [demo.ipynb on Colab](https://colab.research.google.com/github/FrontierDevelopmentLab/2024-hl-radiation-ml/blob/main/public/demo.ipynb).

## 1.4 Reference test-event plots (~33 MB)
- **AWS PATH**: `s3://nasa-radiant-data/helioai-datasets/hl-dosi/models/event_plots/`
- Pre-rendered side-by-side animations and PDF storyboards of the flagship model's predictions on six SPE test events, produced by `scripts/event_plot.py`:

| File | Event | Peak >10 MeV | Seen during training? |
|------|-------|--------------|------------------------|
| `epoch-002-test-event-biosentinel01-58pfu-202302250615-202302280140.{mp4,pdf}` | biosentinel01 | 58 pfu | No |
| `epoch-002-test-event-biosentinel07-620pfu-202307171515-202307191215.{mp4,pdf}` | biosentinel07 (demo event) | 620 pfu | No |
| `epoch-002-test-event-biosentinel19-116pfu-202405101215-202405130310.{mp4,pdf}` | biosentinel19 | 116 pfu | No |
| `epoch-002-test-seen-event-biosentinel04-38pfu-202305071030-202305120850.{mp4,pdf}` | biosentinel04 | 38 pfu | Yes |
| `epoch-002-test-seen-event-biosentinel15-187pfu-202402082245-202402120225.{mp4,pdf}` | biosentinel15 | 187 pfu | Yes |
| `epoch-002-test-seen-event-biosentinel18-208pfu-202405100515-202405111845.{mp4,pdf}` | biosentinel18 | 208 pfu | Yes |

Direct HTTPS browsing also works, e.g. `https://nasa-radiant-data.s3.amazonaws.com/helioai-datasets/hl-dosi/models/event_plots/epoch-002-test-event-biosentinel07-620pfu-202307171515-202307191215.mp4`.

# 2. Access Instructions

Models are stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use the CLI are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). The bucket is public, so `--no-sign-request` works without an AWS account; checkpoints are also reachable over HTTPS at `https://nasa-radiant-data.s3.amazonaws.com/helioai-datasets/hl-dosi/models/<file>` (used by the Colab demo).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-dosi/models/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH>
```
You will need to replace `<AWS PATH>` with the path to the file you want to download and `<LOCAL PATH>` with the path on your local machine where you want to save it.

| Model | AWS Path | Size |
|-------|----------|------|
| RadRecurrentWithSDO (flagship, 20240905) | `s3://nasa-radiant-data/helioai-datasets/hl-dosi/models/epoch-002-model-20240905.pth` | 1.4 GB |
| RadRecurrentWithSDO (`elena-bpd`) | `s3://nasa-radiant-data/helioai-datasets/hl-dosi/models/epoch-009-model-elena-bpd.pth` | 1.4 GB |

# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *create* the model. These can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-hl-radiation-ml).
2. Requirements for *using* the model:

| Component | Minimum |
|-----------|---------|
| **CPU** | Modern multi-core CPU. CPU-only inference of the demo notebook (one event, single MC sample) runs in a few minutes. |
| **RAM** | 8 GB is sufficient for the demo notebook on Colab free tier when streaming the event-subset SDOML-lite tars; 16 GB recommended for batched MC-dropout sampling. |
| **GPU** | Not required for the single-event demo; strongly recommended for training (the published checkpoints were produced on a single NVIDIA A100). |
| **Storage** | ~1.5 GB for one checkpoint; ~5 GB including the single-event demo data subset. |
