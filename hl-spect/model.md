# 1 Model Description

<!--There are three levels of description available for this model:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](https://helioai.org/dev/artifact/f467e243-a299-43b9-871e-8c5c519663a5/details)).
- The full source code used to process the data and create the models (see the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-SPI3S/)).-->

SPI3S delivers a coupled, two-stage modeling system that estimates the Sun's full EUV spectral irradiance at arbitrary heliospheric viewpoints — motivated by the question *"what would MEGS-A measure from Mars?"*

1. **SuNeRF-DT (Solar Neural Radiance Field, Density/Temperature):** reconstructs a time-evolving 3D model of the solar corona (electron density and plasma temperature) from multi-viewpoint EUV observations (SDO/AIA + STEREO/EUVI). Once trained, it can synthesize AIA-like images of the Sun as seen from any heliospheric location, including Mars.
2. **MEGS-AI (KANDEM-Spectrum):** maps a stack of seven AIA EUV images to the full MEGS-A or MEGS-B spectral irradiance, replacing the now-degraded MEGS-A instrument with a learned "Virtual EVE" that works from any image source — including SuNeRF-synthesized Mars views.

End-to-end: multi-viewpoint solar images → SuNeRF-DT → 3D Sun → virtual Mars viewpoint → MEGS-AI → spectral irradiance at Mars, validated against MAVEN/EUVM.


## 1.1 MEGS-AI — KANDEM-Spectrum (MEGS-B, 50 MB)
- **AWS PATH**: `s3://nasa-radiant-data/helioai-datasets/hl-spect/models/megs_ai/KANDEM_baseline_MEGSB.ckpt`
- **Type**: PyTorch Lightning checkpoint. KANDEM-Spectrum architecture (Kolmogorov-Arnold-Network DEM + spectrum head).
- **Input**: stack of 7 AIA EUV images `[94, 131, 171, 193, 211, 304, 335]` at 256×256 resolution, z-score normalized with the UV statistics bundled in the checkpoint.
- **Output**: 3663-bin full MEGS-B spectral irradiance (W m⁻² nm⁻¹).
- Use this as the flagship predictor for the MEGS-B spectrum. The inference demo notebook linked in Section 1.8 uses this checkpoint.

## 1.2 MEGS-AI — KANDEM-Spectrum (MEGS-A, 24 MB)
- **AWS PATH**: `s3://nasa-radiant-data/helioai-datasets/hl-spect/models/megs_ai/kandem_noRadius_MEGSA.ckpt`
- **Type**: same architecture as §2.1, trained to output the 1377-bin MEGS-A spectrum.

## 1.3 MEGS-AI baselines (< 1 MB combined)
- **AWS PATH**: `s3://nasa-radiant-data/helioai-datasets/hl-spect/models/megs_ai/{kan_baseline_mean.ckpt, linear_baseline_7wl_nostd.ckpt}`
- **Type**: lightweight baselines used for comparison during development.
    - `kan_baseline_mean.ckpt` (473 KB): KAN baseline predicting the disk-integrated mean spectrum.
    - `linear_baseline_7wl_nostd.ckpt` (63 KB): linear regression from 7 AIA channels to spectral output.

## 1.4 SuNeRF-DT — PSI checkpoint (144 MB)
- **AWS PATH**: `s3://nasa-radiant-data/helioai-datasets/hl-spect/models/sunerf/PSI/aia_iti_512_log_psi_tnorm/`
- **Type**: PyTorch Lightning checkpoint for SuNeRF-DT trained on Predictive Science Inc. MHD synthetic images. Validates the radiative-transfer-aware architecture against known ground-truth density/temperature.
- **No inference notebook is provided for SuNeRF in this release.** SuNeRF renders Mars views offline; the pre-rendered FITS are included in the companion dataset bucket under `processed_data/mars_views/`. To retrain or render novel viewpoints, see the [SuNeRF repo](https://github.com/FrontierDevelopmentLab/2024-HL-SPI3S-SuNeRF).

## 1.5 SuNeRF-DT — AIA+STEREO Nov-2012 checkpoint (1.7 GB)
- **AWS PATH**: `s3://nasa-radiant-data/helioai-datasets/hl-spect/models/sunerf/aia_stereo_iti_2012-11_checkpoint/`
- **Type**: SuNeRF-DT trained on real multi-viewpoint AIA + STEREO-A/B observations for a November 2012 time window. This is the checkpoint used to generate the Mars-view pre-renders in `processed_data/mars_views/`.

## 1.6 Training configurations
- **AWS PATH**: `s3://nasa-radiant-data/helioai-datasets/hl-spect/models/configs/`
- Full training YAML configurations for both MEGS-AI (`megs_ai_2024_config.yaml`, `dem_2024_config.yaml`) and SuNeRF (`DT_2012_11.yaml`, `psi_193.yaml`, `emission_2012_08-193.yaml`, `render_mhd.yaml`, `sunerfs_simple_star.yaml`, `sunerfs_simple_star_2.yaml`).

## 1.7 Demo data subset (~200 MB)
- **AWS PATH**: `s3://nasa-radiant-data/helioai-datasets/hl-spect/models/data_subset/`
- A small, self-contained subset for running the inference demo notebook on Colab:
    - `earthside_stacks_7ch.npy` (175 MB): 100 pre-processed 7-channel AIA stacks, evenly sampled over 2010-05 to 2014-05.
    - `earthside_eve_14lines.npy` (5 KB): matching 14-line irradiance targets for reference.
    - `earthside_metadata.parquet` (4 KB): timestamps + stack filenames.
    - `mars_stacks_7ch.npy` (24 MB): 14 SuNeRF-rendered Mars-view stacks (2023 dates).
    - `mars_metadata.parquet` (2 KB): Mars-view dates.
    - `eve_14line_normalization.npy`, `eve_14line_wl_names.npy`: normalization parameters and line names.
    - `README.md`: detailed description of each file.

## 1.8 Inference Notebook
- Source: [`public/inference_demo.ipynb`](https://github.com/FrontierDevelopmentLab/2024-HL-SPI3S/blob/main/public/inference_demo.ipynb) in the main GitHub repository.
- A self-contained Google Colab notebook that installs dependencies, clones the `megsai` package, downloads the flagship MEGS-B checkpoint and the demo subset from S3, runs inference on both Earth-view and Mars-view AIA stacks, and plots the predicted EUV spectra.

Open directly in Colab: [inference_demo.ipynb on Colab](https://colab.research.google.com/github/FrontierDevelopmentLab/2024-HL-SPI3S/blob/main/public/inference_demo.ipynb).

# 2 Access Instructions

Models are stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-spect/models/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the file or directory you want to download (see below) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

# 3 System Requirements

There are two sets of system requirements:
1. Requirements to *create* the model. These can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-SPI3S/).
2. Requirements for *using* the model:


| Component | Minimum |
|-----------|---------|
| **CPU** | Modern multi-core CPU |
| **RAM** | 12 GB (for MEGS-AI inference at 256×256; intermediate KAN tensors are memory-heavy). Colab free tier is sufficient. |
| **GPU** | Not required but recommended for batch inference and for any SuNeRF training |
| **Storage** | 500 MB (demo subset + one checkpoint); 4 GB (all MEGS-AI checkpoints + both SuNeRF checkpoints + demo subset) |
