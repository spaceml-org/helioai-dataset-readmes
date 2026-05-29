# 1 Model Description

SPI3S delivers a coupled, physics-informed machine-learning framework designed to estimate the Sun's full EUV spectral irradiance from arbitrary heliospheric viewpoints, with Mars serving as the primary demonstration target.

The challenge addresses a fundamental limitation of traditional solar irradiance measurements: spacecraft observe the Sun from only a small number of locations, while the solar atmosphere is inherently three-dimensional and continuously evolving. As a result, irradiance measured at Earth may differ from that experienced elsewhere in the solar system.

To address this problem, SPI3S combines two complementary AI models:

1. **SuNeRF-DT (Solar Neural Radiance Field – Density & Temperature)**, which reconstructs a time-evolving 3D representation of the solar corona from multi-viewpoint EUV observations.
2. **MEGS-AI (KANDEM-Spectrum)**, which estimates solar spectral irradiance directly from EUV images.

Together, these models form an end-to-end pipeline capable of:

* reconstructing the 4D structure of the Sun,
* generating synthetic observations from arbitrary viewpoints,
* estimating full EUV spectral irradiance from those viewpoints,
* validating predictions against independent spacecraft observations.

The resulting workflow enables, for the first time, estimation of solar spectral irradiance from the perspective of Mars using only solar observations acquired elsewhere in the heliosphere.

**The following resources are available for in-depth dataset descriptions and usage instructions:**

* [Technical Memorandum](https://helioai.org/artifact/f467e243-a299-43b9-871e-8c5c519663a5/details) - full description and results of the FDL Heliolab challenge that produced the model
* [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-SPI3S/) - source code repository with data download scripts  and model files
* [Colab Notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2024-HL-SPI3S/blob/main/public/inference_demo.ipynb) - runs inference on both Earth-view and Mars-view AIA stacks, and plots the predicted EUV spectra (see section 1.8 for more details)


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

## 1.9 Model Training and Evaluation

SPI3S combines two complementary machine-learning models operating within a single end-to-end workflow. SuNeRF-DT reconstructs a four-dimensional (3D + time) representation of the solar corona from multi-viewpoint observations, while MEGS-AI estimates solar spectral irradiance from solar EUV imagery. Together, these models enable estimation of solar spectral irradiance from arbitrary heliospheric viewpoints, including Mars.

### Training strategy

**SuNeRF-DT** is trained using simultaneous multi-viewpoint EUV observations acquired from instruments including SDO/AIA and STEREO/EUVI. Rather than directly reconstructing image intensities, the model learns a continuous representation of the solar corona parameterized by electron density and plasma temperature.

Training incorporates:

- multi-viewpoint EUV observations
- multi-wavelength temperature response functions
- observer viewing geometry
- radiative transfer constraints

The resulting model learns a time-dependent representation of the solar atmosphere that can be queried from arbitrary viewpoints and times within the training interval.

**MEGS-AI** is trained as a supervised image-to-spectrum regression model. Solar EUV images from SDO/AIA are temporally aligned with irradiance measurements from SDO/EVE, allowing the model to learn the relationship between solar image morphology and the corresponding spectral irradiance.

Training inputs include:

- seven-channel EUV image stacks from SDO/AIA
- normalized image representations

Training targets include:

- MEGS-A spectral irradiance
- MEGS-B spectral irradiance
- individual emission-line intensities
- full EUV spectral irradiance distributions

### Evaluation methodology

Evaluation is performed at three levels corresponding to the major components of the SPI3S workflow.

#### SuNeRF-DT validation

Because direct measurements of the Sun's true 4D electron density and plasma temperature structure are not available, SuNeRF-DT is validated against state-of-the-art magnetohydrodynamic (MHD) simulations provided by Predictive Science Inc. (PSI).

Evaluation assesses:

- recovery of coronal density structure
- recovery of plasma temperature distributions
- consistency of synthesized viewpoints
- agreement with simulated ground-truth coronal features

Model-generated views are compared against corresponding views rendered directly from the MHD simulations, enabling qualitative and quantitative assessment of reconstruction fidelity.

#### MEGS-AI validation

MEGS-AI is evaluated using held-out SDO/AIA observations that were not used during training.

Evaluation assesses:

- spectral reconstruction accuracy
- irradiance estimation fidelity
- generalization to unseen solar observations
- performance across the full EUV spectrum

Predicted irradiance spectra are compared directly against corresponding SDO/EVE measurements to evaluate both spectral shape and absolute intensity.

#### End-to-end SPI3S validation

The complete SPI3S pipeline is evaluated by generating synthetic views of the Sun from the perspective of Mars using SuNeRF-DT and then estimating the corresponding irradiance with MEGS-AI.

Evaluation assesses:

- consistency of irradiance predictions from novel viewpoints
- generalization beyond Earth-based observations
- physical realism of synthesized solar imagery
- agreement with independent Mars-based observations

Final irradiance estimates are compared against measurements from MAVEN/EUVM and reference products such as FISM-P, providing a unique cross-planet validation of the complete workflow.

### Scientific evaluation

A key objective of SPI3S is to determine whether a learned 4D representation of the Sun can provide physically meaningful estimates of solar spectral irradiance throughout the heliosphere.

Scientific evaluation therefore focuses on:

- reconstruction of the Sun's evolving 3D coronal structure
- estimation of electron density and plasma temperature
- generation of physically plausible solar views from arbitrary locations
- prediction of full EUV spectral irradiance beyond Earth orbit
- validation of irradiance estimates against independent spacecraft observations

Together, these evaluations assess both the machine-learning performance and the scientific validity of the end-to-end SPI3S framework.


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


<!-- BACKUP
# 1 Access Instructions

Models are is stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/<DATASET_NAME>/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the file or directory you want to download (see below) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

<!-- Add/remove rows as necessary for your project
The ideal case is that within each of these categories, data are uniformly structured.
For example, "processed" may correspond to train/test/validation data, in which we expect a tabular format (consistent column names, different rows) for each training example. 
Different models may have different train/test/validation sets, this can be explained -->


# 2 Model Description

<!-- Add a brief description of the model and the challenge it addresses -->

There are three levels of description available for this model:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](<https://drive.google.com/file/d/1hsxBvVRLFBoHWV8CKWkSo8HYE1TNG6re/view>)).
- The full source code used to process the data and create the models (see the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2024-HL-SPI3S-SuNeRF>)).

The SPI3S challenge produced a coupled, physics-informed + deep-learning modeling system designed to estimate solar spectral irradiance from arbitrary heliospheric viewpoints, specifically targeting Mars as a use case. The architecture combines two primary models:

1. SuNeRF (Solar Neural Radiance Field) - 3D solar reconstruction model
2. MEGS-AI (spectral irradiance model) - spectral prediction model

These are integrated into an end-to-end pipeline capable of mapping multi-viewpoint solar observations → 3D solar structure → spectral irradiance at arbitrary locations.

## 2.1 SuNeRF (Solar Neural Radiance Field)

<!-- Describe the ML models included. For each model, include:
     - Model architecture
     - Purpose (nowcasting, forecasting, classification, etc.)
     - Any caveats on intended use -->

<!-- If applicable, link to inference notebooks or usage examples -->

SuNeRF is a Neural Radiance Field (NeRF)-based model adapted to solar physics, designed to reconstruct the 3D structure of the solar corona and atmosphere from multi-viewpoint observations.

Input:
    - Multi-viewpoint EUV solar images (e.g., SDO + STEREO)
    - Viewing geometry / spacecraft positions
Output:
    - Continuous 3D representation of the solar corona
    - Physical proxies such as:
        - electron density
        - temperature structure

Key characteristics:

- Physics-informed constraints embedded in the model
- Time-dependent (dynamic solar evolution)
- Enables virtual viewpoints anywhere in the heliosphere


## 2.2 MEGS-AI (spectral irradiance model)

MEGS-AI is a deep learning model trained to estimate full solar spectral irradiance from solar observations.

Input:
    - Solar image data (real or synthesized views from SuNeRF)
Output:
    - Spectral irradiance across EUV wavelength bands
    - Virtual replacement for:
         SDO/EVE MEGS-A instrument

Key role in SPI3S:
- Converts geometric / image-based representation to physically meaningful irradiance spectrum
- Extends Virtual EVE concept to off-Earth viewpoints


## 2.3 Coupled SuNeRF + MEGS-AI pipeline

The core innovation of SPI3S is the integration of these two models:

1. SuNeRF reconstructs the 3D Sun
2. Synthetic solar images are generated at a target location (e.g., Mars)
3. MEGS-AI converts those images into spectral irradiance


Multi-viewpoint solar images → SuNeRF → 3D Sun
                                   ↓
                         Virtual viewpoint (Mars)
                                   ↓
                             Synthetic images
                                   ↓
                             MEGS-AI
                                   ↓
                     Spectral irradiance at Mars

                     
## 2.4 Model Output

The SPI3S challenge produced scientific data products demonstrating 3D irradiance modeling capability beyond Earth orbit.

1. Spectral irradiance at Mars (primary output)
     Predicted solar spectral irradiance at:
  	  - Mars orbital position
     Covers:
  	  - EUV wavelength bands
     Time-dependent outputs:
 	   - evolving with solar activity

2. Synthetic solar images from arbitrary viewpoints
     Generated from SuNeRF:
         - virtual images of the Sun
         - from Mars perspective (or other locations)

3. 3D solar atmosphere reconstructions
     - Continuous volumetric representation of the corona
     - Derived physical quantities:
         - density
         - temperature proxies

4. Time-resolved irradiance sequences
 Irradiance predictions as a function of:
    - time
    - solar evolution


# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *create* the model. These can be found in the [GitHub Repository](<LINK_TO_GITHUB_REPO>).
2. Requirements for *using* the model.


| Component | Minimum |
|-----------|---------|
| **CPU** | |
| **RAM** | |
| **GPU** | |
| **Storage** | |
-->
