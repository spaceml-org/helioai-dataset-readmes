# 1. Dataset Description

This dataset corresponds to the **Spectral Irradiance of the 3D Sun (SPI3S) on Mars** challenge from Heliolab 2024. The challenge was designed to estimate solar spectral irradiance from arbitrary viewpoints in the solar system — including Mars — by combining 3D reconstruction of the solar corona with a learned mapping from solar images to irradiance.

The scientific motivation is that solar irradiance is strongly shaped by the Sun's three-dimensional and time-evolving atmosphere, yet direct irradiance measurements are only available from a limited number of spacecraft and viewing geometries. Existing approaches can estimate irradiance at Earth or reconstruct coronal structure from a small number of observations, but they do not independently provide a complete end-to-end system for estimating the Sun's spectral output from other planetary perspectives. The SPI3S workflow was developed to address this challenge.

The dataset combines multi-viewpoint EUV observations, spectral irradiance measurements, and synthetic physics-based simulation products into a unified machine-learning pipeline. In the first stage, a SuNeRF-DT model learns a four-dimensional (3D + time) representation of the solar corona by reconstructing electron density and plasma temperature from simultaneous observations acquired from multiple spacecraft and wavelength channels. In the second stage, an updated MEGS-AI model learns to infer solar spectral irradiance directly from solar EUV imagery.

Together, these components enable the generation of synthetic solar observations from arbitrary heliospheric viewpoints, including Mars, followed by estimation of the irradiance that would be measured at that location. Predictions can then be validated against independent observations from MAVEN/EUVM.

**The following resources are available for in-depth dataset descriptions and usage instructions:**

* [Technical Memorandum](https://helioai.org/dev/artifact/f467e243-a299-43b9-871e-8c5c519663a5/details) - full description and results of the FDL Heliolab challenge that produced the dataset
* [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-SPI3S/) - source code for processing the data and using with the associated models


## 1.1 Processed Data

The processed data products support two distinct stages of the SPI3S workflow. The first stage provides multi-viewpoint solar observations and associated geometry information required to train SuNeRF-DT, which reconstructs a four-dimensional representation of the solar corona. The second stage provides paired solar-image and spectral-irradiance datasets required to train MEGS-AI, which estimates EUV irradiance directly from solar imagery. Additional Mars-view products and validation datasets enable evaluation of the complete end-to-end pipeline.

The raw data undergoes processing to create structured, ML-ready datasets. The processing pipeline includes:

- **AIA image stacking:** Nine SDO/AIA channels (EUV: 94, 131, 171, 193, 211, 304, 335 Å; UV: 1600, 1700 Å) are co-registered per timestamp and written as single-file NumPy stacks at 256×256 resolution.
- **EVE alignment:** SDO/EVE MEGS-A irradiance is resampled and paired with the AIA stacks via a join CSV. The published target variant is a 14-line set (Fe/He/Mg emission lines, z-score normalized).
- **Normalization:** per-line mean/std statistics are provided so users can unnormalize model outputs back to physical units.
- **Mars-view rendering:** Pre-rendered AIA views of the Sun from Mars' orbital position have been generated offline with SuNeRF-DT for selected 2023 dates, one FITS per (date, wavelength).
- **Validation data:** MAVEN/EUVM daily spectra and the FISM-P Mars reference spectrum are included to ground-truth the Mars-side predictions.

## Processed Dataset Schema (Core Training Data)

| Data Type | Shape | Units | Description |
|----------|------|------|-------------|
| AIA stacks | 256×256×9 | normalized intensity | Multi-channel solar images |
| EVE labels | (14,) | normalized | Spectral irradiance lines |
| Normalization | (2×14) | mean/std | Used to recover physical units |
| Wavelength names | (14,) | string | Ion emission line labels |


### MEGS-AI training bundle (~14 GB)

This bundle contains the primary image-to-spectrum training dataset used by MEGS-AI. It links multi-channel AIA image stacks with temporally matched EVE spectral irradiance measurements and includes all normalization metadata required for model training and inference.

- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-spect/processed_data/megs_ai_training/`
- Contents: self-contained image→spectrum training set.
    - `aia_stacks/matches_eve_aia.csv` (3.7 MB): 5,488-row join table mapping AIA stack filenames to EVE indices and dates. Columns: `eve_dates`, `eve_indices`, `time_delta`, `median_dates`, `AIA94`, `AIA131`, `AIA171`, `AIA193`, `AIA211`, `AIA304`, `AIA335`, `AIA1600`, `AIA1700`.
    - `aia_stacks/stacks/aia_<timestamp>.npy` (13 GB): 5,488 preprocessed AIA image stacks at 256×256, 9 channels each (7 EUV + 2 UV), float32, covering 2010-05 to 2014-05.
    - `eve_labels/megsa_converted.npy` (307 KB): `(5488, 14)` float32 — z-score-normalized MEGS-A line intensities. Indexed by row number in the CSV.
    - `eve_labels/megsa_normalization.npy` (240 B): `(2, 14)` — per-line mean (row 0) and std (row 1), for unnormalization.
    - `eve_labels/megsa_wl_names.npy` (492 B): list of 14 line names (Fe XVIII, Fe VIII, Fe XX, Fe IX, Fe X, Fe XI, Fe XII, Fe XIII, Fe XIV, He II, Fe XV, He II, Fe XVI, Mg IX).
    - `config/megs_ai_2024_config.yaml`: training configuration (architecture, train/val/test month splits, wavelength index maps).
- Train/val/test split (per config): val_months `[2, 3]`, test_months `[11, 12]`, holdout_months `[1, 10]`, train = the rest.

### Mars-view pre-renders (~100 MB)

These products represent one of the key outputs of the SuNeRF-DT stage. After reconstructing the 4D solar corona, the model is queried from the perspective of Mars to generate synthetic EUV observations that would be seen by a spacecraft orbiting the planet.

- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-spect/processed_data/mars_views/`
- Contents: SuNeRF-DT rendered Sun-as-seen-from-Mars AIA observations.
    - `mars_<date>_<wavelength>.fits`: per-date, per-wavelength FITS images for 7 EUV channels (94, 131, 171, 193, 211, 304, 335 Å). Multiple dates across 2023.
    - `earth_<date>_<wavelength>.fits`: matched Earth-view renders for comparison.
    - `mars_<date>.jpg`, `mars_<date>_bar.jpg`: quick-look visualizations.

### Validation: MAVEN/EUVM (~60 MB)

These observations provide the primary independent validation dataset for SPI3S. Because MAVEN observes the Sun from Mars orbit, EUVM measurements offer a unique opportunity to assess whether the coupled SuNeRF-DT + MEGS-AI pipeline can accurately predict irradiance beyond Earth-based viewpoints.

- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-spect/processed_data/validation/maven_euvm/`
- Contents: 34 daily CDF files (`mvn_euv_l3_daily_*.cdf`) from the MAVEN Extreme Ultraviolet Monitor, used as the primary validation target for Mars-side irradiance predictions.

### Ancillary (~8 MB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-spect/processed_data/ancillary/`
- Contents:
    - `fism_p_spectrum_mars_l2v01_r00_l3v01_r00_prelim.nc`: FISM-P (Flare Irradiance Spectral Model — Preliminary) Mars-location daily spectra. Used as an independent reference for Mars-view MEGS-AI predictions.


## 1.2 Raw Data

The SPI3S workflow combines observational and simulation datasets spanning both solar imaging and irradiance measurement domains. Multi-viewpoint EUV imagery provides the information needed to reconstruct the Sun's evolving three-dimensional structure, while irradiance measurements supply the targets and validation data required to evaluate spectral predictions. Physics-based MHD simulations provide additional ground truth for assessing the fidelity of the reconstructed coronal representation.

The raw datais publicly available from NASA archives:

- **SDO/AIA:** Atmospheric Imaging Assembly full-disk EUV images across 9 wavelength channels. Available from [JSOC](http://jsoc.stanford.edu/). ML-ready versions are available via the [SDOML dataset](https://sdoml.github.io/).
- **SDO/EVE:** Extreme UltraViolet Variability Experiment Level-2/3 spectral irradiance. Available from [LASP](https://lasp.colorado.edu/eve/data_access/).
- **STEREO/EUVI:** Extreme UltraViolet Imager data on STEREO-A and STEREO-B spacecraft (used for multi-viewpoint SuNeRF training). Available from the [STEREO Science Center](https://stereo-ssc.nascom.nasa.gov/).
- **MAVEN/EUVM:** Mars Atmosphere and Volatile EvolutioN mission Extreme Ultraviolet Monitor. Available from [LASP MAVEN](https://lasp.colorado.edu/maven/sdc/public/data/).
- **PSI MHD simulations:** Synthetic density/temperature cubes from Predictive Science Inc. Available from [predsci.com](https://www.predsci.com/hmi/data_access.php) (not included in this release).


# 2 Access Instructions

Data is stored on Amazon Web Services (AWS). Data access is given via the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-spect/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the data sample you want to download (see table) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

| Data Product | AWS Path | Size | Download time (@100 Mbps) |
|-------------|----------|------|---------------------------|
| Processed | `s3://nasa-radiant-data/helioai-datasets/hl-spect/processed_data/` | ~14 GB | ~20 minutes |

# 3 System Requirements

There are two sets of system requirements:
1. Requirements to *create* the data products. These can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-SPI3S/).
2. Requirements for *using* the data products:


| Component | Minimum |
|-----------|---------|
| **CPU** | Modern multi-core CPU |
| **RAM** | 16 GB (for loading the full 14 GB training bundle into memory); 4 GB is sufficient if streamed |
| **GPU** | Not required for data access; recommended for model training |
| **Storage** | 15 GB for the full processed dataset |

<!-- BACKUP

# 1 Access Instructions

Data is stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/<DATASET_NAME>/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the data sample you want to download (see table) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

| Data Product | AWS Path | Size | Download time (@100 Mbps) |
|-------------|----------|------|---------------------------|
| Processed | `<DATASET_NAME>/processed_data/` | | |
| Raw | `<DATASET_NAME>/raw_data/` | | |
| Results | `<DATASET_NAME>/results/` | | |
| Miscellaneous | `<DATASET_NAME>/miscellaneous/` | | |



# 2 Dataset Description


There are three levels of description available for this dataset:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](<https://drive.google.com/file/d/1hsxBvVRLFBoHWV8CKWkSo8HYE1TNG6re/view>)).
- The full source code used to process the data and create the models (see the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2024-HL-SPI3S-SuNeRF>)).

The SPI3S on Mars challenge was designed to estimate solar spectral irradiance from arbitrary viewpoints in the solar system, including Mars, by combining 3D reconstruction of the solar corona with a learned mapping from solar images to irradiance. The underlying scientific motivation is that solar irradiance is strongly shaped by the Sun's three-dimensional, time-evolving atmosphere, but direct measurements are only available from a limited set of spacecraft and viewpoints. Existing approaches can estimate irradiance at Earth or reconstruct coronal structure from a small number of observations, but they do not by themselves provide a full end-to-end system for estimating the Sun's spectral output at other planetary vantage points. The SPI3S dataset and workflow were assembled to solve this problem.

The challenge combines multi-viewpoint EUV image sequences, spectral irradiance measurements, and synthetic coronal density/temperature data from MHD simulations into a single end-to-end pipeline. In the first stage, a SuNeRF-based model learns a 4D representation of the corona, effectively a time-evolving 3D model of electron density and plasma temperature - from simultaneous observations in multiple EUV channels and from multiple spacecraft viewpoints. In the second stage, an updated MEGS-AI model learns to infer either spectral bands or the full EUV spectral irradiance from solar images. Together, these two components make it possible to synthesize a view of the Sun from Mars and then estimate the irradiance that would be measured there, with validation against MAVEN/EUVM measurements from the Martian perspective.


## 2.1 Processed Data


The processed data for SPI3S include a set of ML-ready training products feeding different parts of the end-to-end pipeline. For the SuNeRF-DT stage, the processed inputs are multi-viewpoint, multi-wavelength EUV image sequences aligned in time and associated with observer geometry so that the model can learn a 4D representation of the corona. 

For the MEGS-AI stage, the processed data consist of solar image representations paired with spectral irradiance targets. The 2024 SPI3S pipeline extends prior MEGS-AI work so that the model estimates not merely a few irradiance bands but the full EUV spectrum. Once SuNeRF-DT has synthesized a novel view of the Sun from Mars or another vantage, that synthetic solar image becomes the processed input to MEGS-AI, which then outputs the irradiance estimate. The memo makes clear that the full pipeline therefore depends on two levels of processed data products: first, geometry-aware multi-viewpoint coronal image sequences for 4D reconstruction, and second, image-to-spectrum training pairs for irradiance inference.

Data processing steps:

- Gather a time sequence of EUV images from multiple instruments and viewpoints, including SDO/ AIA and STEREO/EUVI.

- Use the image sequences together with observer positions and instrument response information to train SuNeRF-DT, a radiative-transfer-aware version of SuNeRF that reconstructs 4D coronal electron density and plasma temperature.

- Synthesize novel solar viewpoints from the trained SuNeRF representation, including the view of the Sun as seen from Mars.

- Train an updated MEGS-AI network to map EUV images to either spectral bands or the full EUV irradiance spectrum.

- Feed the synthetic Mars-view images into MEGS-AI to estimate irradiance at Mars and validate those estimates against MAVEN/EUVM observations.


## 2.2 Raw Data



The raw data for SPI3S includes Extreme UltraViolet satellite images from SDO/AIA and STEREO/EUVI to provide multi-viewpoint solar observations in different wavelength channels. These image data are essential because the SuNeRF component requires simultaneous or near-simultaneous views of the corona from different vantage points to reconstruct the 3D coronal scene. The challenge also uses irradiance measurements from SDO/EVE and MAVEN/EUVM, which serve as the primary targets or validation references for spectral irradiance estimation. In addition, the project uses synthetic density and temperature data derived from MHD simulations by Predictive Science Inc. (PSI), which are used to train and validate the SuNeRF-DT representation against known ground truth.


# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *create* the data products. These can be found in the [GitHub Repository](<https://github.com/emassara/2024-hl-radiation-ml/>).
2. Requirements for *using* the data products


| Component | Minimum |
|-----------|---------|
| **CPU** | |
| **RAM** | |
| **GPU** | |
| **Storage** | |
-->
