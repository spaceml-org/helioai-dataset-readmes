# 1 Access Instructions

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


# 2 Dataset Description

This dataset corresponds to the **Spectral Irradiance of the 3D Sun on Mars (SPI3S)** challenge from Heliolab 2024. It supports estimation of solar EUV spectral irradiance at arbitrary heliospheric viewpoints — including Mars — by combining a SuNeRF-based 3D reconstruction of the solar corona with the MEGS-AI deep-learning model that maps solar images to EUV spectra.

There are three levels of description available for this dataset:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](https://helioai.org/dev/artifact/f467e243-a299-43b9-871e-8c5c519663a5/details)).
- The full source code used to process the data and create the models (see the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-SPI3S/)).


## 2.1 Processed Data

The raw data undergoes processing to create structured, ML-ready datasets. The processing pipeline includes:

- **AIA image stacking:** Nine SDO/AIA channels (EUV: 94, 131, 171, 193, 211, 304, 335 Å; UV: 1600, 1700 Å) are co-registered per timestamp and written as single-file NumPy stacks at 256×256 resolution.
- **EVE alignment:** SDO/EVE MEGS-A irradiance is resampled and paired with the AIA stacks via a join CSV. The published target variant is a 14-line set (Fe/He/Mg emission lines, z-score normalized).
- **Normalization:** per-line mean/std statistics are provided so users can unnormalize model outputs back to physical units.
- **Mars-view rendering:** Pre-rendered AIA views of the Sun from Mars' orbital position have been generated offline with SuNeRF-DT for selected 2023 dates, one FITS per (date, wavelength).
- **Validation data:** MAVEN/EUVM daily spectra and the FISM-P Mars reference spectrum are included to ground-truth the Mars-side predictions.


### MEGS-AI training bundle (~14 GB)
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
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-spect/processed_data/mars_views/`
- Contents: SuNeRF-DT rendered Sun-as-seen-from-Mars AIA observations.
    - `mars_<date>_<wavelength>.fits`: per-date, per-wavelength FITS images for 7 EUV channels (94, 131, 171, 193, 211, 304, 335 Å). Multiple dates across 2023.
    - `earth_<date>_<wavelength>.fits`: matched Earth-view renders for comparison.
    - `mars_<date>.jpg`, `mars_<date>_bar.jpg`: quick-look visualizations.

### Validation: MAVEN/EUVM (~60 MB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-spect/processed_data/validation/maven_euvm/`
- Contents: 34 daily CDF files (`mvn_euv_l3_daily_*.cdf`) from the MAVEN Extreme Ultraviolet Monitor, used as the primary validation target for Mars-side irradiance predictions.

### Ancillary (~8 MB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-spect/processed_data/ancillary/`
- Contents:
    - `fism_p_spectrum_mars_l2v01_r00_l3v01_r00_prelim.nc`: FISM-P (Flare Irradiance Spectral Model — Preliminary) Mars-location daily spectra. Used as an independent reference for Mars-view MEGS-AI predictions.


## 2.2 Raw Data

Raw data is not included in this release as it is publicly available from NASA archives:

- **SDO/AIA:** Atmospheric Imaging Assembly full-disk EUV images across 9 wavelength channels. Available from [JSOC](http://jsoc.stanford.edu/). ML-ready versions are available via the [SDOML dataset](https://sdoml.github.io/).
- **SDO/EVE:** Extreme UltraViolet Variability Experiment Level-2/3 spectral irradiance. Available from [LASP](https://lasp.colorado.edu/eve/data_access/).
- **STEREO/EUVI:** Extreme UltraViolet Imager data on STEREO-A and STEREO-B spacecraft (used for multi-viewpoint SuNeRF training). Available from the [STEREO Science Center](https://stereo-ssc.nascom.nasa.gov/).
- **MAVEN/EUVM:** Mars Atmosphere and Volatile EvolutioN mission Extreme Ultraviolet Monitor. Available from [LASP MAVEN](https://lasp.colorado.edu/maven/sdc/public/data/).
- **PSI MHD simulations:** Synthetic density/temperature cubes from Predictive Science Inc. Available from [predsci.com](https://www.predsci.com/hmi/data_access.php) (not included in this release).


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
