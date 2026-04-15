# 1 Dataset Description

The GEO-CLOAK Heliolab 2024 challenge was designed to move geoeffectiveness forecasting toward an operational, adaptive "Sun-to-ground" ML system. GEO-CLOAK therefore builds on the earlier Multiscale Geoeffectiveness FDL challenge result by pushing the combined SHEATH + DAGGER pipeline into a continual-learning and operational-inference setting.

The dataset behind GEO-CLOAK is a coordinated collection of solar, solar-wind, and ground-perturbation data products spanning multiple forecast horizons. SHEATH uses remote solar measurements to forecast solar-wind conditions at L1 with lead times of multiple days, while DAGGER-CL then refines those forecasts using higher-fidelity in-situ L1 data to produce global ground geomagnetic perturbation estimates with lead times of tens of minutes. This means the processed data product is a multi-stage forecasting dataset connecting solar-disk observations, L1 solar-wind parameters, and ground geomagnetic responses into a unified operational pipeline.

This dataset has two main components: [raw data](#12-raw-data) and [processed data](#11-processed-data).

In addition to the high-level summary of this dataset presented below, a detailed description may be found in the project [Technical Memorandum](https://helioai.org/dev/artifact/89d1911b-7803-44e7-b792-076edb2dc5ed/details); and the full source code used to process the data in the project [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-GeoCL/).

## 1.1 Processed Data

The processed dataset for GEO-CLOAK is a two-stage operational forecasting data product. In the long-horizon branch, solar observations are transformed into forecast solar-wind time series via SHEATH. In the short-horizon branch, those predictions are then refined or superseded by in-situ L1 measurements and used by DAGGER-CL to generate global ground geomagnetic perturbation forecasts.

The raw data undergoes the following processing steps to create a structured dataset suitable for training machine learning models. This involves cleaning, filtering, applying quality standards and transforming the raw measurements into a format that can be used for model training. 

  - Cleaning: Replace sentinel fill-values with NaN; rename columns to a common schema across ACE/DSCOVR/OMNI
  - Resampling: 1-hour for SHEATH, 1-minute for DAGGER-CL
  - Merging: Temporally align solar wind data with geomagnetic indices using backward-looking asof join
  - Derived features: Compute clock angle, trig combinations, dynamic pressure, electric field proxy, dipole tilt (~26 total solar wind features)
  - SDO processing (SHEATH only): Backtrack L1 time to Sun-observation time; segment AIA 193Å into coronal holes / active regions via GMM; extract mean
  intensities per region across all 12 channels (~26 SDO features)
  - Windowing (DAGGER-CL only): Slice 90-minute lookback windows as input; target is SuperMAG dB components at a future timestamp
  - Scaling: Yeo-Johnson power transform + z-score (DAGGER-CL) or z-score only (SHEATH); scalers fitted on training set and saved for inference

Instructions for accessing the following files on Amazon Web Services (AWS) are provided in [Section 2](#2-access-instructions).

### ACE (55 GB)
- AWS PATH: `hl-geo/processed_data/ACE/` 
- Contents: Formatted time-series with fields: `bt`, `bx_gsm`, `by_gsm`, `bz_gsm`, `proton_speed`, `proton_density`, `proton_temperature`. Historical data available from 2001
- Train and test set CSV file available.

### DSCOVR (14 GB)
- AWS PATH: `hl-geo/processed_data/DSCOVR/`
- Content: formatted time-series with fields: `bt`, `bx_gsm`, `by_gsm`, `bz_gsm`, `proton_speed`, `proton_density`, `proton_temperature`. Historical data
  available from 2016-07-26.
- Train and test set CSV file available.

### OMNI (0.5 MB)
- AWS PATH: `hl-geo/processed_data/OMNI/omniweb_formatted_2000.h5`
- Content: ground-truth solar wind labels that SHEATH learns to predict from SDO imagery. 

### SuperMAG (40 GB)
- AWS PATH: `hl-geo/processed_data/SuperMAG/`
- Content: 535 stations x 3 components (dBe, dBn, dBz) of ground magnetic perturbations in nT, stored per-minute, with NaN masks for missing stations,
  StandardScaler-normalized, and time-aligned to ACE/DSCOVR inputs via mapping files. magnetic field components.

### SDO (2 GB)
- AWS PATH: `hl-geo/processed_data/sdo/`
- per-timestamp CSV of 26 features extracted from raw SDO data (from 10 AIA and 3 HMI channels)

### SDO Embeddings (11 MB)
- AWS PATH: `hl-geo/processed_data/sdoembeddings/omniweb_back_tracked_ballistic.csv`
    -   The full pre-split dataset mapping SDO embeddings to OMNI solar wind targets. Each row is one timestamp. Columns:
        - Col 0: OMNI time (datetime at L1)
        - Col 1: SDO time (datetime after ballistic backtracking — used for temporal splitting)
        - Col 2: H5 latent index (integer index into sdo_latent_dataset_21504.h5["latent"] to fetch the 21,504-d embedding)
        - Cols 3–9: The 7 OMNI target variables — Speed, Density, Temperature, Bt, Bx, By, Bz
- AWS PATH: `hl-geo/processed_data/sdoembeddings/scaler_targets.json`
    -   A serialized MinMaxScaler fitted on the 7 target columns during training. Keys: min, scale, data_min, data_max, feature_range, n_features (7), feature_names. Used to inverse-transform
  predictions back to physical units.
- AWS PATH: `hl-geo/processed_data/sdoembeddings/sheath_train_set.csv`
    -   Training split — same columns as above. Everything not in the test intervals or validation years.
- AWS PATH: `hl-geo/processed_data/sdoembeddings/sheath_val_set.csv`
    -   Validation split — rows where SDO time falls in years 2014 or 2017 (excluding test storm intervals).
- AWS PATH: `hl-geo/processed_data/sdoembeddings/sheath_test_set.csv`
    -   Test split — rows from four specific geomagnetic storm periods:
        - 2011-08-04 to 2011-08-08
        - 2017-09-26 to 2017-09-29
        - 2018-08-13 to 2018-08-17
        - 2019-08-29 to 2019-09-01

### SHEATH (2 GB)
- AWS Path: `hl-geo/processed_data/sheath/`
- Training and test data for the SHEATH model.

## 1.2 Raw Data

The raw data for GEO-CLOAK fall into two main branches corresponding to the two major model components. For the DAGGER-CL branch, the input data are solar-wind and geospace drivers suitable for forecasting ground geomagnetic perturbations from near-real-time conditions at or near L1. For the SHEATH branch, the raw data are remote solar measurements that support longer lead-time forecasting of solar-wind conditions before those disturbances reach L1. In both cases, the physical objective is to encode the coupled chain from the Sun, through the solar wind, to Earth's ground magnetic response.

The raw data types inherited by GEO-CLOAK from its parent DAGGER and SHEATH systems include: 
<!-- remote solar observations, in-situ solar-wind parameters, and ground geomagnetic perturbation targets. -->

- ACE & DSCOVR: In-situ solar wind measurements from spacecraft at the L1 Lagrange point (~1.5M km sunward of Earth). They measure the speed, density, and
temperature of the solar wind plasma, plus the interplanetary magnetic field (IMF) vector. DSCOVR replaced ACE as the primary real-time monitor in 2016; the
project uses both for historical coverage.
- OMNI: A NASA-curated compilation of solar wind data from multiple L1 spacecraft (including ACE and DSCOVR), time-shifted to the Earth's bow shock nose and
cross-calibrated. Serves as a clean, gap-filled historical baseline.
- NOAA SWPC RTSW: A real-time JSON feed that blends whichever L1 spacecraft (ACE or DSCOVR) is currently operational. Used for the near-real-time (NRT)
inference pipeline.
- SuperMAG: Ground-truth magnetometer data from ~175+ stations worldwide. Provides geomagnetic perturbation measurements (N/E/Z components) and derived indices
(SME, SML, SMU). This is what the model is ultimately trying to predict.
- SDO (AIA + HMI): Remote-sensing solar disk observations from NASA's Solar Dynamics Observatory. AIA provides EUV images across 10 wavelength channels
(capturing coronal structure); HMI provides vector magnetograms (photospheric magnetic field maps). Consumed via the SDOML Zarr archive and used as input to the
SHEATH model.
- GFZ Geomagnetic Indices: Derived geophysical indices from the GFZ Potsdam observatory network: Kp/ap (global geomagnetic activity), Hp30/ap30 (30-min
high-latitude activity), F10.7 solar radio flux, and sunspot number. Used as additional features characterising solar and geomagnetic conditions.

# 2 Access Instructions

Data is stored on Amazon Web Services (AWS). Data access is given via the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-geo/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request <AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the data sample you want to download (see table) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

| Data Product | AWS Path | Size | Download time (@100 Mbps) |
|-------------|----------|------|---------------------------|
| Processed | `s3://nasa-radiant-data/helioai-datasets/hl-geo/processed_data/` | 123 GB | 3 hours |
| Raw | `s3://nasa-radiant-data/helioai-datasets/hl-geo/raw_data/` | 91 GB | 2 hours |


# 3 System Requirements

There are two sets of system requirements:
1. Requirements to *create* the data products. These can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-GeoCL/).
2. To use the raw and processed data, any modern computer with enough storage is sufficient.
