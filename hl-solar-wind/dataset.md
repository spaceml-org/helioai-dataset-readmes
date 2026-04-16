# 1 Access Instructions

Data is stored on Amazon Web Services (AWS). Data access is given via the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the data sample you want to download (see table) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

| Data Product | AWS Path | Size | Download time (@100 Mbps) |
|-------------|----------|------|---------------------------|
| Processed | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/` | 322 GB | ~7 hours |
| CIPHER Pipeline | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/cipher/` | 350 MB | < 1 minute |



# 2 Dataset Description

This dataset corresponds to the Decoding Solar Wind Structure challenge from Heliolab 2025. It supports the classification and prediction of solar wind phenomena using Parker Solar Probe (PSP) in-situ measurements and Solar Dynamics Observatory (SDO) imagery.

There are three levels of description available for this dataset:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](https://helioai.org/dev/artifact/cfaf8127-3065-48b7-9405-c4dc645327b4/details)).
- The full source code used to process the data and create the models (see the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2025-HL-Solar-Wind)).


## 2.1 Processed Data

The raw data undergoes processing to create structured, ML-ready datasets. The processing pipeline includes:

- **CDF to Zarr conversion:** Raw PSP CDF files are converted to Zarr arrays for efficient random access.
- **Temporal resampling:** Data is resampled to 1-minute cadence (with 12-minute and other cadences also available).
- **Interpolation:** Missing values are linearly interpolated with quality flags (`*_interpflag` columns) preserved.
- **Labeling:** Solar wind structures are classified using the Xu & Borovsky (2015) four-type scheme (Ejecta, Coronal Hole, Sector Reversal, Streamer Belt) stored in the `label_xb2015` column.
- **Temporal alignment:** PSP in-situ measurements are aligned with SDO observations using the Wang-Sheeley-Arge (WSA) solar wind propagation model, accounting for the travel time from the Sun to the spacecraft.
- **Embedding extraction:** SDO HMI magnetic field images (512x512, Bx/By/Bz) are passed through a pretrained Masked Autoencoder (MAE) foundation model to produce fixed-size embeddings (513 tokens x 512 dims per observation).
- **CIPHER pipeline:** A separate semi-automated pipeline (iSAX compression + HDBSCAN clustering + human-in-the-loop validation) produces labeled catalogs of solar wind structures.


### SDO MAE Embeddings (317 GB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/data/sdofm_mae_embeddings.h5`
- Contents: Pre-computed SDO Foundation Model (MAE) encoder embeddings in HDF5 format. 157,351 samples covering 2010–2023. Each sample is 513 tokens x 512 dimensions (float64). HDF5 keys: `latent` (embeddings), `timestamps` (int64 nanoseconds). These embeddings serve as the primary input features for the classification model, replacing the need to process terabytes of raw SDO imagery.

### PSP In-Situ Data, Labeled (2.3 GB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/data/psp_interpolated_1min_labeled.zarr/`
- Contents: Parker Solar Probe measurements at 1-minute cadence, interpolated and labeled. 3.2 million rows x 314 columns (Zarr format). Key fields include:
    - Magnetic field: `psp_fld_l2_mag_RTN_{0,1,2}_mean` (R, T, N components), `psp_fld_l2_mag_RTN_tot_mean` (magnitude)
    - Plasma: `vp_fit_RTN_0_mean` (proton radial velocity), `np_fit_mean` (density), `tp_fit_mean` (temperature)
    - Position: `sc_pos_SH_{lon,lat,r}` (Stonyhurst heliographic), `lon_footpoint`, `lat_footpoint` (WSA-traced magnetic footpoint)
    - Labels: `label_xb2015` (0=Ejecta, 1=Coronal Hole, 2=Sector Reversal, 3=Streamer Belt)

### PSP–SDO Alignment Index, 1-min (2.2 GB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/data/alignment_index.zarr/`
- Contents: Temporal alignment between PSP in-situ measurements and SDO observations. 2.3 million rows x 326 columns (Zarr format). Extends the PSP data with SDO image indices (`idx_131A`, `idx_1600A`, ..., `idx_94A` for each AIA wavelength) and propagation timestamps (`wsa_time`, `ballistic_time`). This is the primary file used during training to look up which SDO embedding corresponds to each PSP measurement.

### PSP–SDO Alignment Index, 12-min (304 MB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/data/combined_12min_2018_2023.zarr/`
- Contents: Same alignment data at 12-minute cadence (matching SDO imaging cadence). Smaller and suitable for quick experiments or prototyping.

### CIPHER Pipeline Outputs (350 MB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/cipher/`
- Contents: Artifacts from the CIPHER semi-automated labeling pipeline:
    - `*_isax_pipeline.pkl` (16 MB): Trained iSAX time series compression model
    - `*_model-dictionary.pkl` (~16 MB each): HDBSCAN cluster models for each PSP parameter (magnetic field components, proton density, temperature, velocity, thermal pressure)
    - `*_transliterate.pkl` (2.2 MB): iSAX symbolic representation mapping
    - `visualizations/*.pdf` (~200 MB): Cluster visualization PDFs (univariate, multivariate, vertical)
    - `visualizations/*.png` (~2 MB): t-SNE cluster plots

### Training Configuration (12 KB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/finetune_solarwind_config.yaml`
- Contents: Full training configuration including model architecture, data paths, train/test splits (train: 2019–2023 months 4–12; val/test: 2019–2023 months 1–3), loss function, and optimizer hyperparameters. Also available in the model bucket.


## 2.2 Raw Data

Raw data is not included in this release as it is publicly available from NASA archives:

- **PSP (Parker Solar Probe):** In-situ solar wind measurements including magnetic field (FIELDS instrument, Level 2) and plasma parameters (SWEAP/SPC and SPAN-I, Level 3). Available from the [NASA SPDF archive](https://spdf.gsfc.nasa.gov/).
- **SDO (Solar Dynamics Observatory):** Multi-wavelength solar disk imagery from AIA (9 EUV channels) and vector magnetograms from HMI (Bx, By, Bz). Available from [JSOC](http://jsoc.stanford.edu/). ML-ready versions are available via the [SDOML dataset](https://sdoml.github.io/).


# 3 System Requirements

There are two sets of system requirements:
1. Requirements to *create* the data products. These can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2025-HL-Solar-Wind).
2. Requirements for *using* the data products:


| Component | Minimum |
|-----------|---------|
| **CPU** | Modern multi-core CPU |
| **RAM** | 16 GB (for loading alignment indices and embeddings subsets) |
| **GPU** | Not required for data access; required for model training |
| **Storage** | 330 GB for full processed dataset; 5 GB for PSP data + alignment only (without embeddings) |
