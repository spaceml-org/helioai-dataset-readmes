# 1 Dataset Description

The **Decoding Solar Wind Structure (Heliolab 2025)** dataset supports the classification and prediction of solar wind regimes by combining:

- **Parker Solar Probe (PSP)** in-situ measurements  
- **Solar Dynamics Observatory (SDO)** remote sensing observations  

The dataset enables learning the relationship between **solar surface magnetic structure** and **solar wind conditions measured in situ**, using physically informed alignment via solar wind propagation models.

There are three levels of description available:
- High-level summary (this document)
- [Technical Memorandum](https://helioai.org/dev/artifact/cfaf8127-3065-48b7-9405-c4dc645327b4/details)
- [GitHub Repository](https://github.com/FrontierDevelopmentLab/2025-HL-Solar-Wind)

---

# 2 Processed Data

The processed dataset, constructed through the processing pipeline described in the next section, is a **time-aligned, multi-modal dataset**, where each row corresponds to a PSP measurement and its associated solar origin, as shown in the schema summarized below.

| Field Group | Example Fields | Units | Description |
|------------|--------------|------|-------------|
| Timestamp | time | datetime | Time of PSP measurement |
| Magnetic Field | psp_fld_l2_mag_RTN_{0,1,2}_mean | nT | Magnetic field vector components (R, T, N) |
| Magnetic Magnitude | psp_fld_l2_mag_RTN_tot_mean | nT | Total magnetic field magnitude |
| Plasma Velocity | vp_fit_RTN_0_mean | km/s | Proton radial velocity |
| Plasma Density | np_fit_mean | cm⁻³ | Proton number density |
| Plasma Temperature | tp_fit_mean | K | Proton temperature |
| Spacecraft Position | sc_pos_SH_{r,lat,lon} | AU, deg | PSP heliographic position |
| Solar Footpoint | lon_footpoint, lat_footpoint | deg | Magnetic footpoint location from WSA mapping |
| Solar Wind Label | label_xb2015 | categorical | Solar wind type (Ejecta, Coronal Hole, etc.) |
| SDO Alignment | idx_* (AIA wavelengths) | index | Index linking to SDO observation |
| Propagation Times | wsa_time, ballistic_time | datetime | Solar origin timestamps |
| SDO Embedding | latent (linked via index) | float | MAE embedding representing solar source region |

> Note: SDO embeddings are stored separately and linked via the alignment index. Each row contains indices pointing to the corresponding embedding rather than the embedding itself.
---

## Processing Pipeline

### 1. Data Conversion
- PSP CDF files → **Zarr format**
- Enables chunked, cloud-efficient access

---

### 2. Temporal Resampling
- PSP data resampled to:
  - **1-minute cadence (primary)**
  - **12-minute cadence (optional, SDO-aligned)**

---

### 3. Missing Data Handling
- Linear interpolation applied
- Missing values tracked via:
  - `*_interpflag` columns

---

### 4. Solar Wind Labeling
- Xu & Borovsky (2015) classification:
  - 0 = Ejecta  
  - 1 = Coronal Hole  
  - 2 = Sector Reversal  
  - 3 = Streamer Belt  

---

### 5. Solar–Wind Propagation Alignment (CRITICAL STEP)

- PSP measurements are mapped back to the Sun using:
  - **Wang-Sheeley-Arge (WSA) model**
- Produces:
  - `ballistic_time` (simple propagation)
  - `wsa_time` (model-informed propagation)

This step establishes which solar region produced the measured solar wind.

---

### 6. SDO Feature Extraction

- SDO HMI magnetograms (Bx, By, Bz) processed via:
  - **Masked Autoencoder (MAE) foundation model**
- Produces:
  - embeddings of shape **(513 tokens × 512 dims)**

---

### 7. Dataset Integration

- PSP + SDO linked via:
  - **alignment index**
- Final dataset is:
  - time-aligned  
  - feature-complete  
  - ML-ready  

---

# 2.1 Processed Data Files

---

## PSP In-Situ Data (Labeled)

| Field | Units | Description |
|------|------|-------------|
| psp_fld_l2_mag_RTN_0_mean | nT | Magnetic field (R component) |
| psp_fld_l2_mag_RTN_1_mean | nT | Magnetic field (T component) |
| psp_fld_l2_mag_RTN_2_mean | nT | Magnetic field (N component) |
| psp_fld_l2_mag_RTN_tot_mean | nT | Total magnetic field magnitude |
| vp_fit_RTN_0_mean | km/s | Proton radial velocity |
| np_fit_mean | cm⁻³ | Proton density |
| tp_fit_mean | K | Proton temperature |
| sc_pos_SH_r | AU | Spacecraft radial distance |
| sc_pos_SH_lat | deg | Spacecraft latitude |
| sc_pos_SH_lon | deg | Spacecraft longitude |
| lon_footpoint | deg | Solar footpoint longitude (WSA) |
| lat_footpoint | deg | Solar footpoint latitude (WSA) |
| label_xb2015 | categorical | Solar wind regime label |

---

## SDO MAE Embeddings

| Field | Units | Description |
|------|------|-------------|
| latent | float | MAE embeddings (513 × 512) |
| timestamps | ns | Observation timestamps |

These replace raw SDO images as model inputs.

---

## PSP–SDO Alignment Index

| Field | Units | Description |
|------|------|-------------|
| idx_131A ... idx_94A | index | SDO image indices per wavelength |
| wsa_time | datetime | Propagation-corrected solar time |
| ballistic_time | datetime | Ballistic propagation estimate |

Defines mapping:
PSP measurement → solar observation

---

## CIPHER Pipeline Outputs

| Component | Description |
|----------|-------------|
| iSAX models | Time-series symbolic compression |
| HDBSCAN models | Unsupervised clustering of solar wind regimes |
| visualization PDFs | Cluster structure analysis |
| t-SNE plots | Low-dimensional structure visualization |

---

## Training Configuration

| Field | Description |
|------|-------------|
| train/test split | temporal split (2019–2023) |
| optimizer | training hyperparameters |
| architecture | model definition |

---

# 2.2 Raw Data Overview

| Source | Cadence | Key Fields | Units | Description | Role |
|-------|--------|------------|-------|-------------|------|
| PSP (FIELDS + SWEAP) | seconds–minutes | B-field, velocity, density, temperature | nT, km/s, cm⁻³, K | In-situ solar wind measurements | Target + features |
| SDO (HMI + AIA) | ~12 min | Magnetograms, EUV images | various | Solar surface structure | Input features |
| WSA model | model-derived | propagation times, footpoints | time, deg | Solar wind source mapping | Alignment |

---

# 2.3 Data Workflow

<p align="center">
  <img src="https://quickchart.io/graphviz?graph=digraph%20G%20%7B%0Arankdir%3DTB%3B%0Anode%20%5Bshape%3Dbox%2C%20style%3Dfilled%2C%20color%3Dlightgray%5D%3B%0APSP%20%5Blabel%3D%22PSP%20In-Situ%20Data%22%5D%3B%0ASDO%20%5Blabel%3D%22SDO%20Images%22%5D%3B%0AWSA%20%5Blabel%3D%22WSA%20Propagation%22%5D%3B%0APROC%20%5Blabel%3D%22Processing%20%5CnResample%20%7C%20Interpolate%20%7C%20Align%22%5D%3B%0AEMB%20%5Blabel%3D%22MAE%20Embeddings%22%5D%3B%0ADATA%20%5Blabel%3D%22Aligned%20Dataset%22%5D%3B%0APSP%20-%3E%20PROC%3B%0ASDO%20-%3E%20EMB%3B%0AEMB%20-%3E%20DATA%3B%0APROC%20-%3E%20DATA%3B%0AWSA%20-%3E%20DATA%3B%0A%7D" width="500">
</p>

---

# 3 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Multi-core | 8+ cores |
| RAM | 16 GB | 32 GB |
| GPU | Not required | Recommended for training |
| Storage | 330 GB (full dataset) | 400+ GB (with working space) |












<!-- # 1 Access Instructions

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
| **Storage** | 330 GB for full processed dataset; 5 GB for PSP data + alignment only (without embeddings) | -->
