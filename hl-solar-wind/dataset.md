# 1. Dataset Description

The **Decoding Solar Wind Structure (Heliolab 2025)** dataset supports the classification and prediction of solar wind regimes by combining:

- **Parker Solar Probe (PSP)** in-situ measurements  
- **Solar Dynamics Observatory (SDO)** remote sensing observations  

The dataset enables learning the relationship between **solar surface magnetic structure** and **solar wind conditions measured in situ**, using physically informed alignment via solar wind propagation models.

In addition to the high-level summary of this dataset, presented below, the full description may be found in the [Technical Memorandum](https://helioai.org/dev/artifact/cfaf8127-3065-48b7-9405-c4dc645327b4/details)); and the full source code used to process the data and create the associated models is contained in the project [GitHub Repository](https://github.com/FrontierDevelopmentLab/2025-HL-Solar-Wind).

# 1.1 Processed Data

The processed dataset, constructed through the processing pipeline described in the next section, is a **time-aligned, multi-modal dataset**, where each row corresponds to a PSP in-situ measurement and its associated solar source-region information derived from SDO observations and propagation modeling. 

| Field Group | Example Fields | Units / Type | Description |
|------------|--------------|-------------|-------------|
| Timestamp | time | datetime | PSP measurement time |
| Magnetic Field | psp_fld_l2_mag_RTN_{0,1,2}_mean | nT | Magnetic field components |
| Magnetic Magnitude | psp_fld_l2_mag_RTN_tot_mean | nT | Total magnetic field |
| Plasma Velocity | vp_fit_RTN_0_mean | km/s | Proton radial velocity |
| Plasma Density | np_fit_mean | cm^-3 | Proton density |
| Plasma Temperature | tp_fit_mean | K | Proton temperature |
| Spacecraft Position | sc_pos_SH_{lon,lat,r} | deg, AU | PSP heliographic position |
| Solar Footpoint | lon_footpoint, lat_footpoint | deg | WSA-mapped origin |
| Propagation Time | wsa_time, ballistic_time | datetime | Solar origin times |
| SDO Alignment | idx_* | index | Linked SDO observation |
| Label | label_xb2015 | categorical | Solar wind type |
| Interpolation Flags | *_interpflag | boolean | Indicates interpolated values |


The primary processed products are stored across multiple files:
- PSP in-situ measurements and labels
- PSP-SDO alignment indices
- SDO MAE embeddings
- CIPHER semi-automated labeling artifacts
- training configuration

Together, these products allow users to train or test models that classify solar wind structure from aligned solar imagery and PSP measurements.

## Processing Pipeline

### 1. Data Conversion
Raw PSP CDF files are converted to Zarr arrays for efficient random access and cloud-based analysis, enabling chunked, cloud-efficient access.

### 2. Temporal Resampling
PSP data are resampled to a consistent cadence:
 - 1-minute cadence for the main labeled PSP dataset
 - 12-minute cadence for smaller PSP-SDO aligned experiments

### 3. Interpolation and quality flags
Missing values are linearly interpolated where appropriate. Interpolation status is preserved using `*_interpflag` columns so users can distinguish measured from interpolated values.

### 4. Solar Wind Labeling
Solar-wind structures are classified using the Xu & Borovsky (2015) four-type scheme:

| Label | Class |      |
|------|------|------|
| 0 | Ejecta | ICME / transient ejecta-like solar wind |
| 1 | Coronal Hole | Fast wind associated with coronal-hole sources  |
| 2 | Sector Reversal | Solar wind near heliospheric current-sheet crossings |
| 3 | Streamer Belt | Slow wind associated with streamer-belt regions  |

These labels are stored in the `label_xb2015` column.

### 5. PSP-SDO temporal alignment

PSP in-situ measurements are aligned with SDO observations using the Wang-Sheeley-Arge (WSA) solar-wind propagation model. This accounts for propagation time between the Sun and PSP and provides solar-source timing information.

The aligned dataset includes:
- `wsa_time` : WSA-model-informed source time
- `ballistic_time` : simpler ballistic propagation estimate
- SDO image indices for AIA/HMI channels

This step establishes which solar region produced the measured solar wind.

### 6. SDO MAE embedding extraction

SDO HMI magnetic-field images are passed through a pretrained Masked Autoencoder (MAE) foundation model.

Input imagery:
  - HMI magnetic field images
  - 512x512 pixels
  - 3 channels: Bx, By, Bz
    
Output embedding:
  - 513 tokens x 512 dimensions per observation
  - stored as precomputed HDF5 embeddings
  - 
These embeddings are the primary model-ready solar-image representation.

7. CIPHER semi-automated labeling pipeline

The CIPHER pipeline provides a complementary semi-automated method for discovering and labeling solar-wind structures. It combines:
  - iSAX symbolic time-series compression
  - HDBSCAN clustering
  - human-in-the-loop validation
  - visualization outputs for cluster interpretation


## Processed Data Products

| Data Product  | AWS Path  |   Size | Format  | Contents / Role |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------- | -----: | ---------------- | --------------------------------------------------------------------------- |
| SDO MAE Embeddings              | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/data/sdofm_mae_embeddings.h5`             | 317 GB | HDF5             | Pre-computed SDO Foundation Model embeddings used as model input            |
| PSP In-Situ Data, Labeled       | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/data/psp_interpolated_1min_labeled.zarr/` | 2.3 GB | Zarr             | 1-minute PSP measurements with interpolation flags and Xu & Borovsky labels |
| PSP–SDO Alignment Index, 1-min  | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/data/alignment_index.zarr/`               | 2.2 GB | Zarr             | PSP data extended with SDO indices and propagation timestamps               |
| PSP–SDO Alignment Index, 12-min | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/data/combined_12min_2018_2023.zarr/`      | 304 MB | Zarr             | Smaller 12-minute alignment dataset for prototyping                         |
| CIPHER Pipeline Outputs         | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/cipher/`                                  | 350 MB | Pickle, PDF, PNG | CIPHER model artifacts and visualizations                                   |
| Training Configuration          | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/finetune_solarwind_config.yaml`           |  12 KB | YAML             | Training config, architecture, split definitions, loss, optimizer settings  |

## SDO MAE Embeddings
| Item            | Description                                                                                         |
| --------------- | --------------------------------------------------------------------------------------------------- |
| AWS path        | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/data/sdofm_mae_embeddings.h5` |
| Size            | 317 GB                                                                                              |
| Format          | HDF5                                                                                                |
| Samples         | 157,351                                                                                             |
| Coverage        | 2010–2023                                                                                           |
| Embedding shape | 513 tokens × 512 dimensions                                                                         |
| Data type       | float64                                                                                             |
| HDF5 keys       | `latent`, `timestamps`                                                                              |
| Role            | Primary SDO image-derived input features for the classification model                               |

## SDO embedding fields
| Field / Key  | Units / Type      | Description                                             |
| ------------ | ----------------- | ------------------------------------------------------- |
| `latent`     | float64 array     | Precomputed MAE encoder embeddings for SDO observations |
| `timestamps` | int64 nanoseconds | Observation timestamps associated with each embedding   |

These embeddings replace the need to process terabytes of raw SDO imagery for standard classifier inference.

## PSP In-Situ Data, Labeled

| Item     | Description                                                                                                     |
| -------- | --------------------------------------------------------------------------------------------------------------- |
| AWS path | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/data/psp_interpolated_1min_labeled.zarr/` |
| Size     | 2.3 GB                                                                                                          |
| Format   | Zarr                                                                                                            |
| Shape    | ~3.2 million rows × 314 columns                                                                                 |
| Cadence  | 1 minute                                                                                                        |
| Role     | Main labeled PSP in-situ dataset                                                                                |

## Key PSP fields
| Field                         | Units         | Description                             |
| ----------------------------- | ------------- | --------------------------------------- |
| `psp_fld_l2_mag_RTN_0_mean`   | nT            | Magnetic field R component              |
| `psp_fld_l2_mag_RTN_1_mean`   | nT            | Magnetic field T component              |
| `psp_fld_l2_mag_RTN_2_mean`   | nT            | Magnetic field N component              |
| `psp_fld_l2_mag_RTN_tot_mean` | nT            | Total magnetic field magnitude          |
| `vp_fit_RTN_0_mean`           | km/s          | Proton radial velocity                  |
| `np_fit_mean`                 | cm^-3         | Proton density                          |
| `tp_fit_mean`                 | K             | Proton temperature                      |
| `sc_pos_SH_lon`               | deg           | PSP Stonyhurst heliographic longitude   |
| `sc_pos_SH_lat`               | deg           | PSP Stonyhurst heliographic latitude    |
| `sc_pos_SH_r`                 | AU            | PSP radial distance from the Sun        |
| `lon_footpoint`               | deg           | WSA-traced magnetic footpoint longitude |
| `lat_footpoint`               | deg           | WSA-traced magnetic footpoint latitude  |
| `label_xb2015`                | integer class | Solar-wind structure label              |

## PSP–SDO Alignment Index, 1-minute
| Item     | Description                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------- |
| AWS path | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/data/alignment_index.zarr/` |
| Size     | 2.2 GB                                                                                            |
| Format   | Zarr                                                                                              |
| Shape    | ~2.3 million rows × 326 columns                                                                   |
| Cadence  | 1 minute                                                                                          |
| Role     | Primary training alignment file linking PSP measurements to SDO observations                      |

## Key alignment fields

| Field   | Units / Type  | Description                                                                  |
| --------------------------------------- | ------------- | ---------------------------------------------------------------------------- |
| `idx_131A`, `idx_1600A`, ..., `idx_94A` | integer index | Index of aligned SDO image for each AIA wavelength                           |
| `wsa_time`                              | datetime      | WSA propagation-corrected solar-source time                                  |
| `ballistic_time`                        | datetime      | Ballistic propagation source-time estimate                                   |
| PSP fields                              | mixed         | PSP in-situ and position fields carried through from the labeled PSP dataset |

This file is the primary file used during training to look up which SDO embedding corresponds to each PSP measurement.

## PSP–SDO Alignment Index, 12-minute
| Item     | Description                                                                                                |
| -------- | ---------------------------------------------------------------------------------------------------------- |
| AWS path | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/data/combined_12min_2018_2023.zarr/` |
| Size     | 304 MB                                                                                                     |
| Format   | Zarr                                                                                                       |
| Cadence  | 12 minutes                                                                                                 |
| Role     | Smaller alignment dataset matching SDO imaging cadence; suitable for quick experiments and prototyping     |

## CIPHER Pipeline Outputs

| Item     | Description                                                                    |
| -------- | ------------------------------------------------------------------------------ |
| AWS path | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/cipher/` |
| Size     | 350 MB                                                                         |
| Role     | Semi-automated labeling and structure-discovery artifacts                      |

| File / Product           | Approx. Size | Description                                |
| ------------------------ | -----------: | ------------------------------------------ |
| `*_isax_pipeline.pkl`    |        16 MB | Trained iSAX time-series compression model |
| `*_model-dictionary.pkl` |  ~16 MB each | HDBSCAN cluster models for PSP parameters  |
| `*_transliterate.pkl`    |       2.2 MB | iSAX symbolic representation mapping       |
| `visualizations/*.pdf`   |      ~200 MB | Cluster visualization PDFs                 |
| `visualizations/*.png`   |        ~2 MB | t-SNE cluster plots                        |


## Training Configuration

| Item                    | Description                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------- |
| AWS path                | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/processed_data/finetune_solarwind_config.yaml` |
| Size                    | 12 KB                                                                                                 |
| Format                  | YAML                                                                                                  |
| Contents                | model architecture, data paths, train/test splits, loss function, optimizer hyperparameters           |
| Training split          | 2019–2023 months 4–12                                                                                 |
| Validation / test split | 2019–2023 months 1–3                                                                                  |
| Also available          | model bucket                                                                                          |


# 1.2 Raw Data 

The raw data used for this project is publicly available from NASA archives.

| Source                           | Cadence            | Key Fields                             | Units                | Description                                                 | Role                                       |
| -------------------------------- | ------------------ | -------------------------------------- | -------------------- | ----------------------------------------------------------- | ------------------------------------------ |
| PSP FIELDS Level 2               | instrument cadence | magnetic field vector                  | nT                   | In-situ magnetic-field measurements from Parker Solar Probe | Solar-wind structure input                 |
| PSP SWEAP/SPC and SPAN-I Level 3 | instrument cadence | velocity, density, temperature         | km/s, cm^-3, K       | In-situ plasma measurements from Parker Solar Probe         | Solar-wind structure input                 |
| SDO/HMI                          | image cadence      | vector magnetograms Bx, By, Bz         | magnetic field units | Photospheric magnetic-field maps                            | Solar source-region input                  |
| SDO/AIA                          | image cadence      | EUV images across multiple wavelengths | intensity units      | Solar atmospheric imagery                                   | Optional / alignment-related solar context |
| WSA model outputs                | model-derived      | footpoints, propagation times          | deg, datetime        | Solar wind source-region and travel-time estimates          | PSP–SDO alignment                          |

## Raw data access

  - PSP (Parker Solar Probe): In-situ solar wind measurements including magnetic field (FIELDS instrument, Level 2) and plasma parameters (SWEAP/SPC and SPAN-I, Level 3). Available from the [NASA SPDF archive](https://spdf.gsfc.nasa.gov/).
  - SDO (Solar Dynamics Observatory): Multi-wavelength solar disk imagery from AIA and vector magnetograms from HMI (Bx, By, Bz). Available from [JSOC](http://jsoc.stanford.edu/). ML-ready versions are available via the [SDOML dataset](https://sdoml.github.io/#/).


# 1.3 Data Workflow

<p align="center">
  <img src="https://quickchart.io/graphviz?graph=digraph%20G%20%7B%0Arankdir%3DTB%3B%0Anode%20%5Bshape%3Dbox%2C%20style%3Dfilled%2C%20color%3Dlightgray%5D%3B%0APSP%20%5Blabel%3D%22PSP%20In-Situ%20Data%22%5D%3B%0ASDO%20%5Blabel%3D%22SDO%20Images%22%5D%3B%0AWSA%20%5Blabel%3D%22WSA%20Propagation%22%5D%3B%0APROC%20%5Blabel%3D%22Processing%20%5CnResample%20%7C%20Interpolate%20%7C%20Align%22%5D%3B%0AEMB%20%5Blabel%3D%22MAE%20Embeddings%22%5D%3B%0ADATA%20%5Blabel%3D%22Aligned%20Dataset%22%5D%3B%0APSP%20-%3E%20PROC%3B%0ASDO%20-%3E%20EMB%3B%0AEMB%20-%3E%20DATA%3B%0APROC%20-%3E%20DATA%3B%0AWSA%20-%3E%20DATA%3B%0A%7D" width="500">
</p>


# 2. Access Instructions
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


# 3 System Requirements

There are two sets of system requirements:
1. Requirements to **create** the data products. These can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2025-HL-Solar-Wind).
2. Requirements for **using** the data products.

| Component   | Minimum                                                                                  | Recommended                                                  |
| ----------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **CPU**     | Modern multi-core CPU                                                                    | 8+ cores for large local processing                          |
| **RAM**     | 16 GB for loading alignment indices and embedding subsets                                | 32 GB for larger Zarr / HDF5 workflows                       |
| **GPU**     | Not required for data access                                                             | Required for model training or MAE inference from raw images |
| **Storage** | 330 GB for full processed dataset; 5 GB for PSP data + alignment only without embeddings | 400+ GB if keeping local working copies and derived outputs  |














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
