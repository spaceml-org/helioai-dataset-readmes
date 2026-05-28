# 1. Dataset Description

Using high-resolution observations from the Helioseismic and Magnetic Imager (HMI, Scherrer et. al 2012) and Atmospheric Imaging Assembly (AIA, Lemen et al. 2012) instruments onboard the Solar Dynamics Observatory (SDO), the Frontier Development Lab (FDL) Heliolab 2025 Active Region Characterization and Analysis of Dynamics and Evolution (ARCADE) system combines physics-based Surface Flux Transport modeling with deep learning to characterize the magnetic evolution of solar active regions that can give rise to flares and coronal mass ejections. The hybrid approach extracts key parameters describing magnetic field structure and integrates uncertainty quantification to produce interpretable forecasts. Results demonstrate accurate short-term predictions of active region emergence up to six hours in advance, providing an important first step for downstream space weather models.

While the ARCADE forecasting model was trained on SDO magnetograms, it was designed to work with following additional SDO/HMI and SDO/AIA data modes: dopplergrams, extreme-ultraviolet images at 171 Å and 304 Å, and continuum-intensity maps. The SDO/HMI exposes continual, full-disk images of the Sun, while the SDO/AIA images the solar atmosphere at 10 different wavelengths, every 10 seconds. 

**In addition to the high-level summary of the dataset provided here, a detailed description may be found in the project [Technical Memorandum](<https://drive.google.com/file/d/1fTI2N0cOcLgbzVkk7QRWpNYnPsgFvwb4/view>).**

<p align="center">
  <img src="https://github.com/spaceml-org/helioai-dataset-readmes/blob/main/hl-arcade/ARCADE_dataset_infographic.png?raw=true" width="400">
</p>

### Dataset Snapshot

| Category | Description |
|---|---|
| Challenge | Heliolab 2025 Active Region Characterization and Analysis of Dynamics and Evolution (ARCADE) |
| Scientific Goal | Forecast the short-term evolution of solar active regions and magnetic structure associated with flares and coronal mass ejections |
| Primary Missions | Solar Dynamics Observatory (SDO) |
| Instruments | HMI (magnetograms, dopplergrams, continuum intensity), AIA (171 Å and 304 Å EUV imagery) |
| Raw Data Format | FITS |
| Processed Data Format | Multi-channel Zarr arrays |
| Spatial Resolution | Full-disk solar observations (4096 × 4096 pixels) |
| Temporal Coverage | Continuous SDO observations |
| Temporal Cadence | HMI full-disk cadence; AIA imaging every ~10 seconds |
| Data Modalities | Magnetograms, Dopplergrams, EUV images, continuum intensity |
| Physics-Based Components | Surface Flux Transport (SFT) / Advective Flux Transport (AFT) modeling |
| ML-Ready Products | Co-registered, normalized, multi-modal solar data cubes |
| Validation Products | Physics-based AFT simulations for magnetic field evolution benchmarking |
| Primary ML Use Case | Physics-informed forecasting of active region emergence and magnetic evolution |


## 1.1 Raw Data

Raw data products are publicly available from solar data archives:

- **SDO HMI and AIA FITS data**  
  - [Joint Science Operations Center (JSOC)](http://jsoc.stanford.edu/)  
  - [Virtual Solar Observatory (VSO)](https://sdac.virtualsolar.org/cgi/search)

These datasets are:
- calibrated  
- science-ready  
- not directly optimized for ML workflows  

The ARCADE pipeline performs the necessary transformations to convert them into analysis-ready formats.

---

## 1.2 Processed Data

The raw SDO archive is transformed into a structured, ML-ready dataset through a multi-stage pipeline.

### 1.2.1 Observational Data Processing

**Data cleaning and correction**
- Removal of:
  - corrupted or incomplete frames  
  - spacecraft anomalies (eclipses, safe modes)  
  - saturated pixels (e.g., during large flares)  
- Correction of Doppler velocities using spacecraft motion metadata  

**Image co-registration**
- Spatial alignment across:
  - time  
  - instruments  
  - wavelengths  

Ensures consistent pixel correspondence for evolving features.  

**Projection correction**
- Correction for:
  - foreshortening near solar limb  
  - line-of-sight vs radial magnetic field differences  

**Normalization**
- Scaling of values to consistent ranges for ML training  

**Removal of geometric effects**
- Mitigation of:
  - limb darkening  
  - spherical projection artifacts  
  - large-scale non-physical gradients  

**Data restructuring**
- Conversion from individual FITS files to **Zarr format**
- Final dataset structure:

| Dimension | Description |
|----------|-------------|
| `t_obs` | Observation timestamps |
| `channel` | Data modality (5 channels) |
| `x, y` | Spatial dimensions (4096 × 4096 pixels) |

This produces a **multi-channel, time-resolved data cube** suitable for deep learning workflows.

### 1.2.2 Simulated Validation Data (AFT)

To complement observational data, ARCADE incorporates physics-based simulations from the **Advective Flux Transport (AFT)** model.

The AFT model implements a **Surface Flux Transport (SFT)** framework, modeling:

- Differential rotation  
- Meridional flow  
- Turbulent diffusion  
- Magnetic flux emergence  

**Key characteristics:**
- Full-disk, time-evolving magnetograms  
- Global radial magnetic field estimates  
- Physically consistent evolution across the entire solar surface  

Unlike SDO magnetograms:
- AFT provides **complete radial field coverage**, not limited to line-of-sight measurements  

This makes AFT critical for:
- evaluating large-scale magnetic evolution  
- validating model generalization beyond observational constraints  

---


# 2. Access Instructions

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
| Processed – Training | 4096×4096 pixel SDO magnetograms stored in a multi-channel Zarr dataset (see [SDOMLv2](https://registry.opendata.aws/sdoml-fdl/) for lower-resolution examples) | TBD | TBD |
| Processed – Validation | `s3://nasa-radiant-data/helioai-datasets/hl-arcade/2025-hl-arcade-development-landing/aft/lisa/AFT_Baseline/{YYYY}/{NN}/AFTmap*.h5` | TBD | TBD |
| Results | [Interactive UI](https://arcade.trillium.tech/) | N/A | N/A |

---

## Note on data availability 

The ARCADE project is currently ongoing. As a result:

- Full processed datasets are not yet finalized for public release  
- Model-ready training subsets may be updated  
- Final dataset sizes and access patterns are subject to change  

This page will be updated as artifacts become available.




<!-- 

# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *create* the data products. These can be found in the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2025-HL-Active-Regions/>).
2. Requirements for *using* the data products


| Component | Minimum |
|-----------|---------|
| **CPU** | |
| **RAM** | |
| **GPU** | 20 GB |
| **Storage** | | 

**Bucket Name:**2025-hl-arcade-development-models
**Size**: 238M
**What's in it:** Raw data
**What it is:** Provides original source data, including cleaned, calibrated, science-ready SDO HMI & AIA image data products accessible from the Joint Science Operations Center (JSOC) at Stanford University (http://jsoc.stanford.edu/)
**What this enables:** Reproducibility and reprocessing from original data sources.

**Bucket Name:** 2025-hl-arcade-development-landing
**Size:** 73G
**What's in it:**Simulated test/validation data
**What it is:**Provides full-Sun, simulated input magnetograms from the SOTA Advective Flux Transport (AFT) model, offering estimates of the magnetic flux of the entire Sun using advective transport, overcoming the line-of-sight observation limitation of real magnetograms.
**What this enables:** Model validation and evaluation. 

**Bucket Name:** 2025-hl-arcade-development-features
**Size:** 4.9G
**What's in it:** Results
**What it is:** Provides input/output files from model validation tests including input  ‘target’ (real) and ‘predicted’ (model-generated) magnetogram images and their ‘difference’ images used to evaluate model performance and quantify data-minus-model residuals.
**What this enables:** Model validation and evaluation. -->


<!-- BACKUP

# 1. Dataset Description

Using high-resolution observations from the Helioseismic and Magnetic Imager (HMI, Scherrer et. al 2012) and Atmospheric Imaging Assembly (AIA, Lemen et al. 2012) instruments onboard the Solar Dynamics Observatory (SDO), the Frontier Development Lab (FDL) Heliolab 2025 Active Region Characterization and Analysis of Dynamics and Evolution (ARCADE) system combines physics-based Surface Flux Transport modeling with deep learning to characterize the magnetic evolution of solar active regions that can give rise to flares and coronal mass ejections. The hybrid approach extracts key parameters describing magnetic field structure and integrates uncertainty quantification to produce interpretable forecasts. Results demonstrate accurate short-term predictions of active region emergence up to six hours in advance, providing an important first step for downstream space weather models.

While the ARCADE forecasting model was trained on SDO magnetograms, it was designed to work with following additional SDO/HMI and SDO/AIA data modes: dopplergrams, extreme-ultraviolet images at 171 Å and 304 Å, and continuum-intensity maps. The SDO/HMI exposes continual, full-disk images of the Sun, while the SDO/AIA images the solar atmosphere at 10 different wavelengths, every 10 seconds. 

In addition to the high-level summary of the dataset provided here, a detailed description may be found in the project [Technical Memorandum](<https://drive.google.com/file/d/1fTI2N0cOcLgbzVkk7QRWpNYnPsgFvwb4/view>).


## 1.1 Processed Data

The raw SDO archive is transformed into a structured, ML-ready dataset through a multi-stage pipeline.

### 1.2.1 Observational Data Processing

#### Data cleaning and correction
- Removal of:
  - corrupted or incomplete frames  
  - spacecraft anomalies (eclipses, safe modes)  
  - saturated pixels (e.g., during large flares)  
- Correction of Doppler velocities using spacecraft motion metadata  

#### Image co-registration
- Spatial alignment across:
  - time  
  - instruments  
  - wavelengths  

Ensures consistent pixel correspondence for evolving features.  

#### Projection correction
- Correction for:
  - foreshortening near solar limb  
  - line-of-sight vs radial magnetic field differences  

#### Normalization
- Scaling of values to consistent ranges for ML training  

#### Removal of geometric effects
- Mitigation of:
  - limb darkening  
  - spherical projection artifacts  
  - large-scale non-physical gradients  

#### Data restructuring
- Conversion from individual FITS files to **Zarr format**
- Final dataset structure:

| Dimension | Description |
|----------|-------------|
| `t_obs` | Observation timestamps |
| `channel` | Data modality (5 channels) |
| `x, y` | Spatial dimensions (4096 × 4096 pixels) |

This produces a **multi-channel, time-resolved data cube** suitable for deep learning workflows.

     
### 1.2.2 Simulated Validation Data (AFT)

To complement observational data, ARCADE incorporates physics-based simulations from the **Advective Flux Transport (AFT)** model.

The AFT model implements a **Surface Flux Transport (SFT)** framework, modeling:

- Differential rotation  
- Meridional flow  
- Turbulent diffusion  
- Magnetic flux emergence  

#### Key characteristics:
- Full-disk, time-evolving magnetograms  
- Global radial magnetic field estimates  
- Physically consistent evolution across the entire solar surface  

Unlike SDO magnetograms:
- AFT provides **complete radial field coverage**, not limited to line-of-sight measurements  

This makes AFT critical for:
- evaluating large-scale magnetic evolution  
- validating model generalization beyond observational constraints  

---

## 1.3 Raw Data

Raw data products are publicly available from solar data archives:

- **SDO HMI and AIA FITS data**  
  - [Joint Science Operations Center (JSOC)](http://jsoc.stanford.edu/)  
  - [Virtual Solar Observatory (VSO)](https://sdac.virtualsolar.org/cgi/search)

These datasets are:
- calibrated  
- science-ready  
- not directly optimized for ML workflows  

The ARCADE pipeline performs the necessary transformations to convert them into analysis-ready formats.

---

# 2. Access Instructions

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


## 2.1 Data Products

| Data Product | AWS Path | Size | Download time (@100 Mbps) |
|-------------|----------|------|---------------------------|
| Processed – Training | 4096×4096 pixel SDO magnetograms stored in a multi-channel Zarr dataset (see [SDOMLv2](https://registry.opendata.aws/sdoml-fdl/) for lower-resolution examples) | TBD | TBD |
| Processed – Validation | `s3://nasa-radiant-data/helioai-datasets/hl-arcade/2025-hl-arcade-development-landing/aft/lisa/AFT_Baseline/{YYYY}/{NN}/AFTmap*.h5` | TBD | TBD |
| Results | [Interactive UI](https://arcade.trillium.tech/) | N/A | N/A |

---

## Data Availability Note

The ARCADE project is currently ongoing. As a result:

- Full processed datasets are not yet finalized for public release  
- Model-ready training subsets may be updated  
- Final dataset sizes and access patterns are subject to change  

This page will be updated as artifacts become available. 
-->









