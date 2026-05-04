# 1. Dataset Description

Using high-resolution observations from the Helioseismic and Magnetic Imager (HMI, Scherrer et. al 2012) and Atmospheric Imaging Assembly (AIA, Lemen et al. 2012) instruments onboard the Solar Dynamics Observatory (SDO), the Frontier Development Lab (FDL) Heliolab 2025 ARCADE system combines physics-based Surface Flux Transport modeling with deep learning to characterize the magnetic evolution of solar active regions that can give rise to flares and coronal mass ejections. The hybrid approach extracts key parameters describing magnetic field structure and integrates uncertainty quantification to produce interpretable forecasts. Results demonstrate accurate short-term predictions of active region emergence up to six hours in advance, providing an important first step for downstream space weather models.

While the ARCADE forecasting model was trained on SDO magnetograms, it was designed to work with following additional SDO/HMI and SDO/AIA data modes: dopplergrams, extreme-ultraviolet images at 171 Å and 304 Å, and continuum-intensity maps. The SDO/HMI exposes continual, full-disk images of the Sun, while the SDO/AIA images the solar atmosphere at 10 different wavelengths, every 10 seconds. 

In addition to the high-level summary of the dataset provided here, a detailed description may be found in the project [Technical Memorandum](<https://drive.google.com/file/d/1fTI2N0cOcLgbzVkk7QRWpNYnPsgFvwb4/view>).
<!-- (see the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2025-HL-Active-Regions/>)).-->

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

<!-- | Raw | `SDO archive FITS images` | | |
| Results | `s3://nasa-radiant-data/helioai-datasets/hl-arcade/2025-hl-arcade-development-features/data/sunpde_output/prod/val_test_*/preds/*png` | | |
| Miscellaneous | `<DATASET_NAME>/miscellaneous/` | | |

# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *create* the data products. These can be found in the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2025-HL-Active-Regions/>).
2. Requirements for *using* the data products


| Component | Minimum |
|-----------|---------|
| **CPU** | |
| **RAM** | |
| **GPU** | 20 GB |
| **Storage** | | -->







