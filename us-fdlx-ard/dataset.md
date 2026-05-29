# 1. Data Description

The **SDOMLv2a** dataset is a comprehensive, machine-learning-ready collection of solar observation data produced by the Frontier Development Lab (FDL). Derived from the Solar Dynamics Observatory (SDO) mission, this dataset serves as a cornerstone for advanced solar research and irradiance prediction.

The raw SDO archive spans approximately 20 petabytes. To address computational and storage challenges, this dataset has been curated, calibrated, and reduced to a high-density, managed size of approximately 8 terabytes without losing physical information relevant to Extreme Ultraviolet (EUV) irradiance tasks.

---

The Analysis Ready Data for Extreme Ultraviolet (ARD-EUV) challenge was created to address two major barriers in heliophysics machine learning.

First, the Solar Dynamics Observatory (SDO) produces an enormous volume of scientifically valuable observations, exceeding 20 PB of data across multiple instruments, formats, cadences, and calibration levels. While these observations contain critical information about the solar atmosphere and solar irradiance, the complexity of accessing and preparing the data makes large-scale machine-learning applications difficult.

Second, the failure of the EVE MEGS-A instrument in 2014 resulted in the loss of direct measurements of many important Extreme Ultraviolet (EUV) spectral lines used in space-weather research and upper-atmosphere modeling.

To address these challenges, the ARD-EUV project developed the SDO Scientific Computing Platform and the SDOMLv2a dataset: a large-scale, machine-learning-ready archive of calibrated AIA, HMI, and EVE observations. The dataset standardizes instrument calibration, spatial alignment, temporal synchronization, and storage format while also supporting derived products such as embeddings and Virtual EVE irradiance predictions.

The resulting dataset provides a unified foundation for solar image analysis, irradiance estimation, foundation-model development, and operational space-weather forecasting applications.

## 1.1 Data Composition

The dataset is multifaceted, incorporating multi-channel and multi-instrument sources. It provides a standardized temporal alignment and spatial resolution across three primary instruments:

## AIA (Atmospheric Imaging Assembly)

<!-- **Channels:** 9 spectral channels (94, 131, 171, 193, 211, 304, 335, 1600, 1700 Å). The 4500 Å channel is excluded.  
**Resolution:** Downsampled to **512x512** pixels.  
**Cadence:** 6 minutes.  
**Format:** Converted from Flexible Image Transport System (.fits) to **.zarr** arrays.  -->

| Property | Value |
|--------|------|
| Channels | 94, 131, 171, 193, 211, 304, 335, 1600, 1700 Å |
| Resolution | 512 × 512 pixels |
| Cadence | 6 minutes |
| Format | Converted from Flexible Image Transport System (.fits) to **.zarr** arrays|

> Note: The 4500 Å channel is excluded.  

### AIA Channels (Physical Meaning)

| Wavelength (Å) | Physical Interpretation |
|---------------|------------------------|
| 94 | Hot flare plasma (~6 MK) |
| 131 | Flares + hot corona |
| 171 | Quiet corona (~0.6 MK) |
| 193 | Corona + coronal holes |
| 211 | Active region corona |
| 304 | Chromosphere / transition region |
| 335 | Active region loops |
| 1600 | Upper photosphere / transition region |
| 1700 | Photosphere |

## HMI (Helioseismic and Magnetic Imager)

<!-- **Components:** 3 vector magnetic field components (*Bx, By, Bz*) derived from line-of-sight data.  
**Resolution:** **512x512** pixels.  
**Cadence:** 12 minutes.  
**Alignment:** Spatially aligned with AIA images (same solar disk size and location).  --> 

| Property | Value |
|--------|------|
| Components | Bx, By, Bz (3-vector magnetic field components derived from line-of-sight data)|
| Units | Gauss |
| Resolution | 512 × 512 |
| Cadence | 12 minutes |
| Alignment | Spatially aligned with AIA images (same solar disk size and location) |

## EVE (EUV Variability Experiment)

<!-- **Data:** 39 ion intensity profiles.  
**Historical Data (2010–2014):** Real measurements (excluding Fe XVI-2 due to sensor defects).  Included in EVE_legacy.zarr folder.  
**Post-2014 (not-included):** "Virtual EVE" spectra generated via ML inference to account for the MEGS-A instrument failure.  -->

| Property | Value | 
|--------|------|
| Data | 39 spectral lines (ion intensity profiles) |
| Units | W/m²/nm |
| Time range | 2010–2014 (real measurements, excluding Fe XVI-2 due to sensor defects) |
| Format | `.zarr` |
| Location | `EVE_legacy.zarr` folder | 

> Note: **Pre-2014**: real measured spectra; **Post-2014**: not included ("Virtual EVE" spectra generated via ML inference to account for the MEGS-A instrument failure) 

## 1.2 Data Processing Pipeline

The generation of SDOMLv2a utilizes the **SDO Scientific Computing Platform**, a Google Cloud Platform (GCP) infrastructure designed for massive parallelization and continuous ingestion.

### Ingestion Sources
Data is ingested via a "Trigger Batching" system from three distinct sources, each requiring specific handling:  

<!-- **HelioCloud:** 5000x5000 images (Level 1.5).  
 **JSOC (Raw):** 4096x4096 images (Level 1).  
 **JSOC (Synoptic):** 1024x1024 images (Level 1.5).  -->

| Source | Resolution | Notes |
|------|-----------|------|
| HelioCloud | 5000×5000 | Level 1.5 |
| JSOC Raw | 4096×4096 | Level 1 |
| JSOC Synoptic | 1024×1024 | Level 1.5 |

Legacy data was downloaded via massively parallelized cloud functions, while new data can be maintained via cron jobs (daily for HMI, every 5 minutes for AIA).

### Calibration Standards 

Regardless of the source quality, all data in SDOMLv2a is processed to achieve a uniform **Level 1.5 calibration**.

**AIA Calibration**  

**Routine:** Python SDO/AIA calibration (Barnes et al 2020).  
**Steps:** Pointing correction, registration, and degradation correction.  
**Standardization:** Exposure time normalization and solar disk size standardization are applied before downsampling to 512x512.

**HMI Calibration**

HMI vectors are not natively available and are derived from four raw .fits files: *azimuth, disambig, field,* and *inclination*.  
**Routine:** Custom Python implementation of IDL routines (Gary & Hagyard 1990).  
**Steps:** Azimuth disambiguation → Coordinate transformation (Native to Spherical CCD) → Registration/Rotation.  
**Artifact Removal:** Unlike standard IDL routines which replace NaNs with zeros (introducing artificial "zero magnetic field" artifacts near the limb), our pipeline preserves NaNs. This prevents interpolation errors during resizing.

---

## 1.3 Technical Specifications & Dataset Structure

| Feature | Specification |
| :----- | :--- |
| **File Format** | .zarr (append-only) |
| **Image Resolution** | 512 x 512 |
| Time Alignment | Multi-instrument synchronized |


```text
📂 us-fdlx-ard-sdomlv2a - Latest version of SDOML in .zarr format (2010-2023)
├── AIA.zarr/ - calibrated AIA data in .zarr format
│   └── YEAR/
│       ├── WAVELENGTH/
│       │   ├── .zarray
│       │   ├── .zattrs
│       │   └── ... 
│       └── .zgroup
│ 
├── EVE_legacy.zarr/ - Legacy EVE data in .zarr format
│   ├──SPECTRAL LINE/
│   │   ├── .zarray
│   │   ├── .zattrs
│   │   └── ...
│   ├── Time/
│   │   ├── .zarray
│   │   ├── 0
│   │   └── ... 
│   └── .zgroup
│ 
└── HMI.zarr/ - calibrated HMI data in .zarr format
    └── YEAR/
        ├── By/
        │   ├── .zarray
        │   ├── .zattrs
        │   └── ... 
        ├── Bx/
        │   ├── .zarray
        │   ├── .zattrs
        │   └── ...  
        ├── Bz/
        │   ├── .zarray
        │   ├── .zattrs
        │   └── ...  
        └── .zgroup
```

To read more about .zarr file and how to use them, see the [Zarr user guide](https://zarr.dev/) and [Zarr-Python documentation](https://zarr.readthedocs.io/en/stable/user-guide/).

---

## 1.4 How the Processed Data Feed the Models

| Stage | Input | Output | Used by |
|---------|---------|---------|---------|
| Raw observations | AIA, HMI, EVE | Instrument-specific science products | Calibration pipeline |
| Calibration and standardization | Raw observations | Level 1.5 aligned images and irradiance measurements | SDOMLv2a |
| Dataset synchronization | AIA, HMI, EVE | Time-aligned ML-ready data | Virtual EVE training |
| Feature extraction | AIA image statistics and image tensors | Structured model inputs | Linear and CNN model branches |
| Virtual EVE inference | AIA observations | Predicted EUV irradiance spectra | Space-weather applications |

## 1.5 Derived Data Products

To aid in computational efficiency, the project repository includes code to generate:  

<!-- **Embeddings:** Lower-dimensional representations of the image data.  
**Virtual EVE:** Predicted irradiance values for dates following the 2014 MEGS-A failure.-->

| Product     | Description                                     |
| ----------- | ----------------------------------------------- |
| Embeddings  | Lower-dimensional representations of image data |
| Virtual EVE | ML-predicted irradiance values for dates following the 2014 MEGS-A failure|

---

## 1.5 SPASE Input Dataset Links:

[AIA 94 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV094/PT12S.html)  
[AIA 131 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV131/PT12S.html)  
[AIA 171 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV171/PT12S.html)  
[AIA 193 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV193/PT12S.html)  
[AIA 211 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV211/PT12S.html)  
[AIA 304 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV304/PT12S.html)  
[AIA 335 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV335/PT12S.html)  
[AIA 1600 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/UV1600/PT24S.html)  
[AIA 1700 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/UV1700/PT24S.html)  
[HMI](https://spase-metadata.org/NASA/NumericalData/SDO/HMI/LOS_Magnetogram/PT720S.html)  
[EVE](https://spase-metadata.org/NASA/NumericalData/SDO/EVE/Level1/Version8/PT0.25S.html)

# 2. Access Instructions

Data is stored on Amazon Web Services (AWS). Data access is given via the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```

# 3. Citation

If you use this dataset in your research, please cite the accompanying paper:

> M. Indaco, D. Gass, W. J. Fawcett, R. Galvez, P. J. Wright, and A. Muñoz-Jaramillo. *Virtual EVE: a Deep Learning Model for Solar Irradiance Prediction*. Machine Learning and the Physical Sciences Workshop, NeurIPS 2023. arXiv:2408.17430. https://doi.org/10.48550/arXiv.2408.17430

BibTeX:

```bibtex
@inproceedings{indaco2024virtualeve,
  title         = {Virtual {EVE}: a Deep Learning Model for Solar Irradiance Prediction},
  author        = {Indaco, M. and Gass, D. and Fawcett, W. J. and Galvez, R. and Wright, P. J. and Mu{\~n}oz-Jaramillo, A.},
  booktitle     = {Machine Learning and the Physical Sciences Workshop, NeurIPS 2023},
  year          = {2024},
  eprint        = {2408.17430},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.SR},
  doi           = {10.48550/arXiv.2408.17430},
  url           = {https://arxiv.org/abs/2408.17430}
}
```

Please also consider citing the underlying SDOMLv2 dataset and the original SDO instrument papers (AIA, HMI, EVE) as appropriate.

<!-- 
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
| Processed | `s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/sdomlv2a/*zarr/` | | |
| Raw | [SDO archive] (<http://jsoc.stanford.edu/>) | | |
| Results | `s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/sdomlv2a/*zarr/` | | |



# 2 Dataset Description



This projects uses SDO data from the AIA and HMI instruments, processed into SDOMLv2 format, with 512x512 image resolution and 2-minute cadence, uploaded in a cloud-friendly format. This data is calibrated to Level 1.5 for AIA data (sourced from synoptic images), with HMI data sourced from line-of-sight magnetograms. Reprojection issues from standard HMI processing (NaN values around solar limb) have been corrected and removed from the v2 version of the SDOML dataset. This dataset covers the entirety of solar cycle 24 and is updating on an ongoing basis to cover the current extent of solar cycle 25, making it an ideal training set for machine learning applications that require large spans of time and high cadences.

There are three levels of description available for this dataset:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](<https://drive.google.com/file/d/1WXEELZ1SLRS7wFgTGafXlC8RVgwCCqd9/view?usp=sharing>)).
- The full source code used to process the data and create the models (see the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD>)).


## 2.1 Processed Data



The processed data products created by ARD-EUV are the project's main contribution. Rather than leaving users to pull and preprocess disparate SDO streams themselves, the project builds a cloud-based large-scale data ingestion, processing, and ML platform that produces synchronized, analysis-ready products suitable for both direct scientific analysis and downstream ML training. The team launched SDOMLv2, with analysis-ready data for AIA, HMI, and EVE spanning 2010-2023, and built a cloud-scale SDO computational platform. Therefore, the processed data here are the infrastructure layer that turns the raw SDO archive into a reproducible ML dataset.

Processing steps:

- Ingest raw SDO AIA, HMI, and EVE streams over the 2010-2023 interval.

- Apply AIA auto-calibration / enhancement methods to improve image consistency.

- Build synchronized, cloud-accessible analysis-ready products for AIA, HMI, and EVE.

- Use the processed image products together with EVE irradiance targets to train Virtual 
  EVE Reloaded, an updated EUV spectral irradiance estimator.
       

## 2.2 Raw Data



The raw data for ARD-EUV come primarily from the three major SDO instrument streams that are most relevant for solar irradiance and image-based ML:

- AIA (Atmospheric Imaging Assembly): multi-wavelength solar images, especially in UV and EUV channels.
- HMI (Helioseismic and Magnetic Imager): photospheric magnetic and continuum measurements.
- EVE (Extreme Ultraviolet Variability Experiment): spectral irradiance measurements that provide the target or reference spectrum for Virtual EVE-style learning.

Physically, these three instrument streams are complementary. AIA provides the coronal and chromospheric image context that can be used to infer irradiance; HMI contributes photospheric and magnetic structure; and EVE provides the spectrally resolved irradiance ground truth required to train or validate virtual-instrument models. SDOML v2 includes analysis-ready data for SDO AIA, HMI, and EVE from 2010 until 2023.

Data format and access:
- AIA/HMI: solar image products from the SDO archive included in SDOMLv2
- EVE: irradiance data stream from the SDO/EVE instrument, included in SDOML v2/ARD-EUV 


# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *create* the data products. These can be found in the [GitHub Repository](<LINK_TO_GITHUB_REPO>).
2. Requirements for *using* the data products


| Component | Minimum |
|-----------|---------|
| **CPU** | |
| **RAM** | |
| **GPU** | NVIDIA V100 GPUs were used to train Virtual EVE, with multiple V100s in parallel |
| **Storage** | The raw SDO dataset is ~20 petabytes, reduced to ~30 terabytes by downscaling images from 4096×4096 to 512×512 |


-->
