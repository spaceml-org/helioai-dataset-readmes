# 1 Dataset Description

The Frontier Development Lab FDL-X Heliolab 2024 **Thermospheric Density Continuous Learning (Thermo-CL)** challenge was designed to push thermospheric density forecasting from a strong, static ML benchmark toward an operational, adaptive forecasting system. The motivation was the rapid growth of satellites and debris in low Earth orbit, where uncertainty in thermospheric density translates directly into uncertainty in drag, orbit prediction, conjunction assessment, and reentry timing. Earlier FDL Karman work had already shown that data-driven models can outperform traditional empirical density models, but this challenge focuses on the next step: continuously updating the forecasting system as new space weather and density data arrive, while preserving historical skill and enabling near-real-time use.

The Thermo-CL dataset is the backbone for Karman-CL -- a continual-learning version of the Karman framework that combines live data ingestion, cloud storage, and transformer-based forecasting models -- and is therefore designed for scientific understanding of thermospheric response, as well as for operational ML deployment. It is a **time-series, physics-informed dataset** that captures the relationship between solar forcing, geomagnetic activity, and thermospheric response, consisting of two main components:
 - [Raw data](#12-raw-data): observational solar, geomagnetic, and orbital inputs
 - [Processed data](#11-processed-data):  time-aligned, machine-learning-ready datasets used for training and inference

In addition to the high-level summary of this dataset presented below, a detailed description may be found in the project [Technical Memorandum](https://helioai.org/dev/artifact/04b6c417-c722-484c-a668-9426bbbb0cd7/details); and the full source code used to process the data in the project [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-hl-thermo-cl).

# 1.1 Processed Data

The processed dataset is a machine-learning-ready, operationally updated forecasting table/tensor stream that aligns thermospheric density targets with live-ingested space weather drivers over long historical windows. Unlike a static benchmark dataset, the processed data are explicitly part of the Karman-CL operational loop: new data are ingested, compared against the existing data distribution, and either passed directly to the current model for live inference or used to trigger model retraining if the distribution has shifted. This means the processed data product is both a forecasting dataset and a data-assimilation/continual-learning substrate.

The raw data (described below) are processed into a ML-ready, structured dataset according to the following recipe, involving cleaning, filtering, applying quality standards, and transforming the raw measurements into a format that can be used for model training.
 
## Processing pipeline:

<!-- - Ingest new space weather driver data through a live ingestion data pipeline.
 - Preprocess and store those data in a cloud environment for ML readiness.

 - Align them with precise orbit determination (POD)-derived thermospheric density targets.
 - Construct model-ready forecasting samples using a historical window and a chosen lead-time/forecast window. 
      - historical window: ~60,000 minutes
 	  - forecast window: 100 minutes.
 - Run the continual-learning trigger:
  	- if the incoming distribution matches the historical one, do live inference,
  	- otherwise retrain candidate models in the ML model zoo and replace underperforming top-K models if warranted.
 - Validate on held-out intervals and major storm cases such as the May 2024 Gannon superstorm.-->


- **Cleaning**
  - Replace missing or fill values with NaN
  - Remove corrupted or invalid records

- **Resampling**
  All input data streams are resampled to a **common cadence (5–60 minutes depending on source)** to ensure all variables are defined on a consistent time grid.

- **Time alignment**
  - All datasets are aligned to a **single reference timestamp**
  - Solar and geomagnetic inputs are synchronized using interpolation or nearest-neighbor matching
  - No explicit propagation delay is applied; instead, **the model learns physical lag relationships**

- **Feature engineering**
  - Derived features include:
    - geomagnetic indices (Kp, ap, Dst)
    - solar flux proxies (F10.7)
    - temporal variables (local time, day-of-year)

- **Scaling**
  - Features are normalized (z-score or similar)
  - Scaling parameters are stored for reproducibility


The resulting processed Thermo-CL dataset is a **time-aligned tabular dataset**, where each row represents a **single timestamp**, where all input variables (solar, geomagnetic, and orbital) are aligned to that time, and the target corresponds to the thermospheric density at that same time.

Instructions for accessing the following list of processed datasets on Amazon Web Services (AWS) are provided in Section 2.

### OMNIWEB data (3.1 GB)
- AWS PATH: `hl-therm/processed_data/physical-drivers-processed/OMNIWEB/{YYYY}/{SUBSET}_omni_{YYYY}_{MM}.parquet`
- Format: tabular (pandas dataframe)
- Available Years: 2000-2025
- Available subsets: `indices`, `magnetic_field`, `solar_wind_velocity`

### SOHO data (0.5GB)
- AWS PATH: `hl-therm/processed_data/physical-drivers-processed/SOHO/{YYYY}/{YY}_{MM}_{DD}_v4.00.parquet`
- Format: tabular (pandas dataframe)
- Available Years: 2000-2025

### NRLMSISE-00 data (4.6 GB)
- AWS PATH: `hl-therm/processed_data/physical-drivers-processed/nrlmsise00_time_series.csv`
- Format: tabular (pandas dataframe)

### GOES data (0.7 GB)
- AWS PATH: `hl-therm/processed_data/satellite-data-processed/GOES/{YYYY}/goes_irradiance_{YYYY}_{WAVELENGTH}nm.parquet`
- Format: tabular (pandas dataframe)
- Available wavelengths: 256, 284, 304, 1175, 1216, 1335, 1405
- Available Years: 2010-2024

### Tudelft data (11 GB)
- AWS PATH: `hl-therm/processed_data/satellite-data-processed/tudelft/version_0{VERSION}/{SATELLITE}_data/`
- Format: tabular (pandas dataframe)
- Available satellites: 
    - `GOCE` (version 1), years 2009-2013
    - `Swarm`(version 1), years 2013-2025
    - `CHAMP`(version 2), years 2000-2010
    - `GRACE-FO` (version 2), years 2018-2025 
    - `GRACE`(version 2), years 2002-2017


### Space Weather Indices & Proxies data (0.2 MB)
- AWS PATH: `hl-therm/processed_data/sw-indices/combined_incides.parquet`
- Format: tabular (pandas dataframe)


## 1.2 Raw Data 

The raw data for this challenge consists of two main source families: thermospheric density targets derived from precise orbit determination (POD) data, and space weather driver data describing the solar and geomagnetic forcing of the upper atmosphere. Karman-CL uses density measurements derived from satellites' POD data as the ground-truth target, because those measurements provide a direct indication of the thermospheric state experienced by spacecraft in low Earth orbit. This makes the target physically meaningful for orbital drag applications rather than a purely modeled proxy. The space weather driver data includes a mixture of upstream space weather inputs from SOHO and OMNIWeb.

Thermospheric density measurements and solar weather information are drawn from publicly available sources (mostly satellite data). These sources are: 

- In-situ density observations from 8 low-Earth-orbit satellites (CHAMP, GOCE, GRACE A/B/C, SWARM A/B/C), provided by TU Delft. 
- GOES EUV Irradiance: Extreme ultraviolet solar irradiance measured by geostationary GOES satellites at 7 wavelengths (25.6–140.5 nm) at 1-minute resolution.
- OMNIWeb Solar Wind & Geomagnetic Data: Interplanetary magnetic field components (Bx, By, Bz), solar wind velocity and plasma properties, and geomagnetic activity indices (AE, AL, AU, SYM/H, etc.) at 1-minute resolution from 2000 onward. 
- SOHO Solar Irradiance: Solar EUV irradiance at 30 nm and 25 nm from the SOHO spacecraft's SEM instrument at 15-second resolution
Space Weather Indices & Proxies: Daily-cadence solar and geomagnetic indices including F10.7, S10.7, M10.7, Y10.7 (solar radio flux and UV/EUV proxies), the Ap geomagnetic index, and dDst/dT (rate of change of the disturbance storm-time index).


# 2 Access Instructions

Data is stored on Amazon Web Services (AWS). Data access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-therm/ 
```

Downloading files is done by e.g. 
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the data sample you want to download (see table) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

| Data Product               | AWS Path    | Size | Download time (@100 Mbps) |
|-----------------------|-------------|-----------|----------|
| Processed             |  `hl-therm/processed_data/`   | 21 GB    | 0.5 hours      |
| Raw      | `hl-therm/raw_data/` | 45 GB     |   1 hour   |


# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *create* the data products. These can be found in the [GitHub Reprository](https://github.com/FrontierDevelopmentLab/2024-hl-thermo-cl).
2. Requirements for *using* the data products
   1. To use the data, any modern computer with sufficient storage is sufficient.
   2. To use the model(s), see the table below

| Component | Minimum |
|-----------|---------|
| **CPU** | Any modern CPU (even 2 cores is fine) |
| **RAM** | 8 GB |
| **GPU** | None — inference runs on CPU |
| **Storage** | ~10 MB for the model weights + storage for inference data|
