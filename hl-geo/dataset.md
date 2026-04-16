# 1 Dataset Description

The GEO-CLOAK Heliolab 2024 challenge was designed to move geoeffectiveness forecasting toward an operational, adaptive "Sun-to-ground" ML system. GEO-CLOAK therefore builds on the earlier Multiscale Geoeffectiveness FDL challenge result by pushing the combined SHEATH + DAGGER pipeline into a continual-learning and operational-inference setting.

The dataset behind GEO-CLOAK is a coordinated collection of solar, solar-wind, and ground-perturbation data products spanning multiple forecast horizons. SHEATH uses remote solar measurements to forecast solar-wind conditions at L1 with lead times of multiple days, while DAGGER-CL then refines those forecasts using higher-fidelity in-situ L1 data to produce global ground geomagnetic perturbation estimates with lead times of tens of minutes. This means the processed data product is a multi-stage forecasting dataset connecting solar-disk observations, L1 solar-wind parameters, and ground geomagnetic responses into a unified operational pipeline.

This dataset has two main components: [raw data](#12-raw-data) and [processed data](#11-processed-data).

In addition to the high-level summary of this dataset presented below, a detailed description may be found in the project [Technical Memorandum](https://helioai.org/dev/artifact/89d1911b-7803-44e7-b792-076edb2dc5ed/details); and the full source code used to process the data in the project [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-GeoCL/).

## 1.1 Processed Data

The processed dataset for GEO-CLOAK is a two-stage operational forecasting data product. In the long-horizon branch, solar observations are transformed into forecast solar-wind time series via SHEATH. In the short-horizon branch, those predictions are then refined or superseded by in-situ L1 measurements and used by DAGGER-CL to generate global ground geomagnetic perturbation forecasts.

The raw data undergo the following preprocessing steps to create a structured dataset suitable for training and inference:

  - **Cleaning:** Replace sentinel fill-values with NaN; rename columns to a common schema across ACE/DSCOVR/OMNI
  - **Resampling:** 1-hour cadence for SHEATH, 1-minute cadence for DAGGER-CL
  - **Merging:** Temporally align solar wind data with geomagnetic indices using backward-looking `asof` joins
  - **Derived features:** Compute clock angle, trigonometric combinations, dynamic pressure, electric field proxy, dipole tilt (~26 total solar wind features)
  - **SDO processing (SHEATH only):** Backtrack L1 time to solar-observation time; segment AIA 193Å imagery into coronal holes, active regions via GMM; extract mean intensities per region across all 12 channels (~26 SDO features)
  - **Windowing (DAGGER-CL only):** Slice 90-minute lookback windows as input; target is SuperMAG dB components at a future timestamp
  - **Scaling:** Yeo-Johnson power transform + z-score (DAGGER-CL) or z-score only (SHEATH); scalers fitted on training set and saved for inference

Instructions for accessing the following files on Amazon Web Services (AWS) are provided in [Section 2](#2-access-instructions).

### Processed Data Products

```markdown
| Data product         | AWS path                                             | Size   | Role in pipeline.                                      | 
|----------------------|------------------------------------------------------|--------|--------------------------------------------------------|
| ACE                  | hl-geo/processed_data/ACE/                           | 55 GB  | Historical L1 solar-wind inputs                        |
| DSCOVR               | hl-geo/processed_data/DSCOVR/                        | 14 GB  | Historical / near-real-time L1 solarwind inputs        |
| OMNI                 | hl-geo/processed_data/OMNI/omniweb_formatted_2000.h5 | 0.5 MB | Solar-wind labels used for SHEATH training             |
| SuperMAG             | hl-geo/processed_data/SuperMAG/                      | 40 GB  | Ground magnetic perturbation targets                   |
| SDO features         | hl-geo/processed_data/sdo/                           | 2 GB   |  Per-timestamp solar feature tables for SHEATH         |
| SDO embeddings       | hl-geo/processed_data/sdoembeddings/                 | 11 MB  | Embedding-based SHEATH inputs and train/val/test splits|
| SHEATH processed set | hl-geo/processed_data/sheath/                        | 2 GB   | Final training / evaluation inputs for SHEATH          |


###ACE / DSCOVR processed columns
These two products use the same core schema.

```markdown
| Column             | Units | Physical meaning                              |
|--------------------|-------|-----------------------------------------------|
| bt                 | nT    | Total interplanetary magnetic field magnitude |
| bx_gsm             | nT    | IMF x-component in GSM coordinates            |
| by_gsm             | nT    | IMF y-component in GSM coordinates            |
| bz_gsm             | nT    | IMF z-component in GSM coordinates            |
| proton_speed       | km/s  | Bulk solar-wind proton speed                  |
| proton_density     | cm^-3 | Solar-wind proton number density              |
| proton_temperature | K     | Solar-wind proton temperature                 |





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
