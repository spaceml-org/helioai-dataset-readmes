# 1. Dataset Description

This dataset corresponds to the **Forecasting Radiation Exposure for Human Space Flight** project (`hl-dosi`) from Heliolab 2024. It supports nowcasting and short-horizon forecasting of the deep-space radiation environment during Solar Proton Events (SPEs) by pairing in-situ dosimeter readings beyond low Earth orbit with co-registered solar imagery and X-ray / proton-flux time series.

In addition to the high-level summary of this dataset provided below, the full source code used to process the data and train the models is contained in the project [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-hl-radiation-ml).

## What a Data Sample Represents

A single training sample is a fixed-length time window (typically several hours of context plus several hours of forecast horizon) at a uniform 15-minute cadence, comprised of:

| Component | Description |
|----------|-------------|
| SDOML-lite frames | 6-channel 512×512 solar images (AIA + HMI) at 15-min cadence |
| GOES-16 XRS | Soft X-ray flux (1–8 Å) |
| GOES-16 SGPS | Integral proton flux > 10 MeV and > 100 MeV |
| RadLab BPD | BioSentinel dose rate (deep-space dosimeter, cislunar) |
| RadLab CRaTER-D1D2 | CRaTER dose rate (lunar orbit, LRO mission) |
| Event label | SPE catalog id and >10 MeV peak flux |

The radiation target variable is BioSentinel BPD dose rate; CRaTER-D1D2 is used as a second deep-space reference. Solar imagery and GOES time-series serve as drivers.

## 1.1 Processed Data

The raw heterogeneous inputs are normalized to a common 15-minute grid and ML-ready formats:

- **GOES XRS / SGPS:** NetCDF level-2 files from NOAA are decoded, the integral proton-flux channels (>10 MeV, >100 MeV) are reconstructed by summing the SGPS differential channels above the threshold, and a per-minute average is written to CSV with a single `datetime` index.
- **SDOML-lite:** SDO/AIA + HMI imagery is reduced to 6 channels (AIA 131, 171, 193, 304 Å and HMI Bx, By) at 512×512 resolution and 15-minute cadence, then sharded into WebDataset tar archives of ~775 MB each. A separate `sdoml-lite-biosentinel` build covers the BioSentinel science window.
- **RadLab:** The community RadLab dose-rate compilation is materialized as a DuckDB database with separate tables for `readings`, `instruments`, `coordinates`, `trajectories`, and `app_metadata`. A 2024-06-25 snapshot is published in both an "original" and a "corrected" form. Per-instrument padded CSVs are also provided for quick prototyping.
- **SDO Latent (sdocore):** A 21504-dimensional VAE embedding of SDOML imagery (FDL-X 2023 Thermospheric Drag Team product) is included for users who want a precomputed compact SDO representation instead of streaming the raw tars.
- **SPE event catalog:** 19 BioSentinel-era events (2023-02 to 2024-05) and 69 CRaTER-era events (2010-08 to 2024-05) are defined in `scripts/events.py` of the GitHub repo, with `date_start` / `date_end` windows that pad each event by 2× the rise time before begin and 6× after maximum. The largest catalogued event is `crater11` (2012-03-04 to 2012-03-15, 6530 pfu).

The repository's `scripts/datasets.py` provides PyTorch `Dataset` wrappers (`GOESXRS`, `GOESSGPS`, `RadLab`, `SDOMLlite`) and a `Sequences` aligner that yields fixed-length windows with all sources synchronized on a 15-minute grid.

## Processed Dataset Schema (Core Training Inputs)

| Source | Shape per step | Units | Description |
|--------|---------------|------|-------------|
| GOES XRS | scalar | W m⁻² | Soft X-ray (1–8 Å) |
| GOES SGPS | (2,) | proton flux units (pfu) | >10 MeV and >100 MeV integral flux |
| RadLab BPD | scalar | µGy/min | BioSentinel BPD dose rate |
| RadLab CRaTER-D1D2 | scalar | µGy/min | LRO/CRaTER dose rate |
| SDOML-lite frame | 6 × 512 × 512 | normalized intensity | AIA 131/171/193/304 + HMI Bx/By |
| SDO latent (optional) | (21504,) | dimensionless | VAE mean+std embedding |

### Solar imagery — SDOML-lite-biosentinel (~397 GB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-dosi/sdoml-lite-biosentinel/`
- Contents: 561 WebDataset tar shards (`sdoml-lite-biosentinel-001.tar` … `sdoml-lite-biosentinel-561.tar`), ~775 MB each. Each shard contains many `<timestamp>.npy` files of shape `(6, 512, 512)` (AIA 131, 171, 193, 304 Å + HMI Bx, By), float32, 15-minute cadence, covering ~2022-11 through 2024-06 (BioSentinel science window).
- Streamable directly with `webdataset` / `torchdata`; per-event subsets are small enough (3–10 shards) to download for inference. The shards covering BioSentinel event `biosentinel07` (2023-07-17 to 2023-07-19, the 620 pfu event used in the demo notebook) are `sdoml-lite-biosentinel-{258,259,260,261}.tar` (~3 GB total).

### GOES-16 X-ray and proton flux (~245 MB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-dosi/goes/`
- Contents:
    - `goes-xrs.csv` (157 MB): GOES-16 EXIS/XRS soft X-ray flux, 1-minute resolution, columns `datetime`, `xrsa` (0.5–4 Å), `xrsb` (1–8 Å). Covers full GOES-16 archive.
    - `goes-sgps.csv` (93 MB): GOES-16 SEISS/SGPS integral proton flux, 1-minute resolution, columns `datetime`, `>10MeV`, `>100MeV`. Reconstructed by summing differential channels p5–p11 (>10 MeV) and p8a–p11 (>100 MeV).

### RadLab dosimeter database (~3.5 GB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-dosi/radlab/`
- Contents:
    - `RadLab-20240625-duck.db` (325 MB) and `RadLab-20240625-duck-corrected.db` (351 MB): DuckDB files with five tables — `readings`, `instruments`, `coordinates`, `trajectories`, `app_metadata`. The "corrected" variant fixes flagged entries (preferred for training).
    - `info.md`: provenance pointer to the upstream Grigorev release.
    - `data_tables/avail_tables.txt`, `data_tables/readings_cadences_variability.csv`: companion metadata.
    - `data_tables/readings_table/readings.csv` (733 MB): full readings table as CSV.
    - `data_tables/readings_table/per_instrument/`, `per_instrument_padded/`: one CSV per instrument; padded variant fills gaps to a uniform cadence.
    - `data_tables/{instruments,coordinates,trajectories,app_metadata}_table/`: the four supporting tables as CSV.
- Key instruments used for `hl-dosi` modeling: **BioSentinel BPD** (90 s cadence, deep space, 2022-11 to 2024-05) as the primary target, **CRaTER-D1D2** (1-hr cadence, lunar orbit, 2009-06 to 2024-06) as a second deep-space reference.

### SDO Latent embeddings — `sdocore` (~24 GB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-dosi/sdocore/`
- Contents:
    - `sdo_core_dataset_21504.h5` (23 GB): HDF5 with `year, month, day, hour, minute, latent_mean, latent_std` for 268,400 samples (6-minute cadence, 2010-05 to 2018-12). Latent dimension 21,504.
    - `sdo_core_dataset_21504.pt` (400 MB): pretrained NVAE encoder/decoder weights (31.7M params).
    - `sdo_core_dataset_21504.txt`: details and citation (FDL-X 2023 Thermospheric Drag Team).
- Useful as a drop-in low-dimensional replacement for the full SDOML-lite tars when raw imagery is not needed.

## 1.2 Raw Data

Raw upstream sources are publicly available and not republished in this bucket beyond the processed forms above:

- **GOES-16 EXIS/XRS, SEISS/SGPS:** NOAA NCEI [Space Weather data](https://www.ncei.noaa.gov/products/goes-r-series). Download scripts: `data_scripts/get_goes_xrs.py`, `data_scripts/get_goes_sgps.py` in the GitHub repo.
- **SDO/AIA, SDO/HMI:** Public via JSOC; ML-ready 6-channel SDOML-lite build is produced by the team.
- **RSTN solar radio bursts (8 frequencies):** NOAA NGDC. Download scripts: `data_scripts/get_rstn_radio.py`, `data_scripts/process_rstn_radio.py`. Not used by the published model but available for downstream work in the repo.
- **RadLab dose-rate compilation:** Original release by K. Grigorev. See `radlab/info.md` for the source link.
- **NOAA SPE catalog:** [Solar Proton Events Affecting the Earth Environment](https://www.ngdc.noaa.gov/stp/space-weather/interplanetary-data/solar-proton-events/SEP%20page%20code.html) — used to define event windows in `scripts/events.py`.

# 2. Access Instructions

Data is stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use the CLI are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). The bucket is public, so `--no-sign-request` works without an AWS account; objects are also reachable over HTTPS at `https://nasa-radiant-data.s3.amazonaws.com/helioai-datasets/hl-dosi/<key>` (used by the Colab demo notebook).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-dosi/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the data sample you want to download (see table) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

| Data Product | AWS Path | Size | Download time (@100 Mbps) |
|-------------|----------|------|---------------------------|
| Solar imagery (SDOML-lite-biosentinel) | `s3://nasa-radiant-data/helioai-datasets/hl-dosi/sdoml-lite-biosentinel/` | ~397 GB | ~9 hours |
| GOES X-ray + proton flux | `s3://nasa-radiant-data/helioai-datasets/hl-dosi/goes/` | ~245 MB | ~20 seconds |
| RadLab dosimeter DB | `s3://nasa-radiant-data/helioai-datasets/hl-dosi/radlab/` | ~3.5 GB | ~5 minutes |
| SDO Latent embeddings | `s3://nasa-radiant-data/helioai-datasets/hl-dosi/sdocore/` | ~24 GB | ~32 minutes |
| Single-event demo subset (`biosentinel07`) | `sdoml-lite-biosentinel-{258..261}.tar` + `goes/*` + `radlab/RadLab-20240625-duck.db` | ~4.5 GB | ~6 minutes |
| Full processed bundle | `s3://nasa-radiant-data/helioai-datasets/hl-dosi/` | ~425 GB | ~10 hours |

# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *create* the data products. These can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-hl-radiation-ml).
2. Requirements for *using* the data products:

| Component | Minimum |
|-----------|---------|
| **CPU** | Modern multi-core CPU |
| **RAM** | 8 GB for the time-series CSVs + DuckDB; 16 GB+ recommended when streaming SDOML-lite shards alongside model inference |
| **GPU** | Not required for data access; recommended for training (the published model was trained on a single NVIDIA A100) |
| **Storage** | ~5 GB for the single-event demo subset; ~30 GB for time series + SDO latents (no raw imagery); ~425 GB for the full processed bundle |
