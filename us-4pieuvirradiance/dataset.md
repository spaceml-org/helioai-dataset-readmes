# 1. Data Description

This dataset is a 14-day multi-viewpoint Extreme Ultraviolet (EUV) imaging set of the Sun, captured by the twin **STEREO Ahead (A)** and **Behind (B)** spacecraft and calibrated to match SDO/AIA's response via **Instrument-to-Instrument (ITI) translation**. The dataset was assembled for the Frontier Development Lab 2022 US 4π EUV Irradiance challenge as the primary training corpus for **SuNeRF** — a Neural Radiance Field (NeRF) model that reconstructs the 3D solar atmosphere from sparse 2D viewpoints.

There are three levels of description available for this dataset:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description in the project [Technical Memorandum](https://helioai.org/dev/artifact/e78110db-de49-4301-86be-fb8188a90d39/details).
- The full source code used to process the data (see the [GitLab Repository](https://gitlab.com/frontierdevelopmentlab/2022-us-4pieuvirradiance/4piuvsun)).

---

## 1.1 Data Composition

The dataset contains four EUV channels from each of two viewpoints (STEREO-A and STEREO-B) spanning **2012-08-24 to 2012-09-06**, at ~hourly cadence.

| Property | Value |
|---|---|
| Instrument | STEREO/SECCHI EUVI (twin spacecraft, A and B) |
| Time range | 2012-08-24T00:00 to 2012-09-06T22:00 |
| Cadence | ~1 hour |
| Channels | 171, 195, 284, 304 Å |
| Spatial resolution | Native EUVI (2048×2048), ITI-converted to AIA resolution |
| Format | FITS (`.fits`), one file per (timestamp, spacecraft) |
| File naming | `<ISO-timestamp>_<A|B>.fits` (e.g., `2012-08-24T07:00:00_A.fits`) |
| Total size | 141 GB |
| File count | 1,452 (363 per channel: 184 STEREO-A + 179 STEREO-B) |
| Unique timestamps per channel | 200 (some hours have only A or only B) |

### EUVI Channels (Physical Meaning)

| Wavelength (Å) | Corresponding AIA channel | Physical Interpretation |
|---|---|---|
| 171 | 171 Å | Quiet corona (~0.6 MK) |
| 195 | 193 Å | Corona + coronal holes |
| 284 | 211 Å | Active region corona |
| 304 | 304 Å | Chromosphere / transition region |

> Note: Native EUVI wavelengths (195, 284) are mapped to the closest AIA equivalents (193, 211) by the ITI calibration step. The on-disk filenames use the EUVI wavelengths.

### Why these dates

The 2012-08-24 → 2012-09-06 window was chosen because:
- STEREO-A and STEREO-B were separated by ~120° from Earth, giving wide-baseline triangulation when combined with SDO at L1 — the geometry that makes 3D reconstruction tractable.
- The window contains a notable filament eruption (2012-08-31), captured by all three viewpoints, which is used as the canonical test case for the SuNeRF eruption model variant.

---

## 1.2 Data Processing Pipeline

### Source data
STEREO/SECCHI EUVI Level 1 (or higher) FITS files from each spacecraft. STEREO-A and STEREO-B Level 1 EUVI data are available from the [STEREO Science Center](https://stereo-ssc.nascom.nasa.gov/).

### ITI calibration

EUVI and AIA observe similar coronal plasma but with different optical responses, point-spread functions, and sensitivities. To train a multi-viewpoint NeRF that treats all images interchangeably, EUVI frames are **passed through an Instrument-to-Instrument (ITI) neural-network translator** that maps them to AIA-equivalent image statistics. The ITI library is available at <https://github.com/RobertJaro/InstrumentToInstrument>.

The pipeline:

1. **Download** STEREO EUVI Level 1 FITS frames for 2012-08-24 → 2012-09-06.
2. **Standardize** image geometry (solar disk centered, north up) and pointing.
3. **ITI-translate** each EUVI frame to an AIA-equivalent radiometric scale using the published ITI EUVI→AIA model.
4. **Resample** to the SuNeRF training resolution (handled at training time, not in the published files).
5. **Persist** as `.fits` files preserving the ITI-converted pixel data and the original WCS / observation metadata, named `<ISO-timestamp>_<A|B>.fits`.

The pipeline code is in `s4pi/data/convert_stereo_to_sdo.py` and `s4pi/data/convert_stereo_to_sdo_full.py` in the source repository.

### Paired SDO/AIA data

SuNeRF training also uses SDO/AIA frames at the matching timestamps as the third viewpoint (L1 perspective). The SDO/AIA data are *not* redistributed here — they are publicly available via SDOML v2 or directly from [JSOC](http://jsoc.stanford.edu/).

---

## 1.3 Technical Specifications & Dataset Structure

| Feature | Specification |
|---|---|
| File format | FITS (Flexible Image Transport System) |
| Header preservation | Original STEREO WCS + ITI provenance keywords |
| Pixel data | ITI-converted (AIA radiometric scale) |
| File size | ~91–106 MB per frame |
| Compression | None |

```text
📂 us-4pieuvirradiance/data/stereo_2012_08/
├── 171/                                    36 GB, 363 files
│   ├── 2012-08-24T00:00:00_A.fits
│   ├── 2012-08-24T00:00:00_B.fits
│   ├── 2012-08-24T01:00:00_A.fits
│   └── ...
├── 195/                                    36 GB, 363 files
│   └── ...
├── 284/                                    36 GB, 363 files
│   └── ...
└── 304/                                    36 GB, 363 files
    └── ...
```

To read FITS files in Python:
```python
from sunpy.map import Map
s_map = Map('2012-08-24T00:00:00_A.fits')
s_map.peek()
```


## 1.4 Companion Models

The models trained on this dataset (SuNeRF base, eruption, ensembles, transfer-learned variants) are published alongside under `s3://nasa-radiant-data/helioai-datasets/us-4pieuvirradiance/models/`. See the companion [model information](https://helioai.org/dev/artifact/6bfca3d1-a16e-47be-9c33-81d6f3852385/details), and the end-to-end [Colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/helioai-notebooks/blob/main/us-4pieuvirradiance/sunerf_tutorial.ipynb) for a runnable inference example.


## 1.5 SPASE Input Dataset Links

[STEREO/SECCHI EUVI](https://spase-metadata.org/SMWG/Observatory/STEREO.html)
[STEREO Science Center](https://stereo-ssc.nascom.nasa.gov/)

# 2. Access Instructions

Data is stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-4pieuvirradiance/data/stereo_2012_08/
```

Downloading files is done by e.g.:
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-4pieuvirradiance/data/stereo_2012_08/ <LOCAL PATH> --recursive
```
You will need to replace `<LOCAL PATH>` with the path on your local machine where you want to save the data.

| Data Product | AWS Path | Size | Download time (@100 Mbps) |
|---|---|---|---|
| Full dataset (all channels) | `us-4pieuvirradiance/data/stereo_2012_08/` | 141 GB | ~3.5 h |
| Single channel (171, 195, 284, or 304) | `us-4pieuvirradiance/data/stereo_2012_08/<channel>/` | 36 GB | ~50 min |

# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *create* the data products. These can be found in the [GitLab Repository](https://gitlab.com/frontierdevelopmentlab/2022-us-4pieuvirradiance/4piuvsun).
2. Requirements for *using* the data products (e.g., reading FITS, training SuNeRF).

| Component | Minimum (read/inspect) | Recommended (training) |
|---|---|---|
| **CPU** | Any modern CPU | 16+ cores |
| **RAM** | 8 GB | 64 GB |
| **GPU** | Not required | NVIDIA, ≥24 GB VRAM |
| **Storage** | 36 GB (one channel) | 141 GB (full dataset) + working space |

# 4. Citation

If you use this dataset in your research, please cite the SuNeRF paper:

> R. Jarolim, B. Tremblay, A. Muñoz-Jaramillo, K. Battams, A. Jungbluth, M. C. M. Cheung, A. Pal, M. Indaco, K. Bingham, S. Pelletier, K. Albanitis, C. Tylor, D. Coronel, T. Y. Chen, R. Galvez, P. J. Wright, S. Kim. *SuNeRF: 3D Reconstruction of the Solar EUV Corona Using Neural Radiance Fields*. The Astrophysical Journal, 2024. https://doi.org/10.3847/2041-8213/ad12d2

Please also cite the underlying STEREO/SECCHI EUVI instrument paper and the ITI translation method:

> Howard, R. A., et al. *Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI)*. Space Sci. Rev. 136, 67–115 (2008).
> Jarolim, R., et al. *Instrument-to-instrument translation: Instrumental advances drive restoration of solar observation series via deep learning*. A&A, 2024.
