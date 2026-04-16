
<!-- # 2 Dataset Description


There are three levels of description available for this dataset:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](https://helioai.org/dev/artifact/89d1911b-7803-44e7-b792-076edb2dc5ed/details)).
- The full source code used to process the data and create the models (see the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-GeoCL/)).

## Project summary

This project forecasts geomagnetic perturbations at ground stations worldwide using two complementary ML models:

- SHEATH: An MLP that translates solar disk imagery (SDO) into solar wind parameter predictions at L1, providing multi-day advance warning of incoming conditions
- DAGGER-CL: A GRU that takes real-time solar wind measurements (ACE/DSCOVR) and predicts magnetic field perturbations (dBe, dBn) at ~535 ground stations with ~30-minute lead time

Together they form a two-stage forecasting pipeline: SHEATH offers early situational awareness from the Sun itself, while DAGGER-CL provides high-fidelity, station-level nowcasts once the solar wind is measured in situ. The dataset includes the raw inputs, processed training data, and trained model weights for both components. -->


## 1 Models Description

The Geoeffectiveness Continual Learning challenge (GEO-CLOAK) produced a two-stage forecasting system for geomagnetic perturbations. The pipeline combines:

1. **SHEATH** — predicts solar-wind conditions at L1 from solar-disk observations, giving multi-day lead time
2. **DAGGER-CL** — predicts ground magnetic perturbations from in-situ solar-wind conditions with shorter lead time but higher local fidelity

Together, these models provide an operational “Sun-to-ground” workflow:
**SDO imagery → SHEATH → L1 solar-wind forecast → DAGGER-CL + real-time L1 inputs → station perturbations → global maps**.

| Model | Purpose | Main input | Main output | Typical lead time |
| :--- | :--- | :--- | :--- | :--- |
| SHEATH | Forecast solar-wind parameters from solar observations | SDO-derived 26-feature vector | Bx, By, Bz, Vx, density, temperature, plus predictive mean and standard deviation | Multi-day |
| DAGGER-CL | Forecast ground magnetic perturbations | L1 solar wind + geomagnetic indices + lookback window | Station-level dBe, dBn, dBz | ~30 minutes |

## 1.2 Models Access

The SHEATH and DAGGER-CL model files are provided below, together with example data for testing.

Instructions for accessing the following files on Amazon Web Services (AWS) are provided in [Section 2](#2-access-instructions).

### SHEATH model

| Item | AWS path | Approx. size | Description |
| :--- | :--- | :--- | :--- |
| model weights | hl-geo/models/ | 300 KB | Trained SHEATH checkpoint |
|  | sheath_latest.pth |  |  |
| example data | hl-geo/models/examples/ | 30 KB | Sample inputs for testing SHEATH |

**Model type**
- multi-layer perceptron (MLP) regression model

**Input**
- 26 engineered solar features derived from 12 SDO channels

**Output**
- 7 solar-wind target variables at L1:
  - `Bx`
  - `By`
  - `Bz`
  - `Vx`
  - `density`
  - `temperature`
  - predictive uncertainty summary


### SHEATH input / output details

**Inputs**
| Input feature group | Count | Units | Description |
| :--- | :--- | :--- | :--- |
| coronal-hole pixel count | 1 | pixels | Count of coronal-hole pixels near central meridian |
| active-region pixel count | 1 | pixels | Count of active-region pixels near central meridian |
| coronal-hole emission by channel | 12 | channel-dependent intensity units | Total signal in coronal-hole mask for each channel |
| active-region emission by channel | 12 | channel-dependent intensity units | Total signal in active-region mask for each channel |

**Outputs**
| Output field | Units | Description |
| :--- | :--- | :--- |
| Bx | nT | IMF x-component |
| By | nT | IMF y -component |
| Bz | nT | IMF z-component |
| vx | km/s | Solar-wind radial speed |
| density | $\mathrm{cm}^{\wedge}-3$ | Proton density |
| temperature | K | Ion / proton temperature |
| mean / std | same units as target | Predictive mean and standard deviation used for uncertainty-aware downstream use |

Notes
- `Vy` and `Vz` are assumed to be zero in the operational coupling used here.
- Dynamic pressure and clock angle are derived downstream from the predicted quantities.

### DAGGER-CL model
| Item | AWS path | Approx. size | Description |
| :--- | :--- | :--- | :--- |
| trained models / checkpoints | hl-geo/models/ DAGGER_CL/ | 120 MB | Trained DAGGER-CL weights and related artifacts |

**Model type**
- GRU-based recurrent model with linear output layers
- continual-learning extensions include replay / resampling and EWC-based regularization
- uncertainty is estimated using deep ensembles

**Input**
- solar-wind measurements at L1
- geomagnetic indices
- 90-minute lookback context
- disturbed-time-focused training windows

**Output**
- station-level ground magnetic perturbations:
  - `dBe`
  - `dBn`
  - `dBz`

Post-processing
- station predictions are passed to a Gaussian-process interpolation module to produce global geomagnetic maps.
- 
# 2 Access Instructions 

Models are stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-geo/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request <AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the data sample you want to download (see table) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

# 3 System Requirements

1. To use the SHEATH model, any modern computer will do (no GPU necessary)
2. The DAGGER-CL model is part of a more complex pipeline, so the requirements depend on how you want to use it. See the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-GeoCL/) for details. 
