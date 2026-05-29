# 1. Models Description

The Frontier Development Lab FDL-X Heliolab 2024 Geoeffectiveness Continual Learning challenge (Geo-CLoak) produced a two-stage forecasting system for geomagnetic perturbations, designed to move geoeffectiveness forecasting toward an operational, adaptive "Sun-to-ground" ML system. 

The Geo-CLoak pipeline combines:

1. **SHEATH** — predicts solar-wind conditions at L1 from solar-disk observations, giving multi-day lead time
2. **DAGGER-CL** — predicts ground magnetic perturbations from in-situ solar-wind conditions with shorter lead time but higher local fidelity

Together, these models provide an operational “Sun-to-ground” workflow:

<!--**SDO imagery → SHEATH → L1 solar-wind forecast → DAGGER-CL + real-time L1 inputs → station perturbations → global maps**-->

<p align="center">
  <img src="https://github.com/spaceml-org/helioai-dataset-readmes/blob/main/hl-geo/geocloak_model_infographic.png?raw=true" width="900">
</p>


| Model | Purpose | Main input | Main output | Typical lead time |
| :--- | :--- | :--- | :--- | :--- |
| SHEATH | Forecast solar-wind parameters from solar observations | SDO-derived 26-feature vector | Bx, By, Bz, Vx, density, temperature, plus predictive mean and standard deviation | Multi-day |
| DAGGER-CL | Forecast ground magnetic perturbations | L1 solar wind + geomagnetic indices + lookback window | Station-level dBe, dBn, dBz | ~30 minutes |

## 1.2 Models 

The SHEATH and DAGGER-CL model files are provided below, together with example data for testing. Instructions for accessing the files on Amazon Web Services (AWS) are provided in Section 2.

### 1.2.1 SHEATH model

| Item | AWS path | Approx. size | Description |
| :--- | :--- | :--- | :--- |
| model weights | hl-geo/models/sheath_latest.pth | 300 KB | Trained SHEATH checkpoint |
| example data | hl-geo/models/examples/ | 30 KB | Sample inputs for testing SHEATH |

Usage instructions are given in this [colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2024-HL-GeoCL/blob/main/public/sheath_inference_quickstart.ipynb).

**Model type**
- multi-layer perceptron (MLP) regression model

**Inputs**: 26 engineered solar features derived from 12 SDO channels
| Input feature group | Count | Units | Description |
| :--- | :--- | :--- | :--- |
| coronal-hole pixel count | 1 | pixels | Count of coronal-hole pixels near central meridian |
| active-region pixel count | 1 | pixels | Count of active-region pixels near central meridian |
| coronal-hole emission by channel | 12 | channel-dependent intensity units | Total signal in coronal-hole mask for each channel |
| active-region emission by channel | 12 | channel-dependent intensity units | Total signal in active-region mask for each channel |

**Outputs**: 7 solar-wind target variables at L1
| Output field | Units | Description |
| :--- | :--- | :--- |
| Bx | nT | IMF x-component |
| By | nT | IMF y -component |
| Bz | nT | IMF z-component |
| vx | km/s | Solar-wind radial speed |
| density | cm^-3 | Proton density |
| temperature | K | Ion / proton temperature |
| mean / std | same units as target | Predictive mean and standard deviation used for uncertainty-aware downstream use |

Notes
- `Vy` and `Vz` are assumed to be zero in the operational coupling used here.
- Dynamic pressure and clock angle are derived downstream from the predicted quantities.

### 1.2.2 DAGGER-CL model

| Item | AWS path | Approx. size | Description |
| :--- | :--- | :--- | :--- |
| trained models / checkpoints | hl-geo/models/ DAGGER_CL/ | 120 MB | Trained DAGGER-CL weights and related artifacts |

**Model type**
- GRU-based recurrent model with linear output layers
- continual-learning extensions include replay / resampling and EWC-based regularization
- uncertainty is estimated using deep ensembles

**Post-processing**
- station predictions are passed to a Gaussian-process interpolation module to produce global geomagnetic maps.

**Inputs**
| Input field group | Units | Description |
| :--- | :--- | :--- |
| Bx, By, Bz | nT | IMF components |
| solar-wind speed | km/s | Upstream solar-wind speed at L1 |
| ion temperature | K | Solar-wind ion temperature |
| geomagnetic indices | mixed | Kp, Hp30, ap30, related geomagnetic context |
| context window | 90 minutes | Historical sequence used by GRU |

**Outputs**: station-level ground magnetic perturbations:
| Output field | Units | Description |
| :--- | :--- | :--- |
| dBe | nT | Predicted eastward ground magnetic perturbation |
| dBn | nT | Predicted northward ground magnetic perturbation |
| dBz | nT | Predicted vertical ground magnetic perturbation |
| ensemble mean | nT | Mean prediction across ensemble members |
| ensemble variance | nT^2 | Predictive variance used as uncertainty estimate |

### 1.2.3 Combined Model Pipeline
| Stage | Input | Output | Used by next stage? |
| :--- | :--- | :--- | :--- |
| SHEATH | SDO-derived solar features | Forecast solar-wind parameters at L1 | Yes |
| DAGGER-CL | Real-time or forecast L1 solar wind + indices | Station perturbations | Yes |
| GP interpolation | Station perturbations + station geometry | Global geomagnetic field estimate | Final product |

## 1.3 Model Training and Evaluation

Geo-CLoak combines two forecasting models operating at different temporal scales. SHEATH provides multi-day forecasts of solar-wind conditions at L1 from remote solar observations, while DAGGER-CL produces short-horizon forecasts of ground magnetic perturbations using in-situ solar-wind measurements and geomagnetic context.

### Training strategy

**SHEATH** is trained as a supervised regression model using paired solar observations and downstream solar-wind measurements. The model learns to map SDO-derived solar features and embeddings to future solar-wind conditions at L1.

Training inputs include:

- coronal-hole and active-region morphology metrics
- multi-channel EUV emission features
- latent solar embeddings

Training targets include:

- IMF components (Bx, By, Bz)
- solar-wind speed
- proton density
- proton temperature

**DAGGER-CL** is trained on historical solar-wind and geomagnetic observations using sequential time-series windows. The model learns the relationship between upstream solar-wind conditions and subsequent ground magnetic perturbations measured by the SuperMAG network.

Training inputs include:

- IMF components
- solar-wind plasma parameters
- geomagnetic activity indices
- 90-minute historical context windows

Training targets include:

- station-level dBe
- station-level dBn
- station-level dBz

The continual-learning framework incorporates replay-based training and Elastic Weight Consolidation (EWC) to improve adaptation while reducing catastrophic forgetting.

### Forecasting configuration

| Model | Forecast target | Typical lead time |
|---------|---------|---------|
| SHEATH | Solar-wind conditions at L1 | Multiple days |
| DAGGER-CL | Ground magnetic perturbations | Tens of minutes |

During operational deployment, SHEATH forecasts may be used as upstream inputs to DAGGER-CL when future L1 observations are unavailable, enabling an end-to-end Sun-to-ground forecasting workflow.

### Evaluation methodology

Evaluation is designed to assess:

- temporal generalization to unseen future periods
- robustness across geomagnetic conditions
- uncertainty estimation quality
- operational forecasting performance at multiple lead times

SHEATH predictions are evaluated against observed L1 solar-wind measurements, while DAGGER-CL forecasts are evaluated against SuperMAG ground magnetometer observations.

Performance assessment includes:

- forecast error metrics
- storm-time case studies
- uncertainty calibration
- lead-time dependent forecast skill

A key objective of Geo-CLoak is operational adaptability. Evaluation therefore extends beyond static benchmark performance to include continual-learning behavior, model stability, and robustness to evolving solar and geomagnetic conditions.

# 2 Access Instructions 

Models are stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-geo/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the data sample you want to download (see table) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

# 3 System Requirements

**SHEATH**
- any modern laptop or workstation should be sufficient for simple inference
- no GPU is required for basic testing
- Python 3.9+ recommended

**DAGGER-CL**

The DAGGER-CL model is part of a more complex pipeline, so the requirements depend on how you want to use it. See the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-GeoCL/) for details. 
  - local inference is lighter-weight than full continual-learning retraining
  - the full continual-learning / near-real-time stack requires more infrastructure, including data ingestion, model registry, and operational orchestration

| Component | Recommendation |
| :--- | :--- |
| CPU | 4+ core CPU |
| RAM | 16 GB recommended |
| GPU | Not required for simple checkpoint inference; useful for training / retraining |
| Storage | Enough storage for checkpoints, processed data subsets, and logs |

# 4 Notes on operational deployment

The full operational GEO-CLOAK framework includes:

- near-real-time inference
- periodic continual-learning retraining
- model registry / storage
- web application serving
- global interpolation of station predictions

These operational components are part of the broader project framework and are more complex than simple local checkpoint inference.



<!-- BACKUP
# 1. Models Description

The Geoeffectiveness Continual Learning challenge (GEO-CLOAK) produced a two-stage forecasting system for geomagnetic perturbations. The pipeline combines:

1. **SHEATH** — predicts solar-wind conditions at L1 from solar-disk observations, giving multi-day lead time
2. **DAGGER-CL** — predicts ground magnetic perturbations from in-situ solar-wind conditions with shorter lead time but higher local fidelity

Together, these models provide an operational “Sun-to-ground” workflow:

**SDO imagery → SHEATH → L1 solar-wind forecast → DAGGER-CL + real-time L1 inputs → station perturbations → global maps**.

<p align="center">
  <img src="https://github.com/spaceml-org/helioai-dataset-readmes/blob/main/hl-geo/geocloak_workflow.png?raw=true" width="600">
</p>

## 1.1 Models Summary

| Model | Purpose | Main input | Main output | Typical lead time |
| :--- | :--- | :--- | :--- | :--- |
| SHEATH | Forecast solar-wind parameters from solar observations | SDO-derived 26-feature vector | Bx, By, Bz, Vx, density, temperature, plus predictive mean and standard deviation | Multi-day |
| DAGGER-CL | Forecast ground magnetic perturbations | L1 solar wind + geomagnetic indices + lookback window | Station-level dBe, dBn, dBz | ~30 minutes |

## 1.2 Models Access

The SHEATH and DAGGER-CL model files are provided below, together with example data for testing. Instructions for accessing the files on Amazon Web Services (AWS) are provided in [Section 2](#2-access-instructions).

### SHEATH model

| Item | AWS path | Approx. size | Description |
| :--- | :--- | :--- | :--- |
| model weights | hl-geo/models/sheath_latest.pth | 300 KB | Trained SHEATH checkpoint |
| example data | hl-geo/models/examples/ | 30 KB | Sample inputs for testing SHEATH |

Usage instructions are given in this [colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2024-HL-GeoCL/blob/main/public/sheath_inference_quickstart.ipynb).

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

**Post-processing**
- station predictions are passed to a Gaussian-process interpolation module to produce global geomagnetic maps.


### DAGGER-CL input / output details

**Inputs**
| Input field group | Units | Description |
| :--- | :--- | :--- |
| Bx, By, Bz | nT | IMF components |
| solar-wind speed | km/s | Upstream solar-wind speed |
| ion temperature | K | Solar-wind ion temperature |
| geomagnetic indices | mixed | Kp, Hp30, ap30, related geomagnetic context |
| context window | 90 minutes | Historical sequence used by GRU |

**Outputs**
| Output field | Units | Description |
| :--- | :--- | :--- |
| dBe | nT | Predicted eastward ground magnetic perturbation |
| dBn | nT | Predicted northward ground magnetic perturbation |
| dBz | nT | Predicted vertical ground magnetic perturbation |
| ensemble mean | nT | Mean prediction across ensemble members |
| ensemble variance | $\mathrm{nT}^2$ | Predictive variance used as uncertainty estimate |

### 1.3 Combined Model Pipeline
| Stage | Input | Output | Used by next stage? |
| :--- | :--- | :--- | :--- |
| SHEATH | SDO-derived solar features | Forecast solar-wind parameters at L1 | Yes |
| DAGGER-CL | Real-time or forecast L1 solar wind + indices | Station perturbations | Yes |
| GP interpolation | Station perturbations + station geometry | Global geomagnetic field estimate | Final product |

# 2 Access Instructions 

Models are stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-geo/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the data sample you want to download (see table) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

# 3 System Requirements

**SHEATH**
- any modern laptop or workstation should be sufficient for simple inference
- no GPU is required for basic testing
- Python $3.9+$ recommended

**DAGGER-CL**

The DAGGER-CL model is part of a more complex pipeline, so the requirements depend on how you want to use it. See the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-GeoCL/) for details. 
  - local inference is lighter-weight than full continual-learning retraining
  - the full continual-learning / near-real-time stack requires more infrastructure, including data ingestion, model registry, and operational orchestration

| Component | Recommendation |
| :--- | :--- |
| CPU | 4+ core CPU |
| RAM | 16 GB recommended |
| GPU | Not required for simple checkpoint inference; useful for training / retraining |
| Storage | Enough storage for checkpoints, processed data subsets, and logs |

# 4 Notes on operational deployment

The full operational GEO-CLOAK framework includes:

- near-real-time inference
- periodic continual-learning retraining
- model registry / storage
- web application serving
- global interpolation of station predictions

These operational components are part of the broader project framework and are more complex than simple local checkpoint inference.
-->


