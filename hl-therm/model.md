# 1 Model Description

The Thermospheric Density Continuous Learning challenge produced a production-oriented, continual-learning forecasting system built around an extension of the Karman thermospheric density modeling framework, referred to as Karman-CL (Continuous Learning). Rather than introducing a single new model architecture, the project's primary innovation lies in combining multiple ML model types with a continual-learning orchestration layer, enabling adaptive updating as new data arrive.

At its core, Karman-CL supports a model zoo of candidate thermospheric density predictors, including:
  - Feedforward Neural Networks (FFNNs)
  - Convolutional Neural Networks (CNNs)
  - Long Short-Term Memory (LSTM) models
    Temporal Fusion Transformer (TFT) forecasting models

These models operate on time-series inputs of:
  - solar and space-weather drivers (e.g., OMNI, SOHO),
  - historical thermospheric density values derived from precise orbit determination (POD) data.

The system evaluates multiple models in parallel and maintains a top-K set of best-performing models, rather than relying on a single fixed architecture.

The defining feature of the challenge is the continual-learning control system, which wraps around the model zoo and governs:
  - Data ingestion and preprocessing (live pipeline)
  - Distribution-shift detection (comparing new data to historical distributions)
  - Retraining triggers
  - Model selection and replacement

The workflow is:
  - New data arrive via live ingestion
  - Data are compared to existing training distribution
  - If no shift → existing models perform inference
  - If shift detected → candidate models retrained
  - If new models outperform → they replace models in the top-K set

This transforms the system from a static ML model into a self-updating forecasting system. 

## 1.1 Model Workflow

<p align="center">
  <img src="https://github.com/spaceml-org/helioai-dataset-readmes/edit/main/hl-therm/thermocl_workflow.png?raw=true" " width="600">
</p>

The Thermo-CL workflow can be summarized as:

**Solar drivers + geomagnetic drivers + satellite density data → processed Thermo-CL dataset → TFT forecast model / nowcasting model → thermospheric density prediction → continual-learning update loop**

## 1.2 Model Summary

| Model | Purpose | Main input | Main output | Role |
| :--- | :--- | :--- | :--- | :--- |
| Temporal Fusion Transformer Forecasting Model | Forecast future thermospheric density | ~7 days of historical space-weather drivers + satellite orbital context + empirical density baseline | Future thermospheric density | Main forecasting model |
| Nowcasting MLP | Estimate current thermospheric density | Instantaneous features only, no time-series history | Current thermospheric density | Instructional / lightweight example model |
| Karman-CL orchestration layer | Manage continual-learning workflow | New observations, model performance metrics, distribution-shift signals | Updated model selection / retraining decisions | Operational control layer |

## 1.3 Model Access

Two ML models are included here: a forecasting model and a nowcasting model. The forecasting model is the main product of this challenge, and is intended to be used for forecasting the thermospheric density. The nowcasting model is a simple model that is not intended for any use other than for instructional purposes. The model files are provided below, along with a sample dataset for testing purposes. 

Instructions for accessing the following files on Amazon Web Services (AWS) are provided in [Section 2](#2-access-instructions).

<!-- ### Temporal Fusion Transformer (TFT) Forecasting Model (4.5 MB)
- AWS PATH: `hl-therm/models/karman_tft_forecast_mape_14.936_params_1074865.torch`
- Usage Instructions: Instructions on how to use the TFT model are given in this [colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2024-HL-Thermo-CL/blob/main/public/inference_forecast_example.ipynb).
- Type: Temporal Fusion Transformer — forecasts density using ~7 days of space-weather history
- Architecture: 1,074,865 parameters (LSTMs + multi-head attention + variable selection networks)
- Accuracy: 14.94% MAPE on validation set -->

## Temporal Fusion Transformer (TFT) Forecasting Model

| Item | Value |
| :--- | :--- |
| AWS path | `hl-therm/models/karman_tft_forecast_mape_14.936_params_1074865.torch` |
| Approx. size | 4.5 MB |
| Model type | Temporal Fusion Transformer |
| Architecture | LSTMs + multi-head attention + variable selection networks |
| Parameters | 1,074,865 |
| Forecast context | ~7 days of space-weather history |
| Validation accuracy | 14.94% MAPE |
| Usage instructions | [TFT inference Colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2024-HL-Thermo-CL/blob/main/public/inference_forecast_example.ipynb) |

The TFT forecasting model is the primary released forecasting model for Thermo-CL. It is designed for multi-horizon time-series forecasting, using recent historical drivers and static orbital features to predict thermospheric density at a future target time.

### TFT model inputs

| Input group | Shape / type | Units | Description |
| :--- | :--- | :--- | :--- |
| Static numeric features | `(N, 8)` | mixed | Satellite orbital and periodic time features, including altitude, latitude, longitude sine/cosine, day-of-year sine/cosine, and time-of-day sine/cosine |
| Historical time-series features | `(N, 100, 25)` | mixed | Approximately 7 days of historical driver data, represented as 100 time steps × 25 space-weather / empirical-model features |
| Future known features | `(N, 1, 1)` | kg/m³ or normalized density units | NRLMSISE-00 density at the forecast target time |
| Normalization dictionary | dict | unitless scaling metadata | Scaling parameters used to normalize model inputs and denormalize predictions |

### Historical time-series feature groups

| Feature group | Count | Units | Description |
| :--- | :--- | :--- | :--- |
| OMNI indices | 6 | mixed / index units | Geomagnetic and solar-wind index-style driver variables |
| OMNI magnetic field | 3 | nT | Interplanetary magnetic field components |
| OMNI solar wind | 4 | km/s, cm^-3, K, mixed | Solar-wind plasma variables |
| SOHO EUV | 2 | irradiance units | SOHO SEM EUV irradiance channels |
| NRLMSISE-00 densities | 10 | kg/m³ or normalized density units | Empirical atmosphere-model density values used as contextual / baseline inputs |

### TFT model outputs

| Output | Shape / type | Units | Description |
| :--- | :--- | :--- | :--- |
| Predicted density | `(N,)` or model-dependent tensor | kg/m³ after denormalization | Forecast thermospheric density at the target time |
| Ground truth density | `(N,)` in example file | kg/m³ | Observed / target thermospheric density used for evaluation |
| NRLMSISE-00 baseline | `(N,)` in example file | kg/m³ | Empirical model baseline density for comparison |

<!-- ### Nowcasting Model (0.1 MB)
- AWS PATH: `hl-therm/models/karman_mlp_nowcast_mape_15.14_params_35585.torch` 
- Usage Instructions: Instructions on how to use the nowcasting model are given in this [colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2024-HL-Thermo-CL/blob/main/public/inference_nowcast_example.ipynb). Note this notebook is for instructional use only.
- Type: Nowcasting MLP — predicts density at the current time using instantaneous features only (no time-series history)
- Architecture: 35,585 parameters — a small feedforward network
- Approach: log_exp_residual — predicts a correction (residual) on top of an exponential atmosphere baseline, in log-space
- Accuracy: 15.14% MAPE on validation set -->

## Nowcasting Model

| Item | Value |
| :--- | :--- |
| AWS path | `hl-therm/models/karman_mlp_nowcast_mape_15.14_params_35585.torch` |
| Approx. size | 0.1 MB |
| Model type | MLP nowcasting model |
| Architecture | Small feedforward network |
| Parameters | 35,585 |
| Approach | `log_exp_residual` |
| Validation accuracy | 15.14% MAPE |
| Usage instructions | Instructional nowcasting Colab notebook |

The nowcasting model predicts thermospheric density at the current time using instantaneous features only, with no historical time-series context. It is intended primarily as a lightweight instructional model, not as the primary forecasting product.

The `log_exp_residual` approach means the model predicts a correction, or residual, on top of an exponential atmosphere baseline in log-space. This makes the model easier to train because it does not need to learn the full density profile from scratch; it learns how observations differ from a physically motivated baseline.

### Nowcasting model inputs

| Input group | Units | Description |
| :--- | :--- | :--- |
| Instantaneous solar / geomagnetic features | mixed | Current driver conditions at the target timestamp |
| Satellite / orbital context | km, degrees, unitless periodic features | Satellite position and time-context variables |
| Empirical baseline density | kg/m³ or normalized density units | Baseline density estimate used for residual correction |

### Nowcasting model output

| Output | Units | Description |
| :--- | :--- | :--- |
| Current thermospheric density | kg/m³ after denormalization | Nowcast estimate of thermospheric density at the current timestamp |


<!-- ### Example TFT Inference Data (1 MB)
- AWS PATH: `hl-therm/models/sample_inputs_tft.pt`
- Description: This file is a PyTorch dictionary containing a sample of everything needed to run TFT inference, shown in the table below. The first three keys (rows) are the direct **model inputs**. The rest is **metadata** for evaluation and denormalization. All values are already preprocessed (scaled/normalized) and ready to feed directly into the TFT. `N` is the number of samples, in this case `N=100`. -->

# 1.4 Example TFT Inference Data

The file `sample_inputs_tft.pt` provides a ready-to-run example input package for the TFT forecasting model.

| Item | Value |
| :--- | :--- |
| AWS path | `hl-therm/models/sample_inputs_tft.pt` |
| Approx. size | 1 MB |
| Format | PyTorch dictionary |
| Number of samples | `N = 100` |
| Purpose | Example data for TFT inference and evaluation |

This file is a PyTorch dictionary containing a sample of everything needed to run TFT inference, shown in the table below. The first three keys (rows) are the direct **model inputs**; the remaining keys provide metadata for evaluation and denormalization. All values are already preprocessed (scaled/normalized), and ready to feed directly into the TFT model.


| Key | Shape | Description |
|-----|-------|-------------|
| `static_feats_numeric` | `(N, 8)` | Satellite orbital parameters: altitude, latitude, longitude sin/cos, day-of-year sin/cos, time-of-day sin/cos |
| `historical_ts_numeric` | `(N, 100, 25)` | ~7-day history (100 steps × 100 min) of 25 space-weather features: OMNI indices (6), OMNI magnetic field (3), OMNI solar wind (4), SOHO EUV (2), NRLMSISE-00 densities (10) |
| `future_ts_numeric` | `(N, 1, 1)` | NRLMSISE-00 density at the forecast target time |
| `target` | `(N,)` | Ground truth density in normalized log-space (what the model is trained to predict) |
| `ground_truth` | `(N,)` | Actual thermospheric density in physical units (kg/m³) |
| `nrlmsise00` | `(N,)` | NRLMSISE-00 baseline density (for comparison) |
| `dates` | list of N strings | Timestamp for each sample |
| `normalization_dict` | dict | The scaling parameters (min/max, quantile transforms) used during preprocessing |

# 1.5 How the Processed Data Feed the Models

| Stage | Input | Output | Used by |
| :--- | :--- | :--- | :--- |
| Raw data ingestion | OMNIWEB, SOHO, GOES, TU Delft density, NRLMSISE-00, space-weather indices | Cleaned source-specific time series | Processing pipeline |
| Processing / alignment | Cleaned multi-source data | Time-aligned model-ready tensors / tables | TFT and nowcasting models |
| TFT forecasting model | Static features, historical time-series features, future known NRLMSISE-00 density | Forecast density at target time | Main forecasting output |
| Nowcasting MLP | Instantaneous features + empirical baseline | Current density estimate | Instructional nowcast output |
| Continual-learning controller | New data + performance checks | Retraining / model replacement decisions | Operational pipeline |

# 2 Access Instructions

Models are stored on Amazon Web Services (AWS), and access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-therm/ 
```

Downloading files is done by e.g. 
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
Replace `<AWS PATH>` with the path to the model or sample input file you want to download, and replace `<LOCAL PATH>` with the path on your local machine where you want to save the file.

