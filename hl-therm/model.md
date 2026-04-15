# 1 Models Description

The Thermospheric Density Continuous Learning challenge produced a production-oriented, continual-learning forecasting system built around an extension of the Karman thermospheric density modeling framework, referred to as Karman-CL (Continuous Learning). Rather than introducing a single new model architecture, the project's primary innovation lies in combining multiple ML model types with a continual-learning orchestration layer, enabling adaptive updating as new data arrive.

At its core, Karman-CL supports a model zoo of candidate thermospheric density predictors, including:
  - Feedforward Neural Networks (FFNNs)
  - Convolutional Neural Networks (CNNs)
  - Long Short-Term Memory (LSTM) models

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

## 1.1 Models Access

Two ML models are included here: a forecasting model and a nowcasting model. The forecasting model is the main product of this work, and is intended to be used for forecasting the thermospheric density. The nowcasting model is a simple model that is not intended for any use other than for instructional purposes. The model files are provided below, along with a sample dataset for testing purposes. 

Instructions for accessing the following files on Amazon Web Services (AWS) are provided in [Section 2](#2-access-instructions).

### Temporal Fusion Transformer (TFT) Forecasting Model (4.5 MB)
- AWS PATH: `hl-therm/models/karman_tft_forecast_mape_14.936_params_1074865.torch`
- Usage Instructions: Instructions on how to use the TFT model are given in this [colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2024-HL-Thermo-CL/blob/main/public/inference_forecast_example.ipynb).
- Type: Temporal Fusion Transformer — forecasts density using ~7 days of space-weather history
- Architecture: 1,074,865 parameters (LSTMs + multi-head attention + variable selection networks)
- Accuracy: 14.94% MAPE on validation set

### Nowcasting Model (0.1 MB)
- AWS PATH: `hl-therm/models/karman_mlp_nowcast_mape_15.14_params_35585.torch` 
- Usage Instructions: Instructions on how to use the nowcasting model are given in this [colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2024-HL-Thermo-CL/blob/main/public/inference_nowcast_example.ipynb). Note this notebook is for instructional use only.
- Type: Nowcasting MLP — predicts density at the current time using instantaneous features only (no time-series history)
- Architecture: 35,585 parameters — a small feedforward network
- Approach: log_exp_residual — predicts a correction (residual) on top of an exponential atmosphere baseline, in log-space
- Accuracy: 15.14% MAPE on validation set

### Example TFT Inference Data (1 MB)
- AWS PATH: `hl-therm/models/sample_inputs_tft.pt`
- Description: This file is a PyTorch dictionary containing a sample of everything needed to run TFT inference, shown in the table below. The first three keys (rows) are the direct **model inputs**. The rest is **metadata** for evaluation and denormalization. All values are already preprocessed (scaled/normalized) and ready to feed directly into the TFT. `N` is the number of samples, in this case `N=100`. 


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
You will need to replace `<AWS PATH>` with the path to the data sample you want to download and `<LOCAL PATH>` with the path on your local machine where you want to save the data).

