# 1. Model Description

<!-- Add a brief description of the model and the challenge it addresses -->

There are three levels of description available for this model:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](<https://drive.google.com/file/d/1ccJgu6uuz_8vGgOAzNdFmL7TmIXQVfBl/view>)).
- The full source code used to process the data and create the models (see the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere>)).
- Instructions on how to use the model(s) are given in this [notebook](<https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere-dataset/blob/main/dataset_example_colab.ipynb>).

The lonosphere-Thermosphere Twin project introduces a modular deep-learning forecasting framework, centered on two primary model families.

## 2.1 IonCast (global TEC forecasting suite)

<!-- Describe the ML models included. For each model, include:
     - Model architecture
     - Purpose (nowcasting, forecasting, classification, etc.)
     - Any caveats on intended use -->

IonCast is a multi-architecture deep-learning framework for global TEC prediction, including:
     - CNN-LSTM encoder-decoder model
     - Encodes TEC maps into latent representations
     - Uses LSTM to model temporal evolution
     - Decodes future TEC maps
     - Graph Neural Network (GNN) model
     - Inspired by GraphCast
     - Operates on spherical meshes
     - Captures global spatial dependencies
     - Spherical Fourier Neural Operator (SFNO)
     - Learns dynamics in frequency space
     - Efficiently models global-scale behavior
     - Handles spherical geometry and periodic structure

These models operate autoregressively, predicting future TEC maps based on past observations and driver variables.

## 2.2 Ionopy (Temporal Fusion Transformer model) 

lonopy is a Temporal Fusion Transformer (TFT) designed for:
     - sparse ionospheric prediction
     - long-range temporal forecasting
     - probabilistic outputs (mean + variance)

It integrates:
     - temporal attention mechanisms
     - static + dynamic feature embeddings
     - multi-source driver inputs

Key modeling innovations
 - Global-scale ML forecasting on a unified multi-modal dataset
 - Integration of:
     - spatial models (CNN, GNN, SFNO)
     - temporal models (LSTM, TFT)
 - Explicit handling of:
     - sparse vs dense observations
     - multi-resolution data
 - Probabilistic forecasting capability

## 2.3 Model Output
1. Global TEC forecasts
     - Predicted TEC maps on a global grid
     - Forecast horizons: minutes to hours
     - Captures:
          - equatorial ionization structure
          - storm-time disturbances
            
2. Probabilistic outputs
     - Mean + variance predictions (lonopy, SFNO variants)
     - Enable uncertainty-aware forecasting
       
3. Time-series forecast sequences
     - Autoregressive predictions across multiple timesteps
     - Capture temporal evolution of ionospheric dynamics
       
4. Evaluation metrics
     - RMSE and MAE vs ground truth (JPL GIM)
     - Performance across:
          - quiet (G0)
          - moderate (G2)
          - severe (G4) conditions

# 2. Access Instructions

Models are is stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/<DATASET_NAME>/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the file or directory you want to download (see below) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

<!-- Add/remove rows as necessary for your project
The ideal case is that within each of these categories, data are uniformly structured.
For example, "processed" may correspond to train/test/validation data, in which we expect a tabular format (consistent column names, different rows) for each training example. 
Different models may have different train/test/validation sets, this can be explained -->
   
# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *create* the model. These can be found in the [GitHub Repository](<LINK_TO_GITHUB_REPO>).
2. Requirements for *using* the model.


| Component | Minimum |
|-----------|---------|
| **CPU** | |
| **RAM** | |
| **GPU** | |
| **Storage** | |
