# 1. Model Description

<!-- Add a brief description of the model and the challenge it addresses -->

There are three levels of description available for this model:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](<https://drive.google.com/file/d/1ccJgu6uuz_8vGgOAzNdFmL7TmIXQVfBl/view>)).
- The full source code used to process the data and create the models (see the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere>)).
- Instructions on how to use the model(s) are given in this [notebook](<https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere-dataset/blob/main/dataset_example_colab.ipynb>).

The lonosphere-Thermosphere Twin project introduces a modular deep-learning forecasting framework, centered on two primary model families.

## 1.1 IonCast (global TEC forecasting suite)

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

## 1.2 Ionopy (Temporal Fusion Transformer model) 

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

## 1.3 Model Output
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

## 1.4 Model Training and Validation

### Processed Data Splitting Strategy
The processed and aligned data product that is input to these models is structured and queried by time. To ensure proper model validation and mitigate data leakage (where portions of the same geomagnetic storm event are scattered across training and validation sets), a novel physics-based classification algorithm was used to divide the entire time interval into sub-intervals associated with a specific storm flag. The classification criteria uses a simple threshold on the Kp time series to identify periods of enhanced geomagnetic activity. 

This criteria is formalized into the Mestici scale (Table 1), which takes into account not only the intensity of Kp (using the NOAA G-levels) but also the duration of the event period. This event catalog is used to identify and exclude the full duration of geomagnetic storm events from the training set, dedicating these critical periods entirely to model validation and testing of out-of-sample performance.

| Mestici Scale | NOAA G-Level (Kp) | Duration (hours) |
|---------------|-------------------|------------------|
| G0H ℓ | Kp < 5 (Calm) | ℓ |
| G1H ℓ | 5 ≤ Kp < 6 (Minor) | ℓ |
| G2H ℓ | 6 ≤ Kp < 7 (Moderate) | ℓ |
| G3H ℓ | 7 ≤ Kp < 8 (Strong) | ℓ |
| G4H ℓ | 8 ≤ Kp < 9 (Severe) | ℓ |
| G5H ℓ | Kp ≥ 9 (Extreme) | ℓ |

**Table 1**: Mestici scale of geomagnetic storms. The scale combines NOAA G-levels (defined by Kp) with storm duration ℓ in hours. For example, G2H6 indicates an event that reached the G2 level lasting at least 6 hours.

To train in a computationally efficient manner, the IonCast LSTM and GNN models train on every 256th sequence of 2 hours (sequences skip 2.66 days between start and end dates).  Test and validation data are removed from the training set for specified dates that span various levels of geomagnetic storms, as defined by the NOAA geomagnetic storm scale (G0, G1, G2, G3, G4, and G5), for a total of 10% of storms at each scale removed from the training set.

Both models are trained to minimize the mean squared error between predictions and ground truth values, for all targets except forcing features. Both the IonCast LSTM and IonCast GNN models are trained with context windows of 8 (3 hours) to predict 1 timestep (15 minutes) ahead, and predict residual targets.

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
