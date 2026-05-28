# 1. Model Description

In an effort to support next-generation space-weather forecasting models and address gaps in current operational frameworks, ﻿the Frontier Development Lab (FDL) Heliolab 2025 lonosphere-Thermosphere Twin research project introduced a novel, modular, deep-learning Total Electron Content (TEC) forecasting model that demonstrates how machine learning (ML) can augment physical understanding of ionospheric variability and advance
operational space weather resilience. The unified model framework centers on two primary model families, IonCast and Ionopy, which together span both dense global maps and sparse observational data, enabling multi-resolution forecasting of ionospheric dynamics.

The following resources are available for in-depth model descriptions and usage instructions:

* [Technical Memorandum](<https://drive.google.com/file/d/1ccJgu6uuz_8vGgOAzNdFmL7TmIXQVfBl/view>) - FDL Heliolab challenge description and results 
* [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere>) - full source code used to create the models and process the input data 

<p align="center">
  <img src="https://github.com/spaceml-org/helioai-dataset-readmes/blob/main/ionosphere-data-public/Ionosphere_Thermosphere_Twin_Model_Workflow.png?raw=true" width="800">
</p>

## 1.1 IonCast (global TEC forecasting suite)

IonCast is a multi-architecture deep-learning framework for global Total Electron Content (TEC) prediction, combining complementary spatial and temporal modeling approaches within a unified forecasting system.

**Model inputs**

| Input Type | Used By | Description |
|-----------|---------|-------------|
| Dense TEC maps | CNN-LSTM, GNN, SFNO | Global TEC maps on a latitude–longitude grid |
| Solar and geomagnetic drivers | CNN-LSTM, GNN, SFNO | Solar wind, IMF, geomagnetic indices, and solar irradiance drivers |
| Auxiliary spatial features | GNN, SFNO | Orbital geometry and quasi-dipole coordinates |
| Forcing variables | GNN | Known future quantities such as Sun/Moon position and orbital geometry |
| State variables | GNN | TEC maps, solar/geomagnetic drivers, and quasi-dipole coordinates predicted autoregressively |


### CNN–LSTM encoder–decoder model

The IonCast LSTM model uses a convolutional encoder–decoder architecture with an LSTM bottleneck to capture both spatial structure and temporal evolution in global TEC maps.

- A convolutional neural network (CNN) encodes each 2D TEC map (180×360) into a compact latent representation  
- A multi-layer LSTM maintains temporal state across sequential inputs, modeling ionospheric dynamics over time  
- A CNN decoder reconstructs future TEC maps from the latent sequence  

In practice, the model uses:
- a six-layer convolutional encoder with circular padding (to respect longitudinal periodicity)  
- a 128-dimensional latent embedding  
- bilinear upsampling and transposed convolutions for reconstruction back to full spatial resolution  

Additional geophysical inputs (e.g., solar and geomagnetic drivers) are incorporated as extra image channels.

The model is trained using:
- batch size: 4  
- dropout: 0.15  
- learning rate: 2e-4  
- weighted loss emphasizing JPLD TEC targets  

**CNN–LSTM output**

- Predicted global TEC maps on a latitude–longitude grid
- Short-term forecast sequences generated autoregressively
- Captures:
  - equatorial ionization structure
  - mid-latitude gradients
  - storm-time disturbances

---

### Graph Neural Network (GNN) model

The IonCast GNN model is based on the GraphCast paradigm, implemented using the NVIDIA PhysicsNeMo framework, and is designed to capture global spatial dependencies on the sphere.

The architecture follows an encoder–processor–decoder structure:

- **Encoder**: maps latitude–longitude grid data onto a spherical icosahedral mesh via graph message passing  
- **Processor**: performs multi-layer message passing across the mesh, enabling long-range spatial interactions  
- **Decoder**: maps predictions back from the mesh to the latitude–longitude grid  

Key modeling features include:

- Representation of the ionosphere on a **multi-resolution spherical mesh**  
- Message passing across up to **32-hop neighborhoods**, enabling global-scale coupling  
- Use of **six mesh levels and six processor layers**  

A key distinction in the GNN formulation is the separation between:

- **Forcing variables** (known at all times, e.g., orbital geometry, Sun/Moon position)  
- **State variables** (TEC maps, solar/geomagnetic drivers, quasi-dipole coordinates)  

During both training and inference:
- forcing variables are provided as known inputs at all timesteps  
- non-forcing variables are predicted autoregressively  

The model operates as follows:
- consumes a context window of past observations (up to time *t*)  
- predicts future states (*t+1 → t+k*) without access to future ground truth  
- concatenates forcing features at all forecast steps to maintain physical consistency  

Training configuration:
- batch size: 1  
- dropout: 0.15  
- learning rate: 3e-4  
- weighted loss emphasizing JPLD TEC targets  


**GNN output**

- Predicted global TEC maps mapped back from the spherical mesh to the latitude–longitude grid
- Autoregressive forecasts for non-forcing variables
- Physically informed rollout using known forcing channels at future timesteps
- Captures global spatial dependencies through spherical mesh message passing

---

### Spherical Fourier Neural Operator (SFNO)

The SFNO model provides a complementary approach by learning ionospheric dynamics in the spectral domain:

- operates in spherical harmonic / Fourier space  
- efficiently captures global-scale behavior  
- naturally handles spherical geometry and periodic structure  

This formulation is particularly well-suited for modeling large-scale ionospheric variability and wave-like dynamics.

**SFNO output**

- Global TEC forecast fields
- Probabilistic forecast variants where available
- Efficient representation of global-scale ionospheric dynamics in frequency space

### Forecasting capability

IonCast models are designed for autoregressive forecasting of global ionospheric state evolution.

Key forecasting capabilities include:

- short-term TEC forecasting on global latitude–longitude grids
- iterative rollout across future timesteps
- integration of multi-modal solar, geomagnetic, and spatial drivers
- forecasting across both quiet and storm-time conditions
- physically informed prediction using forcing variables and spherical geometry

The combination of CNN, GNN, and SFNO architectures enables the framework to capture:

- local spatial structure
- long-range global coupling
- temporal evolution
- large-scale spectral dynamics

This multi-model approach provides complementary forecasting strengths across different ionospheric regimes and spatial scales.


### Performance and interpretability

IonCast is designed to support both operational forecasting and scientific analysis of ionospheric dynamics.

Interpretability arises through:

- explicit separation of forcing and state variables
- physically meaningful spatial representations
- spherical mesh and spectral-domain modeling approaches
- evaluation across geomagnetic storm regimes

Performance is assessed using:

- RMSE and MAE against JPL Global Ionospheric Maps (GIM)
- storm-stratified validation using the Mestici scale
- temporal generalization to unseen geomagnetic events

These evaluation strategies ensure that models are tested not only on average reconstruction quality, but also on robustness during operationally significant space weather conditions.


## 1.2 Ionopy (Temporal Fusion Transformer model) 

Ionopy is a Temporal Fusion Transformer (TFT)-based forecasting model designed to address a complementary problem to IonCast: predicting ionospheric Total Electron Content (TEC) from **sparse, irregular GNSS observations** rather than dense global maps.

Where IonCast focuses on spatially complete global representations, Ionopy is optimized for **data-limited, operational scenarios**, where observations are incomplete, unevenly distributed, and influenced by heterogeneous drivers.

**Model input**

| Input Type | Description |
|-----------|-------------|
| Sparse TEC observations | Point-based TEC measurements from GNSS receivers |
| Solar irradiance drivers | Solar irradiance inputs, including EUV-related information |
| Geomagnetic indices | Driver variables describing magnetospheric forcing |
| Auxiliary geophysical drivers | Additional contextual inputs used to support temporal forecasting |
| Static and dynamic embeddings | Encoded representations of fixed and time-varying features |


**Model output**

| Output Type | Description |
|-------------|-------------|
| Point-based TEC forecasts | Predicted TEC values at GNSS receiver locations |
| Mean TEC prediction | Central forecast estimate for future TEC values |
| Predictive uncertainty | Variance / uncertainty estimates associated with the forecast |
| Multi-horizon forecasts | Predictions generated across multiple future timesteps |
| Temporal attribution | Attention-based interpretability identifying influential timesteps and drivers |

These outputs enable:
- localized forecasting
- validation against real-world GNSS measurements
- uncertainty-aware operational decision making
- forecasting where full global maps are unavailable

---

### Temporal Fusion Transformer (TFT) architecture

Ionopy builds on the Temporal Fusion Transformer architecture, which combines recurrent sequence modeling with attention mechanisms to enable interpretable, multi-horizon forecasting.

Key architectural components include:

- **Sequence encoding with gated recurrent units (GRUs)**  
  - captures local temporal dependencies in TEC and driver time series  

- **Static and dynamic feature embeddings**  
  - integrates heterogeneous inputs, including:
    - GNSS-derived TEC observations  
    - solar irradiance (e.g., EUV)  
    - geomagnetic indices  
    - auxiliary geophysical drivers  

- **Temporal attention mechanism**  
  - identifies which past timesteps and features are most relevant for prediction  
  - provides interpretability through attention weights  

- **Multi-horizon forecasting head**  
  - produces predictions across future time windows in a single forward pass  

This architecture enables Ionopy to model the **nonlinear coupling between solar forcing, geomagnetic activity, and ionospheric response**, which is a key challenge in TEC forecasting.

---

### Sparse-data forecasting paradigm

A defining feature of Ionopy is its ability to operate on **sparse and irregular observational data**, in contrast to grid-based models.

- Inputs consist of **point-based TEC measurements** from GNSS receivers  
- Data are temporally aligned and fused with global driver signals  
- The model learns to infer large-scale ionospheric behavior from partial observations  

This design reflects real operational constraints, where global coverage is incomplete and measurement density varies significantly across regions and time.

---

### Forecasting capability & uncertainty

Ionopy extends beyond deterministic prediction by producing **probabilistic outputs**:

- mean TEC forecast  
- predictive uncertainty (variance)

This enables:
- uncertainty-aware decision making  
- robustness under data gaps and noisy inputs  
- improved interpretability of model confidence  


---

### Performance and interpretability

Experiments across multi-year datasets (2010–2025) demonstrate:

- accurate TEC forecasting up to **24-hour horizons**  
- RMSE values as low as ~3.3 TECU under certain conditions 
- strong dependence on solar EUV irradiance as a predictive driver  

Importantly, the TFT architecture provides **built-in interpretability**:

- attention weights reveal which drivers (e.g., solar vs geomagnetic) dominate predictions  
- temporal attribution highlights critical lead times  

This supports both:
- operational forecasting  
- scientific insight into ionospheric dynamics  

---

# 2. Model Output Summary

The Ionosphere–Thermosphere Twin framework produces complementary outputs from its two primary model families.

| Model Family | Primary Output | Spatial Representation | Forecasting Role |
|-------------|----------------|------------------------|------------------|
| IonCast CNN-LSTM | Global TEC maps | Latitude–longitude grid | Dense global TEC forecasting |
| IonCast GNN | Global TEC maps and autoregressive state forecasts | Spherical mesh mapped back to grid | Global spatial coupling and physics-aware rollout |
| IonCast SFNO | Global TEC fields / probabilistic variants | Spectral / spherical representation | Efficient global-scale dynamics |
| Ionopy TFT | Sparse TEC predictions with uncertainty | GNSS receiver locations / sparse observations | Point-based, probabilistic, multi-horizon forecasting |

Together, these outputs support:
- global ionospheric state prediction
- sparse observational forecasting
- probabilistic uncertainty-aware prediction
- autoregressive time-series forecast sequences
- evaluation across quiet and disturbed geomagnetic conditions


# 3. Model Training and Validation

## 3.1 Shared physics-aware data splitting strategy
The processed and aligned data product that is input to these models is structured and queried by time. To ensure proper model validation and mitigate data leakage (where portions of the same geomagnetic storm event are scattered across training and validation sets), a novel physics-based classification algorithm was used to divide the entire time interval into sub-intervals associated with a specific storm flag. The classification criteria uses a simple threshold on the Kp time series to identify periods of enhanced geomagnetic activity. 

This criteria is formalized into the Mestici scale (Table 1), which takes into account not only the intensity of Kp (using the NOAA G-levels) but also the duration of the event period. This event catalog is used to identify and exclude the full duration of geomagnetic storm events from the training set, dedicating these critical periods entirely to model validation and testing of out-of-sample performance.

## 3.2 Mestici scale for event classification

| Mestici Scale | NOAA G-Level (Kp) | Duration (hours) |
|---------------|-------------------|------------------|
| G0H ℓ | Kp < 5 (Calm) | ℓ |
| G1H ℓ | 5 ≤ Kp < 6 (Minor) | ℓ |
| G2H ℓ | 6 ≤ Kp < 7 (Moderate) | ℓ |
| G3H ℓ | 7 ≤ Kp < 8 (Strong) | ℓ |
| G4H ℓ | 8 ≤ Kp < 9 (Severe) | ℓ |
| G5H ℓ | Kp ≥ 9 (Extreme) | ℓ |

**Table 1**: Mestici scale of geomagnetic storms. The scale combines NOAA G-levels (defined by Kp) with storm duration ℓ in hours. For example, G2H6 indicates an event that reached the G2 level lasting at least 6 hours.

This classification ensures that validation datasets include entire, physically coherent storm events, enabling realistic evaluation of model generalization.

---

## 3.3 Shared training regime

To maintain computational efficiency while preserving temporal diversity:

- Models are trained on sampled sequences (e.g., every 256th 2-hour segment)  
- Training data spans quiet and moderately disturbed conditions  
- Approximately 10% of storm events at each geomagnetic level are withheld for validation/testing  

---

## 3.4 IonCast forecasting setup

All IonCast models are trained under a consistent autoregressive framework:

- Context window: 8 timesteps (~3 hours)  
- Prediction horizon: 1 timestep (15 minutes ahead)  
- Targets: residual TEC values  

The models minimize mean squared error between predictions and ground truth, excluding forcing variables.

---

## 3.5 Ionopy forecasting setup

Ionopy is trained as a sparse, time-series forecasting model using a Temporal Fusion Transformer architecture. Unlike IonCast, which predicts dense global TEC maps, Ionopy is designed to forecast TEC at sparse GNSS observation locations using temporally aligned driver variables and static/dynamic feature embeddings.

Ionopy supports multi-horizon forecasting and probabilistic outputs:

- Inputs: sparse TEC observations, solar irradiance drivers, geomagnetic indices, and auxiliary geophysical features
- Forecast target: point-based TEC predictions at GNSS receiver locations
- Output: mean TEC forecast and predictive uncertainty / variance
- Forecasting mode: multi-horizon time-series prediction
- Interpretability: temporal attention weights and feature attribution

This setup complements IonCast by supporting localized, uncertainty-aware forecasts in settings where full global TEC maps are unavailable or incomplete.

---

## 3.6 Evaluation philosophy & metrics

This training and validation strategy is designed to test:

- temporal generalization (forecasting unseen future states)
- event generalization (performance on unseen storm events)
- robustness across regimes (quiet vs disturbed conditions)

By explicitly separating storm events from training data, the framework ensures that model performance reflects true predictive capability rather than memorization of historical patterns.

Model performance is assessed using both dense global TEC products and sparse observational measurements:

- IonCast models are evaluated primarily against JPL Global Ionospheric Maps (GIM)
- Ionopy models are evaluated primarily against GNSS-derived TEC observations

Primary evaluation metrics include:

- RMSE and MAE
- Performance stratified by geomagnetic conditions:
  - quiet (G0)
  - moderate (G2)
  - severe (G4+)

This regime-based evaluation ensures that models are tested not only on average performance, but also under rare and operationally critical space weather conditions.


# 4. Model Summary

- IonCast → global, map-based forecasting (spatial models)  
- Ionopy → sparse, time-series forecasting (temporal models)  
- Combined → multi-resolution, multi-modal ionospheric forecasting system  

This architecture enables both:
- scientific understanding of ionospheric dynamics  
- operational forecasting under real-world data constraints  

# 5. Access Instructions

The model files and usage instructions may be accessed from the project [Github repository](https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere/).

<!-- Models are is stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/<DATASET_NAME>/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the file or directory you want to download (see below) and `<LOCAL PATH>` with the path on your local machine where you want to save the data. -->

   
# 6. System Requirements

The requirements for *creating* the models may be found in the [Github repository](https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere/). For *using* the models, the following requirements are estimated based on model architecture and dataset characteristics.

| Component | Minimum |
|-----------|---------|
| **CPU** | Modern multi-core CPU |
| **RAM** | 8–16 GB |
| **GPU** | Not required (recommended for GNN/SFNO batch inference) |
| **Storage** | 5–20 GB (model checkpoints + input data subsets) |

<!-- BACKUP
# 1. Model Description

<!-- Add a brief description of the model and the challenge it addresses 

There are three levels of description available for this model:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](<https://drive.google.com/file/d/1ccJgu6uuz_8vGgOAzNdFmL7TmIXQVfBl/view>)).
- The full source code used to process the data and create the models (see the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere>)).
- Instructions on how to use the model(s) are given in this [notebook](<https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere-dataset/blob/main/dataset_example_colab.ipynb>). -->

The lonosphere-Thermosphere Twin project introduces a modular deep-learning forecasting framework centered on two primary model families: IonCast and Ionopy. Together, these models form a unified modeling framework that spans both dense global maps and sparse observational data, enabling multi-resolution forecasting of ionospheric dynamics.

A full decription of the models may be found in the project [Technical Memorandum](<https://drive.google.com/file/d/1ccJgu6uuz_8vGgOAzNdFmL7TmIXQVfBl/view>), and the full source code used to create the models and process the input data in the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere>).


<!-- Describe the ML models included. For each model, include:
     - Model architecture
     - Purpose (nowcasting, forecasting, classification, etc.)
     - Any caveats on intended use -->

## 1.1 IonCast (global TEC forecasting suite)

IonCast is a multi-architecture deep-learning framework for global Total Electron Content (TEC) prediction, combining complementary spatial and temporal modeling approaches within a unified forecasting system.

### CNN–LSTM encoder–decoder model

The IonCast LSTM model uses a convolutional encoder–decoder architecture with an LSTM bottleneck to capture both spatial structure and temporal evolution in global TEC maps.

- A convolutional neural network (CNN) encodes each 2D TEC map (180×360) into a compact latent representation  
- A multi-layer LSTM maintains temporal state across sequential inputs, modeling ionospheric dynamics over time  
- A CNN decoder reconstructs future TEC maps from the latent sequence  

In practice, the model uses:
- a six-layer convolutional encoder with circular padding (to respect longitudinal periodicity)  
- a 128-dimensional latent embedding  
- bilinear upsampling and transposed convolutions for reconstruction back to full spatial resolution  

Additional geophysical inputs (e.g., solar and geomagnetic drivers) are incorporated as extra image channels.

The model is trained using:
- batch size: 4  
- dropout: 0.15  
- learning rate: 2e-4  
- weighted loss emphasizing JPLD TEC targets  

---

### Graph Neural Network (GNN) model

The IonCast GNN model is based on the GraphCast paradigm, implemented using the NVIDIA PhysicsNeMo framework, and is designed to capture global spatial dependencies on the sphere.

The architecture follows an encoder–processor–decoder structure:

- **Encoder**: maps latitude–longitude grid data onto a spherical icosahedral mesh via graph message passing  
- **Processor**: performs multi-layer message passing across the mesh, enabling long-range spatial interactions  
- **Decoder**: maps predictions back from the mesh to the latitude–longitude grid  

Key modeling features include:

- Representation of the ionosphere on a **multi-resolution spherical mesh**  
- Message passing across up to **32-hop neighborhoods**, enabling global-scale coupling  
- Use of **six mesh levels and six processor layers**  

A key distinction in the GNN formulation is the separation between:

- **Forcing variables** (known at all times, e.g., orbital geometry, Sun/Moon position)  
- **State variables** (TEC maps, solar/geomagnetic drivers, quasi-dipole coordinates)  

During both training and inference:
- forcing variables are provided as known inputs at all timesteps  
- non-forcing variables are predicted autoregressively  

The model operates as follows:
- consumes a context window of past observations (up to time *t*)  
- predicts future states (*t+1 → t+k*) without access to future ground truth  
- concatenates forcing features at all forecast steps to maintain physical consistency  

Training configuration:
- batch size: 1  
- dropout: 0.15  
- learning rate: 3e-4  
- weighted loss emphasizing JPLD TEC targets  

---

### Spherical Fourier Neural Operator (SFNO)

The SFNO model provides a complementary approach by learning ionospheric dynamics in the spectral domain:

- operates in spherical harmonic / Fourier space  
- efficiently captures global-scale behavior  
- naturally handles spherical geometry and periodic structure  

This formulation is particularly well-suited for modeling large-scale ionospheric variability and wave-like dynamics.

---

### Autoregressive forecasting framework

All IonCast models operate within a shared autoregressive forecasting paradigm:

- input: context window of 8 timesteps (~3 hours)  
- output: prediction of the next timestep (15 minutes ahead)  
- training objective: minimize mean squared error on residual targets  

This design enables:
- stable multi-step forecasting through iterative rollout  
- consistent integration of multi-modal inputs  
- evaluation across both quiet and disturbed geomagnetic conditions  

## 1.2 Ionopy (Temporal Fusion Transformer model) 

<!-- lonopy is a Temporal Fusion Transformer (TFT) designed for:
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
 - Probabilistic forecasting capability-->


Ionopy is a Temporal Fusion Transformer (TFT)-based forecasting model designed to address a complementary problem to IonCast: predicting ionospheric Total Electron Content (TEC) from **sparse, irregular GNSS observations** rather than dense global maps.

Where IonCast focuses on spatially complete global representations, Ionopy is optimized for **data-limited, operational scenarios**, where observations are incomplete, unevenly distributed, and influenced by heterogeneous drivers.

---

### Temporal Fusion Transformer (TFT) architecture

Ionopy builds on the Temporal Fusion Transformer architecture, which combines recurrent sequence modeling with attention mechanisms to enable interpretable, multi-horizon forecasting.

Key architectural components include:

- **Sequence encoding with gated recurrent units (GRUs)**  
  - captures local temporal dependencies in TEC and driver time series  

- **Static and dynamic feature embeddings**  
  - integrates heterogeneous inputs, including:
    - GNSS-derived TEC observations  
    - solar irradiance (e.g., EUV)  
    - geomagnetic indices  
    - auxiliary geophysical drivers  

- **Temporal attention mechanism**  
  - identifies which past timesteps and features are most relevant for prediction  
  - provides interpretability through attention weights  

- **Multi-horizon forecasting head**  
  - produces predictions across future time windows in a single forward pass  

This architecture enables Ionopy to model the **nonlinear coupling between solar forcing, geomagnetic activity, and ionospheric response**, which is a key challenge in TEC forecasting.

---

### Sparse-data forecasting paradigm

A defining feature of Ionopy is its ability to operate on **sparse and irregular observational data**, in contrast to grid-based models.

- Inputs consist of **point-based TEC measurements** from GNSS receivers  
- Data are temporally aligned and fused with global driver signals  
- The model learns to infer large-scale ionospheric behavior from partial observations  

This design reflects real operational constraints, where global coverage is incomplete and measurement density varies significantly across regions and time.

---

### Probabilistic forecasting capability

Ionopy extends beyond deterministic prediction by producing **probabilistic outputs**:

- mean TEC forecast  
- predictive uncertainty (variance)

This enables:
- uncertainty-aware decision making  
- robustness under data gaps and noisy inputs  
- improved interpretability of model confidence  

---

### Performance and interpretability

Experiments across multi-year datasets (2010–2025) demonstrate:

- accurate TEC forecasting up to **24-hour horizons**  
- RMSE values as low as ~3.3 TECU under certain conditions :contentReference[oaicite:1]{index=1}  
- strong dependence on solar EUV irradiance as a predictive driver  

Importantly, the TFT architecture provides **built-in interpretability**:

- attention weights reveal which drivers (e.g., solar vs geomagnetic) dominate predictions  
- temporal attribution highlights critical lead times  

This supports both:
- operational forecasting  
- scientific insight into ionospheric dynamics  

---

### Role within the Ionosphere–Thermosphere Twin system

Ionopy complements IonCast by addressing a different observational regime:

- **IonCast** → dense, global TEC forecasting (map-based)  
- **Ionopy** → sparse, point-based forecasting (time-series)  

Together, they form a **multi-resolution modeling framework** capable of:

- integrating heterogeneous data sources  
- bridging local observations and global structure  
- supporting both research and operational use cases  

---

### Summary

- TFT → best for **temporal reasoning + sparse data**  
- Ionopy → best for **operational forecasting from real-world observations**  
- Adds **probabilistic + interpretable predictions** to the overall system  
   

## 1.3 Model Output

The Ionosphere–Thermosphere Twin framework produces multiple complementary output types, corresponding to different observational regimes and modeling approaches.

---

### 1. Global TEC forecasts (IonCast)

- Predicted TEC maps on a global latitude–longitude grid  
- Forecast horizons: minutes to hours  
- Captures:
  - equatorial ionization structure  
  - mid-latitude gradients  
  - storm-time disturbances  

These outputs provide a **physically consistent, global view of ionospheric state evolution**, suitable for large-scale analysis and visualization.

---

### 2. Sparse TEC predictions (Ionopy)

- Point-based TEC predictions at GNSS receiver locations  
- Operates on irregular spatial sampling  
- Enables:
  - localized forecasting  
  - validation against real-world measurements  
  - operational deployment where full maps are unavailable  

---

### 3. Probabilistic forecasts

- Mean + variance predictions (Ionopy, SFNO variants)  
- Quantifies uncertainty in model outputs  

Supports:
- uncertainty-aware decision making  
- identification of low-confidence predictions  
- improved robustness under data gaps and noisy inputs  

---

### 4. Time-series forecast sequences

- Autoregressive predictions across multiple timesteps  
- Captures temporal evolution of ionospheric dynamics  
- Enables:
  - short-term forecasting (minutes–hours)  
  - extension to longer horizons via rollout  

---

### 5. Evaluation metrics

Model performance is assessed against JPL Global Ionospheric Maps (GIM) and GNSS-derived TEC observations using:

- RMSE and MAE (primary metrics)  
- Performance stratified by geomagnetic conditions:
  - quiet (G0)  
  - moderate (G2)  
  - severe (G4+)  

This regime-based evaluation ensures that models are tested not only on average performance, but also under **rare and operationally critical space weather conditions**.

## 1.4 Model Training and Validation

### Physics-aware data splitting strategy
The processed and aligned data product that is input to these models is structured and queried by time. To ensure proper model validation and mitigate data leakage (where portions of the same geomagnetic storm event are scattered across training and validation sets), a novel physics-based classification algorithm was used to divide the entire time interval into sub-intervals associated with a specific storm flag. The classification criteria uses a simple threshold on the Kp time series to identify periods of enhanced geomagnetic activity. 

This criteria is formalized into the Mestici scale (Table 1), which takes into account not only the intensity of Kp (using the NOAA G-levels) but also the duration of the event period. This event catalog is used to identify and exclude the full duration of geomagnetic storm events from the training set, dedicating these critical periods entirely to model validation and testing of out-of-sample performance.

### Mestici scale for event classification

| Mestici Scale | NOAA G-Level (Kp) | Duration (hours) |
|---------------|-------------------|------------------|
| G0H ℓ | Kp < 5 (Calm) | ℓ |
| G1H ℓ | 5 ≤ Kp < 6 (Minor) | ℓ |
| G2H ℓ | 6 ≤ Kp < 7 (Moderate) | ℓ |
| G3H ℓ | 7 ≤ Kp < 8 (Strong) | ℓ |
| G4H ℓ | 8 ≤ Kp < 9 (Severe) | ℓ |
| G5H ℓ | Kp ≥ 9 (Extreme) | ℓ |

**Table 1**: Mestici scale of geomagnetic storms. The scale combines NOAA G-levels (defined by Kp) with storm duration ℓ in hours. For example, G2H6 indicates an event that reached the G2 level lasting at least 6 hours.

This classification ensures that validation datasets include entire, physically coherent storm events, enabling realistic evaluation of model generalization.

---

### Training regime

To maintain computational efficiency while preserving temporal diversity:

- Models are trained on sampled sequences (e.g., every 256th 2-hour segment)  
- Training data spans quiet and moderately disturbed conditions  
- Approximately 10% of storm events at each geomagnetic level are withheld for validation/testing  

---

### Forecasting setup

All IonCast models are trained under a consistent autoregressive framework:

- Context window: 8 timesteps (~3 hours)  
- Prediction horizon: 1 timestep (15 minutes ahead)  
- Targets: residual TEC values  

The models minimize mean squared error between predictions and ground truth, excluding forcing variables.

---

### Evaluation philosophy

This training and validation strategy is designed to test:

- **temporal generalization** (forecasting unseen future states)  
- **event generalization** (performance on unseen storm events)  
- **robustness across regimes** (quiet vs disturbed conditions)  

By explicitly separating storm events from training data, the framework ensures that model performance reflects **true predictive capability**, rather than memorization of historical patterns.

---

### System Summary

- IonCast → global, map-based forecasting (spatial models)  
- Ionopy → sparse, time-series forecasting (temporal models)  
- Combined → multi-resolution, multi-modal ionospheric forecasting system  

This architecture enables both:
- scientific understanding of ionospheric dynamics  
- operational forecasting under real-world data constraints  

# 2. Access Instructions

The model files and usage instructions may be accessed from the project [Github repository](https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere/).

<!-- Models are is stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/<DATASET_NAME>/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the file or directory you want to download (see below) and `<LOCAL PATH>` with the path on your local machine where you want to save the data. -->


   
# 3. System Requirements

The requirements for *creating* the models may be found in the [Github repository](https://github.com/FrontierDevelopmentLab/2025-HL-Ionosphere/). For *using* the models, the following requirements are estimated based on model architecture and dataset characteristics.

| Component | Minimum |
|-----------|---------|
| **CPU** | Modern multi-core CPU |
| **RAM** | 8–16 GB |
| **GPU** | Not required (recommended for GNN/SFNO batch inference) |
| **Storage** | 5–20 GB (model checkpoints + input data subsets) | 
-->


