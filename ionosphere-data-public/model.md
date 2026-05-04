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
