# 1. Model Description

The Heliolab 2025 ARCADE system is a physics-informed, hybrid machine learning model designed to forecast the short-term evolution of the solar surface magnetic field, with a primary focus on active region dynamics.

The system combines **multi-modal observational data** from SDO, **first-principles physics** via the Surface Flux Transport (SFT) equation, and **deep learning components** for unresolved processes into a single, end-to-end differentiable forecasting framework.

This enables the model to:
- learn from data  
- enforce physically meaningful evolution  
- produce interpretable and uncertainty-aware forecasts  

In addition the high-level summary of the model presented here, a detailed description may be found in the project ([Technical Memorandum](https://drive.google.com/file/d/1fTI2N0cOcLgbzVkk7QRWpNYnPsgFvwb4/view)). Full source code and trained models will be released upon project completion.

---

## 1.1 Primary model: Physics-informed SFT forecaster

The core ARCADE model is a **hybrid neural–physical architecture** that embeds a differentiable implementation of the Surface Flux Transport (SFT) equation inside a deep learning pipeline.

### Architecture overview

The model consists of four tightly coupled components:

---

### 1. Neural feature extraction

A convolutional neural network (ResNet-style encoder) processes input magnetograms and multi-modal SDO data to extract spatial features.

- Input: full-disk solar images  
- Output: latent feature maps representing magnetic structure  

In the current implementation:

- a ResNet-style convolutional neural network predicts:
  - solar surface flow fields (differential rotation and meridional flow components)
  - flux emergence contributions (source term in the SFT equation)

---

### 2. Differentiable SFT evolution module (PDE)

The physical evolution of the magnetic field is governed by a differentiable SFT model:

- Implements:
  - differential rotation  
  - meridional flow  
  - turbulent diffusion  

- Numerically integrated using:
  - **Neural ODE framework (`torchdiffeq`)**  
  - RK4 time stepping  

This module evolves the magnetic field forward in time:

B(t + Δt) = SFT(B(t), v, η, S)

where:

- v = velocity fields (learned or parameterized)  
- η = diffusion  
- S = flux emergence (learned source term)  
---

### 3. Learned flux emergence (source term)

A neural network predicts the **source term** in the SFT equation:

- captures:
  - unresolved magnetic flux emergence  
  - active region formation  

This is critical because flux emergence is not fully described by classical SFT physics.

---

### 4. Uncertainty quantification head (optional)

An additional neural component estimates **pixel-wise uncertainty**:

- predicts mean + variance  
- trained via negative log-likelihood loss  

Supports:
- confidence-aware forecasting  
- spatial reliability assessment  

---

### End-to-end behavior

The model operates as shown below, with all components trained jointly via backpropagation through the PDE solver.
```text
SDO Observations (HMI + AIA)
        │
        ▼
Multi-modal Data Cube
(magnetograms, EUV, Doppler)
        │
        ▼
CNN / ResNet Encoder
→ Extract magnetic structure
        │
        ▼
Flow + Flux Emergence Networks
→ Predict:
   - velocity fields (DR, MF)
   - source term (flux emergence)
        │
        ▼
Differentiable SFT PDE (Neural ODE)
→ Physical evolution:
   - differential rotation
   - meridional flow
   - diffusion
        │
        ▼
Forecast Magnetogram (t + Δt)
        │
        ├───────────────► Residual Maps
        │
        ├───────────────► Uncertainty Maps
        │
        └───────────────► Physical Parameters (DR/MF)
```

## 1.2 Secondary model: Physical Parameter Estimation

A secondary component estimates physically interpretable parameters of solar surface flows.

Parameters learned:
- Differential rotation coefficients (a, b, c terms)
- Meridional flow parameters

These parameters are learned via gradient-based optimization through the differentiable SFT model during training and are used to:
- validate agreement with known solar physics
- constrain model behavior
- improve interpretability

## 1.3 Model Input

The ARCADE system ingests multi-modal SDO observations:
- Magnetograms (primary physical variable)
- Dopplergrams
- EUV images (171 Å, 304 Å)
- Continuum intensity maps

Inputs are:
- temporally stacked (multiple prior frames)
- co-registered across modalities
- normalized for ML compatibility

## 1.4 Model Output

The ARCADE model produces multiple outputs, reflecting both physical state prediction and diagnostic information.

1. Forecast magnetograms (primary output)
  -Full-disk magnetic field predictions
  -Forecast horizon: ~6 hours
  -Represent future radial magnetic field state

  These outputs serve as the primary forecasting product as well as input to downstream flare and CME prediction systems.

  The [ARCADE interactive demo](https://arcade.trillium.tech/) generates forecasts up to 30 hours into the future of a selected date, based on   the most recent, highly accurate forecasting model.

  <p align="center">
    <img src="https://github.com/spaceml-org/helioai-dataset-readmes/blob/main/hl-arcade/arcade_ui.png?raw=true" width="600">
  </p> 
  
2. Residual and diagnostic maps showing the difference between:
  - prediction and target
  - prediction and input

  Used to:
  - quantify forecast error
- identify spatial structure in model failures
        
3. Pixel-wise uncertainty maps
      - Standard deviation estimates per pixel
      - Represent aleatoric uncertainty in the forecast

     These outputs provide:
      - confidence-aware predictions
      - spatially reliability assessment

4. Learned physical parameter estimates 
      - Differential rotation coefficients
      - Meridional flow parameters

     Used to:
      - validate physical consistency
      - compare against classical solar models
   
5. Learned flux emergence fields
      - Neural estimates of the source term in the SFT equation
      - Capture:
        - unresolved magnetic flux emergence
        - active region formation dynamics


## 1.5 Forecasting Framework

The model is trained to predict the evolution of the magnetic field over short time horizons.

Typical configuration:
- Input: magnetogram at time *t*
- Target: magnetogram at *t+Δt* (e.g., 6-12 hours)
- Time integration:
     - continuous evolution via Neural ODE
     - discrete supervision via MSE or likelihood loss

Training objective:
- minimize mean squared error between predicted and observed fields
- optionally include probabilistic loss (NLL) for uncertainty modeling

## 1.6 Model Availability

The ARCADE project is currently ongoing.
- Model weights are not yet publicly released
- Full training pipelines and inference code will be published upon completion

An interactive demonstration is available:
https://arcade.spaceml.org/app

This interface allows:
- selection of input dates
- generation of forecasts up to 30 hours ahead



<!-- # 2. Access Instructions

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
     
# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *create* the model. These can be found in the [GitHub Repository](<LINK_TO_GITHUB_REPO>).
2. Requirements for *using* the model.


| Component | Minimum |
|-----------|---------|
| **CPU** | |
| **RAM** | |
| **GPU** | |
| **Storage** | | --> 

