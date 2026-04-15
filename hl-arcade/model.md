# 1 Access Instructions

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


# 2 Model Description

<!-- Add a brief description of the model and the challenge it addresses -->

There are three levels of description available for this model:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](<https://drive.google.com/file/d/1fTI2N0cOcLgbzVkk7QRWpNYnPsgFvwb4/view>)).
- Work on this project is still ongoing; when completed, the full source code used to process the data and create the models will be linked here. 
<!--- The full source code used to process the data and create the models (see the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2025-HL-Active-Regions/>)).-->

The ARCADE challenge produced a physics-informed, hybrid deep-learning forecasting model for predicting the short-term evolution of the solar surface magnetic field, with a primary focus on active region dynamics. The system combines data-driven learning from multi-modal SDO observations with a differentiable implementation of the Surface Flux Transport (SFT) equation, forming an end-to-end trainable pipeline.

## 2.1 Primary model: Physics-informed SFT forecaster

<!-- Describe the ML models included. For each model, include:
     - Model architecture
     - Purpose (nowcasting, forecasting, classification, etc.)
     - Any caveats on intended use -->

<!-- If applicable, link to inference notebooks or usage examples 
Instructions on how to use the model(s) are given in this [notebook](<LINK_TO_NOTEBOOK>).-->

The central model is a hybrid architecture consisting of:

- Neural feature extraction (CNN/ResNet-style encoder)
- Differentiable SFT evolution module (PDE-based)
- Learned flux emergence (source-term) model
- Uncertainty quantification head

This structure enables the model to learn solar magnetic field evolution directly from observations while enforcing physically meaningful dynamics.

    
## 2.2 Secondary model (supporting component)

A secondary model estimates solar Differential Rotation (DR) and Meridional Flow (MF) parameters using optimization within the SFT equation.

This model:
- fits physically interpretable parameters
- validates consistency with known solar physics

## 2.3 Model Input

Both the primary and secondary ARCADE mdoels were designed to ingest multi-modal SDO data, including:
- Magnetograms (primary physical variable)
- Dopplergrams
- EUV images (171 Å, 304 Å)
- Continuum intensity

These inputs are temporally stacked, co-registered, and normalized into a unified data representation.

## 2.3 Model Output

1. Forecast magnetograms (primary output)
     - Predicted full-disk magnetograms at ~6-hour lead time
     - Represent the future radial magnetic field state
     - Generated as time-ordered image sequences
       
      Interactive UI: https://arcade.spaceml.org/app

These serve as the operational output for active region tracking and downstream flare and CME prediction models.
  
2. Residual/error maps (diagnostic outputs), showing differences between prediction and target, and prediction and input, used to quantify model performance and identify spatially structured forecast errors.
        
3. Pixel-wise uncertainty maps
      - Standard deviation estimates per pixel
      - Represent aleatoric uncertainty in the forecast

     These outputs provide:
      - confidence-aware predictions
      - spatially resolved reliability estimates

4. Learned physical parameter estimates (supporting outputs)
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

