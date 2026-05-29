# 1. Model Description

NASA's Solar Dynamics Observatory (SDO) carries the EVE (EUV Variability Experiment) instrument, which measures solar spectral irradiance. In 2014, a capacitor failure destroyed the MEGS-A module, eliminating measurements of 14 key extreme ultraviolet (EUV) spectral lines.

**Virtual EVE** restores this lost measurement capability using deep learning. By training on four years of overlapping SDO/AIA imagery and EVE measurements (2010–2014), the model learns to predict what MEGS-A *would* have measured — effectively virtualizing the broken instrument without any hardware repair.

In the ARD-EUV challenge, the Virtual EVE functionality was extended into a more production-ready, deep-learning pipeline trained in supervised fashion using Huber loss, with PyTorch and PyTorch Lightning used for model development and training. The model takes analysis-ready AIA/HMI image products as input and produces inferred irradiance-related outputs.

**The following resources are available for in-depth model descriptions and usage instructions:**

* [Technical Memorandum](https://drive.google.com/file/d/1WXEELZ1SLRS7wFgTGafXlC8RVgwCCqd9/view) - full description and results of the FDL Heliolab challenge that produced the model
* [GitHub Repository](https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE) - source code for downloading input data and using with the associated models
* [Colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE/blob/main/public/virtual_eve_tutorial.ipynb) - instructions on how to use the model 

<p align="center">
  <img src="https://github.com/nbonaventura/helioai-dataset-readmes/blob/main/us-fdlx-ard/infographics_ard_euv_model.png?raw=true" width="900">
</p>

---

The Virtual EVE architecture combines two complementary approaches. A linear component captures the strong global relationship between overall solar brightness and irradiance, while a convolutional neural network (CNN) captures complex spatial structures such as active regions, coronal holes, and flare-related morphology. Together these components form a hybrid architecture that balances physical interpretability with predictive performance.

## 1.1 Model Pipeline

The model learns the mapping: SDO AIA images → EUV spectral irradiance (EVE). This enables reconstruction of missing irradiance measurements after the 2014 EVE MEGS-A failure.

| Stage | Input | Output | Description |
|------|------|--------|-------------|
| Linear branch | AIA summary statistics | Irradiance estimate | Captures global intensity relationships |
| CNN branch | Full AIA images (512×512×9) | Irradiance estimate | Captures spatial structure |
| Hybrid combination | Linear + CNN outputs | Final irradiance prediction | Weighted sum with learnable parameter |


## 1.2 Model Inputs

| Input | Shape | Units | Description |
|------|------|------|-------------|
| AIA images | 512×512×9 | intensity (DN/s normalized) | Multi-channel solar images |
| AIA statistics | 18 features | normalized | Per-channel mean + std |
| (Optional) HMI | 512×512×3 | Gauss | Magnetic field input (if used in extensions) |


## 1.3 Model Outputs

| Output | Dimension | Units | Description |
|-------|----------|------|-------------|
| EVE spectral lines | 38 values | W/m²/nm | Solar EUV irradiance at specific ion wavelengths |


## 1.4 Virtual EVE Hybrid Model Checkpoint File (42 MB)

  - **AWS PATH:** `us-fdlx-ard/virtual-eve/AIA_MEGS_20_30_epochs_36min.ckpt`
  - **Usage Instructions:** See [colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE/blob/main/public/virtual_eve_tutorial.ipynb)
  - **Type:** Hybrid Linear + CNN - predicts solar EUV irradiance from AIA imagery

The model checkpoint file, containing a snapshot of model weights and parameters during training, is a PyTorch dictionary containing:

| Key | Description |
|-----|-------------|
| `model` | The trained `HybridIrradianceModel` instance (ready for inference) |
| `normalizations` | Per-channel AIA normalization statistics (`mean`, `std`) used during training |
| `sci_parameters` | Scientific metadata: `aia_wavelengths` (list of 9 input channels) and `eve_ions` (list of 38 output spectral lines) |


### Architecture

The model contained in this checkpoint is a two-component hybrid model combining a linear and a CNN model. The **linear model** is a single, fully-connected layer that maps per-channel mean and standard deviation statistics (18 features from 9 AIA channels) to irradiance predictions. The **CNN model** is an EfficientNet-B5 backbone (~30M parameters) that extracts spatial features from the full 512×512 pixel images.

- **Linear model**
  - Fully connected layer
  - Input: 18 statistical features
  - Output: irradiance

- **CNN model**
  - EfficientNet-B5 (~30M parameters)
  - Input: full-resolution images
  - Output: irradiance

The outputs are combined as `O_total = O_linear + λ * O_CNN`, where λ is a learnable blending parameter.

---

### Training
The linear model is trained first (20 epochs), then the CNN is activated while the linear weights are frozen (30 epochs), ensuring stable convergence. 

  - Two-phase training:
    1. Linear model (20 epochs)
    2. CNN added (30 epochs)
  - Linear weights frozen during phase 2
  - Loss function: **Huber Loss**

The model is trained on the [SDOML v2 dataset](https://sdoml.org), which is publicly available on the same S3 bucket:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/sdomlv2a/
```
## 1.5 Model Training and Evaluation

The Virtual EVE model is trained as a supervised image-to-spectrum regression system. The objective is to learn the relationship between solar observations from AIA and the corresponding EUV irradiance measurements recorded by EVE during the period when both instruments were simultaneously operational.

### Training strategy

Training uses synchronized observations from the SDOMLv2a dataset covering the period 2010–2014.

Training inputs include:

* Nine calibrated AIA wavelength channels
* Per-channel image statistics (mean and standard deviation)
* Optional HMI magnetic-field products for experimental configurations

Training targets include:

* 38 EVE spectral irradiance channels spanning both MEGS-A and MEGS-B observations

The hybrid architecture is trained in two stages:

1. Linear component training
   * CNN frozen
   * Learns the dominant global irradiance relationships

2. CNN refinement stage
   * Linear component frozen
   * EfficientNet CNN activated
   * Learns residual spatial information not captured by the linear model

Training uses Huber Loss, providing robustness to outliers while maintaining sensitivity to prediction accuracy.

### Forecasting configuration

| Parameter | Value |
|------------|---------|
| Input cadence | Typically 6–120 minutes |
| Input channels | 9 AIA channels |
| Target outputs | 38 EVE spectral lines |
| CNN backbone | EfficientNet (B3/B5/B7 tested) |
| Loss function | Huber Loss |
| Training approach | Two-stage hybrid training |

### Evaluation methodology

Evaluation is designed to assess:

* irradiance reconstruction accuracy
* performance across individual spectral lines
* sensitivity to input cadence
* contribution of AIA versus HMI observations
* robustness across quiet and active solar conditions

Model performance is evaluated using held-out EVE observations collected during the operational lifetime of MEGS-A.

Performance assessment includes:

* Huber Loss
* Mean Absolute Error (MAE)
* Relative Absolute Error (RAE)
* channel-by-channel spectral analysis
* predicted-versus-observed irradiance comparisons

A key finding of the project is that AIA imagery alone provides nearly all of the predictive information required for irradiance reconstruction, while HMI magnetograms provide little additional benefit for most forecasting scenarios.

# 2 Access Instructions

Models are stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/virtual-eve/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/virtual-eve/AIA_MEGS_20_30_epochs_36min.ckpt <LOCAL PATH>
```
You will need to replace `<LOCAL PATH>` with the path on your local machine where you want to save the model checkpoint.

# 3 System Requirements

There are two sets of system requirements:
1. Requirements to *create* the model. These can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE).
2. Requirements for *using* the model (inference only).

| Component | Minimum |
|-----------|---------|
| **CPU** | Any modern CPU (inference runs on CPU) |
| **RAM** | 4 GB |
| **GPU** | Not required (CPU-only inference supported) |
| **Storage** | ~50 MB for model checkpoint |

The model can also be run for free on [Google Colab](https://colab.research.google.com/github/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE/blob/main/public/virtual_eve_tutorial.ipynb).





<!--# 1 Model Description

NASA's Solar Dynamics Observatory (SDO) carries the EVE (EUV Variability Experiment) instrument, which measures solar spectral irradiance. In 2014, a capacitor failure destroyed the MEGS-A module, eliminating measurements of 14 key extreme ultraviolet (EUV) spectral lines.

**Virtual EVE** restores this lost measurement capability using deep learning. By training on four years of overlapping SDO/AIA imagery and EVE measurements (2010–2014), the model learns to predict what MEGS-A *would* have measured — effectively virtualizing the broken instrument without any hardware repair.

In the ARD-EUV challenge, the Virtual Eve functionality was extended into a more production-ready, deep-learning pipeline trained in supervised fashion using Huber loss, with PyTorch and PyTorch Lightning used for model development and training. The model takes analysis-ready AIA/HMI image products as input and produces inferred irradiance-related outputs. A detailed description of the model may be found in the project [Technical Memorandum](https://drive.google.com/file/d/1WXEELZ1SLRS7wFgTGafXlC8RVgwCCqd9/view).

The model files are provided below, along with sample training data for testing purposes. Instructions for accessing the files on Amazon Web Services (AWS) are provided in [Section 2](#2-access-instructions).

## 1.1 Virtual EVE Hybrid Model (42 MB)

- **AWS PATH:** `us-fdlx-ard/virtual-eve/AIA_MEGS_20_30_epochs_36min.ckpt`
- **Usage Instructions:** Instructions on how to use the model are given in this [colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE/blob/main/public/virtual_eve_tutorial.ipynb).
- **Type:** Hybrid Linear + CNN — predicts solar EUV irradiance from AIA imagery
- **Architecture:** A two-component hybrid model combining:
  - A **Linear model** — a single fully-connected layer that maps per-channel mean and standard deviation statistics (18 features from 9 AIA channels) to irradiance predictions.
  - A **CNN model** — an EfficientNet-B5 backbone (~30M parameters) that extracts spatial features from the full 512×512 px images.
  - The outputs are combined as: $O_{\text{total}} = O_{\text{linear}} + \lambda \cdot O_{\text{CNN}}$, where $\lambda$ is a learnable blending parameter.
- **Training:** Two-phase training procedure — the linear model is trained first (20 epochs), then the CNN is activated while the linear weights are frozen (30 epochs), ensuring stable convergence. Loss function: Huber Loss.
- **Input:** 9 SDO/AIA EUV/UV channels (94, 131, 171, 193, 211, 304, 335, 1600, 1700 Å) at 512×512 px resolution.
- **Output:** 38 EVE ion spectral lines — covering both MEGS-A and MEGS-B wavelength ranges.

### Model Checkpoint Contents

The model checkpoint file, containing a snapshot of model weights and parameters during training, is a PyTorch dictionary containing:

| Key | Description |
|-----|-------------|
| `model` | The trained `HybridIrradianceModel` instance (ready for inference) |
| `normalizations` | Per-channel AIA normalization statistics (`mean`, `std`) used during training |
| `sci_parameters` | Scientific metadata: `aia_wavelengths` (list of 9 input channels) and `eve_ions` (list of 38 output spectral lines) |

### Training Data

The model is trained on the [SDOML v2 dataset](https://sdoml.org), which is publicly available on the same S3 bucket:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/sdomlv2a/
```

# 2 Access Instructions

Models are stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/virtual-eve/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/virtual-eve/AIA_MEGS_20_30_epochs_36min.ckpt <LOCAL PATH>
```
You will need to replace `<LOCAL PATH>` with the path on your local machine where you want to save the model checkpoint.

# 3 System Requirements

There are two sets of system requirements:
1. Requirements to *create* the model. These can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE).
2. Requirements for *using* the model (inference only).

| Component | Minimum |
|-----------|---------|
| **CPU** | Any modern CPU (inference runs on CPU) |
| **RAM** | 4 GB |
| **GPU** | Not required (CPU-only inference supported) |
| **Storage** | ~50 MB for model checkpoint |

The model can also be run for free on [Google Colab](https://colab.research.google.com/github/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE/blob/main/public/virtual_eve_tutorial.ipynb). -->


# 4 Reference

Indaco, M., Gass, D., Fawcett, W. J., Galvez, R., Wright, P. J., & Muñoz-Jaramillo, A. (2024). *Virtual EVE: a Deep Learning Model for Solar Irradiance Prediction.* [arXiv:2408.17430](https://arxiv.org/abs/2408.17430)


<!-- BACKUP
# 1 Access Instructions

Models are stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/virtual-eve/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/virtual-eve/AIA_MEGS_20_30_epochs_36min.ckpt <LOCAL PATH>
```
You will need to replace `<LOCAL PATH>` with the path on your local machine where you want to save the model checkpoint.


# 2 Model Description

There are three levels of description available for this model:
- A high-level summary (this document) for users to quickly become familiar with the model.
- A detailed description (see the [Technical Memorandum](https://helioai.org)).
- The full source code used to process the data and train the model (see the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE)).

## Project Summary

NASA's Solar Dynamics Observatory (SDO) carries the EVE (EUV Variability Experiment) instrument, which measures solar spectral irradiance. In 2014, a capacitor failure destroyed the MEGS-A module, eliminating measurements of 14 key extreme ultraviolet (EUV) spectral lines.

**Virtual EVE** restores this lost measurement capability using deep learning. By training on four years of overlapping SDO/AIA imagery and EVE measurements (2010–2014), the model learns to predict what MEGS-A *would* have measured — effectively virtualizing the broken instrument without any hardware repair.

## 2.1 Virtual EVE Hybrid Model (42 MB)

- **AWS PATH:** `us-fdlx-ard/virtual-eve/AIA_MEGS_20_30_epochs_36min.ckpt`
- **Usage Instructions:** Instructions on how to use the model are given in this [colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE/blob/main/public/virtual_eve_tutorial.ipynb).
- **Type:** Hybrid Linear + CNN — predicts solar EUV irradiance from AIA imagery
- **Architecture:** A two-component hybrid model combining:
  - A **Linear model** — a single fully-connected layer that maps per-channel mean and standard deviation statistics (18 features from 9 AIA channels) to irradiance predictions.
  - A **CNN model** — an EfficientNet-B5 backbone (~30M parameters) that extracts spatial features from the full 512×512 px images.
  - The outputs are combined as: $O_{\text{total}} = O_{\text{linear}} + \lambda \cdot O_{\text{CNN}}$, where $\lambda$ is a learnable blending parameter.
- **Training:** Two-phase training procedure — the linear model is trained first (20 epochs), then the CNN is activated while the linear weights are frozen (30 epochs), ensuring stable convergence. Loss function: Huber Loss.
- **Input:** 9 SDO/AIA EUV/UV channels (94, 131, 171, 193, 211, 304, 335, 1600, 1700 Å) at 512×512 px resolution.
- **Output:** 38 EVE ion spectral lines — covering both MEGS-A and MEGS-B wavelength ranges.

### Checkpoint Contents

The checkpoint file is a PyTorch dictionary containing:

| Key | Description |
|-----|-------------|
| `model` | The trained `HybridIrradianceModel` instance (ready for inference) |
| `normalizations` | Per-channel AIA normalization statistics (`mean`, `std`) used during training |
| `sci_parameters` | Scientific metadata: `aia_wavelengths` (list of 9 input channels) and `eve_ions` (list of 38 output spectral lines) |

### Data

The model is trained on the [SDOML v2 dataset](https://sdoml.org), which is publicly available on the same S3 bucket:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-fdlx-ard/sdomlv2a/
```


# 3 System Requirements

There are two sets of system requirements:
1. Requirements to *create* the model. These can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE).
2. Requirements for *using* the model (inference only).

| Component | Minimum |
|-----------|---------|
| **CPU** | Any modern CPU (inference runs on CPU) |
| **RAM** | 4 GB |
| **GPU** | Not required (CPU-only inference supported) |
| **Storage** | ~50 MB for model checkpoint |

The model can also be run for free on [Google Colab](https://colab.research.google.com/github/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE/blob/main/public/virtual_eve_tutorial.ipynb).


# 4 Reference

Indaco, M., Gass, D., Fawcett, W. J., Galvez, R., Wright, P. J., & Muñoz-Jaramillo, A. (2024). *Virtual EVE: a Deep Learning Model for Solar Irradiance Prediction.* [arXiv:2408.17430](https://arxiv.org/abs/2408.17430)

-->
