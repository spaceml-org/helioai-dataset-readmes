# 1 Model Description

<!--There are three levels of description available for this model:
- A high-level summary (this document) for users to quickly become familiar with the model.
- A detailed description (see the [Technical Memorandum](https://helioai.org)).
- The full source code used to process the data and train the model (see the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE)).

## Project Summary-->

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

The model can also be run for free on [Google Colab](https://colab.research.google.com/github/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE/blob/main/public/virtual_eve_tutorial.ipynb).


# 4 Reference

Indaco, M., Gass, D., Fawcett, W. J., Galvez, R., Wright, P. J., & Muñoz-Jaramillo, A. (2024). *Virtual EVE: a Deep Learning Model for Solar Irradiance Prediction.* [arXiv:2408.17430](https://arxiv.org/abs/2408.17430)
