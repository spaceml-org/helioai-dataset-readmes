
<!-- # 2 Dataset Description


There are three levels of description available for this dataset:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](https://helioai.org/dev/artifact/89d1911b-7803-44e7-b792-076edb2dc5ed/details)).
- The full source code used to process the data and create the models (see the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-GeoCL/)).

## Project summary

This project forecasts geomagnetic perturbations at ground stations worldwide using two complementary ML models:

- SHEATH: An MLP that translates solar disk imagery (SDO) into solar wind parameter predictions at L1, providing multi-day advance warning of incoming conditions
- DAGGER-CL: A GRU that takes real-time solar wind measurements (ACE/DSCOVR) and predicts magnetic field perturbations (dBe, dBn) at ~535 ground stations with ~30-minute lead time

Together they form a two-stage forecasting pipeline: SHEATH offers early situational awareness from the Sun itself, while DAGGER-CL provides high-fidelity, station-level nowcasts once the solar wind is measured in situ. The dataset includes the raw inputs, processed training data, and trained model weights for both components. -->


## 1 Models Description

The Geoeffectiveness Continual Learning challenge (GEO-CLOAK) produced a multi-stage, continual-learning geoeffectiveness forecasting system, extending the earlier Multiscale Geoeffectiveness (SHEATH + DAGGER++) architecture into a fully adaptive, operational ML pipeline.

Rather than introducing a single new standalone model, GEO-CLOAK integrates the following model components to forecast geomagnetic perturbations at ground stations worldwide: 

1. SHEATH (solar-wind forecasting model):  An MLP that translates solar disk imagery (SDO) into solar wind parameter predictions at L1, providing multi-day advance warning of incoming conditions.
2. DAGGER-CL (continual-learning geomagnetic perturbation model): A GRU that takes real-time solar wind measurements (ACE/DSCOVR) and predicts magnetic field perturbations (dBe, dBn) at ~535 ground stations with ~30-minute lead time.

Together, they form a two-stage forecasting pipeline, with SHEATH offering early situational awareness from the Sun itself, and DAGGER-CL providing high-fidelity, station-level nowcasts once the solar wind is measured in situ. 


## 1.1 Models Access

The SHEATH and DAGGER-CL model files are provided below, along with a sample dataset for testing purposes. 

Instructions for accessing the following files on Amazon Web Services (AWS) are provided in [Section 2](#2-access-instructions).

### SHEATH model (300 KB)
- AWS PATH: `hl-geo/models/sheath_latest.pth`
- Usage instructions are given in this [colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2024-HL-GeoCL/blob/main/public/sheath_inference_quickstart.ipynb).
- Type: 3-layer MLP 

### SHEATH example data (30 KB)
- AWS Path: `hl-geo/models/examples/`
- A sub-sample example data ready for usage with SHEATH. Example given in the SHEATH colab notebook linked above.

### DAGGER-CL (120 MB)
- AWS Path: `hl-geo/models/DAGGER_CL/`
- Contents: Trained models and checkpoints of the DAGGER-CL pipeline. As this is a continuuous learning pipeline this requires some infrastructure to set up, details can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-GeoCL/).

# 2 Access Instructions 

Models are stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-geo/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request <AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the data sample you want to download (see table) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.

# 3 System Requirements

1. To use the SHEATH model, any modern computer will do (no GPU necessary)
2. The DAGGER-CL model is part of a more complex pipeline, so the requirements depend on how you want to use it. See the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2024-HL-GeoCL/) for details. 
