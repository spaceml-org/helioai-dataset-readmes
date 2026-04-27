# 1. Model Description

This project classifies solar wind into four physical regimes using a two-stage ML pipeline:

1. **SDO Foundation Model (MAE)**  
   Encodes SDO HMI magnetic field images (512×512, 3 channels) into embeddings  
   → output: 513 tokens × 512 dimensions  

2. **SkipLinear Classifier Head**  
   Takes embeddings + spacecraft information and predicts solar wind class:

    - Ejecta  
    - Coronal Hole  
    - Sector Reversal  
    - Streamer Belt  

The embeddings are precomputed, enabling efficient inference without running the full MAE backbone.

Instructions on how to run inference with the models are provided in this [Colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2025-HL-Solar-Wind/blob/main/public/inference_demo.ipynb).


## Model Pipeline 

| Stage | Input | Output | Role |
|------|------|--------|------|
| MAE Backbone | SDO HMI images (Bx, By, Bz) | Embeddings (513×512) | Feature extraction |
| SkipLinear Head | Embeddings + position features | Solar wind class | Classification |

---

# 1.1 Solar Wind Classifier — Full Checkpoint

| Property | Value |
|--------|------|
| AWS Path | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/models/models/mae_skip_linear_best.ckpt` |
| Size | 7.1 GB |
| Type | PyTorch Lightning checkpoint |
| Components | MAE backbone + SkipLinear head |
| Validation Performance | val_loss = 0.06, val_f1 = 0.31 |

### Use Case
- End-to-end inference from raw SDO images  
- Fine-tuning backbone on new tasks  


## Model Inputs (Full Checkpoint)

| Input | Shape | Description |
|------|------|-------------|
| SDO HMI images | 512×512×3 | Magnetic field (Bx, By, Bz) |


## Model Outputs

| Output | Type | Description |
|-------|------|-------------|
| Class label | integer (0–3) | Solar wind regime |


# 1.2 Solar Wind Classifier — Head Weights Only

| Property | Value |
|--------|------|
| AWS Path | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/models/models/head_weights.pt` |
| Size | 2.1 GB |
| Type | PyTorch state dict |
| Parameters | 545.8 million |

### Architecture

- 8-layer MLP  
- 1024 hidden units  
- Skip connection at layer 4  


## Model Inputs (Head Only)

| Input | Shape | Description |
|------|------|-------------|
| MAE embedding | 262,656 dims (flattened) | Precomputed SDO features |
| Position encoding | 264 dims | Sinusoidal encoding of spacecraft location |
| Radial distance | 1 dim | Normalized PSP distance |

---

## Model Outputs

| Output | Type | Description |
|-------|------|-------------|
| Class probabilities | vector (4) | Probability of each solar wind type |
| Class label | integer | Predicted class |

---

## Use Case

- Lightweight inference using precomputed embeddings  
- Primary method used in demo notebook  

---

# 1.3 Pretrained MAE Backbone

| Property | Value |
|--------|------|
| AWS Path | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/models/models/pretrained_mae_e128.ckpt` |
| Size | 1.2 GB |
| Type | PyTorch Lightning checkpoint |


## Architecture

| Parameter | Value |
|----------|------|
| Model type | Vision Transformer (ViT) |
| Patch size | 16 |
| Embedding dim | 768 |
| Depth | 12 |
| Heads | 12 |
| Training | 128 epochs |


## Inputs / Outputs

| Input | Output |
|------|--------|
| SDO images | latent embeddings |


## Use Case

- Fine-tuning for new downstream tasks  
- Recomputing embeddings if needed  


# 1.4 NVAE Embeddings Model

| Property | Value |
|--------|------|
| AWS Path | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/models/models/sdofm_nvae_embeddings.pt` |
| Size | 381 MB |
| Type | PyTorch tensor file |


## Description

Alternative embedding representation using NVAE model.


# 1.5 Demo Data Subset

| Property | Value |
|--------|------|
| AWS Path | `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/models/data_subset/` |
| Size | 502 MB |


## Contents

| File | Size | Description |
|-----|-----|-------------|
| embeddings_subset.npy | 501 MB | 500 MAE embeddings (125 per class) |
| metadata_subset.parquet | 39 KB | Labels, positions, timestamps |
| normalization_stats.json | <1 KB | Radial distance normalization |


## Example Inference Schema

| Field | Description |
|------|-------------|
| embedding | MAE feature vector |
| label | true solar wind class |
| position | spacecraft location |
| radial distance | normalized input |
| timestamp | observation time |


# 1.6 Model Workflow

<p align="center">
  <img src="https://quickchart.io/graphviz?graph=digraph%20G%20%7B%0Arankdir%3DTB%3B%0Anode%20%5Bshape%3Dbox%2C%20style%3Dfilled%2C%20color%3Dlightgray%2C%20fontname%3DHelvetica%5D%3B%0A%0ASDO%20%5Blabel%3D%22SDO%20Images%5Cn(HMI%20Bx%2FBy%2FBz)%22%5D%3B%0AMAE%20%5Blabel%3D%22MAE%20Backbone%5Cn(Vision%20Transformer)%22%5D%3B%0AEMB%20%5Blabel%3D%22Embeddings%5Cn(513%20tokens%20x%20512%20dims)%22%5D%3B%0APSP%20%5Blabel%3D%22PSP%20Position%20Features%5Cn(lat%2C%20lon%2C%20r)%22%5D%3B%0AHEAD%20%5Blabel%3D%22SkipLinear%20Classifier%20Head%5Cn(MLP%20with%20skip)%22%5D%3B%0AOUT%20%5Blabel%3D%22Solar%20Wind%20Class%5Cn(Ejecta%20%7C%20CH%20%7C%20SR%20%7C%20SB)%22%5D%3B%0A%0ASDO%20-%3E%20MAE%3B%0AMAE%20-%3E%20EMB%3B%0AEMB%20-%3E%20HEAD%3B%0APSP%20-%3E%20HEAD%3B%0AHEAD%20-%3E%20OUT%3B%0A%7D" width="500">
</p>


# 2. Access Instructions

Models are stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/models/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the file or directory you want to download (see below) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.


# 3 System Requirements

There are two sets of system requirements:
1. Requirements to *create* the model. These can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2025-HL-Solar-Wind).
2. Requirements for *using* the model:


| Component | Minimum |
|-----------|---------|
| **CPU** | Modern multi-core CPU |
| **RAM** | 8 GB (for inference with head weights + demo subset); 16 GB (for full checkpoint) |
| **GPU** | Not required for inference from pre-computed embeddings; required for training or running the MAE backbone |
| **Storage** | 3 GB (head weights + demo subset); 11 GB (all model files + demo subset) | 


<!--# 2 Model Description


There are three levels of description available for this model:
- A high-level summary (this document) for users to quickly become familiar with the dataset.
- A detailed description (see the [Technical Memorandum](https://helioai.org/dev/artifact/cfaf8127-3065-48b7-9405-c4dc645327b4/details)).
- The full source code used to process the data and create the models (see the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2025-HL-Solar-Wind)).

## Project Summary

This project classifies solar wind into four physical types using a two-stage ML pipeline:

1. **SDO Foundation Model (MAE):** A pretrained Vision Transformer masked autoencoder encodes SDO HMI magnetic field images (512x512, 3 channels) into compact embeddings (513 tokens x 512 dims).
2. **SkipLinear Classifier Head:** An MLP with skip connections takes the MAE embeddings plus spacecraft position and radial distance as input, and predicts one of four solar wind classes: Ejecta, Coronal Hole, Sector Reversal, or Streamer Belt (following the Xu & Borovsky 2015 classification scheme).

The embeddings are pre-computed and stored in the companion dataset bucket, so inference with the classifier head requires only the head weights (~2 GB) and the embeddings — not the full MAE backbone.

Instructions on how to use the models are given in this [colab notebook](https://colab.research.google.com/github/FrontierDevelopmentLab/2025-HL-Solar-Wind/blob/main/public/inference_demo.ipynb).

## 2.1 Solar Wind Classifier — Full Checkpoint (7.1 GB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/models/models/mae_skip_linear_best.ckpt`
- Type: PyTorch Lightning checkpoint containing both the MAE backbone and the SkipLinear classifier head.
- Best validation performance: val_loss=0.06, val_f1=0.31 (macro-averaged across 4 classes).
- Use this checkpoint if you want to run the full end-to-end pipeline from SDO images, or to fine-tune the backbone on a different task.

## 2.2 Solar Wind Classifier — Head Weights Only (2.1 GB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/models/models/head_weights.pt`
- Type: PyTorch state dict containing only the SkipLinear classifier head (545.8M parameters).
- Use this for lightweight inference from pre-computed MAE embeddings. This is what the demo notebook uses.
- Architecture: 8-layer MLP with 1024 hidden units and a skip connection at layer 4. Inputs are the flattened embedding (262,656 dims) concatenated with sinusoidal position encoding (264 dims) and normalized radial distance (1 dim).

## 2.3 Pretrained MAE Backbone (1.2 GB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/models/models/pretrained_mae_e128.ckpt`
- Type: PyTorch Lightning checkpoint of the MAE backbone pretrained for 128 epochs on SDO imagery.
- Use this as a starting point for fine-tuning on new downstream tasks. The backbone is a Vision Transformer (ViT) with patch_size=16, embed_dim=768, depth=12, num_heads=12.

## 2.4 NVAE Embeddings Model (381 MB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/models/models/sdofm_nvae_embeddings.pt`
- Type: PyTorch tensor file containing NVAE model embeddings.

## 2.5 Demo Data Subset (502 MB)
- AWS PATH: `s3://nasa-radiant-data/helioai-datasets/hl-solar-wind/models/data_subset/`
- Contents: A small, balanced subset for running the inference demo notebook:
    - `embeddings_subset.npy` (501 MB): 500 pre-computed MAE embeddings (125 per class) from the 2023 test split (January–March), stored as float32.
    - `metadata_subset.parquet` (39 KB): Labels, spacecraft positions (lon/lat), magnetic footpoint positions, radial distance (raw and normalized), and timestamps.
    - `normalization_stats.json` (< 1 KB): Radial distance normalization parameters (mean, std) computed from the test subset.



# 3 System Requirements

There are two sets of system requirements:
1. Requirements to *create* the model. These can be found in the [GitHub Repository](https://github.com/FrontierDevelopmentLab/2025-HL-Solar-Wind).
2. Requirements for *using* the model:


| Component | Minimum |
|-----------|---------|
| **CPU** | Modern multi-core CPU |
| **RAM** | 8 GB (for inference with head weights + demo subset); 16 GB (for full checkpoint) |
| **GPU** | Not required for inference from pre-computed embeddings; required for training or running the MAE backbone |
| **Storage** | 3 GB (head weights + demo subset); 11 GB (all model files + demo subset) | -->
