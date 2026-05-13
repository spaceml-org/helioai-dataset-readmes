# 1. Model Description

**SuNeRF (Solar Neural Radiance Field)** is a deep learning model that reconstructs the **3D structure of the solar atmosphere** from sparse 2D EUV imaging. By training jointly on observations from SDO/AIA (Earth view) and STEREO-A/B (off-Earth views), the model learns a continuous 4D representation (x, y, z, t) of EUV emission and absorption, enabling **novel-view synthesis** — rendering synthetic EUV images from arbitrary heliocentric viewpoints, including positions never directly observed.

The published checkpoints in this collection were produced during the Frontier Development Lab 2022 US 4π EUV Irradiance challenge and during follow-on transfer-learning experiments in 2023.

There are three levels of description available:
- A high-level summary (this document).
- A detailed description in the [SuNeRF paper](https://doi.org/10.3847/1538-4357/ad38c9).
- The full source code (see the [GitLab Repository](https://gitlab.com/frontierdevelopmentlab/2022-us-4pieuvirradiance/4piuvsun)).

Instructions on how to use the models are given in this [Colab notebook](<COLAB_LINK_PLACEHOLDER>).

---

## 1.1 Architecture

SuNeRF is a standard hierarchical NeRF adapted to the solar setting:

| Component | Specification |
|---|---|
| Input | (x, y, z, t) — 3D position + time, 4D total |
| Output | (emission, absorption) — 2 channels per ray sample |
| Positional encoder | Fourier (10 frequencies per spatial dim, log-spaced) |
| Activation | **Sine activation** (`w0=1`) — differs from standard ReLU NeRF, improves smoothness of the recovered atmosphere |
| Coarse model | 8 layers × 512 filters, with hierarchical sampling |
| Fine model | 8 layers × 512 filters, refines on importance-sampled rays |
| Stratified samples per ray | 64 |
| Hierarchical samples per ray | 128 |
| Volume render | Standard NeRF emission–absorption integration |

The model is trained per-wavelength: separate networks for 171 Å, 193/195 Å, 211/284 Å, and 304 Å.

### Time normalization

Time is normalized as `(date - 2010-01-01) / 30 days × 2π`, so that the time axis maps cleanly onto a periodic representation usable by the Fourier encoder.

### Spatial coordinates

Coordinates are heliocentric Cartesian in units of solar radii, with the volume of interest constrained to roughly `[-1.3, 1.3] R☉` around the Sun.

---

## 1.2 Model Variants

The published bundle contains **6 model families** plus a shared pretrained checkpoint. All variants share the architecture above; they differ in **training data**, **wavelength**, and **training schedule**.

| Variant | AWS path (under `s3://nasa-radiant-data/helioai-datasets/us-4pieuvirradiance/models/`) | Wavelength(s) | Training data | Purpose |
|---|---|---|---|---|
| **Pretrained checkpoint** | `transfer.ckpt` | base 193 Å | SDO + STEREO 2012-08 | Starting point for transfer learning and the eruption model |
| **Base high-resolution** | `sunerf_1024/save_state.snf` | 193 Å | SDO + STEREO 2012-08 | Canonical SuNeRF, trained at 1024² ray resolution |
| **Base ensemble (uncertainty)** | `sunerf_ensemble/ensemble_{1..5}/save_state.snf` | 193 Å | SDO + STEREO 2012-08 | 5-member ensemble for uncertainty quantification |
| **Eruption-specialized** | `eruption/save_state.snf` | 304 Å | SDO + STEREO 2012-08, subframe around CME region | Fine-tuned around the 2012-08-31 filament eruption (hgc_lon=90°, hgc_lat=-20°, 1024×1024 subframe) |
| **PSI per-wavelength** | `psi_models/psi_{171,193,211}.snf` | 171, 193, 211 Å | [PSI 3D MHD simulation](https://www.predsci.com/) of the corona | Ground-truth-validated models trained on synthetic AIA from PSI |
| **PSI ensemble** | `psi_ensemble/psi_ensemble_{1..4}.snf` | 193 Å | PSI simulation | 4-member ensemble on PSI synthetic data |
| **Transfer-learned per-wavelength** | `transfer_runs/{171,211,304,2022_02}/save_state.snf` | 171, 211, 304 Å + 2022-02 epoch | SDO + STEREO 2012-08, fine-tuned from `transfer.ckpt` | Per-wavelength specialization; `2022_02/` is a separate 2022 February observation window |

---

## 1.3 Inputs

At inference time, the model takes a **camera pose** specifying the viewpoint:

| Parameter | Type | Description |
|---|---|---|
| `lat` | float | Heliographic latitude of the observer (degrees) |
| `lon` | float | Heliographic longitude (Stonyhurst) of the observer (degrees) |
| `time` | datetime | Observation time (must fall in the model's training time range) |
| `distance` | float | Observer distance from Sun center, in solar radii (default: 1 AU ≈ 215 R☉) |
| `resolution` | int | Output image resolution in pixels (default: model-trained value, typically 1024 or 2048) |
| `focal` | float | Focal length (px) — controls field of view |

The model returns rendered images for the time-and-viewpoint pair; no actual sensor observation is required as input.

## 1.4 Outputs

Each `load_observer_image(...)` call returns a dict:

| Output | Shape | Units | Description |
|---|---|---|---|
| `channel_map` | (H, W) | normalized EUV intensity | The synthetic AIA-equivalent image at the requested viewpoint |
| `height_map` | (H, W) | solar radii | Per-pixel weighted-mean ray depth — a 2D projection of the 3D atmosphere height |
| `absorption_map` | (H, W) | normalized | Per-pixel integrated absorption along the ray |

---

## 1.5 Checkpoint formats

The bundle contains two file formats:

| Extension | What | When to use |
|---|---|---|
| `.snf` | **SuNeRF state** — a `torch.save` dict with `coarse_model`, `fine_model`, `encoder_kwargs`, `sampling_kwargs`, `wavelength`, `start_time`, `end_time`, and `config` keys. Self-contained for inference. | **For inference.** Use `SuNeRFLoader` (below). |
| `.ckpt` | **PyTorch Lightning checkpoint** — full training state including optimizer, scheduler, and model weights. | For resuming or fine-tuning training. |

For most users, the `.snf` files are what you want.

### Loading example

```python
from datetime import datetime
from s4pi.maps.evaluation.loader import SuNeRFLoader

# Load any .snf file
loader = SuNeRFLoader(
    'sunerf_ensemble/ensemble_1/save_state.snf',
    resolution=2048,   # optional; defaults to training resolution
)

# Render a synthetic image from any viewpoint
outputs = loader.load_observer_image(
    lat=0.0,                                        # heliographic latitude (deg)
    lon=0.0,                                        # heliographic longitude (deg)
    time=datetime(2012, 8, 31, 12, 0, 0),           # observation time (UTC)
    distance=215.0,                                 # ~1 AU in solar radii
)

channel_map    = outputs['channel_map']     # (H, W) synthetic AIA image
height_map     = outputs['height_map']      # (H, W) atmosphere height
absorption_map = outputs['absorption_map']  # (H, W) absorption
```

A working end-to-end example is in the [Colab notebook](<COLAB_LINK_PLACEHOLDER>).

---

# 2. Access Instructions

Models are stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-4pieuvirradiance/models/
```

Downloading files is done by e.g.:
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-4pieuvirradiance/models/sunerf_ensemble/ensemble_1/save_state.snf <LOCAL PATH>
```
You will need to replace `<LOCAL PATH>` with the path on your local machine where you want to save the file.

Listing one of the variants in detail:
```
aws s3 ls --recursive --no-sign-request s3://nasa-radiant-data/helioai-datasets/us-4pieuvirradiance/models/psi_models/
```

| Bundle | AWS Path | Size |
|---|---|---|
| All models | `models/` | 1.2 GB |
| Eruption | `models/eruption/` | 58 MB |
| PSI ensemble | `models/psi_ensemble/` | 58 MB |
| PSI per-wavelength | `models/psi_models/` | 44 MB |
| SuNeRF 1024 | `models/sunerf_1024/` | 228 MB |
| SuNeRF ensemble | `models/sunerf_ensemble/` | 418 MB |
| Transfer runs | `models/transfer_runs/` | 360 MB |
| Pretrained | `models/transfer.ckpt` | 44 MB |

---

# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *train* the models from scratch (see the [GitLab Repository](https://gitlab.com/frontierdevelopmentlab/2022-us-4pieuvirradiance/4piuvsun)).
2. Requirements for *running inference* with the published checkpoints.

| Component | Minimum (inference, low-res) | Recommended (inference, 1024²+) |
|---|---|---|
| **CPU** | Any modern CPU | 8+ cores |
| **RAM** | 8 GB | 16 GB |
| **GPU** | Not required (CPU inference works) | NVIDIA, ≥8 GB VRAM (much faster) |
| **Storage** | ~60 MB per `.snf` checkpoint | 1.2 GB for the full bundle |

Inference can also be run for free on [Google Colab](<COLAB_LINK_PLACEHOLDER>) with the free-tier GPU.

---

# 4. Citation

If you use these models in your research, please cite:

> R. Jarolim, B. Tremblay, A. Muñoz-Jaramillo, K. Battams, A. Jungbluth, M. C. M. Cheung, A. Pal, M. Indaco, K. Bingham, S. Pelletier, K. Albanitis, C. Tylor, D. Coronel, T. Y. Chen, R. Galvez, P. J. Wright, S. Kim. *SuNeRF: 3D Reconstruction of the Solar EUV Corona Using Neural Radiance Fields*. The Astrophysical Journal, 2024. https://doi.org/10.3847/1538-4357/ad38c9

BibTeX:

```bibtex
@article{jarolim2024sunerf,
  title   = {{SuNeRF}: 3{D} Reconstruction of the Solar {EUV} Corona Using Neural Radiance Fields},
  author  = {Jarolim, R. and Tremblay, B. and Mu{\~n}oz-Jaramillo, A. and Battams, K.
             and Jungbluth, A. and Cheung, M. C. M. and Pal, A. and Indaco, M.
             and Bingham, K. and Pelletier, S. and Albanitis, K. and Tylor, C.
             and Coronel, D. and Chen, T. Y. and Galvez, R. and Wright, P. J. and Kim, S.},
  journal = {The Astrophysical Journal},
  year    = {2024},
  doi     = {10.3847/1538-4357/ad38c9}
}
```

Please also cite the companion dataset (see [dataset.md](./dataset.md)) and the underlying STEREO/SECCHI EUVI instrument paper.
