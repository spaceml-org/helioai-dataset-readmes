# **SDOMLv2a Dataset Documentation**

**Overview**

The **SDOMLv2a** dataset is a comprehensive, machine-learning-ready collection of solar observation data produced by the Frontier Development Lab (FDL). Derived from the Solar Dynamics Observatory (SDO) mission, this dataset serves as a cornerstone for advanced solar research and irradiance prediction.

The raw SDO archive spans approximately 20 petabytes. To address computational and storage challenges, this dataset has been curated, calibrated, and reduced to a high-density, managed size of approximately 8 terabytes without losing physical information relevant to Extreme Ultraviolet (EUV) irradiance tasks.

---

** 1. Data Composition**

The dataset is multifaceted, incorporating multi-channel and multi-instrument sources. It provides a standardized temporal alignment and spatial resolution across three primary instruments:

** 1.1 AIA (Atmospheric Imaging Assembly)**

**Channels:** 9 spectral channels (94, 131, 171, 193, 211, 304, 335, 1600, 1700 Å). The 4500 Å channel is excluded.  
**Resolution:** Downsampled to **512x512** pixels.  
**Cadence:** 6 minutes.  
**Format:** Converted from Flexible Image Transport System (.fits) to **.zarr** arrays.  

** 1.2 HMI (Helioseismic and Magnetic Imager)**

**Components:** 3 vector magnetic field components (*Bx, By, Bz*) derived from line-of-sight data.  
**Resolution:** **512x512** pixels.  
**Cadence:** 12 minutes.  
**Alignment:** Spatially aligned with AIA images (same solar disk size and location).  

** 1.3 EVE (EUV Variability Experiment)**

**Data:** 39 ion intensity profiles.  
**Historical Data (2010–2014):** Real measurements (excluding Fe XVI-2 due to sensor defects).  Included in EVE_legacy.zarr folder.  
**Post-2014 (not-included):** "Virtual EVE" spectra generated via ML inference to account for the MEGS-A instrument failure.  

---

** 2. Data Processing Pipeline**

The generation of SDOMLv2a utilizes the **SDO Scientific Computing Platform**, a Google Cloud Platform (GCP) infrastructure designed for massive parallelization and continuous ingestion.

** 2.1 Ingestion Sources**
Data is ingested via a "Trigger Batching" system from three distinct sources, each requiring specific handling:  
 **HelioCloud:** 5000x5000 images (Level 1.5).  
 **JSOC (Raw):** 4096x4096 images (Level 1).  
 **JSOC (Synoptic):** 1024x1024 images (Level 1.5).  

Legacy data was downloaded via massively parallelized cloud functions, while new data can be maintained via cron jobs (daily for HMI, every 5 minutes for AIA).

** 2.2 Calibration Standards **

Regardless of the source quality, all data in SDOMLv2a is processed to achieve a uniform **Level 1.5 calibration**.

** AIA Calibration**  
**Routine:** Python SDO/AIA calibration (Barnes et al 2020).  
**Steps:** Pointing correction, registration, and degradation correction.  
**Standardization:** Exposure time normalization and solar disk size standardization are applied before downsampling to 512x512.

** HMI Calibration**
HMI vectors are not natively available and are derived from four raw .fits files: *azimuth, disambig, field,* and *inclination*.  
**Routine:** Custom Python implementation of IDL routines (Gary & Hagyard 1990).  
**Steps:** Azimuth disambiguation → Coordinate transformation (Native to Spherical CCD) → Registration/Rotation.  
**Artifact Removal:** Unlike standard IDL routines which replace NaNs with zeros (introducing artificial "zero magnetic field" artifacts near the limb), our pipeline preserves NaNs. This prevents interpolation errors during resizing.

---

** 3. Technical Specifications & Dataset Structure**

| Feature | Specification |
| :----- | :--- |
| **File Format** | .zarr (append-only) |
| **Image Resolution** | 512 x 512 |


```text
📂 us-fdlx-ard-sdomlv2a - Latest version of SDOML in .zarr format (2010-2023)
├── AIA.zarr/ - calibrated AIA data in .zarr format
│   └── YEAR/
│       ├── WAVELENGTH/
│       │   ├── .zarray
│       │   ├── .zattrs
│       │   └── ... 
│       └── .zgroup
│ 
├── EVE_legacy.zarr/ - Legacy EVE data in .zarr format
│   ├──SPECTRAL LINE/
│   │   ├── .zarray
│   │   ├── .zattrs
│   │   └── ...
│   ├── Time/
│   │   ├── .zarray
│   │   ├── 0
│   │   └── ... 
│   └── .zgroup
│ 
└── HMI.zarr/ - calibrated HMI data in .zarr format
    └── YEAR/
        ├── By/
        │   ├── .zarray
        │   ├── .zattrs
        │   └── ... 
        ├── Bx/
        │   ├── .zarray
        │   ├── .zattrs
        │   └── ...  
        ├── Bz/
        │   ├── .zarray
        │   ├── .zattrs
        │   └── ...  
        └── .zgroup
```

To read more about .zarr file and how to use them, see the [Zarr user guide](https://zarr.dev/) and [Zarr-Python documentation](https://zarr.readthedocs.io/en/stable/user-guide/).

---

** 4. Derived Data Products**

To aid in computational efficiency, the project repository includes code to generate:  
**Embeddings:** Lower-dimensional representations of the image data.  
**Virtual EVE:** Predicted irradiance values for dates following the 2014 MEGS-A failure.

---

** 5. SPASE Input Dataset Links:**

[AIA 94 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV094/PT12S.html)  
[AIA 131 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV131/PT12S.html)  
[AIA 171 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV171/PT12S.html)  
[AIA 193 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV193/PT12S.html)  
[AIA 211 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV211/PT12S.html)  
[AIA 304 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV304/PT12S.html)  
[AIA 335 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/EUV335/PT12S.html)  
[AIA 1600 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/UV1600/PT24S.html)  
[AIA 1700 Å](https://spase-metadata.org/NASA/NumericalData/SDO/AIA/UV1700/PT24S.html)  
[HMI](https://spase-metadata.org/NASA/NumericalData/SDO/HMI/LOS_Magnetogram/PT720S.html)  
[EVE](https://spase-metadata.org/NASA/NumericalData/SDO/EVE/Level1/Version8/PT0.25S.html)


# 4 Citation

If you use this dataset in your research, please cite the accompanying paper:

> M. Indaco, D. Gass, W. J. Fawcett, R. Galvez, P. J. Wright, and A. Muñoz-Jaramillo. *Virtual EVE: a Deep Learning Model for Solar Irradiance Prediction*. Machine Learning and the Physical Sciences Workshop, NeurIPS 2023. arXiv:2408.17430. https://doi.org/10.48550/arXiv.2408.17430

BibTeX:

```bibtex
@inproceedings{indaco2024virtualeve,
  title         = {Virtual {EVE}: a Deep Learning Model for Solar Irradiance Prediction},
  author        = {Indaco, M. and Gass, D. and Fawcett, W. J. and Galvez, R. and Wright, P. J. and Mu{\~n}oz-Jaramillo, A.},
  booktitle     = {Machine Learning and the Physical Sciences Workshop, NeurIPS 2023},
  year          = {2024},
  eprint        = {2408.17430},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.SR},
  doi           = {10.48550/arXiv.2408.17430},
  url           = {https://arxiv.org/abs/2408.17430}
}
```

Please also consider citing the underlying SDOMLv2 dataset and the original SDO instrument papers (AIA, HMI, EVE) as appropriate.
