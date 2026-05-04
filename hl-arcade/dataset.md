# 1. Dataset Description

Using high-resolution observations from the Helioseismic and Magnetic Imager (HMI, Scherrer et. al 2012) and Atmospheric Imaging Assembly (AIA, Lemen et al. 2012) instruments onboard the Solar Dynamics Observatory (SDO), the Frontier Development Lab (FDL) Heliolab 2025 ARCADE system combines physics-based Surface Flux Transport modeling with deep learning to characterize the magnetic evolution of solar active regions that can give rise to flares and coronal mass ejections. The hybrid approach extracts key parameters describing magnetic field structure and integrates uncertainty quantification to produce interpretable forecasts. Results demonstrate accurate short-term predictions of active region emergence up to six hours in advance, providing an important first step for downstream space weather models.

While the ARCADE forecasting model was trained on SDO magnetograms, it was designed to work with following additional SDO/HMI and SDO/AIA data modes: dopplergrams, extreme-ultraviolet images at 171 Å and 304 Å, and continuum-intensity maps. The SDO/HMI exposes continual, full-disk images of the Sun, while the SDO/AIA images the solar atmosphere at 10 different wavelengths, every 10 seconds. 

In addition to the high-level summary of the dataset provided here, a detailed description may be found in the project [Technical Memorandum](<https://drive.google.com/file/d/1fTI2N0cOcLgbzVkk7QRWpNYnPsgFvwb4/view>).
<!-- (see the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2025-HL-Active-Regions/>)).-->

## 1.1 Processed Data

### Observed Data

To process and prepare SDO archive data for input to the ARCADE machine-learning models, the following steps were taken:

 - Data cleaning and correction: Removal of bad frames, corrupted files, artifacts, images with spacecraft anomalies (eclipses, safe modes) and saturated pixels (e.g., during large flares in AIA); and subtraction of spacecraft/observer motion from Dopplergram velocities using available metadata or velocity correction maps.

 - Image co-registration: Aligning images so that the same solar features fall on the same pixels over time and across instruments/wavelengths.

 - Adjusting for projection effects: Correction for the fact that features away from the solar disk center are foreshortened and that HMI line-of-sight measurements are not equal to radial values except at the disk center.

 - Normalization: Scale intensities/values to a consistent range suitable for ML training.

 - Removal of geometric effects: Removal of large-scale patterns caused by the Sun’s spherical shape, limb darkening, and other geometric distortions unrelated to physical evolution.

 - Re-structuring: SDO data in the form of individual FITS files, corresponding to each of the five independent data types, were combined into a single Zarr-formatted file, resulting in a data cube with a *t_obs* axis containing the timestamps of each data set; a channel axis containing the 5 data modes; and *x* and *y* axes of length 4096 each to match the pixel dimensions of the data images.
     
### Simulated Validation Data

The ARCADE forecasting system also incorporates physics-based simulated data from the Advective Flux Transport (AFT) model as a complementary dataset to the observational SDO inputs. AFT produces time-evolving, full-Sun maps of the radial magnetic field using a Surface Flux Transport (SFT) framework that models key physical processes including differential rotation, meridional flow, turbulent diffusion, and flux emergence.

Unlike observational magnetograms, which are limited to line-of-sight measurements, AFT provides a physically consistent estimate of the global radial magnetic field across the entire solar surface, making it a critical resource for evaluating large-scale magnetic evolution.

The simulated AFT baseline data contain:
- Full-disk, time-evolving magnetogram maps generated via SFT physics
- Radial magnetic field estimates over the entire solar surface

## 1.2 Raw Data

The raw data (i.e., before being processed specifically for input to a machine learning model), includes cleaned and calibrated, science-ready SDO HMI and AIA FITS image data products downloadable from the primary SDO data archive hosted by the [Joint Science Operations Center (JSOC)](http://jsoc.stanford.edu/) at Stanford University; as well as the [Virtual Solar Observatory (VSO)](https://sdac.virtualsolar.org/cgi/search) search interface that accesses multiple solar data archives. 

# 2. Access Instructions

Data is stored on Amazon Web Services (AWS). Access is given through the AWS Command Line Interface (CLI). Instructions on how to install and use are given in the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Listing files is done by e.g.:
```
aws s3 ls --no-sign-request s3://nasa-radiant-data/helioai-datasets/<DATASET_NAME>/
```

Downloading files is done by e.g.
```
aws s3 cp --no-sign-request s3://nasa-radiant-data/helioai-datasets/<AWS PATH> <LOCAL PATH> --recursive
```
You will need to replace `<AWS PATH>` with the path to the data sample you want to download (see table) and `<LOCAL PATH>` with the path on your local machine where you want to save the data.


| Data Product | AWS Path | Size | Download time (@100 Mbps) |
|-------------|----------|------|---------------------------|
| Processed - Training | [SDOMLv2](https://registry.opendata.aws/sdoml-fdl/) Magnetograms (single channel of multi-channel Zarr files) | | |
| Processed - Validation | `s3://nasa-radiant-data/helioai-datasets/hl-arcade/2025-hl-arcade-development-landing/aft/lisa/AFT_Baseline/{YYYY}/{NN}/AFTmap*.h5`, obtained from [here](https://data.boulder.swri.edu/lisa/AFT_Baseline/) and provided by [Lisa Upton](https://coffies.stanford.edu/people/lisa_upton) | | |
| Results |  [Interactive UI](https://arcade.spaceml.org/app) | | |

<!-- | Raw | `SDO archive FITS images` | | |
| Results | `s3://nasa-radiant-data/helioai-datasets/hl-arcade/2025-hl-arcade-development-features/data/sunpde_output/prod/val_test_*/preds/*png` | | |
| Miscellaneous | `<DATASET_NAME>/miscellaneous/` | | |

# 3. System Requirements

There are two sets of system requirements:
1. Requirements to *create* the data products. These can be found in the [GitHub Repository](<https://github.com/FrontierDevelopmentLab/2025-HL-Active-Regions/>).
2. Requirements for *using* the data products


| Component | Minimum |
|-----------|---------|
| **CPU** | |
| **RAM** | |
| **GPU** | 20 GB |
| **Storage** | | -->







