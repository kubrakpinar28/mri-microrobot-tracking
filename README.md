# MRI Microrobot Detection and Tracking

Deep learning-based detection and tracking of multiple magnetic microrobots in MRI images, using a YOLO11n object detector and BoT-SORT multi-object tracking.

## Table of Contents

- [Motivation](#motivation)
- [Project Summary](#project-summary)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
  - [1. Why Synthetic Data](#1-why-synthetic-data)
  - [2. Source MRI Datasets](#2-source-mri-datasets)
  - [3. Physics-Based Artifact Simulation](#3-physics-based-artifact-simulation)
  - [4. Aspect Ratio Constraint](#4-aspect-ratio-constraint)
  - [5. Quality Filtering (CNR)](#5-quality-filtering-cnr)
  - [6. Data Augmentation and Splits](#6-data-augmentation-and-splits)
  - [7. Detection Model: YOLO11n](#7-detection-model-yolo11n)
  - [8. Test Set Evaluation](#8-test-set-evaluation)
  - [9. Center Localization Validation (Hungarian Matching)](#9-center-localization-validation-hungarian-matching)
  - [10. Multi-Robot Tracking: BoT-SORT](#10-multi-robot-tracking-bot-sort)
- [Pipeline Execution Order](#pipeline-execution-order)
- [Installation](#installation)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Author](#author)
- [References](#references)

## Motivation

Magnetically-actuated microrobots are a promising technology for minimally invasive medical procedures: targeted drug delivery, localized cancer treatment, and catheter-based interventions. For these systems to move from research to clinical use, the robot's position inside the body must be tracked in real time and with high precision.

MRI offers a natural platform for this. A magnetic microrobot disturbs the local magnetic field during an MRI scan, producing a characteristic susceptibility artifact: a dark void surrounded by a bright halo. If this artifact's location can be reliably detected, the robot's spatial position can be recovered without any additional sensor or hardware.

This project builds a complete deep learning pipeline to solve that detection and tracking problem, starting from data that does not exist in the real world (no public, labeled dataset of microrobot MRI scans is available) and ending with a trained, evaluated, multi-robot tracking system.

## Project Summary

The pipeline consists of five stages:

1. **Synthetic data generation** — physics-based simulation of microrobot MRI artifacts, placed onto real anatomical MRI backgrounds across four organs (brain, heart, breast, knee)
2. **Quality filtering** — discarding synthetic samples whose artifact is too weak to be realistically detectable
3. **Data augmentation** — expanding the training set only, while keeping validation/test data clean
4. **Detection model training** — a YOLO11n object detector trained to localize microrobots and output their center coordinates
5. **Evaluation and tracking** — standard detection metrics, sub-pixel center accuracy validation via optimal (Hungarian) matching, and a multi-robot catheter-tracking demonstration using BoT-SORT

## Key Results

| Metric | Value |
|---|---|
| Test mAP50 | 0.957 |
| Test mAP50-95 | 0.645 |
| Precision | 0.894 |
| Recall | 0.949 |
| Mean center error (Hungarian matching, full test set) | 0.393 px |
| RMS center error | 1.884 px |
| Median center error | 0.204 px |
| Bounding box Dice (mean) | 0.878 |
| Bounding box Dice (median) | 0.918 |
| V-curve tracking — unique IDs recovered | 5 / 5 |
| V-curve tracking — ID consistency | 100% |
| V-curve tracking — false positives per frame | 0.00 |

The Hungarian-matching evaluation covers the **full test set** (173 positive images, 737 matched robot instances), not a sample.

## Repository Structure
.

├── README.md

├── requirements.txt

├── .gitignore

│

├── brain_mri_synthetic.py        # Core synthetic MRI artifact generator (physics-based)

├── filter_by_cnr.py              # CNR-based quality filter

├── augment_dataset.py            # Data augmentation (train split only)

├── nii_to_png.py                 # NIfTI → PNG conversion (heart dataset)

├── pck_to_png_knee.py            # .pck → PNG conversion (knee dataset)

├── merge_datasets.py             # Merges per-organ datasets into one

├── split_dataset.py              # Train / val / test split

├── main.py                       # Entry point tying the pipeline together

│

├── tracking_linear.py            # Linear-motion multi-robot tracking scenario

├── tracking_vcurve.py            # V-curve tracking scenario (v1)

├── tracking_vcurve_v2.py         # V-curve tracking scenario (v2, distance-constrained + spline)

│

├── inference_comparison_v5.png   # Final visualization: GT (red) vs predicted (green) centers

├── vcurve_v3_tracking.gif        # Final V-curve tracking animation (100% ID consistency)

├── project_summary.json          # Machine-readable summary of key metrics

│

└── synthetic_dataset_yolo_augmented/   # Generated dataset (not tracked in git — see .gitignore)

├── train/{images,labels}/

├── val/{images,labels}/

└── test/{images,labels}/

Trained model weights (`best.pt`) and the full generated dataset are not included in this repository due to file size; they are stored separately (Google Drive). Contact the author for access if needed.

## Methodology

### 1. Why Synthetic Data

No public, labeled dataset of microrobot MRI scans exists. Acquiring real labeled data would require coordinating an actual microrobot experiment with MRI scan time, which was outside the scope of this internship. Instead, the project takes a **simulation-based approach**: real anatomical MRI images are used as backgrounds, and a physically-motivated artifact model is used to synthesize the microrobot's signature on top of them. This is a common strategy in medical imaging ML research when real annotated data is scarce.

A single real magnetic artifact template (provided by the lab from a previous physical experiment) was used to derive the shape of the void/halo pattern that the physics model is layered onto.

### 2. Source MRI Datasets

Four open-access MRI datasets were used as anatomical backgrounds, one per organ, each with a pixel spacing value (mm per pixel) documented in the literature — required to convert robot diameter from physical units (mm) into pixels:

| Organ | Source dataset | Pixel spacing |
|---|---|---|
| Brain | MR-ART (Narai et al., *Scientific Data*, 2022) | 0.80 mm/px |
| Heart | Medical Decathlon, Task02_Heart | 1.25 mm/px |
| Breast | Kaggle Breast Cancer MRI's | 0.75 mm/px (literature estimate — see Limitations) |
| Knee | KneeMRI Rijeka (Stajduhar et al., 2017) | 1.80 mm/px |

`nii_to_png.py` and `pck_to_png_knee.py` handle the format-specific conversion of the heart (NIfTI) and knee (`.pck`) datasets into standard PNG images that the rest of the pipeline can process uniformly.

### 3. Physics-Based Artifact Simulation

The core synthetic data generator, `brain_mri_synthetic.py`, models three physical effects:

**Dipole magnetic field.** A magnetic microrobot is approximately spherical, so its field distortion follows the classic point-dipole approximation:
B_dipole(r, θ) = (μ₀·m / 4π) · (2cosθ / r³)

The artifact intensity falls off with the cube of distance and depends on the angle from the dipole axis. This governs how the void/halo pattern is shaped around the robot's placement point.

**Depth-dependent visibility.** As the robot's true depth (`h`, in mm) relative to the imaging plane increases, the artifact becomes progressively harder to see, eventually vanishing:
visibility = max(0, 1 − (h / h_threshold)^1.2)     where h_threshold = 9.0 mm

**Rice noise.** MRI image noise does not follow a Gaussian distribution — it follows a Rice distribution, especially at low signal-to-noise ratios. This is simulated as:
I_Rice = √[(I + n₁)² + n₂²]     where n₁, n₂ ~ N(0, σ)

Robot diameter is randomized between **10–20 mm (1–2 cm)** per instance and converted to pixels using the organ's pixel spacing, so robot size in the image is physically grounded rather than arbitrary.

### 4. Aspect Ratio Constraint

Because the microrobot is spherical, its theoretical MRI artifact is isotropic — meaning every bounding box must be square (aspect ratio = 1.0). Two specific design choices guarantee this:

1. The artifact patch is cropped to a square before being placed onto the background.
2. Motion-stretch / elongation effects are disabled entirely (`motion_vec = None` for all robots), since an active motion-stretch would anisotropically elongate the patch and break the AR=1.0 constraint.

A further important detail: the **ground-truth bounding box center is always taken from the robot's true physical placement coordinate, not from the visible artifact mask**. This matters specifically at tissue edges — if part of a robot's artifact falls outside the tissue region and is not rendered, the label center remains correct rather than shifting toward whatever fragment of the artifact happened to be visible.

### 5. Quality Filtering (CNR)

Every generated image is scored using Contrast-to-Noise Ratio (CNR):
CNR = |mean(artifact) − mean(surrounding tissue)| / std(surrounding tissue)

`filter_by_cnr.py` discards images whose CNR falls below an organ-specific threshold, since a sufficiently faint artifact would not be realistically detectable by any model (or radiologist) and would only inject label noise into training.

### 6. Data Augmentation and Splits

`augment_dataset.py` applies horizontal/vertical flips, brightness/contrast jitter, additional Rice noise, and rotation — **only to the training split**. Validation and test sets remain clean, unaugmented copies of the originally generated images, so all reported metrics reflect performance on realistic, unmodified data rather than inflated performance on augmented samples.

`merge_datasets.py` and `split_dataset.py` handle combining the four per-organ datasets and producing the final train/val/test partition in YOLO label format.

### 7. Detection Model: YOLO11n

A YOLO11n object detector (2.59M parameters) was trained with transfer learning from COCO-pretrained weights.

**Why detection instead of segmentation:** the goal is to recover the microrobot's center point, not its exact pixel-level shape. A bounding box center gives this directly. A segmentation model would add label complexity (polygon masks) without adding value for this specific goal, and would be more fragile in the presence of artifact-boundary ambiguity.

Training configuration:

| Parameter | Value |
|---|---|
| Image size | 320 × 320 |
| Batch size | 16 |
| Optimizer | AdamW (lr=0.002, momentum=0.9) |
| Max epochs | 100 |
| Early stopping patience | 20 |
| Actual stop / best epoch | epoch 93 / best at epoch 73 |
| Hardware | Google Colab, Tesla T4 GPU |
| Best validation mAP50 | 0.956 |

### 8. Test Set Evaluation

The model was evaluated on a held-out, augmentation-free test set of 191 images (741 robot instances). Results:

| Metric | Value |
|---|---|
| mAP50 | 0.957 |
| mAP50-95 | 0.645 |
| Precision | 0.894 |
| Recall | 0.949 |

### 9. Center Localization Validation (Hungarian Matching)

Standard detection metrics (mAP, precision, recall) confirm *that* the model finds robots, but not *how accurately* it locates their center. To answer that, predicted bounding box centers were matched against ground-truth centers using the **Hungarian algorithm** (optimal one-to-one assignment), which avoids the incorrect pairings that a naive nearest-neighbor approach can produce when multiple robots are close together. Matches beyond 50 px were rejected as invalid.

This evaluation was run on the **entire test set** (173 positive images, 737 matched robots), not a sample:

| Metric | Value |
|---|---|
| Matched robots | 737 |
| RMS center error | 1.884 px |
| Mean center error | 0.393 px |
| Median center error | 0.204 px |
| Bounding box Dice (mean) | 0.878 |
| Bounding box Dice (median) | 0.918 |

In visualizations (`inference_comparison_v5.png`), red points mark ground-truth centers, green boxes/points mark model predictions, and yellow lines connect matched pairs — since the mean error is under half a pixel, red and green points overlap almost completely in most images.

### 10. Multi-Robot Tracking: BoT-SORT

YOLO11n is trained on static images; tracking across a sequence of frames requires a separate multi-object tracking algorithm layered on top. **BoT-SORT** (Aharon et al., 2022) was used, with its ReID (re-identification) component disabled, since appearance features are not discriminative in grayscale MRI.

Two scenarios were tested:

- **Linear motion** (`tracking_linear.py`): 4 robots moving independently in straight lines across 30 frames.
- **V-curve catheter formation** (`tracking_vcurve_v2.py`), used as a **proof-of-concept**: 5 robots arranged along a parabolic curve —
y(x) = (1 / 2d) · (x − x_center)² + y_center

  — simulating robots moving in sequence inside a catheter. Robot positions are placed using arc-length parametrization (so spacing along the curve is even) and an iterative distance-constraint algorithm enforces consistent spacing between consecutive robots frame to frame. The curve itself is smoothed for visualization using spline interpolation.

  This scenario was run on a **knee MRI background only**, as a single representative demonstration of catheter-like sequential tracking — it was **not** repeated independently across all four organs. The detection model itself, by contrast, was trained and evaluated across all four organs.

Final results (`vcurve_v3_tracking.gif`):

| Metric | Linear (4 robots, 30 frames) | V-curve (5 robots, 40 frames) |
|---|---|---|
| Unique IDs recovered | 4 / 4 | 5 / 5 |
| ID consistency | 75.0% | 100.0% |
| Avg. detections/frame | 3.90 | 4.42 |
| False positives/frame | 0.63 | 0.00 |

Note that the average detections/frame in the V-curve scenario (4.42) is below 5 by design, not by error: as robots approach the edge of the tissue region near the end of the sequence, some become physically occluded/invisible — the model correctly does not detect them, which is the expected behavior, not a tracking failure.

## Pipeline Execution Order

```bash
# 1. Convert organ-specific raw formats to PNG
python nii_to_png.py            # heart dataset (NIfTI)
python pck_to_png_knee.py       # knee dataset (.pck)

# 2. Generate synthetic microrobot artifacts across all organs
python brain_mri_synthetic.py

# 3. Filter out low-quality (low-CNR) samples
python filter_by_cnr.py

# 4. Merge per-organ datasets and create train/val/test split
python merge_datasets.py
python split_dataset.py

# 5. Apply data augmentation (training split only)
python augment_dataset.py

# 6. Train YOLO11n (run in a GPU environment, e.g. Google Colab)
#    See training configuration in Methodology, Section 7

# 7. Run tracking scenarios
python tracking_linear.py
python tracking_vcurve_v2.py
```

`main.py` provides a single entry point that can be adapted to run the above stages sequentially.

## Installation

```bash
pip install -r requirements.txt
```

This project was trained and evaluated using `ultralytics==8.4.70` and `torch==2.11.0` (CUDA 12.8) on Google Colab. See `requirements.txt` for the full dependency list.

## Known Limitations

- The model is trained and validated entirely on synthetic data; no real in-vivo or in-vitro microrobot MRI scans were used.
- Brain MRI images can contain natural susceptibility sources (blood vessels, calcifications) that occasionally cause false positives, since these visually resemble the synthetic artifact pattern.
- Breast MRI pixel spacing (0.75 mm/px) is a literature estimate rather than a value extracted from DICOM metadata, since the source dataset does not include this metadata.
- The V-curve geometric constraint (catheter shape) is applied as post-processing on top of YOLO detections; it is not learned by the model itself.
- V-curve tracking was demonstrated on a single organ (knee) as a proof-of-concept; it was not independently repeated across brain, heart, and breast.

## Future Work

- Validation and fine-tuning against real microrobot MRI scans (in vitro phantom or in vivo)
- Organ-specific post-processing to reduce brain false positives
- 3D tracking from volumetric MRI data, rather than single 2D slices
- Shape-aware tracking that incorporates catheter/curve geometry directly into the model rather than as post-processing
- Real-time closed-loop MRI-robot control system

## Author

Kübra Akpınar - Computer Engineer —(kubra.akpinar028@gmail.com, 05454911561) Internship Project, ODTÜ ROMER (Biomedical Robotics Research Laboratory), 2026

## References

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO11. https://github.com/ultralytics/ultralytics
- Narai, Á. et al. (2022). Movement-related artefacts (MR-ART) dataset of matched motion-corrupted and clean structural MRI brain scans. *Scientific Data*, 9, 630.
- Stajduhar, I. et al. (2017). Semi-automated detection of anterior cruciate ligament injury from MRI. *Computer Methods and Programs in Biomedicine*.
- Simpson, A. L. et al. (2019). A large annotated medical image dataset for the development and evaluation of segmentation algorithms. arXiv:1902.09063.
- Aharon, N., Orfaig, R., & Bobrovsky, B. Z. (2022). BoT-SORT: Robust Associations Multi-Pedestrian Tracking. arXiv:2206.14651.
- Nelson, B. J., Kaliakatsos, I. K., & Abbott, J. J. (2010). Microrobots for minimally invasive medicine. *Annual Review of Biomedical Engineering*, 12, 55–85.