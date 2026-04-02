# CEG3004 Environmental Sound Classification

> DSP-based audio classification pipeline for the ESC-50 dataset, robust to noise and bandwidth distortions.

---

## Project Overview

This project implements a robust **Environmental Sound Classification (ESC)** pipeline as part of CEG3004. The system classifies audio clips into **50 sound classes** and is designed to perform well under clean, noisy, and band-limited conditions.

**Dataset:** Derived from ESC-50 — 2,000 audio clips (5s each, mono, 40 clips/class)  
**Submission set:** Each clip appears in three versions — Clean, Noisy, and Band-limited

---

## Repository Structure

```
CEG3004-DSP-Project/
│
├── data/                        # (gitignored) Local data directory
│   ├── train/
│   │   ├── audio/               # Training .wav files
│   │   └── labels.csv
│   └── submission/
│       ├── audio/               # Submission .wav files
│       └── metadata.csv
│
├── src/
│   ├── features.py              # All DSP feature extraction functions
│   ├── preprocessing.py         # Audio loading, trimming, normalization
│   ├── augmentation.py          # Noise, gain, bandpass, shift augmentations
│   ├── train.py                 # Model training and cross-validation
│   └── predict.py               # Inference and CSV generation
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_feature_analysis.ipynb # Feature visualization and ablation
│   └── 03_experiments.ipynb     # Model experiments and comparisons
│
├── outputs/                     # (gitignored) Generated model and predictions
│   ├── Pr_7_model.joblib
│   └── Pr_7_predictions.csv
│
├── consolidated_code.py         # Single-file version (for Colab submission)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CEG3004-DSP-Project.git
cd CEG3004-DSP-Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python consolidated_code.py
```

This will:
- Download and extract the dataset automatically
- Extract DSP features from all training audio
- Apply data augmentation (noise, gain, bandpass, shift)
- Train a Random Forest classifier with 5-fold cross-validation
- Save the model as `Pr_7_model.joblib`
- Generate predictions as `Pr_7_predictions.csv`

---

## DSP Pipeline

### Preprocessing
- **Silence trimming** via `librosa.effects.trim` (top_db=20)
- **Peak normalization** to [-1, 1]
- **Pre-emphasis filter** (α = 0.97) to boost high frequencies
- **Fixed-length padding/truncation** to 5 seconds at 16 kHz

### Feature Extraction
Features are extracted globally and across 4 temporal segments for richer temporal representation:

| Feature | Description |
|---|---|
| MFCC (20 coeffs) + Δ + ΔΔ | Captures timbral texture and dynamics |
| Log-Mel Spectrogram (40 bands) | Energy distribution across frequency |
| Spectral Centroid | Brightness of the sound |
| Spectral Bandwidth | Spread of frequencies |
| Spectral Rolloff | High-frequency energy drop-off point |
| Spectral Contrast | Peak vs. valley energy contrast |
| Zero Crossing Rate | Percussiveness / noisiness indicator |
| RMS Energy | Overall loudness |

### Data Augmentation
Each training sample is augmented 8× (4 types × 2 repetitions) to improve robustness:

| Augmentation | Purpose |
|---|---|
| Additive Gaussian noise | Simulates noisy conditions |
| Random gain scaling (0.7–1.3×) | Volume variation robustness |
| Random bandpass filter (200–6000 Hz) | Band-limited robustness |
| Random time shift (±20%) | Temporal invariance |

### Classifier
- **Random Forest** with 300 trees, `max_depth=15`, balanced class weights
- 5-fold **Stratified Cross-Validation** for reliable macro-F1 estimation
- Trained on clean + all augmented samples

---

## Results

| Split | Macro F1 |
|---|---|
| CV (train, augmented) | *reported during training* |
| Validation (held-out 20%) | *reported during training* |

*(Update this table after running the pipeline)*

---

## Requirements

```
librosa
numpy
pandas
scikit-learn
scipy
matplotlib
tqdm
joblib
gdown
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

- Random seed fixed at `42` throughout (train/test split, cross-validation, Random Forest)
- All augmentation parameters are logged in `src/augmentation.py`
- Feature vector dimension is deterministic given the same audio input

---

## Team

**Team ID:** Pr_7  

