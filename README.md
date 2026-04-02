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

### 4. Run experiments (optional)

```
Open corresponding experiment notebook in IDE, run in sequential order (01_eda.ipynb -> 02_feature_analysis.ipynb -> 03_experiments.ipynb)
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

Macro-F1: 0.6198997668997669
| label            | precision | recall | f1-score | support |
|------------------|-----------|--------|----------|---------|
| airplane         | 0.60      | 0.60   | 0.60     | 5       |
| breathing        | 0.50      | 0.40   | 0.44     | 5       |
| brushing_teeth   | 1.00      | 1.00   | 1.00     | 5       |
| can_opening      | 1.00      | 1.00   | 1.00     | 5       |
| car_horn         | 0.50      | 0.20   | 0.29     | 5       |
| cat              | 1.00      | 0.20   | 0.33     | 5       |
| chainsaw         | 0.17      | 0.20   | 0.18     | 5       |
| chirping_birds   | 0.57      | 0.80   | 0.67     | 5       |
| church_bells     | 1.00      | 0.80   | 0.89     | 5       |
| clapping         | 0.80      | 0.80   | 0.80     | 5       |
| clock_alarm      | 1.00      | 0.60   | 0.75     | 5       |
| clock_tick       | 0.00      | 0.00   | 0.00     | 5       |
| coughing         | 0.30      | 0.60   | 0.40     | 5       |
| cow              | 0.44      | 1.00   | 0.62     | 4       |
| crackling_fire   | 0.80      | 0.80   | 0.80     | 5       |
| crickets         | 1.00      | 0.60   | 0.75     | 5       |
| crow             | 1.00      | 0.60   | 0.75     | 5       |
| crying_baby      | 0.40      | 1.00   | 0.57     | 4       |
| dog              | 0.75      | 0.60   | 0.67     | 5       |
| door_wood_creaks | 1.00      | 0.25   | 0.40     | 4       |
| door_wood_knock  | 0.43      | 0.60   | 0.50     | 5       |
| drinking_sipping | 0.75      | 0.60   | 0.67     | 5       |
| engine           | 0.00      | 0.00   | 0.00     | 5       |
| fireworks        | 0.75      | 0.60   | 0.67     | 5       |
| footsteps        | 0.80      | 0.80   | 0.80     | 5       |
| frog             | 1.00      | 1.00   | 1.00     | 4       |
| glass_breaking   | 0.75      | 0.75   | 0.75     | 4       |
| hand_saw         | 0.62      | 1.00   | 0.77     | 5       |
| helicopter       | 0.50      | 0.40   | 0.44     | 5       |
| hen              | 0.00      | 0.00   | 0.00     | 5       |
| insects          | 0.75      | 0.75   | 0.75     | 4       |
| keyboard_typing  | 0.50      | 0.80   | 0.62     | 5       |
| laughing         | 0.20      | 0.25   | 0.22     | 4       |
| mouse_click      | 1.00      | 0.40   | 0.57     | 5       |
| pig              | 0.71      | 1.00   | 0.83     | 5       |
| pouring_water    | 0.71      | 1.00   | 0.83     | 5       |
| rain             | 0.62      | 1.00   | 0.77     | 5       |
| rooster          | 0.57      | 0.80   | 0.67     | 5       |
| sea_waves        | 0.83      | 1.00   | 0.91     | 5       |
| sheep            | 1.00      | 0.75   | 0.86     | 4       |
| siren            | 0.83      | 1.00   | 0.91     | 5       |
| sneezing         | 0.60      | 0.60   | 0.60     | 5       |
| snoring          | 0.50      | 0.40   | 0.44     | 5       |
| thunderstorm     | 0.83      | 1.00   | 0.91     | 5       |
| toilet_flush     | 0.38      | 1.00   | 0.56     | 5       |
| train            | 0.80      | 1.00   | 0.89     | 4       |
| vacuum_cleaner   | 1.00      | 0.40   | 0.57     | 5       |
| washing_machine  | 1.00      | 0.40   | 0.57     | 5       |
| water_drops      | 0.67      | 0.50   | 0.57     | 4       |
| wind             | 0.50      | 0.40   | 0.44     | 5       |
| accuracy         |           |        | 0.64     | 240     |
| macro avg        | 0.67      | 0.65   | 0.62     | 240     |
| weighted avg     | 0.67      | 0.64   | 0.62     | 240     |

---
## Experimental Results

### Classifier Comparison

Two classifiers were evaluated using 5-fold stratified cross-validation on clean features (no augmentation). Random Forest (300 trees) achieved a mean macro-F1 of **0.5375 ± 0.0329**, outperforming SVM with RBF kernel at **0.5134 ± 0.0181**. Random Forest was selected as the final classifier due to its higher score and greater variance tolerance across folds, which suggests better generalisation.

### Augmentation Ablation

To assess the contribution of each augmentation strategy, models were trained with one augmentation type removed at a time and evaluated on a held-out 20% validation set.

| Configuration | Macro-F1 |
|---|---|
| Without bandpass | **0.6599** |
| All augmentations | 0.6349 |
| No augmentation | 0.6175 |
| Without shift | 0.5867 |
| Without noise | 0.5732 |
| Without gain | 0.5726 |

Removing bandpass filtering *improved* performance, suggesting it may introduce frequency distortions inconsistent with the test distribution. In contrast, noise and gain augmentation contributed most to robustness — their removal caused the largest performance drops (~4.5pp each). All augmentation configurations outperformed the no-augmentation baseline, confirming the overall value of data augmentation for generalisation under distortion.

### Per-Class Performance

Final model validation macro-F1: **0.6349** across 50 classes. Performance varied substantially by sound category. Highly distinctive sounds achieved perfect classification — `brushing_teeth`, `can_opening`, `frog`, and `thunderstorm` all scored F1 = 1.0. The hardest classes were `hen`, `engine`, and `laughing` (F1 = 0.0), likely due to spectral overlap with acoustically similar categories. The most frequent confusions were `chainsaw → toilet_flush` (4 errors) and several pairs involving impulsive or broadband sounds such as `car_horn`, `engine`, and `mouse_click`.

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


## Reproducibility

- Random seed fixed at `42` throughout (train/test split, cross-validation, Random Forest)
- All augmentation parameters are logged in `src/augmentation.py`
- Feature vector dimension is deterministic given the same audio input

---


## Team

**Team ID:** Pr_7  

