"""
train.py
--------
Model training, cross-validation, and evaluation.

Run this script directly to train and save the model:
    python -m src.train
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay

from src.features import extract_features
from src.augmentation import AUGMENTATION_TYPES

# ---------------------------------------------------------------------------
# Paths — update DATA_ROOT to match your environment
# ---------------------------------------------------------------------------
DATA_ROOT      = os.environ.get("DATA_ROOT", "data")
TRAIN_DIR      = os.path.join(DATA_ROOT, "train")
LABELS_CSV     = os.path.join(TRAIN_DIR, "labels.csv")
AUDIO_TRAIN    = os.path.join(TRAIN_DIR, "audio")
MODEL_PATH     = "Pr_7_model.joblib"
AUGMENTS_PER_TYPE = 2   # how many augmented copies per augmentation type


def build_label_maps(df):
    classes      = sorted(df["label"].unique().tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    return classes, label_to_idx, idx_to_label


def extract_all_features(df, audio_dir, label_to_idx):
    """Extract clean features for every clip in df."""
    X, y, paths = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        wav_path = os.path.join(audio_dir, f"{row['clip_id']}.wav")
        X.append(extract_features(wav_path, augment_type=None))
        y.append(label_to_idx[row["label"]])
        paths.append(wav_path)
    return np.stack(X), np.array(y, dtype=np.int64), paths


def build_augmented_set(paths_tr, y_tr):
    """
    Expand the training set with augmented copies.
    Each sample → 1 clean + (len(AUGMENTATION_TYPES) × AUGMENTS_PER_TYPE) augmented = 9×.
    """
    X_aug, y_aug = [], []
    for wav_path, label in tqdm(
        zip(paths_tr, y_tr), total=len(paths_tr), desc="Augmenting training data"
    ):
        X_aug.append(extract_features(wav_path, augment_type=None))
        y_aug.append(label)
        for aug_type in AUGMENTATION_TYPES:
            for _ in range(AUGMENTS_PER_TYPE):
                X_aug.append(extract_features(wav_path, augment_type=aug_type))
                y_aug.append(label)
    return np.stack(X_aug), np.array(y_aug, dtype=np.int64)


def build_model():
    # TODO: swap out the classifier or tune hyperparameters here
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])


def plot_confusion_matrix(y_true, y_pred, idx_to_label, save_path="confusion_matrix.png"):
    classes = [idx_to_label[i] for i in range(len(idx_to_label))]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to {save_path}")
    plt.show()


def main():
    assert os.path.exists(TRAIN_DIR), f"Train directory not found: {TRAIN_DIR}"

    df = pd.read_csv(LABELS_CSV)
    df["clip_id"] = df["clip_id"].astype(str)
    df["label"]   = df["label"].astype(str)
    print(f"Loaded {len(df)} training samples across {df['label'].nunique()} classes.")

    classes, label_to_idx, idx_to_label = build_label_maps(df)

    # --- Extract clean features ---
    X, y, paths = extract_all_features(df, AUDIO_TRAIN, label_to_idx)
    print(f"Feature matrix shape: {X.shape}")

    # --- Train / validation split (stratified) ---
    X_tr, X_va, y_tr, y_va, paths_tr, paths_va = train_test_split(
        X, y, paths, test_size=0.2, random_state=42, stratify=y
    )

    # --- Build augmented training set ---
    X_tr_aug, y_tr_aug = build_augmented_set(paths_tr, y_tr)
    print(f"Augmented training set shape: {X_tr_aug.shape}")

    # --- Cross-validation ---
    model = build_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tr_aug, y_tr_aug, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"\nCV Macro-F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # --- Final fit on full augmented training data ---
    model.fit(X_tr_aug, y_tr_aug)

    # --- Validation evaluation ---
    y_pred = model.predict(X_va)
    print("\n" + classification_report(y_va, y_pred, target_names=[idx_to_label[i] for i in range(len(classes))]))
    print(f"Validation Macro-F1: {f1_score(y_va, y_pred, average='macro'):.4f}")

    plot_confusion_matrix(y_va, y_pred, idx_to_label)

    # --- Save model ---
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return model, idx_to_label


if __name__ == "__main__":
    main()
