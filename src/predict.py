"""
predict.py
----------
Load a trained model and generate predictions for the submission set.

Run directly:
    python -m src.predict
"""

import os
import numpy as np
import pandas as pd
import joblib

from tqdm import tqdm
from src.features import extract_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT       = os.environ.get("DATA_ROOT", "data")
SUBMISSION_DIR  = os.path.join(DATA_ROOT, "submission")
METADATA_CSV    = os.path.join(SUBMISSION_DIR, "metadata.csv")
AUDIO_SUB_DIR   = os.path.join(SUBMISSION_DIR, "audio")
MODEL_PATH      = "Pr_7_model.joblib"
OUT_CSV         = "Pr_7_predictions.csv"


def predict(model_path=MODEL_PATH, out_csv=OUT_CSV):
    assert os.path.exists(model_path), f"Model not found at {model_path}. Train first."
    assert os.path.exists(SUBMISSION_DIR), f"Submission directory not found: {SUBMISSION_DIR}"

    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    # Recover idx → label mapping from the model's training classes
    # (RandomForest stores classes_ on the named step 'clf')
    clf = model.named_steps["clf"]
    classes = list(clf.classes_)

    sub_meta = pd.read_csv(METADATA_CSV)
    sub_meta["clip_id"] = sub_meta["clip_id"].astype(str)
    print(f"Submission clips: {len(sub_meta)}")

    pred_rows = []
    for _, row in tqdm(sub_meta.iterrows(), total=len(sub_meta), desc="Predicting"):
        clip_id  = row["clip_id"]
        wav_path = os.path.join(AUDIO_SUB_DIR, f"{clip_id}.wav")
        feat     = extract_features(wav_path)
        pred_idx = int(model.predict(feat.reshape(1, -1))[0])
        pred_rows.append((clip_id, pred_idx))

    out = pd.DataFrame(pred_rows, columns=["clip_id", "predicted_label"])
    out.to_csv(out_csv, index=False)
    print(f"\nPredictions saved to {out_csv}")
    return out


if __name__ == "__main__":
    predict()
