"""
preprocessing.py
----------------
Audio loading and preprocessing utilities.
"""

import numpy as np
import librosa


def load_audio(path, sr=16000):
    """
    Load a .wav file as mono audio, resampled to the given sample rate.

    Args:
        path (str): Path to the audio file.
        sr (int): Target sample rate. Default: 16000 Hz.

    Returns:
        tuple: (y, sr) — audio signal as float32 numpy array and sample rate.
    """
    y, sr_out = librosa.load(path, sr=sr, mono=True)
    y = np.nan_to_num(y).astype(np.float32)
    return y, sr_out


def preprocess_audio(y, sr, target_len=5.0):
    """
    Apply a standard preprocessing chain to a raw audio signal:
      1. Silence trimming
      2. Peak normalization
      3. Pre-emphasis filtering (boosts high frequencies)
      4. Fixed-length padding / truncation

    Args:
        y (np.ndarray): Raw audio signal.
        sr (int): Sample rate.
        target_len (float): Target duration in seconds. Default: 5.0.

    Returns:
        np.ndarray: Preprocessed audio signal of fixed length (target_len * sr samples).
    """
    # 1. Remove silence from edges
    y, _ = librosa.effects.trim(y, top_db=20)

    # 2. Peak normalization to [-1, 1]
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # 3. Pre-emphasis filter — boosts high-frequency content
    #    y[n] = x[n] - 0.97 * x[n-1]
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # 4. Pad or truncate to fixed length
    target_samples = int(sr * target_len)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    return y
