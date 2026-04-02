"""
features.py
-----------
DSP feature extraction for environmental sound classification.

All features are computed over the full signal and over 4 equal temporal
segments to capture time-varying behaviour. This gives the classifier
both global and local spectral/temporal information.
"""

import numpy as np
import librosa

from src.preprocessing import load_audio, preprocess_audio
from src.augmentation import apply_augmentation


def _segment_stats(x, n_segments=4):
    """
    Compute per-segment mean and std across time, plus global stats.

    For a 2D feature matrix x of shape (n_features, n_frames):
      - Global: mean and std across all frames  → 2 * n_features values
      - Per-segment: mean and std within each of n_segments windows
                     → n_segments * 2 * n_features values

    Args:
        x (np.ndarray): Feature matrix of shape (n_features, n_frames).
        n_segments (int): Number of temporal segments. Default: 4.

    Returns:
        np.ndarray: Flattened array of statistics.
    """
    n_frames = x.shape[1]
    seg_size = n_frames // n_segments
    segment_stats = []
    for i in range(n_segments):
        seg = x[:, i * seg_size:(i + 1) * seg_size]
        if seg.shape[1] > 0:
            segment_stats.extend([seg.mean(axis=1), seg.std(axis=1)])
        else:
            segment_stats.extend([np.zeros(x.shape[0]), np.zeros(x.shape[0])])
    return np.concatenate([
        x.mean(axis=1), x.std(axis=1),  # global stats
        *segment_stats                   # per-segment stats
    ])


def extract_features(path, sr=16000, augment_type=None):
    """
    Load, preprocess, optionally augment, and extract DSP features from an audio file.

    Features extracted:
        - MFCC (20 coeffs) + first and second order deltas
        - Log-Mel Spectrogram (40 bands)
        - Spectral Centroid
        - Spectral Bandwidth
        - Spectral Rolloff
        - Spectral Contrast
        - Zero Crossing Rate (ZCR)
        - RMS Energy

    Each feature is summarised with global mean/std and per-segment (4×)
    mean/std, giving rich temporal context.

    Args:
        path (str): Path to a .wav audio file.
        sr (int): Sample rate. Default: 16000 Hz.
        augment_type (str or None): Augmentation to apply before feature
            extraction. One of 'noise', 'gain', 'bandpass', 'shift', or None.

    Returns:
        np.ndarray: 1D float32 feature vector.
    """
    y, sr = load_audio(path, sr)
    y = preprocess_audio(y, sr)

    if augment_type is not None:
        y = apply_augmentation(y, sr, augment_type)

    # --- MFCC + deltas ---
    # Captures timbral texture; deltas encode rate-of-change dynamics
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # --- Log-Mel Spectrogram ---
    # Perceptually-weighted energy distribution across frequency
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=1024, hop_length=256)
    log_mel = librosa.power_to_db(mel)

    # --- Spectral shape features ---
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)           # "brightness"
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)          # frequency spread
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)            # high-freq cutoff

    # --- Spectral contrast ---
    # Ratio of spectral peaks to valleys per sub-band; separates tonal vs noisy sounds
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=1024, hop_length=256)

    # --- Temporal features ---
    zcr = librosa.feature.zero_crossing_rate(y)   # percussiveness / noisiness
    rms = librosa.feature.rms(y=y)                # overall loudness envelope

    features = np.concatenate([
        _segment_stats(mfcc),
        _segment_stats(mfcc_delta),
        _segment_stats(mfcc_delta2),
        _segment_stats(centroid),
        _segment_stats(bandwidth),
        _segment_stats(rolloff),
        _segment_stats(zcr),
        _segment_stats(log_mel),
        _segment_stats(contrast),
        _segment_stats(rms),
    ])

    return features.astype(np.float32)
