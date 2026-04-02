"""
augmentation.py
---------------
Audio augmentation functions used to improve model robustness against
noise and bandwidth distortions — mirroring real-world test conditions.
"""

import numpy as np
from scipy.signal import butter, lfilter


def augment_noise(y, noise_factor=None):
    """
    Add Gaussian white noise to the signal.

    Simulates noisy recording conditions. The noise factor is randomised
    each call unless explicitly provided, producing diverse augmented samples.

    Args:
        y (np.ndarray): Audio signal.
        noise_factor (float, optional): Noise amplitude scale. If None,
            sampled uniformly from [0.001, 0.02].

    Returns:
        np.ndarray: Noisy audio signal.
    """
    if noise_factor is None:
        noise_factor = np.random.uniform(0.001, 0.02)
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def augment_gain(y, gain_range=(0.7, 1.3)):
    """
    Multiply the signal by a random gain factor.

    Simulates volume variation between recordings.

    Args:
        y (np.ndarray): Audio signal.
        gain_range (tuple): (min_gain, max_gain). Default: (0.7, 1.3).

    Returns:
        np.ndarray: Gain-scaled audio signal.
    """
    gain = np.random.uniform(*gain_range)
    return y * gain


def bandpass_filter(y, sr, low=None, high=None):
    """
    Apply a Butterworth bandpass filter to restrict frequency content.

    Simulates band-limited recording or transmission. Cutoff frequencies
    are randomised each call unless provided.

    Args:
        y (np.ndarray): Audio signal.
        sr (int): Sample rate.
        low (float, optional): Low cutoff frequency in Hz.
            If None, sampled from [200, 1000].
        high (float, optional): High cutoff frequency in Hz.
            If None, sampled from [3000, 6000].

    Returns:
        np.ndarray: Band-filtered audio signal.
    """
    if low is None:
        low = np.random.uniform(200, 1000)
    if high is None:
        high = np.random.uniform(3000, 6000)
    if low >= high:
        low, high = 200, 4000
    b, a = butter(4, [low / (sr / 2), high / (sr / 2)], btype='band')
    return lfilter(b, a, y)


def augment_shift(y, max_shift=0.2):
    """
    Randomly shift the audio signal in time (circular).

    Provides temporal invariance — the same sound starting at different
    time offsets should produce the same label.

    Args:
        y (np.ndarray): Audio signal.
        max_shift (float): Maximum shift as a fraction of total length.
            Default: 0.2 (±20%).

    Returns:
        np.ndarray: Time-shifted audio signal.
    """
    shift = int(np.random.uniform(-max_shift, max_shift) * len(y))
    return np.roll(y, shift)


# Convenience map for use in training loops
AUGMENTATION_TYPES = ["noise", "gain", "bandpass", "shift"]


def apply_augmentation(y, sr, augment_type):
    """
    Dispatch function — apply a named augmentation to a signal.

    Args:
        y (np.ndarray): Audio signal.
        sr (int): Sample rate.
        augment_type (str): One of 'noise', 'gain', 'bandpass', 'shift'.

    Returns:
        np.ndarray: Augmented audio signal.
    """
    if augment_type == "noise":
        return augment_noise(y)
    elif augment_type == "gain":
        return augment_gain(y)
    elif augment_type == "bandpass":
        return bandpass_filter(y, sr)
    elif augment_type == "shift":
        return augment_shift(y)
    else:
        raise ValueError(f"Unknown augmentation type: '{augment_type}'")
