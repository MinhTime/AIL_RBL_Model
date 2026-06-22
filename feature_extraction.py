"""
feature_extraction.py
======================
Feature extraction module (paper Section III-F).

For every (pre-processed) heartbeat segment the following features are computed
and concatenated into a single fixed-length feature vector:

Amplitude / temporal / statistical features
--------------------------------------------
* R-peaks detected with ``peakutils.indexes(thres=0.5, min_dist=100)``.
* QRS duration estimated from the differences (``numpy.diff``) of R-peak
  locations.
* Statistical metrics of the R-peak amplitudes: mean, median, sum, std.

High-Order Statistics (HOS)
---------------------------
* Skewness  = sum((x_i - mean)^3) / (n * sigma^3)
* Kurtosis  = sum((x_i - mean)^4) / (n * sigma^4) - 3   (excess kurtosis)

Frequency-domain features
--------------------------
* Magnitude spectrum from the Fast Fourier Transform (FFT) of the signal.

The scalar features are placed first, followed by the FFT magnitude spectrum,
giving every beat a feature vector of identical length.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import peakutils

logger = logging.getLogger(__name__)

# Number of scalar (non-spectral) features produced per beat. Kept as a module
# constant so downstream code can reason about the feature layout if needed.
N_SCALAR_FEATURES = 9


def skewness(x: np.ndarray) -> float:
    """Skewness = sum((x_i - mean)^3) / (n * sigma^3)  (paper eq. 2)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0
    mean = x.mean()
    sigma = x.std()  # population standard deviation
    if sigma == 0:
        return 0.0
    return float(np.sum((x - mean) ** 3) / (n * sigma ** 3))


def kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis = sum((x_i - mean)^4) / (n * sigma^4) - 3  (paper eq. 3)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0
    mean = x.mean()
    sigma = x.std()
    if sigma == 0:
        return 0.0
    return float(np.sum((x - mean) ** 4) / (n * sigma ** 4) - 3.0)


def detect_r_peaks(signal: np.ndarray, thres: float = 0.5, min_dist: int = 100) -> np.ndarray:
    """Return the indices of R-peaks using peakutils with the paper's settings."""
    signal = np.asarray(signal, dtype=np.float64)
    # peakutils expects a normalised threshold in [0, 1] relative to the signal
    # amplitude range, which is exactly how the paper uses threshold=0.5.
    return peakutils.indexes(signal, thres=thres, min_dist=min_dist)


def _rpeak_amplitude_stats(signal: np.ndarray, peak_idx: np.ndarray) -> List[float]:
    """mean, median, sum, std of the R-peak amplitudes (0 if no peaks)."""
    if peak_idx.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    amps = signal[peak_idx]
    return [
        float(np.mean(amps)),
        float(np.median(amps)),
        float(np.sum(amps)),
        float(np.std(amps)),
    ]


def _qrs_duration_stats(peak_idx: np.ndarray) -> List[float]:
    """mean and std of the inter-R-peak intervals (proxy for QRS timing).

    QRS duration is estimated from numpy.diff() of the R-peak positions, as the
    paper specifies. With fewer than two peaks the interval is undefined -> 0.
    """
    if peak_idx.size < 2:
        return [0.0, 0.0]
    intervals = np.diff(peak_idx).astype(np.float64)
    return [float(np.mean(intervals)), float(np.std(intervals))]


def fft_magnitude(signal: np.ndarray) -> np.ndarray:
    """Magnitude spectrum (one-sided) of the FFT of ``signal`` (paper eq. 5)."""
    signal = np.asarray(signal, dtype=np.float64)
    spectrum = np.fft.rfft(signal)
    return np.abs(spectrum)


def extract_features(signal: np.ndarray, thres: float = 0.5, min_dist: int = 100) -> np.ndarray:
    """Compute the full feature vector for a single beat segment.

    Layout
    ------
    [ n_rpeaks,
      amp_mean, amp_median, amp_sum, amp_std,
      qrs_mean, qrs_std,
      skewness, kurtosis,
      <FFT magnitude spectrum ...> ]
    """
    signal = np.asarray(signal, dtype=np.float64)

    peak_idx = detect_r_peaks(signal, thres=thres, min_dist=min_dist)

    scalar_features: List[float] = []
    scalar_features.append(float(peak_idx.size))               # number of R-peaks
    scalar_features.extend(_rpeak_amplitude_stats(signal, peak_idx))  # 4 stats
    scalar_features.extend(_qrs_duration_stats(peak_idx))      # 2 stats
    scalar_features.append(skewness(signal))                   # HOS
    scalar_features.append(kurtosis(signal))                   # HOS

    spectral_features = fft_magnitude(signal)

    return np.concatenate([np.asarray(scalar_features, dtype=np.float64), spectral_features])


def extract_features_batch(
    signals: np.ndarray,
    thres: float = 0.5,
    min_dist: int = 100,
) -> np.ndarray:
    """Run :func:`extract_features` over every row of a 2-D array of beats."""
    signals = np.asarray(signals, dtype=np.float64)
    features = [extract_features(sig, thres=thres, min_dist=min_dist) for sig in signals]
    matrix = np.vstack(features)
    logger.info("Extracted feature matrix of shape %s.", matrix.shape)
    return matrix
