"""
preprocess.py
=============
Pre-processing module for ECG signals.

Two stages, exactly as described in the paper (Section III-C):

1. Wavelet denoising
   A 9-level ``bior4.4`` wavelet decomposition. Coefficients from index 0 up to
   ``cutoff_low`` and from ``cutoff_high`` to the end are zeroed out, which
   removes the corresponding (lowest- and highest-frequency) sub-bands before
   the signal is reconstructed.

2. Baseline fitting / removal
   Multiple median filtration of different widths estimates the slowly varying
   baseline (wander), which is then subtracted, keeping only the signal of
   interest.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pywt
from scipy.signal import medfilt

logger = logging.getLogger(__name__)

DEFAULT_WAVELET = "bior4.4"
DEFAULT_LEVEL = 9


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    cutoff_low: int = 1,
    cutoff_high: int = 1,
) -> np.ndarray:
    """Denoise a 1-D ECG signal via band-limited wavelet reconstruction.

    The signal is decomposed with ``pywt.wavedec`` into a coefficient list
    ``[cA_n, cD_n, cD_{n-1}, ..., cD_1]`` (low frequency first, high frequency
    last). The first ``cutoff_low`` coefficient arrays (lowest frequencies,
    i.e. baseline / very-low-frequency content) and the last ``cutoff_high``
    coefficient arrays (highest frequencies, i.e. high-frequency noise) are set
    to zero. The remaining mid-band coefficients are kept and the signal is
    reconstructed with ``pywt.waverec``.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    wavelet : str, default 'bior4.4'
        Mother wavelet.
    level : int, default 9
        Number of decomposition levels.
    cutoff_low : int, default 1
        Number of leading (lowest-frequency) coefficient arrays to zero out.
    cutoff_high : int, default 1
        Number of trailing (highest-frequency) coefficient arrays to zero out.

    Returns
    -------
    np.ndarray
        The denoised, length-matched signal.
    """
    signal = np.asarray(signal, dtype=np.float64)
    original_length = signal.shape[0]

    # Clamp the requested level to what the signal length actually supports.
    max_level = pywt.dwt_max_level(original_length, pywt.Wavelet(wavelet).dec_len)
    used_level = min(level, max_level)

    coeffs = pywt.wavedec(signal, wavelet, level=used_level)
    n = len(coeffs)

    # Zero the lowest-frequency arrays (0 .. cutoff_low) ...
    low = max(0, min(cutoff_low, n))
    for i in range(low):
        coeffs[i] = np.zeros_like(coeffs[i])

    # ... and the highest-frequency arrays (cutoff_high .. end).
    high = max(0, min(cutoff_high, n))
    for i in range(n - high, n):
        coeffs[i] = np.zeros_like(coeffs[i])

    reconstructed = pywt.waverec(coeffs, wavelet)
    # waverec may return a signal 1 sample longer due to padding; trim it.
    return reconstructed[:original_length]


def median_baseline_removal(
    signal: np.ndarray,
    widths: Sequence[int] = (71, 215),
) -> np.ndarray:
    """Estimate and subtract the baseline using successive median filters.

    A cascade of median filters with increasing (odd) widths progressively
    removes the QRS complexes, P and T waves, leaving an estimate of the
    low-frequency baseline wander. Subtracting that estimate from the original
    signal corrects the baseline.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal.
    widths : sequence of int, default (71, 215)
        Odd median-filter window widths applied in sequence. At 360 Hz these
        correspond to roughly 200 ms and 600 ms windows. Even values are
        incremented by 1 to stay odd (required by ``scipy.signal.medfilt``).

    Returns
    -------
    np.ndarray
        The baseline-corrected signal.
    """
    signal = np.asarray(signal, dtype=np.float64)
    baseline = signal.copy()
    for w in widths:
        kernel = w if w % 2 == 1 else w + 1
        kernel = min(kernel, _largest_odd_le(len(baseline)))
        if kernel < 1:
            continue
        baseline = medfilt(baseline, kernel_size=kernel)
    return signal - baseline


def _largest_odd_le(n: int) -> int:
    """Largest odd integer <= n (median kernels must be odd and <= signal len)."""
    if n <= 0:
        return 1
    return n if n % 2 == 1 else n - 1


def preprocess_signal(
    signal: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = DEFAULT_LEVEL,
    cutoff_low: int = 1,
    cutoff_high: int = 1,
    median_widths: Sequence[int] = (71, 215),
) -> np.ndarray:
    """Full single-signal pre-processing: wavelet denoise -> baseline removal."""
    denoised = wavelet_denoise(
        signal, wavelet=wavelet, level=level,
        cutoff_low=cutoff_low, cutoff_high=cutoff_high,
    )
    return median_baseline_removal(denoised, widths=median_widths)


def preprocess_batch(
    signals: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Apply :func:`preprocess_signal` to every row of a 2-D array of beats."""
    signals = np.asarray(signals, dtype=np.float64)
    out = np.empty_like(signals)
    for i in range(signals.shape[0]):
        out[i] = preprocess_signal(signals[i], **kwargs)
    logger.info("Pre-processed %d signals.", signals.shape[0])
    return out
