"""
synthetic.py
============
Synthetic minority over-sampling for the ECG classification pipeline.

This module replaces the original *duplication*-based balancing (which used
``sklearn.utils.resample`` with ``replace=True`` and therefore produced exact
copies of existing minority beats) with **SMOTE-family synthetic sample
generation**. New minority examples are *interpolated between real neighbours*
instead of being duplicated, which increases the diversity of the training set
and reduces the over-fitting / optimistic-bias risk of naive duplication.

Design notes
------------
* All routines operate on generic matrices ``X`` of shape ``(n_samples,
  n_features)`` and an integer label vector ``y``. Because the operation is
  agnostic to what each column represents, the *same* function balances either
  the raw R-peak-centred beat segments (**signal space**) or the extracted
  feature vectors (**feature space**).
* Balancing is only ever applied to the *training* split (the caller is
  responsible for splitting first), so no information leaks into the held-out
  test set.
* The majority class is left completely untouched; every other class is grown
  up to ``target_per_minority`` samples, exactly reproducing the original
  normalisation target (20,000 per minority) but with synthetic rather than
  duplicated data.

Two back-ends are provided:

``imbalanced-learn`` (preferred, already declared in ``requirements.txt``)
    Exposes the canonical, peer-reviewed implementations of SMOTE,
    Borderline-SMOTE and ADASYN.
NumPy fallback
    A small, self-contained implementation of vanilla SMOTE, used automatically
    when ``imbalanced-learn`` is not importable. It is kept deliberately simple
    and readable so the algorithm is transparent for the report.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional imbalanced-learn back-end (preferred when available).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - depends on the local environment
    from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE  # type: ignore

    _HAS_IMBLEARN = True
except Exception:  # ImportError or any transitive failure
    _HAS_IMBLEARN = False


# Methods that create *synthetic* data. ``duplicate`` is kept only so that the
# original behaviour can still be selected for A/B comparison in the report.
SYNTHETIC_METHODS: Tuple[str, ...] = ("smote", "borderline", "adasyn")
ALL_METHODS: Tuple[str, ...] = ("duplicate",) + SYNTHETIC_METHODS


# --------------------------------------------------------------------------- #
# Sampling-strategy helper.
# --------------------------------------------------------------------------- #
def _build_sampling_strategy(
    y: np.ndarray, target_per_minority: int
) -> Tuple[Dict[int, int], Optional[int]]:
    """Return ``({class -> desired_count}, majority_label)``.

    The largest class is treated as the majority and excluded (left untouched).
    Every other class whose current count is *below* ``target_per_minority`` is
    scheduled to be grown up to that target. A class already at/above the target
    is left as-is (SMOTE can only oversample, never shrink).
    """
    classes, counts = np.unique(y, return_counts=True)
    if classes.size == 0:
        return {}, None

    majority = int(classes[int(np.argmax(counts))])
    strategy: Dict[int, int] = {}
    for cls, cnt in zip(classes, counts):
        cls = int(cls)
        if cls == majority:
            continue
        if cnt < target_per_minority:
            strategy[cls] = int(target_per_minority)
        else:
            logger.info(
                "Class %d already has %d >= target %d; leaving untouched.",
                cls, int(cnt), target_per_minority,
            )
    return strategy, majority


# --------------------------------------------------------------------------- #
# NumPy SMOTE (fallback, and used to explain the algorithm in the report).
# --------------------------------------------------------------------------- #
def _smote_numpy_for_class(
    X_min: np.ndarray,
    n_synthetic: int,
    k_neighbors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate ``n_synthetic`` SMOTE samples for a single minority class.

    For each synthetic sample:
      1. pick a random base sample ``x_i`` from the minority class,
      2. pick one of its ``k`` nearest (same-class) neighbours ``x_j``,
      3. draw ``gap ~ U(0, 1)`` and set ``x_new = x_i + gap * (x_j - x_i)``.

    This places the new point somewhere on the segment joining two real
    same-class beats, i.e. inside the minority manifold rather than on top of an
    existing point (which is what plain duplication does).
    """
    from sklearn.neighbors import NearestNeighbors

    m, n_features = X_min.shape
    if n_synthetic <= 0:
        return np.empty((0, n_features), dtype=np.float64)
    if m == 1:
        # A single sample has no neighbours; fall back to replication.
        logger.warning("Only one sample for a minority class; replicating it.")
        return np.repeat(X_min.astype(np.float64), n_synthetic, axis=0)

    k = int(min(k_neighbors, m - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)  # +1 -> first is self
    neigh_idx = nn.kneighbors(X_min, return_distance=False)[:, 1:]  # drop self

    base = rng.integers(0, m, size=n_synthetic)
    steps = rng.random(size=n_synthetic)
    picks = rng.integers(0, k, size=n_synthetic)

    x_i = X_min[base]
    x_j = X_min[neigh_idx[base, picks]]
    synthetic = x_i + steps[:, None] * (x_j - x_i)
    return synthetic.astype(np.float64)


def _balance_numpy(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: Dict[int, int],
    k_neighbors: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy SMOTE balancing (keeps all originals, appends synthetics)."""
    rng = np.random.default_rng(random_state)
    X_parts = [X.astype(np.float64)]
    y_parts = [y]

    for cls, target in sampling_strategy.items():
        mask = y == cls
        X_cls = X[mask]
        n_new = int(target - X_cls.shape[0])
        if n_new <= 0:
            continue
        synth = _smote_numpy_for_class(X_cls, n_new, k_neighbors, rng)
        X_parts.append(synth)
        y_parts.append(np.full(synth.shape[0], cls, dtype=y.dtype))

    return np.vstack(X_parts), np.concatenate(y_parts)


# --------------------------------------------------------------------------- #
# imbalanced-learn back-end.
# --------------------------------------------------------------------------- #
def _balance_imblearn(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: Dict[int, int],
    method: str,
    k_neighbors: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance via imbalanced-learn's SMOTE / BorderlineSMOTE / ADASYN."""
    # ``k`` must be strictly smaller than the smallest class being oversampled.
    min_count = min(int(np.sum(y == c)) for c in sampling_strategy)
    k = max(1, min(int(k_neighbors), min_count - 1))
    if k != k_neighbors:
        logger.info("Clamped k_neighbors %d -> %d (smallest minority = %d).",
                    k_neighbors, k, min_count)

    common = dict(sampling_strategy=sampling_strategy, random_state=random_state)
    if method == "smote":
        sampler = SMOTE(k_neighbors=k, **common)
    elif method == "borderline":
        sampler = BorderlineSMOTE(k_neighbors=k, **common)
    elif method == "adasyn":
        sampler = ADASYN(n_neighbors=k, **common)
    else:  # pragma: no cover - guarded by balance_xy
        raise ValueError(f"Unknown synthetic method: {method!r}")

    X_res, y_res = sampler.fit_resample(X, y)
    return np.asarray(X_res, dtype=np.float64), np.asarray(y_res)


# --------------------------------------------------------------------------- #
# Duplication (original behaviour) - kept for A/B comparison only.
# --------------------------------------------------------------------------- #
def _balance_duplicate(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: Dict[int, int],
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Original method: random over-sampling *with replacement* (exact copies)."""
    from sklearn.utils import resample

    X_parts, y_parts = [], []
    for cls in np.unique(y):
        cls = int(cls)
        mask = y == cls
        if cls in sampling_strategy:
            X_cls, y_cls = resample(
                X[mask], y[mask], replace=True,
                n_samples=sampling_strategy[cls], random_state=random_state,
            )
        else:
            X_cls, y_cls = X[mask], y[mask]
        X_parts.append(X_cls)
        y_parts.append(y_cls)

    return np.vstack(X_parts), np.concatenate(y_parts)


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def balance_xy(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "smote",
    target_per_minority: int = 20_000,
    k_neighbors: int = 5,
    random_state: int = 42,
    prefer_imblearn: bool = True,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance ``(X, y)`` by growing every minority class to ``target_per_minority``.

    Parameters
    ----------
    X, y
        Feature/signal matrix and integer labels (training split only!).
    method
        One of ``{"smote", "borderline", "adasyn", "duplicate"}``. The first
        three synthesise new samples; ``"duplicate"`` reproduces the original
        copy-with-replacement behaviour and exists purely for comparison.
    target_per_minority
        Desired sample count for each minority class (paper's normalisation
        target of 20,000).
    k_neighbors
        Number of nearest same-class neighbours used for interpolation.
    random_state
        Reproducibility seed.
    prefer_imblearn
        Use imbalanced-learn when available; otherwise the NumPy SMOTE fallback
        is used (``borderline``/``adasyn`` gracefully degrade to plain SMOTE).
    shuffle
        Shuffle the returned dataset so classes are interleaved.

    Returns
    -------
    (X_resampled, y_resampled)
    """
    method = method.lower()
    if method not in ALL_METHODS:
        raise ValueError(f"method must be one of {ALL_METHODS}, got {method!r}")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    sampling_strategy, majority = _build_sampling_strategy(y, target_per_minority)
    if not sampling_strategy:
        logger.info("Nothing to balance (already balanced or single class).")
        return X, y
    logger.info("Majority class = %s; balancing plan = %s (method=%s).",
                majority, sampling_strategy, method)

    if method == "duplicate":
        X_res, y_res = _balance_duplicate(X, y, sampling_strategy, random_state)
    elif prefer_imblearn and _HAS_IMBLEARN:
        try:
            X_res, y_res = _balance_imblearn(
                X, y, sampling_strategy, method, k_neighbors, random_state
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("imbalanced-learn failed (%s); using NumPy SMOTE.", exc)
            X_res, y_res = _balance_numpy(X, y, sampling_strategy, k_neighbors, random_state)
    else:
        if method in ("borderline", "adasyn"):
            logger.warning(
                "Method %r needs imbalanced-learn (not installed); "
                "falling back to plain NumPy SMOTE.", method,
            )
        X_res, y_res = _balance_numpy(X, y, sampling_strategy, k_neighbors, random_state)

    if shuffle:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(y_res))
        X_res, y_res = X_res[perm], y_res[perm]

    return X_res, y_res
