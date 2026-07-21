"""
dataset.py
==========
Data Loader & Balancer for the MIT-BIH Arrhythmia ECG classification pipeline.

Responsibilities
----------------
1. Load the MIT-BIH Arrhythmia database (raw WFDB records) and extract individual
   heartbeat segments centered on the annotated R-peaks.
2. Map the 15 original annotation symbols into the 5 AAMI target classes:
   N, SVEB, VEB, FB (F), Q.
3. Split the data into 80% training / 20% testing (stratified).
4. Balance the *training* set by up-sampling each minority class to a target
   count. Two strategies are supported:
      * ``duplicate`` - random over-sampling with replacement (the original,
        paper-faithful behaviour: exact copies of existing beats);
      * ``smote`` / ``borderline`` / ``adasyn`` - **synthetic** minority
        over-sampling that interpolates new beats between real neighbours
        (see ``synthetic.py``). This is the modification introduced in this
        work: instead of duplicating beats we *generate* new, diverse ones.

Notes
-----
The MIT-BIH Arrhythmia Database must be available locally as WFDB records
(.dat / .hea / .atr). It can be downloaded from PhysioNet:
    https://physionet.org/content/mitdb/1.0.0/
or programmatically with ``wfdb.dl_database('mitdb', dl_dir=...)``.

Balancing always runs *after* the train/test split and only on the training
data, so no synthetic information can leak into the held-out test set.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import synthetic as syn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AAMI mapping: 15 raw MIT-BIH annotation symbols -> 5 target classes.
#
# The paper (Table 1) summarises the 15 MIT-BIH heartbeat types into the five
# AAMI EC57 classes. We use the de Chazal et al. mapping, which is the
# community-standard interpretation of the AAMI grouping:
#   N    (Normal)                     : N, L, R, e, j
#   SVEB (Supraventricular ectopic)   : A, a, J, S
#   VEB  (Ventricular ectopic)        : V, E
#   F    (Fusion)                     : F
#   Q    (Unknown / paced)            : /, f, Q
# ---------------------------------------------------------------------------
AAMI_CLASSES: Tuple[str, ...] = ("N", "SVEB", "VEB", "FB", "Q")

SYMBOL_TO_AAMI: Dict[str, str] = {
    # --- N : Normal -------------------------------------------------------
    "N": "N",   # Normal beat
    "L": "N",   # Left bundle branch block beat
    "R": "N",   # Right bundle branch block beat
    "e": "N",   # Atrial escape beat
    "j": "N",   # Nodal (junctional) escape beat
    # --- SVEB : Supraventricular ectopic ---------------------------------
    "A": "SVEB",  # Atrial premature beat
    "a": "SVEB",  # Aberrated atrial premature beat
    "J": "SVEB",  # Nodal (junctional) premature beat
    "S": "SVEB",  # Supraventricular premature beat
    # --- VEB : Ventricular ectopic ---------------------------------------
    "V": "VEB",   # Premature ventricular contraction
    "E": "VEB",   # Ventricular escape beat
    # --- FB : Fusion ------------------------------------------------------
    "F": "FB",    # Fusion of ventricular and normal beat
    # --- Q : Unknown / paced ---------------------------------------------
    "/": "Q",     # Paced beat
    "f": "Q",     # Fusion of paced and normal beat
    "Q": "Q",     # Unclassifiable beat
}

# Integer label encoding used throughout the pipeline (and in the confusion
# matrix order: 0=N, 1=SVEB, 2=VEB, 3=FB, 4=Q).
CLASS_TO_INT: Dict[str, int] = {c: i for i, c in enumerate(AAMI_CLASSES)}
INT_TO_CLASS: Dict[int, str] = {i: c for c, i in CLASS_TO_INT.items()}

# Standard MIT-BIH sampling frequency.
SAMPLING_RATE_HZ: int = 360


@dataclass
class BeatDataset:
    """Container for extracted heartbeat segments and their integer labels."""

    X: np.ndarray  # shape (n_beats, window_length), raw signal segments
    y: np.ndarray  # shape (n_beats,), integer class labels

    def __len__(self) -> int:
        return len(self.y)

    def class_distribution(self) -> Dict[str, int]:
        unique, counts = np.unique(self.y, return_counts=True)
        return {INT_TO_CLASS[int(u)]: int(c) for u, c in zip(unique, counts)}


def _list_record_names(data_dir: str) -> List[str]:
    """Return the base names of every WFDB record (.hea) found in ``data_dir``."""
    records = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(data_dir)
        if f.endswith(".hea")
    )
    if not records:
        raise FileNotFoundError(
            f"No WFDB header (.hea) files found in '{data_dir}'. "
            "Download the MIT-BIH Arrhythmia Database from PhysioNet first."
        )
    return records


def load_mitbih_beats(
    data_dir: str,
    window_size: int = 360,
    channel: int = 0,
    records: Sequence[str] | None = None,
    signal_preprocessor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> BeatDataset:
    """Load every annotated heartbeat from the MIT-BIH records in ``data_dir``.

    For each annotation that maps to one of the five AAMI classes, a fixed-length
    window centered on the R-peak sample is extracted.

    Parameters
    ----------
    data_dir : str
        Directory containing the WFDB records (.dat/.hea/.atr).
    window_size : int, default 360
        Length (in samples) of each extracted beat segment (~1 s at 360 Hz).
    channel : int, default 0
        Which ECG lead/channel to read.
    records : sequence of str, optional
        Explicit list of record names. If ``None``, every record in the
        directory is used.
    signal_preprocessor : callable, optional
        If given, applied to the *full-length* channel signal of each record
        before beats are segmented. This is the faithful way to run the paper's
        9-level ``bior4.4`` wavelet denoising, which requires a long signal
        (a single 360-sample beat only supports ~5 levels).

    Returns
    -------
    BeatDataset
    """
    import wfdb  # imported lazily so the module loads even without wfdb installed

    record_names = list(records) if records is not None else _list_record_names(data_dir)
    half = window_size // 2

    segments: List[np.ndarray] = []
    labels: List[int] = []

    for rec in record_names:
        rec_path = os.path.join(data_dir, rec)
        try:
            record = wfdb.rdrecord(rec_path)
            annotation = wfdb.rdann(rec_path, "atr")
        except Exception as exc:  # pragma: no cover - depends on local files
            logger.warning("Skipping record %s (%s)", rec, exc)
            continue

        signal = record.p_signal[:, channel].astype(np.float64)
        if signal_preprocessor is not None:
            signal = np.asarray(signal_preprocessor(signal), dtype=np.float64)
        n_samples = signal.shape[0]

        for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
            aami = SYMBOL_TO_AAMI.get(symbol)
            if aami is None:
                continue  # symbol not part of the 5-class scheme -> discard
            start = sample_idx - half
            end = start + window_size
            if start < 0 or end > n_samples:
                continue  # incomplete window at signal boundary
            segments.append(signal[start:end])
            labels.append(CLASS_TO_INT[aami])

    if not segments:
        raise RuntimeError("No valid heartbeat segments were extracted.")

    X = np.asarray(segments, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    logger.info("Loaded %d beats from %d records.", len(y), len(record_names))
    return BeatDataset(X=X, y=y)


def split_train_test(
    dataset: BeatDataset,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[BeatDataset, BeatDataset]:
    """Stratified 80/20 train-test split (preserves class proportions)."""
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.y,
    )
    logger.info(
        "Split: train=%d (%.0f%%), test=%d (%.0f%%)",
        len(y_train), (1 - test_size) * 100, len(y_test), test_size * 100,
    )
    return BeatDataset(X_train, y_train), BeatDataset(X_test, y_test)


def balance_by_upsampling(
    dataset: BeatDataset,
    target_per_minority: int = 20_000,
    random_state: int = 42,
) -> BeatDataset:
    """Up-sample every *minority* class by DUPLICATION (with replacement).

    This is the original, paper-faithful strategy: each minority class is
    resampled *with replacement* up to ``target_per_minority`` samples, so the
    added beats are exact copies of existing ones. The majority class is left
    untouched. Kept as a baseline for comparison against the synthetic
    (``balance_by_smote``) strategy.
    """
    counts = dataset.class_distribution()
    majority_class = max(counts, key=counts.get)
    majority_int = CLASS_TO_INT[majority_class]
    logger.info("Majority class detected: %s (%d samples).", majority_class, counts[majority_class])

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for class_name, class_int in CLASS_TO_INT.items():
        mask = dataset.y == class_int
        X_cls = dataset.X[mask]
        y_cls = dataset.y[mask]
        if X_cls.shape[0] == 0:
            logger.warning("Class %s has no samples; skipping.", class_name)
            continue
        if class_int == majority_int:
            # Keep the majority class exactly as-is.
            X_parts.append(X_cls)
            y_parts.append(y_cls)
            continue
        # Up-sample the minority class with replacement to the target size.
        X_up, y_up = resample(
            X_cls,
            y_cls,
            replace=True,
            n_samples=target_per_minority,
            random_state=random_state,
        )
        X_parts.append(X_up)
        y_parts.append(y_up)
        logger.info("Duplicated %s: %d -> %d.", class_name, X_cls.shape[0], target_per_minority)

    X_balanced = np.vstack(X_parts)
    y_balanced = np.concatenate(y_parts)

    # Shuffle the combined dataset so classes are interleaved.
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(y_balanced))
    balanced = BeatDataset(X_balanced[perm], y_balanced[perm])
    logger.info("Balanced (duplicate) training distribution: %s", balanced.class_distribution())
    return balanced


def balance_by_smote(
    dataset: BeatDataset,
    target_per_minority: int = 20_000,
    k_neighbors: int = 5,
    random_state: int = 42,
    method: str = "smote",
) -> BeatDataset:
    """Up-sample every minority class with SYNTHETIC beats (SMOTE family).

    Unlike :func:`balance_by_upsampling`, the added beats are *not* copies:
    each new beat is interpolated between a real minority beat and one of its
    ``k_neighbors`` nearest same-class neighbours (see ``synthetic.py``). This
    yields a more diverse, less redundant training set.

    Parameters
    ----------
    dataset : BeatDataset
        The (training) dataset to balance - raw R-peak-centred beat segments.
    target_per_minority : int, default 20_000
        Desired sample count for each minority class.
    k_neighbors : int, default 5
        Number of nearest neighbours used for interpolation.
    random_state : int, default 42
        Reproducibility seed.
    method : {"smote", "borderline", "adasyn"}, default "smote"
        Which synthetic-oversampling variant to use.
    """
    before = dataset.class_distribution()
    X_res, y_res = syn.balance_xy(
        dataset.X,
        dataset.y,
        method=method,
        target_per_minority=target_per_minority,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    balanced = BeatDataset(X=X_res.astype(np.float64), y=y_res.astype(np.int64))
    logger.info("Balanced (%s) training distribution: %s -> %s",
                method, before, balanced.class_distribution())
    return balanced


def balance_training_set(
    dataset: BeatDataset,
    method: str = "smote",
    target_per_minority: int = 20_000,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> BeatDataset:
    """Dispatch balancing to the chosen strategy.

    ``method="duplicate"`` reproduces the original copy-with-replacement
    behaviour; any other value ("smote", "borderline", "adasyn") produces
    synthetic beats.
    """
    if method == "duplicate":
        return balance_by_upsampling(
            dataset, target_per_minority=target_per_minority, random_state=random_state
        )
    return balance_by_smote(
        dataset,
        target_per_minority=target_per_minority,
        k_neighbors=k_neighbors,
        random_state=random_state,
        method=method,
    )
