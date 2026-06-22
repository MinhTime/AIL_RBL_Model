"""
main.py
=======
End-to-end driver for the ECG classification pipeline, tying together
``dataset.py``, ``preprocess.py``, ``feature_extraction.py`` and ``trainer.py``.

Pipeline (paper Figure 4)
-------------------------
    load raw beats
        -> stratified 80/20 split
        -> balance training set (up-sample minorities to 20,000)
        -> pre-process (wavelet denoise + median baseline removal)
        -> feature extraction (R-peaks, QRS, stats, HOS, FFT)
        -> train Random Forest  -> checkpoint
        -> evaluate (confusion matrix, accuracy, sensitivity, specificity, PPV)

Usage
-----
    # Train (and immediately evaluate) on the MIT-BIH records in ./mitdb
    python main.py train --data-dir ./mitdb --model-out rf_model.joblib

    # Evaluate independently using a saved checkpoint
    python main.py test  --data-dir ./mitdb --model-in  rf_model.joblib

The split uses a fixed random seed, so the held-out test set is identical
between the ``train`` and ``test`` runs.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

import dataset as ds
import feature_extraction as fe
import preprocess as pp
import trainer as tr


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_record_preprocessor(args: argparse.Namespace):
    """Build a full-record signal preprocessor (faithful 9-level denoising)."""
    if args.preprocess_level != "record":
        return None

    def _prep(signal: np.ndarray) -> np.ndarray:
        return pp.preprocess_signal(
            signal,
            level=args.wavelet_level,
            cutoff_low=args.cutoff_low,
            cutoff_high=args.cutoff_high,
            median_widths=tuple(args.median_widths),
        )

    return _prep


def _build_feature_matrices(
    train_beats: ds.BeatDataset,
    test_beats: ds.BeatDataset,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-process (if per-beat) and feature-extract both splits.

    When ``--preprocess-level record`` is used, denoising already happened on
    the full record inside the loader, so beats are passed straight to feature
    extraction. When ``--preprocess-level beat`` is used, each beat segment is
    denoised here instead.
    """
    if args.preprocess_level == "beat":
        X_train_sig = pp.preprocess_batch(
            train_beats.X,
            level=args.wavelet_level,
            cutoff_low=args.cutoff_low,
            cutoff_high=args.cutoff_high,
            median_widths=tuple(args.median_widths),
        )
        X_test_sig = pp.preprocess_batch(
            test_beats.X,
            level=args.wavelet_level,
            cutoff_low=args.cutoff_low,
            cutoff_high=args.cutoff_high,
            median_widths=tuple(args.median_widths),
        )
    else:
        X_train_sig, X_test_sig = train_beats.X, test_beats.X

    X_train = fe.extract_features_batch(X_train_sig, thres=args.peak_thres, min_dist=args.peak_min_dist)
    X_test = fe.extract_features_batch(X_test_sig, thres=args.peak_thres, min_dist=args.peak_min_dist)
    return X_train, train_beats.y, X_test, test_beats.y


def run_train(args: argparse.Namespace) -> None:
    beats = ds.load_mitbih_beats(
        args.data_dir,
        window_size=args.window_size,
        signal_preprocessor=_make_record_preprocessor(args),
    )
    train_beats, test_beats = ds.split_train_test(
        beats, test_size=args.test_size, random_state=args.seed
    )
    train_beats = ds.balance_by_upsampling(
        train_beats, target_per_minority=args.target_per_minority, random_state=args.seed
    )

    X_train, y_train, X_test, y_test = _build_feature_matrices(train_beats, test_beats, args)

    model = tr.train_random_forest(X_train, y_train, random_state=args.seed)
    tr.save_model(model, args.model_out)

    report = tr.evaluate(model, X_test, y_test, class_names=list(ds.AAMI_CLASSES))
    print(report)


def run_test(args: argparse.Namespace) -> None:
    beats = ds.load_mitbih_beats(
        args.data_dir,
        window_size=args.window_size,
        signal_preprocessor=_make_record_preprocessor(args),
    )
    # Reproduce the identical split (same seed) to recover the held-out test set.
    _, test_beats = ds.split_train_test(
        beats, test_size=args.test_size, random_state=args.seed
    )

    if args.preprocess_level == "beat":
        X_test_sig = pp.preprocess_batch(
            test_beats.X,
            level=args.wavelet_level,
            cutoff_low=args.cutoff_low,
            cutoff_high=args.cutoff_high,
            median_widths=tuple(args.median_widths),
        )
    else:
        X_test_sig = test_beats.X
    X_test = fe.extract_features_batch(X_test_sig, thres=args.peak_thres, min_dist=args.peak_min_dist)

    model = tr.load_model(args.model_in)
    report = tr.evaluate(model, X_test, test_beats.y, class_names=list(ds.AAMI_CLASSES))
    print(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MIT-BIH ECG classification pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", required=True, help="Directory with WFDB records (.dat/.hea/.atr).")
        p.add_argument("--window-size", type=int, default=360, help="Beat window length in samples.")
        p.add_argument("--test-size", type=float, default=0.20, help="Test split fraction.")
        p.add_argument("--seed", type=int, default=42, help="Random seed (also fixes the split).")
        p.add_argument("--preprocess-level", choices=["record", "beat"], default="record",
                       help="Denoise the full record (faithful 9-level) or each beat segment.")
        p.add_argument("--wavelet-level", type=int, default=9, help="Wavelet decomposition levels.")
        p.add_argument("--cutoff-low", type=int, default=1, help="Wavelet low-frequency arrays to zero.")
        p.add_argument("--cutoff-high", type=int, default=1, help="Wavelet high-frequency arrays to zero.")
        p.add_argument("--median-widths", type=int, nargs="+", default=[71, 215],
                       help="Median-filter widths for baseline removal.")
        p.add_argument("--peak-thres", type=float, default=0.5, help="peakutils threshold.")
        p.add_argument("--peak-min-dist", type=int, default=100, help="peakutils minimum distance.")
        p.add_argument("--verbose", action="store_true", help="Enable INFO logging.")

    p_train = sub.add_parser("train", help="Train + checkpoint + evaluate.")
    add_common(p_train)
    p_train.add_argument("--model-out", default="rf_model.joblib", help="Checkpoint output path.")
    p_train.add_argument("--target-per-minority", type=int, default=20_000,
                         help="Up-sampling target for each minority class.")

    p_test = sub.add_parser("test", help="Evaluate using a saved checkpoint.")
    add_common(p_test)
    p_test.add_argument("--model-in", default="rf_model.joblib", help="Checkpoint to load.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(getattr(args, "verbose", False))

    if args.command == "train":
        run_train(args)
    elif args.command == "test":
        run_test(args)


if __name__ == "__main__":
    main()
