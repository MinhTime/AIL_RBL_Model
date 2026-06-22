"""
trainer.py
==========
Classifier, tester & checkpoint module (paper Section III-H / IV).

* Trains a ``RandomForestClassifier`` with the paper's hyper-parameters
  (n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1).
* Saves / loads the trained model with joblib so testing can be run
  independently of training.
* Evaluates the model, producing a confusion matrix together with per-class
  Accuracy, Sensitivity, Specificity and Positive Predictive Value (PPV),
  plus the overall accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

# Class order for the confusion matrix / reports (matches dataset.AAMI_CLASSES).
DEFAULT_CLASS_NAMES: List[str] = ["N", "SVEB", "VEB", "FB", "Q"]


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Instantiate the Random Forest with the exact paper hyper-parameters."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",   # paper: "auto"/"sqrt" -> sqrt of n_features
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit the Random Forest classifier on the training features."""
    model = build_random_forest(random_state=random_state)
    logger.info("Training RandomForest on %d samples, %d features ...",
                X_train.shape[0], X_train.shape[1])
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


def save_model(model: RandomForestClassifier, path: str) -> None:
    """Checkpoint the trained model to disk (joblib)."""
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path: str) -> RandomForestClassifier:
    """Load a previously checkpointed model from disk."""
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


@dataclass
class EvaluationReport:
    """Holds all evaluation outputs."""

    confusion: np.ndarray
    overall_accuracy: float
    class_names: List[str]
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __str__(self) -> str:  # pretty, console-friendly summary
        lines: List[str] = []
        lines.append(f"Overall accuracy: {self.overall_accuracy:.4f}")
        lines.append("")
        header = f"{'Class':<8}{'Accuracy':>10}{'Sensitivity':>13}{'Specificity':>13}{'PPV':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        for cls in self.class_names:
            m = self.per_class[cls]
            lines.append(
                f"{cls:<8}{m['accuracy']:>10.4f}{m['sensitivity']:>13.4f}"
                f"{m['specificity']:>13.4f}{m['ppv']:>10.4f}"
            )
        lines.append("")
        lines.append("Confusion matrix (rows = actual, cols = predicted):")
        col_header = "        " + "".join(f"{c:>8}" for c in self.class_names)
        lines.append(col_header)
        for i, cls in enumerate(self.class_names):
            row = "".join(f"{int(v):>8}" for v in self.confusion[i])
            lines.append(f"{cls:>8}{row}")
        return "\n".join(lines)


def evaluate(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str] | None = None,
) -> EvaluationReport:
    """Evaluate the model and compute per-class clinical metrics.

    For each class, the standard one-vs-rest quantities are derived from the
    confusion matrix:
        TP : correctly predicted instances of the class
        FN : actual class instances predicted as another class
        FP : other-class instances predicted as this class
        TN : everything else
    and:
        Accuracy    = (TP + TN) / (TP + TN + FP + FN)
        Sensitivity = TP / (TP + FN)        (recall / true positive rate)
        Specificity = TN / (TN + FP)        (true negative rate)
        PPV         = TP / (TP + FP)        (precision)
    """
    class_names = class_names or DEFAULT_CLASS_NAMES
    labels = list(range(len(class_names)))

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    total = cm.sum()
    overall_accuracy = float(np.trace(cm) / total) if total else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    for i, cls in enumerate(class_names):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - tp)
        fp = float(cm[:, i].sum() - tp)
        tn = float(total - tp - fn - fp)

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        accuracy = (tp + tn) / total if total else 0.0

        per_class[cls] = {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
        }

    report = EvaluationReport(
        confusion=cm,
        overall_accuracy=overall_accuracy,
        class_names=class_names,
        per_class=per_class,
    )
    logger.info("Evaluation complete. Overall accuracy = %.4f", overall_accuracy)
    return report
