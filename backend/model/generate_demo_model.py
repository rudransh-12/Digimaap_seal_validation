"""
SealScan -- Demo Model Generator
Generates and saves a demo RandomForestClassifier trained on synthetic
seal similarity data. Run this script once to produce tampering_classifier.pkl.

Usage:
    python model/generate_demo_model.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "tampering_classifier.pkl"

# Feature order: [cosine, orb, ssim, edge, hist, shape]
# Risk: 0=LOW, 1=MEDIUM, 2=HIGH


def _generate_samples(n: int, risk: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate synthetic feature samples for a given risk class.
    Higher-risk samples have lower similarity metrics and higher difference metrics.
    """
    if risk == 0:   # LOW -- very similar images
        cosine = rng.uniform(0.85, 1.00, n)
        orb    = rng.uniform(0.60, 1.00, n)
        ssim   = rng.uniform(0.80, 1.00, n)
        edge   = rng.uniform(0.00, 0.10, n)
        hist   = rng.uniform(0.00, 0.10, n)
        shape  = rng.uniform(0.00, 0.08, n)
    elif risk == 1:  # MEDIUM -- some differences
        cosine = rng.uniform(0.50, 0.85, n)
        orb    = rng.uniform(0.30, 0.65, n)
        ssim   = rng.uniform(0.45, 0.82, n)
        edge   = rng.uniform(0.10, 0.35, n)
        hist   = rng.uniform(0.10, 0.40, n)
        shape  = rng.uniform(0.08, 0.35, n)
    else:            # HIGH -- very different / tampered
        cosine = rng.uniform(0.00, 0.55, n)
        orb    = rng.uniform(0.00, 0.35, n)
        ssim   = rng.uniform(0.00, 0.50, n)
        edge   = rng.uniform(0.35, 1.00, n)
        hist   = rng.uniform(0.40, 1.00, n)
        shape  = rng.uniform(0.35, 1.00, n)

    return np.column_stack([cosine, orb, ssim, edge, hist, shape])


def main() -> None:
    rng = np.random.default_rng(2024)

    n_per_class = 500
    X_low    = _generate_samples(n_per_class, 0, rng)
    X_medium = _generate_samples(n_per_class, 1, rng)
    X_high   = _generate_samples(n_per_class, 2, rng)

    X = np.vstack([X_low, X_medium, X_high])
    y = np.array(["LOW"] * n_per_class + ["MEDIUM"] * n_per_class + ["HIGH"] * n_per_class)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("=== Classification Report (test set) ===")
    print(classification_report(y_test, clf.predict(X_test)))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
