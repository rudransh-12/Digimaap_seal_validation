"""
SealScan -- Image Blur Classifier Service.

Loads the pretrained blur classification RandomForest model from disk once at startup
and provides feature extraction and prediction (P(BLURRY) vs P(NOT_BLURRY)).
Falls back gracefully to a heuristic if the model file cannot be loaded.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import joblib
import numpy as np

try:
    import skimage.measure
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

from app.config.settings import (
    BLUR_CLASSIFIER_PATH,
    BLUR_CLASSIFIER_FEATURE_ORDER,
    QUALITY_THRESHOLDS,
)

logger = logging.getLogger("sealscan.blur_classifier")


class BlurPrediction(NamedTuple):
    passed: bool
    probability_not_blurry: float
    probability_blurry: float
    discrete_prediction: str
    feature_dict: dict[str, float]


def extract_blur_features(gray: np.ndarray) -> dict[str, float]:
    """
    Extract the 7 numerical sharpness / blur features in the exact order
    expected by the trained RandomForest classifier.
    """
    h, w = gray.shape[:2]

    # 1. Laplacian variance
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 2. Edge strength (mean magnitude of Sobel gradient vectors)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

    # 3. Noise (Gaussian residual std-dev)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    diff = gray.astype(np.float32) - blurred.astype(np.float32)
    noise = float(np.std(diff))

    # 4. Brenner sharpness (horizontal second-difference)
    if w > 2:
        diff_x = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
        brenner = float(np.mean(diff_x ** 2))
    else:
        brenner = 0.0

    # 5. Canny edge density with thresholds (100, 200)
    edges = cv2.Canny(gray, 100, 200)
    canny_density = float(np.count_nonzero(edges) / max(edges.size, 1))

    # 6. FFT high-frequency ratio
    cy, cx = h // 2, w // 2
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    mag = np.abs(f)
    r = max(1, int(min(h, w) * 0.1))
    y, x = np.ogrid[:h, :w]
    dc_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
    total_energy = float(np.sum(mag))
    high_freq_energy = float(np.sum(mag[~dc_mask]))
    fft_ratio = float(high_freq_energy / (total_energy + 1e-8))

    # 7. Blur effect via skimage.measure.blur_effect
    if _HAS_SKIMAGE:
        try:
            blur_eff = float(skimage.measure.blur_effect(gray))
        except Exception:
            blur_eff = 0.5
    else:
        blur_eff = 0.5

    return {
        "laplacian_variance": laplacian_var,
        "edge_strength": edge_strength,
        "noise": noise,
        "brenner_sharpness": brenner,
        "canny_edge_density": canny_density,
        "fft_high_frequency_ratio": fft_ratio,
        "blur_effect": blur_eff,
    }


class BlurClassifier:
    """
    Loads and wraps the pretrained blur classifier.
    """

    def __init__(self) -> None:
        self._model = None
        self._mode: str = "heuristic"
        self._load_model()

    def _load_model(self) -> None:
        path = Path(BLUR_CLASSIFIER_PATH)
        if path.is_file():
            try:
                loaded = joblib.load(path)
                if isinstance(loaded, dict) and "model" in loaded:
                    self._model = loaded["model"]
                else:
                    self._model = loaded
                self._mode = "sklearn"
                logger.info("Blur classifier loaded successfully from %s", path)
            except Exception as exc:
                logger.error("Failed to load blur classifier from %s: %s. Using heuristic fallback.", path, exc)
                self._model = None
                self._mode = "heuristic"
        else:
            logger.warning("Blur classifier model not found at %s. Using heuristic fallback.", path)
            self._mode = "heuristic"

    def predict(self, gray: np.ndarray, threshold: float | None = None) -> BlurPrediction:
        """
        Run blur prediction on a greyscale image.

        Args:
            gray: 2D uint8 greyscale image
            threshold: optional custom threshold for probability_not_blurry (defaults to settings threshold)
        """
        thresh = threshold if threshold is not None else float(QUALITY_THRESHOLDS.get("min_blur_classifier_score", 0.50))
        features = extract_blur_features(gray)

        if self._mode == "sklearn" and self._model is not None:
            return self._predict_sklearn(features, thresh)
        return self._predict_heuristic(features, thresh)

    def _predict_sklearn(self, features: dict[str, float], thresh: float) -> BlurPrediction:
        vec = np.array([[features[k] for k in BLUR_CLASSIFIER_FEATURE_ORDER]], dtype=np.float64)
        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(vec)[0]
            # Class 0: BLURRY, Class 1: NOT_BLURRY
            p_blurry = float(probs[0])
            p_not_blurry = float(probs[1]) if len(probs) > 1 else float(1.0 - p_blurry)
        else:
            pred = self._model.predict(vec)[0]
            p_not_blurry = 1.0 if pred == 1 else 0.0
            p_blurry = 1.0 - p_not_blurry

        passed = p_not_blurry >= thresh
        discrete = "NOT_BLURRY" if passed else "BLURRY"

        return BlurPrediction(
            passed=passed,
            probability_not_blurry=round(p_not_blurry, 4),
            probability_blurry=round(p_blurry, 4),
            discrete_prediction=discrete,
            feature_dict=features,
        )

    def _predict_heuristic(self, features: dict[str, float], thresh: float) -> BlurPrediction:
        """
        Heuristic fallback based on laplacian variance and canny edge density.
        """
        lap = features.get("laplacian_variance", 0.0)
        canny = features.get("canny_edge_density", 0.0)

        # Normalize score into [0, 1]
        score = min(1.0, max(0.0, (lap / 200.0) * 0.7 + (canny / 0.05) * 0.3))
        passed = score >= thresh

        return BlurPrediction(
            passed=passed,
            probability_not_blurry=round(score, 4),
            probability_blurry=round(1.0 - score, 4),
            discrete_prediction="NOT_BLURRY" if passed else "BLURRY",
            feature_dict=features,
        )

    @property
    def mode(self) -> str:
        return self._mode
