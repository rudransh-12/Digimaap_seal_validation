"""
SealScan -- Tampering Classifier Service.

Loads a pretrained sklearn-compatible model from disk once at startup and
exposes a predict() method. Falls back to a deterministic heuristic when no
model file is present (development / demo mode).

The classifier MUST NOT be retrained during an API request.
"""
from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np

from app.config.settings import (
    CLASSIFIER_PATH,
    CLASSIFIER_FEATURE_ORDER,
    RISK_CLASSES,
    HEURISTIC_THRESHOLDS,
)

logger = logging.getLogger("sealscan.tampering_classifier")

RiskClass = Literal["LOW", "MEDIUM", "HIGH"]

# Metric weights used by the FALLBACK heuristic only.
# Higher-is-better metrics are inverted so the composite score
# represents tampering risk (0 = clearly intact, 1 = likely tampered).
# SIFT is given 80% dominant weight (-0.80) for overwhelming authority
# on rotation, scale, and perspective invariant matching.
_HEURISTIC_WEIGHTS: dict[str, float] = {
    "sift_match_ratio":     -0.80,   # dominant invariant feature matcher (80%)
    "cosine_similarity":   -0.03,   # auxiliary global intensity alignment (3%)
    "ssim_score":          -0.05,   # auxiliary structural similarity (5%)
    "edge_difference":      0.04,   # auxiliary edge map difference (4%)
    "histogram_difference": 0.04,   # auxiliary color distribution difference (4%)
    "shape_difference":     0.04,   # auxiliary contour moment difference (4%)
}


class TamperingClassifier:
    """
    Loads and wraps the pretrained tampering classifier.

    Usage:
        classifier = TamperingClassifier()          # load once
        result = classifier.predict(aggregated_metrics)
    """

    def __init__(self) -> None:
        self._model = None
        self._mode: str = "heuristic"
        self._load_model()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        path = Path(CLASSIFIER_PATH)
        if path.is_file():
            try:
                with open(path, "rb") as f:
                    self._model = pickle.load(f)
                self._mode = "sklearn"
                logger.info("Tampering classifier loaded from %s", path)
            except Exception as exc:
                logger.error("Failed to load classifier from %s: %s. Using heuristic.", path, exc)
                self._model = None
                self._mode = "heuristic"
        else:
            logger.warning(
                "Classifier model not found at %s. Using built-in heuristic fallback.", path
            )
            self._mode = "heuristic"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def predict(self, aggregated_metrics: dict) -> tuple[float, RiskClass]:
        """
        Predict tampering risk from the aggregated metric dict.

        Returns:
            tampering_score (float in [0, 1]) -- higher = higher risk
            risk_class      (str)             -- LOW | MEDIUM | HIGH

        Note: tampering_score is NOT a calibrated probability unless
        the loaded model has been explicitly calibrated (CalibratedClassifierCV).
        """
        if self._mode == "sklearn" and self._model is not None:
            return self._predict_sklearn(aggregated_metrics)
        return self._predict_heuristic(aggregated_metrics)

    # ------------------------------------------------------------------
    # Internal: sklearn model
    # ------------------------------------------------------------------
    def _predict_sklearn(self, metrics: dict) -> tuple[float, RiskClass]:
        feature_vec = np.array(
            [[metrics[k] for k in CLASSIFIER_FEATURE_ORDER]], dtype=np.float64
        )

        # Try probability estimation first; fall back to decision function
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(feature_vec)[0]
            # Assume class ordering matches RISK_CLASSES index
            score = float(np.max(proba))
            class_idx = int(np.argmax(proba))
            risk_class: RiskClass = RISK_CLASSES[min(class_idx, len(RISK_CLASSES) - 1)]
        elif hasattr(self._model, "decision_function"):
            raw = self._model.decision_function(feature_vec)[0]
            # Normalise to [0,1] via sigmoid
            if isinstance(raw, np.ndarray):
                raw = float(raw[np.argmax(raw)])
            score = float(1 / (1 + np.exp(-raw)))
            label = self._model.predict(feature_vec)[0]
            risk_class = str(label).upper() if str(label).upper() in RISK_CLASSES else "MEDIUM"
        else:
            label = self._model.predict(feature_vec)[0]
            score = 0.5  # unknown confidence
            risk_class = str(label).upper() if str(label).upper() in RISK_CLASSES else "MEDIUM"

        return round(score, 6), risk_class

    # ------------------------------------------------------------------
    # Internal: heuristic fallback
    # ------------------------------------------------------------------
    def _predict_heuristic(self, metrics: dict) -> tuple[float, RiskClass]:
        """
        Weighted linear combination that produces a tampering risk score in [0,1].
        Tuned so:
            * Identical images  -> score ~0.0 -> LOW
            * Significantly different images -> score ~1.0 -> HIGH
        """
        raw_score = 0.0
        for metric, weight in _HEURISTIC_WEIGHTS.items():
            val = float(metrics.get(metric, 0.0))
            raw_score += weight * val

        # Shift to [0,1] range (raw_score in [-0.88, +0.12])
        score = float(np.clip((raw_score + 0.88) / 1.00, 0.0, 1.0))

        if score <= HEURISTIC_THRESHOLDS["LOW"]:
            risk_class: RiskClass = "LOW"
        elif score <= HEURISTIC_THRESHOLDS["MEDIUM"]:
            risk_class = "MEDIUM"
        else:
            risk_class = "HIGH"

        logger.info("Heuristic classifier | score=%.4f risk_class=%s", score, risk_class)
        return round(score, 6), risk_class

    @property
    def mode(self) -> str:
        return self._mode
