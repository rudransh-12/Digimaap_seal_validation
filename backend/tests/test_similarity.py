"""
Tests for Feature Extraction, Aggregation, and Classifier services.
"""
import numpy as np
import pytest

from app.services.preprocessing import ImagePreprocessor
from app.services.feature_extraction import FeatureExtractor
from app.services.reference_aggregation import aggregate
from app.services.tampering_classifier import TamperingClassifier
from app.config.settings import CLASSIFIER_FEATURE_ORDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gradient_bgr(size: int = 512, variant: int = 0) -> np.ndarray:
    """
    Horizontal gradient + stripe texture + mild noise.
    Passes quality checks and has good edge structure for ORB/SSIM tests.
    """
    rng = np.random.default_rng(42 + variant)
    x = np.linspace(0, 255, size).astype(np.uint8)
    base = np.tile(x, (size, 1))
    for i in range(variant * 7, size, 32):
        base[i:i + 16, :] = np.clip(base[i:i + 16, :].astype(np.int32) + 40, 0, 255).astype(np.uint8)
    noise = rng.integers(-5, 6, (size, size)).astype(np.int16)
    gray = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _solid(val: int = 128, size: int = 512) -> np.ndarray:
    return np.full((size, size, 3), val, dtype=np.uint8)



_preprocessor = ImagePreprocessor()
_extractor = FeatureExtractor()


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------
class TestPreprocessing:
    def test_output_size(self):
        from app.config.settings import PREPROCESSING
        target_w, target_h = PREPROCESSING["target_size"]
        bgr = _gradient_bgr(size=1024)
        prep = _preprocessor.process(bgr)
        assert prep.bgr.shape[:2] == (target_h, target_w)
        assert prep.gray.shape == (target_h, target_w)
        assert prep.hsv.shape[:2] == (target_h, target_w)
        assert prep.edges.shape == (target_h, target_w)
        assert prep.gray_norm.shape == (target_h, target_w)

    def test_gray_norm_range(self):
        bgr = _gradient_bgr()
        prep = _preprocessor.process(bgr)
        assert prep.gray_norm.min() >= 0.0
        assert prep.gray_norm.max() <= 1.0


# ---------------------------------------------------------------------------
# Feature Extraction tests
# ---------------------------------------------------------------------------
class TestFeatureExtraction:
    def _prep(self, bgr: np.ndarray):
        return _preprocessor.process(bgr)

    def test_identical_images_high_similarity(self):
        img = _gradient_bgr()
        p = self._prep(img)
        metrics = _extractor.compute(p, p)
        assert metrics.cosine_similarity > 0.99
        assert metrics.ssim_score > 0.99
        assert metrics.edge_difference < 0.01
        assert metrics.histogram_difference < 0.05

    def test_different_images_lower_similarity(self):
        img_a = _gradient_bgr(variant=0)
        img_b = _solid(200)
        p_a = self._prep(img_a)
        p_b = self._prep(img_b)
        metrics = _extractor.compute(p_a, p_b)
        # These images are clearly different
        assert metrics.ssim_score < 0.9
        assert metrics.edge_difference > 0.0

    def test_sift_ratio_in_bounds(self):
        img = _gradient_bgr()
        p = self._prep(img)
        metrics = _extractor.compute(p, p)
        assert 0.0 <= metrics.sift_match_ratio <= 1.0

    def test_shape_difference_identical(self):
        img = _gradient_bgr()
        p = self._prep(img)
        metrics = _extractor.compute(p, p)
        assert metrics.shape_difference < 0.05

    def test_histogram_difference_identical(self):
        img = _gradient_bgr()
        p = self._prep(img)
        metrics = _extractor.compute(p, p)
        assert metrics.histogram_difference < 0.05



# ---------------------------------------------------------------------------
# Reference Aggregation tests
# ---------------------------------------------------------------------------
class TestAggregation:
    def _make_comparisons(self):
        return [
            {
                "reference_id": "reference_1",
                "cosine_similarity": 0.90,
                "sift_match_ratio": 0.70,
                "ssim_score": 0.85,
                "edge_difference": 0.10,
                "histogram_difference": 0.15,
                "shape_difference": 0.05,
            },
            {
                "reference_id": "reference_2",
                "cosine_similarity": 0.60,
                "sift_match_ratio": 0.40,
                "ssim_score": 0.55,
                "edge_difference": 0.35,
                "histogram_difference": 0.40,
                "shape_difference": 0.30,
            },
        ]

    def test_best_match_selects_best(self):
        comps = self._make_comparisons()
        result = aggregate(comps, method="best_match")
        # Best match by ssim_score should be reference_1
        assert result["ssim_score"] == pytest.approx(0.85)

    def test_mean_aggregation(self):
        comps = self._make_comparisons()
        result = aggregate(comps, method="mean")
        assert result["ssim_score"] == pytest.approx((0.85 + 0.55) / 2, abs=1e-4)

    def test_median_aggregation(self):
        comps = self._make_comparisons()
        result = aggregate(comps, method="median")
        assert result["ssim_score"] == pytest.approx((0.85 + 0.55) / 2, abs=1e-4)

    def test_all_feature_keys_present(self):
        comps = self._make_comparisons()
        result = aggregate(comps, method="mean")
        for key in CLASSIFIER_FEATURE_ORDER:
            assert key in result

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate([])


# ---------------------------------------------------------------------------
# Classifier tests (heuristic mode)
# ---------------------------------------------------------------------------
class TestClassifier:
    def _classifier(self) -> TamperingClassifier:
        # Always returns heuristic instance (no .pkl in tests)
        return TamperingClassifier()

    def _low_risk_metrics(self) -> dict:
        return {
            "cosine_similarity": 0.97,
            "sift_match_ratio": 0.85,
            "ssim_score": 0.96,
            "edge_difference": 0.02,
            "histogram_difference": 0.03,
            "shape_difference": 0.01,
        }

    def _high_risk_metrics(self) -> dict:
        return {
            "cosine_similarity": 0.10,
            "sift_match_ratio": 0.05,
            "ssim_score": 0.08,
            "edge_difference": 0.90,
            "histogram_difference": 0.85,
            "shape_difference": 0.80,
        }

    def test_low_risk_classification(self):
        clf = self._classifier()
        score, risk = clf.predict(self._low_risk_metrics())
        assert risk == "LOW"
        assert 0.0 <= score <= 1.0

    def test_high_risk_classification(self):
        clf = self._classifier()
        score, risk = clf.predict(self._high_risk_metrics())
        assert risk == "HIGH"
        assert score > 0.5

    def test_score_in_bounds(self):
        clf = self._classifier()
        for metrics in [self._low_risk_metrics(), self._high_risk_metrics()]:
            score, _ = clf.predict(metrics)
            assert 0.0 <= score <= 1.0

    def test_risk_class_valid(self):
        from app.config.settings import RISK_CLASSES
        clf = self._classifier()
        _, risk = clf.predict(self._low_risk_metrics())
        assert risk in RISK_CLASSES
