"""
Tests for the Image Quality Service.
"""
import numpy as np
import pytest

from app.services.image_quality import check_quality
from app.config.settings import QUALITY_THRESHOLDS


def _make_bgr(width: int, height: int, mean: int = 128, std: int = 50) -> np.ndarray:
    """Create a synthetic BGR image with controlled statistics."""
    rng = np.random.default_rng(42)
    gray_vals = np.clip(rng.normal(mean, std, (height, width)), 0, 255).astype(np.uint8)
    bgr = np.stack([gray_vals, gray_vals, gray_vals], axis=-1)
    return bgr


def _make_sharp_bgr(width: int = 512, height: int = 512) -> np.ndarray:
    """Create a high-contrast checkerboard (high Laplacian variance = sharp)."""
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    tile = 16
    for y in range(height):
        for x in range(width):
            if (x // tile + y // tile) % 2 == 0:
                bgr[y, x] = 255
    return bgr


def _make_blurry_bgr(width: int = 512, height: int = 512) -> np.ndarray:
    """Constant-value image = zero Laplacian variance."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Resolution tests
# ---------------------------------------------------------------------------
class TestResolution:
    def test_resolution_pass(self):
        bgr = _make_sharp_bgr(640, 480)
        passed, metrics, failed = check_quality(bgr)
        assert metrics.resolution.passed is True
        assert metrics.resolution.width == 640
        assert metrics.resolution.height == 480

    def test_resolution_fail_too_small(self):
        bgr = _make_sharp_bgr(100, 100)
        passed, metrics, failed = check_quality(bgr)
        assert metrics.resolution.passed is False
        assert any(m.metric == "resolution" for m in failed)


# ---------------------------------------------------------------------------
# Sharpness tests
# ---------------------------------------------------------------------------
class TestSharpness:
    def test_sharp_image_passes(self):
        bgr = _make_sharp_bgr()
        passed, metrics, failed = check_quality(bgr)
        assert metrics.sharpness.passed is True

    def test_blurry_image_fails(self):
        bgr = _make_blurry_bgr()
        passed, metrics, failed = check_quality(bgr)
        assert metrics.sharpness.passed is False
        sharpness_failures = [m for m in failed if m.metric == "sharpness"]
        assert len(sharpness_failures) == 1
        assert sharpness_failures[0].value < QUALITY_THRESHOLDS["min_sharpness"]
        assert "blurry" in sharpness_failures[0].message.lower()


# ---------------------------------------------------------------------------
# Brenner Sharpness tests
# ---------------------------------------------------------------------------
class TestBrennerSharpness:
    def test_sharp_image_passes(self):
        bgr = _make_sharp_bgr()
        passed, metrics, failed = check_quality(bgr)
        assert metrics.brenner_sharpness.passed is True
        assert metrics.brenner_sharpness.value >= QUALITY_THRESHOLDS["min_brenner_sharpness"]

    def test_blurry_image_fails(self):
        bgr = _make_blurry_bgr()
        passed, metrics, failed = check_quality(bgr)
        assert metrics.brenner_sharpness.passed is False
        failures = [m for m in failed if m.metric == "brenner_sharpness"]
        assert len(failures) == 1


# ---------------------------------------------------------------------------
# Canny Edge Density tests
# ---------------------------------------------------------------------------
class TestCannyEdgeDensity:
    def test_structured_image_passes(self):
        bgr = _make_sharp_bgr()
        passed, metrics, failed = check_quality(bgr)
        assert metrics.canny_edge_density.passed is True
        assert metrics.canny_edge_density.value >= QUALITY_THRESHOLDS["min_canny_edge_density"]

    def test_flat_image_fails(self):
        bgr = _make_blurry_bgr()
        passed, metrics, failed = check_quality(bgr)
        assert metrics.canny_edge_density.passed is False
        failures = [m for m in failed if m.metric == "canny_edge_density"]
        assert len(failures) == 1


# ---------------------------------------------------------------------------
# Brightness tests
# ---------------------------------------------------------------------------
class TestBrightness:
    def test_normal_brightness_passes(self):
        bgr = _make_bgr(512, 512, mean=128)
        passed, metrics, failed = check_quality(bgr)
        assert metrics.brightness.passed is True

    def test_too_dark_fails(self):
        bgr = _make_bgr(512, 512, mean=5, std=2)
        passed, metrics, failed = check_quality(bgr)
        assert metrics.brightness.passed is False
        dark_failures = [m for m in failed if m.metric == "brightness"]
        assert dark_failures

    def test_overexposed_fails(self):
        bgr = np.full((512, 512, 3), 250, dtype=np.uint8)
        passed, metrics, failed = check_quality(bgr)
        bright_failures = [m for m in failed if m.metric == "brightness"]
        assert bright_failures


# ---------------------------------------------------------------------------
# Contrast tests
# ---------------------------------------------------------------------------
class TestContrast:
    def test_low_contrast_fails(self):
        # Very narrow range of values = near-zero std
        bgr = np.full((512, 512, 3), 128, dtype=np.uint8)
        bgr[:, :, 0] = 129   # tiny variation
        passed, metrics, failed = check_quality(bgr)
        contrast_failures = [m for m in failed if m.metric == "contrast"]
        assert contrast_failures


# ---------------------------------------------------------------------------
# All metrics collected (not stop-at-first-failure)
# ---------------------------------------------------------------------------
class TestCollectsAllFailures:
    def test_multiple_failures_collected(self):
        # Very dark + constant (low contrast + dark brightness)
        bgr = np.full((100, 100, 3), 5, dtype=np.uint8)
        passed, metrics, failed = check_quality(bgr)
        assert passed is False
        # Should have at least resolution + brightness + contrast + sharpness failures
        assert len(failed) >= 3

    def test_structure_of_failed_metric(self):
        bgr = _make_blurry_bgr()
        _, _, failed = check_quality(bgr)
        for m in failed:
            assert hasattr(m, "metric")
            assert hasattr(m, "value")
            assert hasattr(m, "passed")
            assert m.passed is False
